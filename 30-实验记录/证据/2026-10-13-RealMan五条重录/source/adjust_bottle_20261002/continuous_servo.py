"""Independent simulation-only continuous SDK6 hand trajectory helper.

No import of SAPIEN, CuRobo, native_robot, ROS or robot network interfaces.
Direct integer SDK APIs remain unchanged. The optional drive writer only uses
already configured articulation drives; it does not change gains/effort/axes,
create tendons, set qpos, step a scene, or communicate with physical hardware.

The gap template is simulation-tuned geometry, not an SDK/TCP/force calibration.
A commanded velocity bound does not prove actual drive stability: audit both
returned qvel and position finite differences in a separate physics trace.
"""
from __future__ import annotations

import math
import numbers
from typing import Mapping, Sequence
import numpy as np

READY_NEAR_SDK6 = (561., 617., 627., 619., 927., 0.)
OPEN_SDK6 = (1000., 1000., 1000., 1000., 1000., 0.)
GAP_MM_PER_SDK_UNIT = (.114, .175, .194, .177, .0852)


def _finite_vector(values, size, label):
    raw = np.asarray(values, dtype=object)
    if raw.shape != (size,):
        raise ValueError(f"{label} must have shape ({size},)")
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Real)
           or not math.isfinite(float(v)) for v in raw):
        raise ValueError(f"{label} must contain finite real numbers, not booleans")
    return np.asarray(raw, dtype=np.float64)


def quintic(u):
    """Normalized C2 profile and exact derivative with respect to u."""
    if not math.isfinite(float(u)) or not 0. <= u <= 1.:
        raise ValueError("quintic phase must be in [0,1]")
    u = float(u)
    # Evaluate the complement near u=1 to keep SDK endpoints in bounds under
    # floating-point roundoff; this is the same polynomial, not qpos clipping.
    w = u if u <= .5 else 1.-u
    h = w**3*(10.-15.*w+6.*w*w)
    if u > .5:h = 1.-h
    return h, 30*u*u*(1-u)*(1-u)


class ContinuousHandMapping:
    """Unrounded version of the existing model-limit SDK convention.

    q = lower + (1 - SDK/1000) * (upper-lower); SDK is float [0,1000].
    Mimic position is a*q_parent+b, velocity is a*qdot_parent. This is a
    model mapping with vendor direction; no encoder calibration is implied.
    """
    def __init__(self, metadata: Mapping):
        self.channels = tuple(metadata['channels'])
        self.limits = {n: dict(lim) for n, lim in metadata['limits'].items()}
        self.mimic = {n: dict(m) for n, m in metadata['mimic'].items()}
        if len(self.channels) != 6 or len(set(self.channels)) != 6:
            raise ValueError("Expected six distinct primary SDK channels")
        if set(self.channels) & set(self.mimic):
            raise ValueError("A primary joint cannot also be a mimic follower")
        known = set(self.channels); self.mimic_order = []
        pending = dict(self.mimic)
        while pending:
            ready = [n for n, m in pending.items() if m['joint'] in known]
            if not ready:
                raise ValueError("Cyclic or missing mimic dependency")
            for n in ready:
                m = pending.pop(n)
                a, b = float(m.get('multiplier', 1.)), float(m.get('offset', 0.))
                if not math.isfinite(a) or not math.isfinite(b):
                    raise ValueError("Mimic coefficients must be finite")
                self.mimic_order.append((n, m['joint'], a, b)); known.add(n)
        if known != set(self.limits):
            raise ValueError("Hand limits must cover exactly all primary and mimic joints")
        self.joint_names = tuple(self.limits)
        self.followers = tuple(n for n in self.joint_names if n not in self.channels)
        for n, lim in self.limits.items():
            lo, hi, speed = float(lim['lower']), float(lim['upper']), float(lim['velocity'])
            if not all(math.isfinite(x) for x in (lo, hi, speed)) or lo >= hi or speed <= 0.:
                raise ValueError("Invalid position/velocity limits for " + n)
            lim.update(lower=lo, upper=hi, velocity=speed)
        self.lower = np.asarray([self.limits[n]['lower'] for n in self.channels])
        self.width = np.asarray([self.limits[n]['upper']-self.limits[n]['lower'] for n in self.channels])

    def state(self, sdk6, sdk_rate_units_s=None):
        sdk = _finite_vector(sdk6, 6, 'SDK6')
        if np.any(sdk < 0.) or np.any(sdk > 1000.):
            raise ValueError("Continuous SDK6 must be in [0,1000]; -1 is not a waypoint")
        rate = np.zeros(6) if sdk_rate_units_s is None else _finite_vector(sdk_rate_units_s, 6, 'SDK rate')
        pos = dict(zip(self.channels, self.lower+(1.-sdk/1000.)*self.width))
        vel = dict(zip(self.channels, -rate*self.width/1000.))
        for name, parent, a, b in self.mimic_order:
            pos[name] = a*pos[parent]+b
            vel[name] = a*vel[parent]
        self.validate(pos, vel)
        return ({n: float(pos[n]) for n in self.joint_names},
                {n: float(vel[n]) for n in self.joint_names})

    def validate(self, positions, velocities, *, position_tolerance_rad=1e-10,
                 check_velocity=True, velocity_tolerance_rad_s=1e-10):
        if set(positions) != set(self.joint_names) or set(velocities) != set(self.joint_names):
            raise ValueError("Position and velocity must cover the complete hand")
        for n in self.joint_names:
            q, qd, lim = float(positions[n]), float(velocities[n]), self.limits[n]
            if not math.isfinite(q) or not math.isfinite(qd):
                raise ValueError("Nonfinite hand state " + n)
            if q < lim['lower']-position_tolerance_rad or q > lim['upper']+position_tolerance_rad:
                raise ValueError("Continuous commanded position exceeds " + n + " limit")
            if check_velocity and abs(qd) > lim['velocity']+velocity_tolerance_rad_s:
                raise ValueError("Continuous commanded velocity exceeds " + n + " limit")
        for n, parent, a, b in self.mimic_order:
            if abs(float(positions[n])-a*float(positions[parent])-b) > position_tolerance_rad:
                raise ValueError("Mimic position mismatch " + n)
            if abs(float(velocities[n])-a*float(velocities[parent])) > velocity_tolerance_rad_s:
                raise ValueError("Mimic velocity mismatch " + n)

    def minimum_duration_s(self, start_sdk6, end_sdk6, *, velocity_fraction=1.):
        if not math.isfinite(velocity_fraction) or not 0. < velocity_fraction <= 1.:
            raise ValueError("velocity_fraction must be in (0,1]")
        p0, _ = self.state(start_sdk6); p1, _ = self.state(end_sdk6)
        spans = {n: abs(p1[n]-p0[n])/(self.limits[n]['velocity']*velocity_fraction)
                 for n in self.joint_names}
        worst = max(spans, key=spans.get)
        return 1.875*spans[worst], worst


class SDKWaypointProfile:
    """Serial quintic SDK6 waypoints, zero speed/acceleration at every waypoint.

    Duration requests are lengthened if needed for every primary and follower
    velocity limit. Rows share segment boundary endpoints. sample(i) remains at
    the final target with zero velocity after motion ends, including padding.
    """
    def __init__(self, mapping: ContinuousHandMapping, waypoints: Sequence,
                 *, dt=.004, segment_duration_s=None, velocity_fraction=1.,
                 post_padding_steps=0, annotations=None):
        if isinstance(dt, bool) or not math.isfinite(dt) or dt <= 0.:
            raise ValueError("dt must be a positive finite physics interval")
        if isinstance(post_padding_steps, bool) or int(post_padding_steps) != post_padding_steps or post_padding_steps < 0:
            raise ValueError("post_padding_steps must be a nonnegative integer")
        if not math.isfinite(velocity_fraction) or not 0. < velocity_fraction <= 1.:
            raise ValueError("velocity_fraction must be in (0,1]")
        self.mapping, self.dt = mapping, float(dt)
        self.velocity_fraction = float(velocity_fraction)
        self.waypoints = tuple(_finite_vector(w, 6, 'Waypoint SDK6').copy() for w in waypoints)
        if len(self.waypoints) < 2:
            raise ValueError("At least two SDK6 waypoints are required")
        for w in self.waypoints: mapping.state(w)
        count = len(self.waypoints)-1
        durations = [0.]*count if segment_duration_s is None else _finite_vector(segment_duration_s, count, 'Segment durations')
        if np.any(np.asarray(durations) < 0.):raise ValueError("Requested duration cannot be negative")
        self.segments = []; last_step = 0
        for index, (start, end) in enumerate(zip(self.waypoints[:-1], self.waypoints[1:])):
            minimum, worst = mapping.minimum_duration_s(start, end, velocity_fraction=velocity_fraction)
            requested = float(durations[index]); intervals = max(1, int(math.ceil(max(minimum, requested)/self.dt)))
            duration = intervals*self.dt
            p0, _ = mapping.state(start); p1, _ = mapping.state(end)
            peaks = {n: 1.875*abs(p1[n]-p0[n])/duration for n in mapping.joint_names}
            self.segments.append({'index': index, 'start_step': last_step,
                                  'end_step': last_step+intervals, 'intervals': intervals,
                                  'duration_s': duration, 'requested_duration_s': requested,
                                  'minimum_duration_s': minimum, 'limiting_joint': worst,
                                  'peak_joint_velocity_rad_s': peaks,
                                  'start_sdk6': start.tolist(), 'end_sdk6': end.tolist()})
            last_step += intervals
        self.motion_steps = last_step+1
        self.padding_steps = int(post_padding_steps)
        self.num_steps = self.motion_steps+self.padding_steps
        self.annotations = dict(annotations or {})

    def sample(self, index):
        if isinstance(index, bool) or int(index) != index or index < 0:
            raise ValueError("Sample index must be a nonnegative integer")
        index = int(index)
        if index >= self.motion_steps-1:
            sdk = self.waypoints[-1].copy(); rate = np.zeros(6)
            seg_index = len(self.segments)-1; u = 1.
        else:
            seg = next(s for s in self.segments if index <= s['end_step'])
            seg_index = seg['index']; u = (index-seg['start_step'])/seg['intervals']
            h, dh = quintic(u)
            start, end = self.waypoints[seg_index:seg_index+2]
            sdk = start+(end-start)*h
            rate = (end-start)*dh/seg['duration_s']
        positions, velocities = self.mapping.state(sdk, rate)
        max_ratio = max(abs(velocities[n])/self.mapping.limits[n]['velocity'] for n in self.mapping.joint_names)
        return {'sample_index': index, 'nominal_time_s': index*self.dt,
                'segment_index': seg_index, 'segment_phase': float(u),
                'padding': index >= self.motion_steps, 'motion_complete': index >= self.motion_steps-1,
                'sdk6': sdk.tolist(), 'sdk_rate_units_s': rate.tolist(),
                'joint_position_rad': positions, 'joint_velocity_rad_s': velocities,
                'primary_velocity_rad_s': {n: velocities[n] for n in self.mapping.channels},
                'max_joint_velocity_ratio': float(max_ratio)}

    def arrays(self):
        samples = [self.sample(i) for i in range(self.num_steps)]
        names = self.mapping.joint_names
        return {'joint_names': names,
                'time_s': np.asarray([s['nominal_time_s'] for s in samples]),
                'sdk6': np.asarray([s['sdk6'] for s in samples]),
                'sdk_rate_units_s': np.asarray([s['sdk_rate_units_s'] for s in samples]),
                'position': np.asarray([[s['joint_position_rad'][n] for n in names] for s in samples]),
                'velocity': np.asarray([[s['joint_velocity_rad_s'][n] for n in names] for s in samples])}

    def metadata(self):
        return {'scope': 'simulation_only_continuous_model_sdk_mapping',
                'sdk_mapping_calibrated': False, 'dt_s': self.dt,
                'motion_steps': self.motion_steps, 'padding_steps': self.padding_steps,
                'num_steps': self.num_steps, 'motion_duration_s': (self.motion_steps-1)*self.dt,
                'velocity_fraction': self.velocity_fraction, 'primary_joints': list(self.mapping.channels),
                'all_joint_names': list(self.mapping.joint_names), 'segments': self.segments,
                'annotations': self.annotations,
                'warning': 'Commanded speed bounds do not validate actual physics stability; log qpos finite differences separately from returned qvel.'}


def gap_closure_sdk6(ready_near_sdk6, common_gap_mm, *, mm_per_sdk_unit=GAP_MM_PER_SDK_UNIT):
    """Decrease five flexion SDK targets by a common nominal gap displacement.

    Rates are local simulation geometry estimates in mm/SDK unit, not calibrated
    aperture, force or physical encoder conversion. Rotation channel is held.
    """
    near = _finite_vector(ready_near_sdk6, 6, 'Ready-near SDK6')
    rates = _finite_vector(mm_per_sdk_unit, 5, 'Simulation gap rates')
    if np.any(near < 0.) or np.any(near > 1000.) or np.any(rates <= 0.):
        raise ValueError("SDK6/rates outside the gap template domain")
    if isinstance(common_gap_mm, bool) or not math.isfinite(common_gap_mm) or common_gap_mm < 0.:
        raise ValueError("common_gap_mm must be finite and nonnegative")
    end = near.copy(); end[:5] -= common_gap_mm/rates
    if np.any(end < 0.) or np.any(end > 1000.):
        raise ValueError("Gap closure would exceed SDK limits; choose a smaller gap, do not clip")
    return end


def two_stage_gap_profile(metadata, *, dt=.004, start_sdk6=OPEN_SDK6,
                          ready_near_sdk6=READY_NEAR_SDK6, common_gap_mm=1.,
                          mm_per_sdk_unit=GAP_MM_PER_SDK_UNIT,
                          ready_near_duration_s=3., gap_peak_rate_mm_s=.5,
                          velocity_fraction=1., post_padding_steps=125):
    """Open->ready-near, then a slow nominal common-gap closure template.

    Callers must validate exact mesh clearance/contact phase and actual dynamics.
    This template supplies command waypoints only; it does not close a force loop.
    """
    if isinstance(gap_peak_rate_mm_s, bool) or not math.isfinite(gap_peak_rate_mm_s) or gap_peak_rate_mm_s <= 0.:
        raise ValueError("gap_peak_rate_mm_s must be finite and positive")
    mapping = metadata if isinstance(metadata, ContinuousHandMapping) else ContinuousHandMapping(metadata)
    end = gap_closure_sdk6(ready_near_sdk6, common_gap_mm, mm_per_sdk_unit=mm_per_sdk_unit)
    gap_duration = 1.875*common_gap_mm/gap_peak_rate_mm_s
    return SDKWaypointProfile(mapping, [start_sdk6, ready_near_sdk6, end], dt=dt,
                              segment_duration_s=[ready_near_duration_s, gap_duration],
                              velocity_fraction=velocity_fraction, post_padding_steps=post_padding_steps,
                              annotations={'template': 'ready_near_then_common_gap',
                                           'gap_rates_status': 'simulation_tuned_not_calibrated',
                                           'gap_mm_per_sdk_unit': list(mm_per_sdk_unit),
                                           'nominal_common_gap_mm': float(common_gap_mm),
                                           'nominal_peak_gap_rate_mm_s': float(gap_peak_rate_mm_s)})


def apply_servo_sample(robot, side, mapping, sample, *, command_scalar=None,
                       apply_passive_force=False):
    """Optional simulation drive writer; call once before a physics scene step.

    Preconditions: physical mimic coupling is installed on this fresh articulation
    and follower drive properties are already zero. This function does not change
    any drive property or caller method. Only primary6 get analytic velocity; all
    follower velocity targets remain zero. Direct SDK integer APIs are untouched.
    A latched/contact-regulated finger must override its primary target and set
    velocity to zero before stepping; do not retain its free-profile feedforward.
    """
    if side not in ('left', 'right'):
        raise ValueError("side must be left/right")
    if tuple(robot._manifest['hands'][side]['channels']) != mapping.channels:
        raise ValueError("Mapping belongs to a different hand")
    positions, velocities = sample['joint_position_rad'], sample['joint_velocity_rad_s']
    mapping.validate(positions, velocities)
    sdk = _finite_vector(sample['sdk6'], 6, 'Sample SDK6')
    expected_pos, expected_vel = mapping.state(sdk, sample['sdk_rate_units_s'])
    if any(abs(positions[n]-expected_pos[n]) > 1e-10 or abs(velocities[n]-expected_vel[n]) > 1e-10
           for n in mapping.joint_names):
        raise ValueError("Sample SDK and full joint state are inconsistent")
    if command_scalar is not None:
        if isinstance(command_scalar, bool) or not isinstance(command_scalar, numbers.Real) or not math.isfinite(command_scalar) or not 0. <= command_scalar <= 1.:
            raise ValueError("Official command scalar must be in [0,1]")
    joints = {n: robot._hand_joints[n] for n in mapping.joint_names}
    for n, joint in joints.items():
        joint.set_drive_target(float(positions[n]))
        joint.set_drive_velocity_target(float(velocities[n]) if n in mapping.channels else 0.)
    robot._hand_sdk_target[side] = sdk.tolist()
    if command_scalar is not None:setattr(robot, side+'_gripper_val', float(command_scalar))
    if apply_passive_force:robot._entity_qf(getattr(robot, side+'_entity'))


def run_cpu_tests(metadata=None):
    """Pure NumPy/CPU formula, bounds, padding, mapping and mock-drive checks."""
    from types import SimpleNamespace
    fake = {'channels': ['p'+str(i) for i in range(6)],
            'limits': {('p'+str(i)): {'lower': 0., 'upper': 1., 'velocity': 1.} for i in range(6)},
            'mimic': {'f1': {'joint': 'p0', 'multiplier': 1.1, 'offset': .1},
                      'f2': {'joint': 'f1', 'multiplier': -.5, 'offset': .65}}}
    fake['limits'].update(f1={'lower': .1, 'upper': 1.3, 'velocity': .8},
                          f2={'lower': 0., 'upper': .7, 'velocity': .3})
    mp = ContinuousHandMapping(fake); checks = []
    q, v = mp.state([100.25, 200.5, 300.75, 400., 500., 600.], [10, 0, 0, 0, 0, 0])
    assert abs(q['p0']-.89975) < 1e-12
    assert abs(q['f1']-1.1*q['p0']-.1) < 1e-12 and abs(q['f2']+.5*q['f1']-.65) < 1e-12
    assert abs(v['f2']+.5*v['f1']) < 1e-12; checks.append('float_mapping_nested_mimic_position_velocity')
    profile = SDKWaypointProfile(mp, [[1000]*6, [900.25, 800.5, 700.75, 600., 500., 400.], [1000]*6],
                                 dt=.004, segment_duration_s=[.1, .1], velocity_fraction=.5, post_padding_steps=23)
    arr = profile.arrays(); ratios = np.abs(arr['velocity'])/np.array([mp.limits[n]['velocity'] for n in mp.joint_names])
    assert ratios.max() <= .5+1e-12
    for idx in [0, *(s['end_step'] for s in profile.segments), profile.motion_steps, profile.num_steps+100]:
        assert all(x == 0. for x in profile.sample(idx)['joint_velocity_rad_s'].values())
    assert np.allclose(profile.sample(profile.num_steps+100)['sdk6'], [1000]*6)
    assert all(s['duration_s'] >= s['minimum_duration_s'] for s in profile.segments)
    checks.append('all_joint_speed_retiming_segment_endpoints_padding_zero')
    points = np.linspace(.01, .99, 100); eps = 1e-6
    h = lambda z: 10*z**3-15*z**4+6*z**5
    numeric = (h(points+eps)-h(points-eps))/(2*eps)
    analytic = 30*points**2*(1-points)**2
    for edge in [0., np.nextafter(0., 1.), 1e-12, .5, 1.-1e-12, np.nextafter(1., 0.), 1.]:
        hh, dd = quintic(edge)
        assert 0. <= hh <= 1. and dd >= 0.
    derivative_error = float(np.max(np.abs(numeric-analytic)))
    assert derivative_error < 1e-8; checks.append('analytic_quintic_derivative_vs_continuous_finite_difference')
    template = two_stage_gap_profile(mp, common_gap_mm=.5)
    start, near, end = template.waypoints
    assert np.array_equal(near, np.array(READY_NEAR_SDK6))
    assert np.allclose((near[:5]-end[:5])*np.array(GAP_MM_PER_SDK_UNIT), .5)
    assert end[5] == near[5]; checks.append('two_stage_common_gap_float_template')
    for bad in ([-1]*6, [1001]*6, [float('nan')]*6, [True]*6):
        try:mp.state(bad)
        except ValueError:pass
        else:raise AssertionError('invalid SDK accepted')
    try:gap_closure_sdk6([0]*6, 1.)
    except ValueError:pass
    else:raise AssertionError('out-of-range gap silently clipped')
    checks.append('invalid_sdk_and_gap_rejected_without_clipping')
    class Joint:
        def __init__(self):self.position = None;self.velocity = None;self.properties = (5., .1, 10.)
        def set_drive_target(self, value):self.position = value
        def set_drive_velocity_target(self, value):self.velocity = value
    robot = SimpleNamespace(_manifest={'hands': {'left': fake}},
                            _hand_joints={n: Joint() for n in mp.joint_names},
                            _hand_sdk_target={'left': None}, left_gripper_val=1.)
    direct_api = lambda *a: None; robot.set_hand_angle = direct_api
    sample = profile.sample(profile.segments[0]['intervals']//2)
    apply_servo_sample(robot, 'left', mp, sample, command_scalar=.7)
    assert robot.set_hand_angle is direct_api and robot.left_gripper_val == .7
    assert all(robot._hand_joints[n].properties == (5., .1, 10.) for n in mp.joint_names)
    assert all(robot._hand_joints[n].velocity == 0. for n in mp.followers)
    assert any(robot._hand_joints[n].velocity != 0. for n in mp.channels)
    checks.append('mock_drive_primary_analytic_followers_zero_sdk_api_properties_unchanged')
    result = {'passed': True, 'checks': checks, 'analytic_derivative_max_error': derivative_error}
    if metadata is not None:
        native = ContinuousHandMapping(metadata)
        actual = two_stage_gap_profile(native, common_gap_mm=1.)
        arrays = actual.arrays()
        ratios = np.abs(arrays['velocity'])/np.array([native.limits[n]['velocity'] for n in native.joint_names])
        assert ratios.max() <= 1.+1e-12
        for s in actual.segments:
            assert max(s['peak_joint_velocity_rad_s'][n]/native.limits[n]['velocity'] for n in native.joint_names) <= 1.+1e-12
        assert np.all(arrays['velocity'][actual.motion_steps:] == 0.)
        result['native_template'] = actual.metadata()
        result['native_max_sampled_speed_ratio'] = float(ratios.max())
        result['checks'].append('native_gen4_all_joint_limits_mimic_speeds_padding')
    return result


if __name__ == '__main__':
    import argparse, json
    from pathlib import Path
    parser = argparse.ArgumentParser(description='Pure CPU continuous hand servo formula tests')
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--side', choices=('left', 'right'), default='left')
    args = parser.parse_args()
    meta = json.loads(args.manifest.read_text())['hands'][args.side] if args.manifest else None
    print(json.dumps(run_cpu_tests(meta), ensure_ascii=False, indent=2))
