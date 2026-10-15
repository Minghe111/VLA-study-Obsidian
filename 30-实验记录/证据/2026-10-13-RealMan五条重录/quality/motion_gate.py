"""Independent read-only gate for five re-recorded upright expert scenes.

Numerical motion quality is distinct from actual collision/physical success.
No source arrays, simulated states, robot, model, or recording are modified.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

DEG = 180. / np.pi
ARM_BASE = {'left': 0, 'right': 12}
CONFIG = Path(__file__).with_name('thresholds_confirmed.json')
DENSE_BEFORE = 'realman_dexterous_physics/measured_state24_before'
DENSE_AFTER = 'realman_dexterous_physics/measured_state24_after'
DENSE_COMMAND = 'realman_dexterous_physics/applied_command24'
TIME_BEFORE = 'realman_dexterous_physics/sim_time_before_s'
TIME_AFTER = 'realman_dexterous_physics/sim_time_after_s'
LOW_STATE = 'realman_dexterous/measured_state24'
LOW_COMMAND = 'realman_dexterous/next_command24'
LOW_TIME = 'realman_dexterous/sample_time_s'
NEXT_TIME = 'realman_dexterous/next_sample_time_s'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_json(value):
    if isinstance(value, dict):
        return value
    if value is None:
        return None
    return json.loads(Path(value).read_text())


def load_numeric(path, arm):
    path = Path(path)
    keys = [DENSE_BEFORE, DENSE_AFTER, DENSE_COMMAND, TIME_BEFORE, TIME_AFTER,
            LOW_STATE, LOW_COMMAND, LOW_TIME, NEXT_TIME,
            f'vision/cam_{arm}_wrist/extrinsics_matrix', f'state/{arm}_ee_poses']
    if path.suffix == '.npz':
        with np.load(path, allow_pickle=False) as f:
            return {key: f[key] for key in keys if key in f}
    import h5py
    with h5py.File(path, 'r') as f:
        return {key: f[key][...] for key in keys if key in f}


def angle_metric(values, near2pi_tolerance=.1):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6 or len(values) == 0:
        raise ValueError('Six ordered arm joint channels and nonempty trajectory required')
    unwrapped = np.unwrap(values, axis=0)
    diff = np.diff(values, axis=0)
    udiff = np.diff(unwrapped, axis=0)
    return {
        'samples': len(values),
        'per_joint_raw_range_deg': (np.ptp(values, axis=0) * DEG).tolist(),
        'per_joint_unwrapped_range_deg': (np.ptp(unwrapped, axis=0) * DEG).tolist(),
        'per_joint_net_deg': ((unwrapped[-1] - unwrapped[0]) * DEG).tolist(),
        'per_joint_total_variation_deg': (np.abs(udiff).sum(axis=0) * DEG).tolist(),
        'per_joint_maximum_raw_step_deg': (np.abs(diff).max(axis=0) * DEG).tolist() if len(diff) else [0.] * 6,
        'maximum_range_deg': float(np.ptp(unwrapped, axis=0).max() * DEG),
        'maximum_total_variation_deg': float(np.abs(udiff).sum(axis=0).max() * DEG),
        'maximum_raw_step_deg': float(np.abs(diff).max() * DEG) if len(diff) else 0.,
        'raw_steps_over_pi': int(np.sum(np.abs(diff) > np.pi)),
        'near2pi_jump_count': int(np.sum((np.abs(np.abs(diff) - 2 * np.pi) <= near2pi_tolerance)
                                       & (np.abs(diff) > np.pi))),
        'unwrapping_adjusted_steps': int(np.sum(np.abs(diff - udiff) > 1e-6)),
        'six_joint_path_lengths_were_not_summed': True,
    }


def path_metric(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return None
    steps = np.linalg.norm(np.diff(values, axis=0), axis=1)
    length = float(steps.sum())
    displacement = float(np.linalg.norm(values[-1] - values[0]))
    return {'samples': len(values), 'path_m': length, 'straight_displacement_m': displacement,
            'path_to_straight_ratio': length / displacement if displacement > 1e-6 else None,
            'maximum_sample_step_m': float(steps.max()) if len(steps) else 0.,
            'start_m': values[0].tolist(), 'end_m': values[-1].tolist(),
            'shorter_path_feasibility_was_not_inferred': True}


def rotation_integrity(matrices):
    rotations = np.asarray(matrices, dtype=np.float64)[:, :3, :3]
    error = np.swapaxes(rotations, 1, 2) @ rotations - np.eye(3)[None]
    determinant = np.linalg.det(rotations)
    return {'maximum_orthogonality_residual': float(np.abs(error).max()),
            'maximum_determinant_minus_one': float(np.abs(determinant - 1).max())}


def first_closure(applied, times, base, threshold):
    hits = np.flatnonzero(np.any(applied[:, base + 6:base + 11] < threshold, axis=1))
    if not len(hits):
        return {'found': False, 'time_s': None, 'row': None, 'initially_closed': False}
    row = int(hits[0])
    return {'found': True, 'time_s': float(times[row]), 'row': row, 'initially_closed': row == 0,
            'threshold_sdk': threshold, 'thumb_rotation_excluded': True,
            'definition_is_not_actual_contact': True}


def numerical_motion(arrays, metadata, cfg, actual_first_target_contact_time_s=None):
    """Motion-only evaluation. Caller separately enforces scope and other gates."""
    arm = metadata['arm']
    base = ARM_BASE[arm]
    required = [DENSE_BEFORE, DENSE_AFTER, DENSE_COMMAND, TIME_BEFORE, TIME_AFTER,
                LOW_STATE, LOW_COMMAND, LOW_TIME, NEXT_TIME,
                f'vision/cam_{arm}_wrist/extrinsics_matrix']
    missing = [key for key in required if key not in arrays]
    if missing:
        return {'passed': False, 'assessable': False, 'failures': ['required_numeric_fields_missing'], 'missing_fields': missing}
    if not all(np.isfinite(arrays[key]).all() for key in required):
        return {'passed': False, 'assessable': False, 'failures': ['nonfinite_numeric_values']}
    before, after, command = [np.asarray(arrays[key], dtype=np.float64) for key in (DENSE_BEFORE, DENSE_AFTER, DENSE_COMMAND)]
    tb, ta, tl, tn = [np.asarray(arrays[key], dtype=np.float64) for key in (TIME_BEFORE, TIME_AFTER, LOW_TIME, NEXT_TIME)]
    lowstate, lowcommand = [np.asarray(arrays[key], dtype=np.float64) for key in (LOW_STATE, LOW_COMMAND)]
    wrist = np.asarray(arrays[f'vision/cam_{arm}_wrist/extrinsics_matrix'], dtype=np.float64)
    if (before.ndim != 2 or before.shape[1] != 24 or before.shape != after.shape or before.shape != command.shape
            or len(tb) != len(before) or len(ta) != len(before) or len(before) < 2
            or lowstate.shape != lowcommand.shape or lowstate.ndim != 2 or lowstate.shape[1] != 24
            or len(tl) != len(lowstate) or len(tn) != len(tl) or len(tl) < 2
            or wrist.shape != (len(tl), 4, 4)):
        return {'passed': False, 'assessable': False, 'failures': ['numeric_shapes_inconsistent']}
    failures, checks = [], []
    def check(name, passed, value=None, limit=None, category='motion'):
        checks.append({'name': name, 'passed': bool(passed), 'value': value, 'limit': limit, 'category': category})
        if not passed:
            failures.append(name)
    check('physical_clock_contiguous', np.all(np.diff(tb) > 0) and
          np.max(np.abs(np.diff(tb) - cfg['physics_dt_s'])) <= cfg['clock_tolerance_s'] and
          np.max(np.abs(ta - tb - cfg['physics_dt_s'])) <= cfg['clock_tolerance_s'], category='integrity')
    check('observation_clock_contiguous', np.all(np.diff(tl) > 0) and
          np.max(np.abs(np.diff(tl) - cfg['observation_dt_s'])) <= cfg['clock_tolerance_s'] and
          np.max(np.abs(tn - tl - cfg['observation_dt_s'])) <= cfg['clock_tolerance_s'], category='integrity')
    chain = float(np.abs(before[1:] - after[:-1]).max())
    check('measured_state_chain_contiguous', chain <= cfg['state_chain_tolerance'], chain, cfg['state_chain_tolerance'], 'integrity')
    dense = np.vstack((before[:1], after))
    td = np.r_[tb[:1], ta]
    idx = np.clip(np.searchsorted(tb, tl), 0, len(tb) - 1)
    prev = np.clip(idx - 1, 0, len(tb) - 1)
    idx = np.where(np.abs(tb[prev] - tl) < np.abs(tb[idx] - tl), prev, idx)
    alignment = float(np.abs(lowstate - before[idx]).max())
    check('low_measured_matches_physical_state', alignment <= cfg['state_chain_tolerance'], alignment, cfg['state_chain_tolerance'], 'integrity')
    check('low_observation_timestamp_matches_physical_clock', float(np.abs(tb[idx] - tl).max()) <= cfg['clock_tolerance_s'], category='integrity')
    closure = first_closure(command, tb, base, cfg['approach_boundary']['threshold_sdk'])
    check('substantial_commanded_closure_observed', closure['found'])
    check('recording_did_not_start_already_closed', not closure['initially_closed'])
    full_actual = angle_metric(dense[:, base:base + 6], cfg['near2pi_jump_tolerance_rad'])
    full_command = angle_metric(np.vstack((before[:1, base:base + 6], command[:, base:base + 6])), cfg['near2pi_jump_tolerance_rad'])
    full_lowcommand = angle_metric(np.vstack((lowstate[:1, base:base + 6], lowcommand[:, base:base + 6])), cfg['near2pi_jump_tolerance_rad'])
    for name, metric in [('actual', full_actual), ('command', full_command), ('next_command', full_lowcommand)]:
        check(name + '_no_raw_pi_or_2pi_jumps', metric['raw_steps_over_pi'] == 0 and metric['near2pi_jump_count'] == 0,
              {'over_pi': metric['raw_steps_over_pi'], 'near2pi': metric['near2pi_jump_count']}, 0, 'integrity')
    check('camera_rotation_matrix_valid', rotation_integrity(wrist)['maximum_orthogonality_residual'] < 1e-4 and
          rotation_integrity(wrist)['maximum_determinant_minus_one'] < 1e-4, category='integrity')
    windows = {}
    if closure['found']:
        windows['before_substantial_commanded_closure'] = closure['time_s']
    if actual_first_target_contact_time_s is not None:
        contact = float(actual_first_target_contact_time_s)
        check('actual_contact_timestamp_within_recording', td[0] <= contact <= td[-1], contact, [float(td[0]), float(td[-1])], 'integrity')
        if td[0] <= contact <= td[-1]:
            windows['before_actual_first_target_contact'] = contact
    metrics = {}
    for window, end in windows.items():
        keep_dense, keep_cmd, keep_low, keep_next = [times <= end + 1e-10 for times in (td, tb, tl, tn)]
        check(window + '_nonempty_prefix', keep_dense.sum() >= 2 and keep_cmd.sum() >= 2 and keep_low.sum() >= 2)
        if min(keep_dense.sum(), keep_cmd.sum(), keep_low.sum()) < 2:
            continue
        actual = angle_metric(dense[keep_dense, base:base + 6], cfg['near2pi_jump_tolerance_rad'])
        # Initial physical state is included to avoid omitting a first command branch jump.
        cmd = angle_metric(np.vstack((before[:1, base:base + 6], command[keep_cmd, base:base + 6])), cfg['near2pi_jump_tolerance_rad'])
        low = angle_metric(np.vstack((lowstate[:1, base:base + 6], lowcommand[keep_next, base:base + 6])), cfg['near2pi_jump_tolerance_rad']) if keep_next.any() else None
        cam = path_metric(wrist[keep_low, :3, 3])
        ee = path_metric(np.asarray(arrays[f'state/{arm}_ee_poses'])[keep_low, :3]) if f'state/{arm}_ee_poses' in arrays else None
        metrics[window] = {'end_time_s': float(end), 'last_actual_observation_s': float(tl[keep_low][-1]),
                           'actual_joints': actual, 'command_joints': cmd, 'next_command_joints': low,
                           'wrist_center_world_model_matrix': cam, 'legacy_ee_position_report_only': ee,
                           'legacy_ee_is_not_calibrated_real_tcp': True}
        check(window + '_actual_single_joint_range', actual['maximum_range_deg'] <= cfg['maximum_actual_joint_range_deg'], actual['maximum_range_deg'], cfg['maximum_actual_joint_range_deg'])
        check(window + '_command_single_joint_range', cmd['maximum_range_deg'] <= cfg['maximum_command_joint_range_deg'], cmd['maximum_range_deg'], cfg['maximum_command_joint_range_deg'])
        check(window + '_actual_single_joint_travel', actual['maximum_total_variation_deg'] <= cfg['maximum_actual_joint_total_variation_deg'], actual['maximum_total_variation_deg'], cfg['maximum_actual_joint_total_variation_deg'])
        check(window + '_command_single_joint_travel', cmd['maximum_total_variation_deg'] <= cfg['maximum_command_joint_total_variation_deg'], cmd['maximum_total_variation_deg'], cfg['maximum_command_joint_total_variation_deg'])
        check(window + '_actual4ms_steps', actual['maximum_raw_step_deg'] <= cfg['maximum_actual4ms_joint_step_deg'], actual['maximum_raw_step_deg'], cfg['maximum_actual4ms_joint_step_deg'])
        check(window + '_command4ms_steps', cmd['maximum_raw_step_deg'] <= cfg['maximum_command4ms_joint_step_deg'], cmd['maximum_raw_step_deg'], cfg['maximum_command4ms_joint_step_deg'])
        check(window + '_command60ms_steps', low is not None and low['maximum_raw_step_deg'] <= cfg['maximum_next_command60ms_joint_step_deg'], None if low is None else low['maximum_raw_step_deg'], cfg['maximum_next_command60ms_joint_step_deg'])
        check(window + '_wrist_center_path', cam['path_m'] <= cfg['maximum_wrist_center_path_m'], cam['path_m'], cfg['maximum_wrist_center_path_m'])
        ratio_applicable = cam['straight_displacement_m'] >= cfg['minimum_displacement_for_ratio_gate_m']
        check(window + '_wrist_center_detour', not ratio_applicable or (cam['path_to_straight_ratio'] is not None and cam['path_to_straight_ratio'] <= cfg['maximum_wrist_center_path_to_straight_ratio']),
              {'ratio': cam['path_to_straight_ratio'], 'applicable': ratio_applicable, 'displacement_m': cam['straight_displacement_m']}, cfg['maximum_wrist_center_path_to_straight_ratio'])
    return {'passed': not failures, 'assessable': True, 'failures': failures, 'checks': checks,
            'first_substantial_commanded_closure': closure,
            'actual_contact_timestamp_provided': actual_first_target_contact_time_s is not None,
            'sdk_boundary_does_not_prove_first_physical_contact': True,
            'full_actual_joints_report_only': full_actual, 'full_command_joints_report_only': full_command,
            'metrics': metrics, 'shorter_route_safety_was_not_assumed': True}


def validate_periodic_branch_receipt(receipt, cfg):
    """Checks actual recorded planner candidates algebraically, not path safety."""
    if receipt is None:
        return {'passed': False, 'assessable': False, 'failures': ['actual_planner_branch_receipt_missing']}
    if receipt.get('schema') != 'nearest-periodic-branch-v1' or not receipt.get('events'):
        return {'passed': False, 'assessable': False, 'failures': ['actual_planner_branch_receipt_schema_or_events_missing']}
    failures, events = [], []
    for number, event in enumerate(receipt['events']):
        issue = []
        try:
            arm = event['arm']
            limits = np.asarray(receipt['arm_joint_limits_rad'][arm], dtype=np.float64)
            kinds = receipt['arm_joint_types'][arm]
            raw, reference, chosen = [np.asarray(event[key], dtype=np.float64) for key in ('raw_ik_q_rad', 'reference_q_rad', 'selected_q_rad')]
            if limits.shape != (6, 2) or any(x.shape != (6,) for x in (raw, reference, chosen)):
                raise ValueError('Expected six joint coordinates and limit pairs')
            if not all(np.isfinite(x).all() for x in (limits, raw, reference, chosen)) or not np.all(limits[:, 0] <= limits[:, 1]):
                raise ValueError('Nonfinite/invalid values')
            if kinds != ['revolute'] * 6 and kinds != ['continuous'] * 6:
                raise ValueError('Recorded revolute joint semantics required')
            candidates = []
            options = []
            for j in range(6):
                lo = int(np.ceil((limits[j, 0] - raw[j] - 1e-8) / (2 * np.pi)))
                hi = int(np.floor((limits[j, 1] - raw[j] + 1e-8) / (2 * np.pi)))
                if hi < lo or hi - lo > 8:
                    raise ValueError('Missing/bad legal periodic branch range')
                options.append([raw[j] + 2 * np.pi * k for k in range(lo, hi + 1)])
            if np.prod([len(x) for x in options]) > 4096:
                raise ValueError('Unbounded candidate enumeration')
            for q in itertools.product(*options):
                candidates.append(np.asarray(q))
            candidate_array = np.asarray(candidates)
            scores = np.linalg.norm(candidate_array - reference, axis=1)
            chosen_score = float(np.linalg.norm(chosen - reference))
            minimum_score = float(scores.min())
            periodic_residual = float(np.abs((chosen - raw) - np.rint((chosen - raw) / (2 * np.pi)) * (2 * np.pi)).max())
            if periodic_residual > cfg['branch_periodic_equivalence_tolerance_rad']:
                issue.append('chosen_goal_is_not_periodic_equivalent_to_recorded_raw_ik')
            if np.any(chosen < limits[:, 0] - 1e-8) or np.any(chosen > limits[:, 1] + 1e-8):
                issue.append('chosen_goal_outside_recorded_joint_limits')
            if chosen_score > minimum_score + cfg['branch_selection_distance_tolerance_rad']:
                issue.append('chosen_goal_is_not_nearest_legal_periodic_equivalent')
            recorded_candidates = np.asarray([x['q_rad'] for x in event['candidates']], dtype=np.float64)
            if recorded_candidates.ndim != 2 or recorded_candidates.shape[1] != 6 or not np.isfinite(recorded_candidates).all():
                issue.append('recorded_candidates_invalid')
            else:
                misses = sum(np.linalg.norm(recorded_candidates - q[None], axis=1).min() > cfg['branch_periodic_equivalence_tolerance_rad'] for q in candidate_array)
                if misses:
                    issue.append('recorded_candidates_do_not_include_all_legal_periodic_equivalents')
                if np.linalg.norm(recorded_candidates - chosen[None], axis=1).min() > cfg['branch_periodic_equivalence_tolerance_rad']:
                    issue.append('chosen_goal_not_present_in_actual_candidate_receipt')
            if event.get('selection_policy') != 'nearest_periodic_equivalent':
                issue.append('planner_selection_policy_differs')
            if event.get('reference_kind') not in ('measured', 'commanded'):
                issue.append('planner_reference_kind_missing')
            detail = {'number': number, 'arm': arm, 'stage': event.get('stage'),
                      'sim_time_s': event.get('sim_time_s'), 'reference_kind': event.get('reference_kind'),
                      'raw_q_rad': raw.tolist(), 'chosen_q_rad': chosen.tolist(),
                      'chosen_distance_from_reference_rad': chosen_score,
                      'minimum_legal_periodic_distance_rad': minimum_score,
                      'periodic_residual_rad': periodic_residual,
                      'independently_enumerated_legal_candidates': len(candidates),
                      'passed': not issue, 'failures': issue,
                      'this_branch_check_does_not_establish_collision_free_trajectory': True}
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            issue.append('planner_receipt_fields_incomplete:' + str(exc))
            detail = {'number': number, 'passed': False, 'failures': issue}
        events.append(detail)
        failures.extend(f'event_{number}:{x}' for x in issue)
    return {'passed': not failures, 'assessable': True, 'events': events, 'failures': failures,
            'actual_candidate_receipt_required': True,
            'pose_periodicity_checked_from_recorded_revolute_joint_coordinates': True,
            'candidate_and_actual_rollout_safety_is_a_separate_required_gate': True}


WINDOW_REJECTION_REASON = 'Outside independent near numeric planning window; physical vendor goal is not modified'


def validate_auxiliary_numeric_window(meta, actualfull, indices):
    """Read-only proof of an independent window; vendor/FMA limits stay intact."""
    window = meta.get('auxiliary_numeric_joint_window')
    if window is None:
        return None
    if not isinstance(window, dict):
        raise ValueError('Incomplete auxiliary numeric window receipt')
    required_true = ('only_selected_auxiliary_limit_changed',
        'source_model_limits_state_and_collision_poses_unchanged',
        'source_collision_geometry_and_local_shape_poses_cloned',
        'original_FMA_and_physical_limits_must_still_be_audited',
        'actual_vendor_FMA_model_not_replaced', 'auxiliary_planning_pk_is_separate_from_auditor_pk')
    if any(window.get(key) is not True for key in required_true) or window.get('path_postprocessing_wrap_or_clip_allowed') is not False:
        raise ValueError('Auxiliary window did not preserve original model/geometry/limits')
    for key in ('source_kinematic_urdf_sha256', 'canonical_original_urdf_sha256', 'auxiliary_derived_urdf_sha256'):
        value = window.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Auxiliary window original/derived URDF identity incomplete')
    inventory = window.get('collision_shape_inventory')
    if not isinstance(inventory, dict) or not inventory or any(not isinstance(k, str) or type(v) is not int or v < 0 for k, v in inventory.items()):
        raise ValueError('Auxiliary window native collision inventory missing')
    if window.get('native_FK_checked_link_count') != 93:
        raise ValueError('Auxiliary window did not preserve original 93-link FK inventory')
    for key in ('native_full_FK_initial_matrix_max_error', 'native_collision_link_initial_matrix_max_error'):
        value = float(window[key])
        if not np.isfinite(value) or not 0 <= value <= 1e-8:
            raise ValueError('Auxiliary window original FK/collision pose mismatch')
    if type(window.get('original_physics_exemption_pair_count')) is not int or window['original_physics_exemption_pair_count'] < 0:
        raise ValueError('Original physics exemptions were not recorded')
    selected = window['selected_full_q_index']
    names = window['actual_full_joint_names']
    limits = np.asarray(window['actual_full_vendor_limits_rad'], dtype=float)
    arm_names = meta['selected_arm_joint_names']
    vendor = np.asarray(window['original_vendor_limits_rad'], dtype=float)
    commanded = np.asarray(meta['native_commanded_limits_rad'], dtype=float)
    bounds = np.asarray(window['auxiliary_window_rad'], dtype=float)
    if (type(selected) is not int or selected != indices[-1] or len(names) != 36 or len(set(names)) != 36
            or not all(isinstance(n, str) for n in names) or limits.shape != (36, 2)
            or vendor.shape != (2,) or commanded.shape != (6, 2) or bounds.shape != (2,)
            or not all(np.isfinite(x).all() for x in (limits, vendor, commanded, bounds))
            or not np.all(limits[:, 0] <= limits[:, 1])):
        raise ValueError('Auxiliary window exact full ordering/bounds incomplete')
    if arm_names != [names[i] for i in indices] or window['selected_joint'] != names[selected]:
        raise ValueError('Auxiliary window selected native joint differs from actual arm')
    # Exported CPU bounds and PhysX readback can differ at float32 roundoff;
    # preserve original 1e-5 commanded and 1e-4 measured/start FMA tolerances.
    if (np.max(abs(limits[indices] - commanded)) > 1e-5
            or np.max(abs(vendor - commanded[-1])) > 1e-5
            or np.max(abs(vendor - limits[selected])) > 1e-10):
        raise ValueError('Auxiliary window source bounds differ from original vendor readback')
    current = float(window['current_q_rad'])
    if not np.isfinite(current) or current != float(actualfull[selected]) or not vendor[0] <= current <= vendor[1]:
        raise ValueError('Auxiliary window current does not match original actual measured state')
    expected = np.array([max(vendor[0], current - (np.pi - 1e-5)),
                         min(vendor[1], current + (np.pi - 1e-5))])
    width = float(window['auxiliary_width_rad'])
    if (not 0 < bounds[1] - bounds[0] < 2 * np.pi or not np.isfinite(width)
            or np.max(abs(bounds - expected)) > 1e-10 or abs(width - (bounds[1] - bounds[0])) > 1e-10):
        raise ValueError('Auxiliary window is not the exact current-centered finite intersection')
    if np.max(np.maximum(np.maximum(limits[:, 0] - actualfull, actualfull - limits[:, 1]), 0.)) > 1e-4:
        raise ValueError('Actual initial full state exceeds original measured/start FMA tolerance')
    return {'receipt': window, 'full_limits': limits, 'selected_index': selected, 'bounds': bounds}


def legal_goals_from_actual_collision_receipt(endpoint, declared, meta, actualfull, indices, window):
    """Exclude only proven window rejections; missing collision truth stays fatal."""
    checks = endpoint['declared_goal_collision_checks']
    if window is None:
        # Preserve the historical legitimate schema and its strict missing-key
        # failure. A generic missing collision_count must never be ignored.
        return [np.asarray(item['full_qpos_rad'])[indices].tolist() for item in checks if item['collision_count'] == 0], []
    if len(checks) != len(declared):
        raise ValueError('Window collision receipt does not cover the actual native declaration')
    frozen = np.ones(36, dtype=bool); frozen[indices] = False
    commanded = np.asarray(meta['native_commanded_limits_rad'], dtype=float)
    limits, selected, bounds = window['full_limits'], window['selected_index'], window['bounds']
    original = np.asarray(endpoint['equivalent_goal_declaration']['original_full_IK_qpos_rad'], dtype=float)
    if original.shape != (36,) or not np.isfinite(original).all() or not np.array_equal(original[frozen], actualfull[frozen]):
        raise ValueError('Actual native declaration changed frozen original measured joints')
    seen, legal, excluded = set(), [], []
    for item in checks:
        number = item['declared_goal_index']
        q = np.asarray(item['full_qpos_rad'], dtype=float)
        if type(number) is not int or number in seen or not 0 <= number < len(declared):
            raise ValueError('Window goal index is absent/duplicate/outside actual declaration')
        seen.add(number)
        if q.shape != (36,) or not np.isfinite(q).all():
            raise ValueError('Window goal must contain exact finite full36 numeric state')
        if (not np.array_equal(q[indices], np.asarray(declared[number]['arm_qpos_rad'], dtype=float))
                or not np.array_equal(q[frozen], actualfull[frozen])):
            raise ValueError('Window goal differs from actual native declaration or frozen actual joints')
        if (not np.all(q[indices] >= commanded[:, 0]) or not np.all(q[indices] <= commanded[:, 1])
                or np.max(np.maximum(np.maximum(limits[frozen, 0] - q[frozen], q[frozen] - limits[frozen, 1]), 0.)) > 1e-4):
            raise ValueError('Window goal violates original native active/frozen bounds contract')
        inside = bool(bounds[0] <= q[selected] <= bounds[1])
        if 'collision_count' not in item:
            if item.get('rejection_reason') != WINDOW_REJECTION_REASON or inside:
                raise ValueError('Missing collision check is not an explicitly proven outside-window rejection')
            excluded.append({'declared_goal_index': number, 'selected_joint_q_rad': float(q[selected]),
                             'auxiliary_window_rad': bounds.tolist(), 'actual_collision_check_executed': False,
                             'reason': WINDOW_REJECTION_REASON})
            continue
        count = item['collision_count']
        if type(count) is not int or count < 0 or item.get('rejection_reason') or not inside:
            raise ValueError('Real collision-count entry is invalid or outside its numeric window')
        if count == 0:
            legal.append(q[indices].tolist())
    return legal, excluded


def validate_planner_source_binding(receipt, arrays, metadata, cfg, numeric_path):
    """Bind branch algebra to accepted source metadata and actually applied targets.

    This does not re-run IK, FK, collision, planning or physics. The source
    declarations remain actual planner evidence; numeric execution is checked
    independently against the original 4ms drive-target recording.
    """
    if receipt is None or not receipt.get('events'):
        return {'passed': False, 'assessable': False,
                'failures': ['actual_planner_source_evidence_missing']}
    if any(key not in arrays for key in (TIME_BEFORE, DENSE_BEFORE, DENSE_COMMAND)):
        return {'passed': False, 'assessable': False,
                'failures': ['actual_physics_fields_for_source_binding_missing']}
    failures, events = [], []
    if receipt.get('events_from_actual_planner_receipts') is not True:
        failures.append('actual_planner_origin_not_declared')
    if receipt.get('synthetic_component_fixture_only') or receipt.get('future_original_commands_used') is not False:
        failures.append('synthetic_or_original_future_commands_are_not_actual_new_planning')
    if receipt.get('source_slot_index') != metadata.get('index'):
        failures.append('planner_source_slot_differs')
    numeric_path = Path(numeric_path).resolve()
    expected_owner = numeric_path.parent.parent if numeric_path.suffix in ('.hdf5', '.h5') else None
    source_paths = []
    tol = cfg['branch_periodic_equivalence_tolerance_rad']
    def same(a, b, tolerance=tol):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        return a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all() and bool(np.allclose(a, b, rtol=0, atol=tolerance))
    for number, event in enumerate(receipt['events']):
        issue = []
        detail = {'number': number, 'arm': event.get('arm'), 'stage': event.get('stage')}
        def check(name, passed):
            if not passed:
                issue.append(name)
        try:
            source = Path(event['actual_source_receipt_path']).resolve()
            source_paths.append(source)
            check('source_file_exists_and_SHA_matches', source.is_file() and sha(source) == event['actual_source_receipt_sha256'])
            if not source.is_file():
                raise ValueError('Actual source receipt file missing')
            if expected_owner is not None:
                check('source_receipt_belongs_to_same_new_candidate', source.parent == expected_owner)
            data = json.loads(source.read_text())
            meta = data['planning_metadata']
            arm = event['arm']; base = ARM_BASE[arm]
            check('source_is_successful_actual_approach', data['status'] == 'Success' and data.get('full_mesh_audit_passed') is True)
            check('source_arm_and_stage_match', data['side'] == arm == metadata['arm'] and
                  event['stage'] == 'free_approach_' + str(data['index']) and source.stem == event['stage'])
            check('planner_preserved_actual_scene', meta.get('physical_state_unchanged') is True and meta.get('no_attached_objects') is True)
            indices = meta['selected_arm_indices']
            if len(indices) != 6 or len(set(indices)) != 6:
                raise ValueError('Actual commanded native joint indices must identify six joints')
            actualfull = np.asarray(meta['initial_actual_full_qpos_rad'], dtype=float)
            if actualfull.shape != (36,):
                raise ValueError('Actual before full-q must contain36 joint coordinates')
            accepted = [row for row in meta['RRT_attempts'] if 'rejection_reason' not in row and
                        row.get('RRT_status', '').rstrip('.') == 'Exact solution']
            if len(accepted) != 1:
                raise ValueError('Unique actually accepted original RRT row required')
            row = accepted[0]; selection = row['near_numeric_goal_selection']
            endpoint = next(item for item in meta['endpoint_attempts'] if item['attempt'] == row['IK_attempt'])
            declaration = endpoint['equivalent_goal_declaration']
            declared = [item for item in declaration['goal_attempts'] if item['inside_exact_native_commanded_limits'] and item.get('native_FK_equivalent')]
            check('chosen_branch_matches_original_accepted_branch', event['selected_RRT_branch'] == row['branch'] == meta['selected_branch'])
            check('reference_is_original_actual_before_q', event['reference_kind'] == 'measured' and same(event['reference_q_rad'], actualfull[indices]))
            check('raw_IK_binds_chosen_original_family', same(event['raw_ik_q_rad'], np.asarray(row['original_IK_full_qpos_rad'])[indices]) and
                  same(row['original_IK_full_qpos_rad'], declaration['original_full_IK_qpos_rad']))
            check('selected_goal_is_original_near_selection', same(event['selected_q_rad'], selection['chosen_arm_qpos_rad']) and
                  same(event['selected_q_rad'], meta['selected_numeric_goal_arm_qpos_rad']))
            check('actual_limits_and_types_preserved', same(receipt['arm_joint_limits_rad'][arm], meta['native_commanded_limits_rad']) and
                  receipt['arm_joint_types'][arm] == meta['native_commanded_URDF_types'] and
                  same(meta['native_commanded_limits_rad'], declaration['commanded_native_limits_rad']) and
                  meta['native_commanded_URDF_types'] == declaration['commanded_native_URDF_types'])
            check('candidate_list_comes_from_actual_goal_declaration', len(event['candidates']) == len(declared) and
                  same([item['q_rad'] for item in event['candidates']], [item['arm_qpos_rad'] for item in declared]) and
                  all(item.get('native_FK_equivalent') is True for item in event['candidates']))
            check('actual_all93_link_FK_declaration_preserved', declaration['native_FK_link_count'] == 93 and
                  declaration.get('frozen_joints_bitwise_unchanged') is True and all(item['native_FK_checked_links'] == 93 and
                  item['native_FK_max_matrix_error'] <= item['native_FK_matrix_tolerance'] for item in declared))
            check('actual_full_native_collision_checks_preserved', event['original_full_native_collision_checks'] == endpoint['declared_goal_collision_checks'])
            window = validate_auxiliary_numeric_window(meta, actualfull, indices)
            legal, excluded = legal_goals_from_actual_collision_receipt(endpoint, declared, meta, actualfull, indices, window)
            detail['explicit_proven_window_rejected_goal_count'] = len(excluded)
            detail['explicit_proven_window_rejected_goals'] = excluded
            legal = sorted(legal, key=lambda q: np.linalg.norm(np.asarray(q) - actualfull[indices]))
            check('actual_legal_goal_family_preserved', len(legal) == selection['original_legal_goal_count'] and
                  same(legal, np.asarray(selection['original_legal_goals_full_qpos_rad'])[:, indices]))
            check('only_one_near_numeric_goal_reaches_actual_RRT', selection['RRT_goal_count'] == 1 and
                  selection['far_equivalent_goal_fallback_allowed'] is False and
                  same(row['predeclared_native_free_goals_arm_rad'], [event['selected_q_rad']]))
            proof = row['chosen_goal_target_FK_proof']
            check('chosen_task_target_FK_proof_is_original', event['chosen_target_FK_proof'] == proof and
                  proof['passed'] is True and proof['native_FK_checked_links'] == 93)
            for key in ('numeric_endpoint_match', 'raw_joint_limit_audit', 'retimed_joint_limit_audit',
                        'retimed_numeric_endpoint_match', 'audited_vertices_numeric_endpoint_match',
                        'motion_quality_raw', 'motion_quality_retimed'):
                check('original_' + key + '_passed', row[key]['passed'] is True)
            check('original_actual_full_native_geometry_passed', row['audit_passed'] is True and row['audit']['passed'] is True)
            for key in ('raw_path_arm_rad', 'retimed_commanded_position_arm_rad', 'actual_TOPP_audit_vertices_arm_rad'):
                p = np.asarray(row[key], dtype=float)
                check(key + '_same_specific_numeric_endpoint', p.ndim == 2 and p.shape[1] == 6 and len(p) >= 2 and
                      same(p[0], actualfull[indices], 1e-7) and same(p[-1], event['selected_q_rad'], 1e-6))
                if window is not None:
                    check(key + '_inside_exact_auxiliary_numeric_window', p.ndim == 2 and p.shape[1] == 6 and
                          np.isfinite(p).all() and bool(np.all(p[:, -1] >= window['bounds'][0]) and
                          np.all(p[:, -1] <= window['bounds'][1])))
            time_s = float(event['sim_time_s'])
            check('planning_timestamp_is_original_actual_clock', abs(time_s - float(meta['rerecord_planning_sim_time_s'])) <= cfg['clock_tolerance_s'])
            times = np.asarray(arrays[TIME_BEFORE]); start = int(np.argmin(abs(times - time_s)))
            check('planning_timestamp_exists_in_actual_physics_recording', abs(float(times[start]) - time_s) <= cfg['clock_tolerance_s'])
            check('planner_reference_matches_actual_recorded_measurement', same(arrays[DENSE_BEFORE][start, base:base + 6], event['reference_q_rad'], cfg['state_chain_tolerance']))
            retimed = np.asarray(row['retimed_commanded_position_arm_rad'], dtype=float)
            applied = np.asarray(arrays[DENSE_COMMAND])[start:start + len(retimed), base:base + 6]
            # Native PhysX targets may be stored in float32. Compare either the
            # original numbers or their exact native float32 representation;
            # no angles/path are wrapped, clipped, resampled or re-executed.
            native_float32 = retimed.astype(np.float32).astype(np.float64)
            command_match = same(applied, retimed, 1e-7) or same(applied, native_float32, 1e-7)
            check('actually_applied_4ms_targets_equal_original_retimed_path', command_match)
            original_target_error = float(np.abs(applied - retimed).max()) if applied.shape == retimed.shape else None
            native_float32_error = float(np.abs(applied - native_float32).max()) if applied.shape == retimed.shape else None
            detail.update(actual_source_receipt_path=str(source), actual_source_receipt_sha256=sha(source),
                          accepted_original_RRT_branch=row['branch'], original_IK_attempt=row['IK_attempt'],
                          planning_time_s=time_s, actual_physics_start_row=start,
                          original_retimed_command_rows=len(retimed), actual_applied_command_rows=len(applied),
                          actual_vs_original_retimed_max_error_rad=original_target_error,
                          actual_vs_original_native_float32_max_error_rad=native_float32_error,
                          command_numeric_comparison_tolerance_rad=1e-7,
                          native_float32_storage_representation_checked=True)
        except (KeyError, TypeError, ValueError, IndexError, OSError, StopIteration) as exc:
            issue.append('actual_source_binding_incomplete:' + str(exc))
        detail.update(passed=not issue, failures=issue)
        events.append(detail)
        failures.extend(f'event_{number}:{item}' for item in issue)
    if len(set(source_paths)) != len(source_paths):
        failures.append('duplicate_actual_approach_source_events')
    if expected_owner is not None:
        actual_sources = []
        for source in expected_owner.glob('free_approach_*.json'):
            try:
                if json.loads(source.read_text()).get('status') == 'Success':
                    actual_sources.append(source.resolve())
            except (ValueError, OSError):
                failures.append('candidate_source_receipt_unreadable')
        if set(actual_sources) != set(source_paths):
            failures.append('branch_events_do_not_cover_all_actual_successful_approach_sources')
    return {'passed': not failures, 'assessable': True, 'events': events, 'failures': failures,
            'actual_source_SHA_and_fields_checked': True, 'actual_drive_target_execution_checked': True,
            'source_IK_FK_collision_evidence_was_not_fabricated_or_recomputed': True,
            'simulation_or_model_executed_by_binding_validator': False}


def upstream_gate(receipts):
    required = ('original_physical_success', 'original_airhold_acceptance', 'original_safety_acceptance',
                'official_demo_randomized_actual_parameters', 'actual_clutter_collision_guard',
                'hardware_camera_physics_clock_unchanged')
    if receipts is None:
        return {'passed': False, 'assessable': False, 'failures': ['original_upstream_acceptance_receipts_missing']}
    failures, checks = [], []
    for key in required:
        value = receipts.get(key)
        if not isinstance(value, dict) or value.get('passed') is not True or not value.get('evidence_path') or not value.get('evidence_sha256'):
            failures.append(key + '_actual_evidence_missing_or_failed')
            checks.append({'name': key, 'passed': False})
            continue
        path = Path(value['evidence_path'])
        passed = path.is_file() and sha(path) == value['evidence_sha256']
        if not passed:
            failures.append(key + '_actual_evidence_sha_mismatch')
        checks.append({'name': key, 'passed': passed, 'evidence_path': str(path), 'evidence_sha256': value['evidence_sha256']})
    return {'passed': not failures, 'assessable': True, 'checks': checks, 'failures': failures,
            'upstream_truth_preserved_without_relabelling': True}


def evaluate_candidate(numeric_path_or_hdf, metadata, config_path=CONFIG,
                       branch_receipt_path=None, upstream_acceptance_path=None,
                       actual_first_target_contact_time_s=None):
    metadata, cfg = load_json(metadata), load_json(config_path)
    path = Path(numeric_path_or_hdf)
    initial_stat = path.stat()
    arrays = load_numeric(path, metadata['arm'])
    motion = numerical_motion(arrays, metadata, cfg, actual_first_target_contact_time_s)
    branch_receipt = load_json(branch_receipt_path)
    branch = validate_periodic_branch_receipt(branch_receipt, cfg)
    source_binding = validate_planner_source_binding(branch_receipt, arrays, metadata, cfg, path)
    upstream = upstream_gate(load_json(upstream_acceptance_path))
    binding = cfg['scene_bindings'].get(str(metadata.get('index')), {})
    scope = (metadata.get('index') in cfg['scope']['indices'] and
             metadata.get('pose_class') == cfg['scope']['pose_class'] and
             metadata.get('brand') == binding.get('brand') and metadata.get('arm') == binding.get('arm'))
    after_stat = path.stat()
    unchanged = (initial_stat.st_size, initial_stat.st_mtime_ns) == (after_stat.st_size, after_stat.st_mtime_ns)
    overall = bool(scope and cfg.get('approved') is True and motion['passed'] and branch['passed'] and source_binding['passed'] and upstream['passed'] and unchanged)
    return {'schema': 'rm65b-upright5-candidate-quality-result-v1', 'metadata': metadata,
            'numeric_source': str(path), 'numeric_source_sha256': sha(path), 'numeric_source_unchanged': unchanged,
            'config_approved': cfg.get('approved') is True, 'config_sha256': sha(config_path),
            'within_confirmed_five_upright_scope': scope, 'motion': motion, 'planner_branch': branch,
            'planner_source_binding': source_binding,
            'original_upstream_acceptance': upstream, 'overall_publish_gate_passed': overall,
            'original_initial_pose_match_is_not_required_for_new_randomized_scenes': True,
            'simulation_or_model_executed_by_validator': False, 'hardware_io': False,
            'original_dataset_was_not_modified': True,
            'outside_scope_original45_are_not_claimed_motion_quality_passed': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--numeric', required=True)
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--config', default=str(CONFIG))
    parser.add_argument('--branch-receipt')
    parser.add_argument('--upstream-acceptance')
    parser.add_argument('--actual-first-target-contact-time-s', type=float)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = evaluate_candidate(args.numeric, args.metadata, args.config,
                                args.branch_receipt, args.upstream_acceptance,
                                args.actual_first_target_contact_time_s)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
    print(json.dumps({'motion_passed': result['motion']['passed'],
                      'branch_passed': result['planner_branch']['passed'],
                      'planner_source_binding_passed': result['planner_source_binding']['passed'],
                      'upstream_passed': result['original_upstream_acceptance']['passed'],
                      'overall_publish_gate_passed': result['overall_publish_gate_passed']}, ensure_ascii=False))
    return 0 if result['overall_publish_gate_passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
