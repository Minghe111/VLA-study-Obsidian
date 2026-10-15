"""CPU synthetic-contact counterexamples using frozen controller arithmetic.

No SAPIEN/robot/physics run is performed. The LP fixture is explicitly synthetic
to test handoff plumbing; it is never evidence of a feasible physical wrench.
"""
from __future__ import annotations
import ast
from contextlib import contextmanager, redirect_stdout
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
sys.dont_write_bytecode = True
DIRECTORY = Path('/home/admin123/liminghe/VLA-WAM-RL/30-实验记录/证据/2026-10-13-RealMan五条重录/preflight/remote_161_v6/server_bundle_20261014_v6_hybrid_candidate7f/payload/deploy/realman_gen4/rgb50_motion_quality_20261014_v6_161_hybrid_v7f/jobs/slot17_runtime/planning')
LEGACY = Path('/home/admin123/liminghe/VLA-WAM-RL/30-实验记录/证据/2026-10-13-RealMan五条重录/preflight/remote_161_v6/server_bundle_20261014_v6_hybrid_candidate7f/payload/deploy/realman_gen4/rgb50_motion_quality_20261014_v6_161_hybrid_v7f/jobs/slot17_runtime/source/adjust_bottle_20261002')
if str(DIRECTORY) not in sys.path:
    sys.path.insert(0, str(DIRECTORY))
from grasp_acquisition_guard import GraspAcquisitionGuard, bind_session, MAX_EXTRA_STEPS, STABLE_PAIR_STEPS, _sha
spec = importlib.util.spec_from_file_location('acquisition_frozen_servo_CPU', LEGACY / 'continuous_servo.py')
servo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(servo)

class Joint:

    def __init__(self, q, *, float32_target=False):
        self.float32_target = float32_target
        self.q = float(q)
        self.target = float(np.float32(q)) if float32_target else float(q)
        self.velocity = 0.0

    def get_drive_target(self):
        return [self.target]

    def get_drive_velocity_target(self):
        return [self.velocity]

    def set_drive_target(self, value):
        self.target = float(np.float32(value)) if self.float32_target else float(value)

    def set_drive_velocity_target(self, value):
        self.velocity = float(value)

class Hand:

    def __init__(self, prefix, *, endpoint_upper=None):
        self.channels = tuple((prefix + t for t in ('little', 'ring', 'middle', 'index', 'thumb', 'rotate')))
        self.limits = {n: {'lower': 0.0, 'upper': 1.4, 'velocity': 1.0} for n in self.channels}
        if endpoint_upper is not None:
            self.limits[self.channels[-1]]['upper'] = float(endpoint_upper)
        self.limits[prefix + 'middle_follower'] = {'lower': 0.0, 'upper': 1.4, 'velocity': 1.0}
        self.metadata = {'channels': list(self.channels), 'limits': self.limits, 'mimic': {prefix + 'middle_follower': {'joint': self.channels[2], 'multiplier': 1.0, 'offset': 0.0}}}
        self.mapping = servo.ContinuousHandMapping(self.metadata)

    def coupled(self, primary):
        output = dict(primary)
        for (n, p, a, b) in self.mapping.mimic_order:
            output[n] = a * output[p] + b
        return output

    def to_sdk(self, positions):
        return [float(np.clip(1000.0 * (1.0 - (positions[n] - self.limits[n]['lower']) / (self.limits[n]['upper'] - self.limits[n]['lower'])), 0.0, 1000.0)) for n in self.channels]

class Robot:
    communication_flag = False

    def __init__(self, side, *, endpoint_upper=None, float32_target=False):
        self.side = side
        self._hand_maps = {s: Hand('l_' if s == 'left' else 'r_', endpoint_upper=endpoint_upper if s == side else None) for s in ('left', 'right')}
        self._manifest = {'hands': {s: h.metadata for (s, h) in self._hand_maps.items()}}
        self._hand_joints = {n: Joint(0.0, float32_target=float32_target) for h in self._hand_maps.values() for n in h.limits}
        self._hand_sdk_target = {s: [1000.0] * 6 for s in ('left', 'right')}
        self.left_arm_joints = [Joint(0.0) for _ in range(6)]
        self.right_arm_joints = [Joint(0.0) for _ in range(6)]
        self.left_gripper_val = self.right_gripper_val = 0.0
        self.left_entity = SimpleNamespace()
        self.right_entity = self.left_entity

    def _side(self, value):
        return str(value)

    def _entity_qf(self, entity):
        pass

    def _measured_positions(self):
        return {n: j.q for (n, j) in self._hand_joints.items()}

    def get_hand_sdk(self, side):
        return self._hand_maps[side].to_sdk(self._measured_positions())

    def get_state24(self, measured=True):
        result = []
        for side in ('left', 'right'):
            result.extend((j.q for j in getattr(self, side + '_arm_joints')))
            result.extend(self.get_hand_sdk(side))
        return result

class Scene:

    def __init__(self, robot, pattern, *, parallel=False):
        (self.robot, self.pattern, self.parallel) = (robot, pattern, parallel)
        (self.extra, self.initial) = (0, True)

    def get_timestep(self):
        return 0.004

    def forces(self):
        return [0.0, 0.0, 0.03, 0.03, 0.0] if self.initial else self.pattern(self.extra)

    def get_contacts(self):
        result = []
        prefix = 'l_' if self.robot.side == 'left' else 'r_'
        for (i, (token, force)) in enumerate(zip(('little', 'ring', 'middle', 'index', 'thumb'), self.forces())):
            if force <= 0:
                continue
            normal = np.array([-1.0, 0.0, 0.0]) if i == 4 and (not self.parallel) else np.array([1.0, 0.0, 0.0])
            point = SimpleNamespace(normal=normal, impulse=normal * force * self.get_timestep(), position=np.array([0.0, float(i) * 0.01, 0.0]))
            result.append(SimpleNamespace(bodies=[SimpleNamespace(entity=SimpleNamespace(name='001_bottle')), SimpleNamespace(entity=SimpleNamespace(name=prefix + token))], points=[point]))
        return result

def synthetic_lp(usable=True):
    return SimpleNamespace(allocate_contact_forces=lambda *args, **kwargs: {'usable': usable, 'target_normal_forces_N': [0.0, 0.0, 0.12, 0.12, 0.24], 'reason': 'explicit CPU synthetic LP fixture'})

class Session:

    def __init__(self, out, pattern, *, side='left', parallel=False, usable_lp=True, endpoint_upper=None, float32_target=False):
        (self.side, self.out) = (side, Path(out))
        robot = Robot(side, endpoint_upper=endpoint_upper, float32_target=float32_target)
        scene = Scene(robot, pattern, parallel=parallel)
        component = SimpleNamespace(cmass_local_pose=SimpleNamespace(to_transformation_matrix=lambda : np.eye(4)), mass=0.1)
        self.task = SimpleNamespace(robot=robot, scene=scene, bottle=SimpleNamespace(actor=SimpleNamespace(get_components=lambda : [component]), get_pose=lambda : SimpleNamespace(to_transformation_matrix=lambda : np.eye(4))))
        self.stage = {'name': 'close', 'gripper': True, 'dense_call': 3}
        self.setting = {'open_sdk': [1000, 1000, 1000, 1000, 1000, 0], 'close_sdk': [650, 670, 610, 610, 887, 0], 'finger_force_allocation_N': [0.0, 0.0, 0.12, 0.12, 0.24], 'feedback_joint_rate_cap_rad_s': 0.16, 'lp_min_main_force_N': 1.15}
        self.spec = {'grasp_setting': copy.deepcopy(self.setting)}
        (self.steps, self.dense_calls, self.holding) = (1475, 3, False)
        (self.traces, self.telemetry, self.sample_metadata) = ([], [], [])
        (self.reference, self.carry_support_baseline) = (None, None)
        (self.report, self.call_log, self.inject_guard_failure_step) = ({}, [], None)
        h = robot._hand_maps[side]
        (initial, _) = h.mapping.state(self.setting['close_sdk'])
        (initial[h.channels[2]], initial[h.channels[3]]) = (0.516, 0.52)
        for (n, q) in h.coupled({n: initial[n] for n in h.channels}).items():
            robot._hand_joints[n] = Joint(q, float32_target=float32_target)
        namespace = {'np': np, 'MethodType': MethodType, 'RealManRobot': Robot, 'ContinuousHandMapping': servo.ContinuousHandMapping, 'SDKWaypointProfile': servo.SDKWaypointProfile, 'stage': self.stage, 'sapien': SimpleNamespace(physx=SimpleNamespace(PhysxRigidDynamicComponent=type(component)))}
        tree = ast.parse((LEGACY / 'contact_capture_control_v26.py').read_text())
        selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('_feedback_rate_limits', 'install_contact_capture')]
        if side == 'right':

            class Reflect(ast.NodeTransformer):

                def visit_Constant(self, node):
                    if isinstance(node.value, str):
                        value = 'right' if node.value == 'left' else 'r_' + node.value[2:] if node.value.startswith('l_') else node.value
                        return ast.copy_location(ast.Constant(value), node)
                    return node

                def visit_Attribute(self, node):
                    self.generic_visit(node)
                    if node.attr.startswith('left_'):
                        node.attr = 'right_' + node.attr[5:]
                    return node
            selected = [Reflect().visit(n) for n in selected]
        extracted = ast.Module(body=selected, type_ignores=[])
        ast.fix_missing_locations(extracted)
        exec(compile(extracted, str(LEGACY / 'contact_capture_control_v26.py') + ':CPU_synthetic', 'exec'), namespace)
        (self.feedback, self.contact_update) = namespace['install_contact_capture'](self.task, self.setting)
        robot.set_gripper(0.0, side)
        self.feedback['index'] = self.feedback['profile_motion_steps'] + 1
        with redirect_stdout(io.StringIO()):
            self.contact_update()
        scene.initial = False
        self.registration = {'installer_sha256': _sha(DIRECTORY / 'grasp_acquisition_guard.py'), 'CPU_synthetic_contacts_and_LP_not_physical_validation': True}
        self.lp_fixture = synthetic_lp(usable_lp)

    def grip(self):
        f = self.task.scene.forces()
        return {'thumb_normal_N': float(f[4]), 'fingers_normal_N': float(sum(f[:4])), 'opposing_normals_dot': 1.0 if self.task.scene.parallel or min(f[4], sum(f[:4])) == 0 else -1.0, 'relative_bottle_palm': (np.eye(4) + np.diag([0.0, 0.0, 0.0, 0.0])).tolist(), 'contacts': []}

    def bottle_gap(self):
        return 0.012

    def step(self):
        self.contact_update()
        for joint in self.task.robot._hand_joints.values():
            joint.q = joint.target
        self.steps += 1
        self.task.scene.extra += 1
        self.telemetry.append((self.steps,))
        self.traces.append({'step': self.steps, 'phase': copy.deepcopy(self.stage), 'grip': copy.deepcopy(self.grip())})
        if self.steps % 15 == 0:
            self.sample_metadata.append({'physics_step': self.steps, 'sim_time_s': self.steps * 0.004})
        if self.inject_guard_failure_step == self.task.scene.extra:
            raise RuntimeError('CPU original-step guard counterexample')

    def plan_path(self, *args, **kwargs):
        self.call_log.append('original_CMA_carry_audit')
        return {'status': 'Success', 'CPU_fixture_only': True}

    def dense(self, sequence, **kwargs):
        self.call_log.append('original_dense_or_holding')
        return None

    def save_evidence(self):
        self.call_log.append('original_save')

@contextmanager
def fixture(pattern, **kwargs):
    with tempfile.TemporaryDirectory(prefix='realman_acquisition_CPU_') as temp:
        session = Session(temp, pattern, **kwargs)
        guard = bind_session(session, session.task.robot._hand_maps[session.side].mapping, session.registration)
        with patch.dict(sys.modules, {'purecontact_lp_v26': session.lp_fixture}), redirect_stdout(io.StringIO()):
            yield (session, guard)

class AcquisitionCounterexamples(unittest.TestCase):

    def test_float32_nominal_upper_readback_keeps_full_step_and_contact_gate(self):
        for side in ('left', 'right'):
            with self.subTest(side=side):
                with fixture(lambda _: [0.0] * 5, side=side, endpoint_upper=1.1641, float32_target=True) as (s, g):
                    name = g.mapping.channels[-1]
                    upper = g.mapping.limits[name]['upper']
                    rounded = float(np.float32(upper))
                    self.assertEqual(rounded, 1.1641000509262085)
                    self.assertEqual(s.task.robot._hand_joints[name].get_drive_target()[0], rounded)
                    with self.assertRaisesRegex(RuntimeError, 'Carry started without actual'):
                        s.plan_path(None, None)
                    self.assertEqual(s.steps, 1475 + MAX_EXTRA_STEPS)
                    self.assertEqual(len(g.rows), MAX_EXTRA_STEPS)
                    self.assertEqual(s.call_log, [])
                    self.assertFalse(g.completed)
                    self.assertTrue(all((row['contact_update']['float32_nominal_upper_readbacks'] == [name] for row in g.rows)))
                    self.assertTrue(all((len(row['applied_command24']) == 24 and all((0.0 <= v <= 1000.0 for v in row['applied_command24'])) for row in g.rows)))

    def test_next_float32_above_nominal_upper_is_still_rejected_before_step(self):
        for side in ('left', 'right'):
            with self.subTest(side=side):
                with fixture(lambda _: [0.0] * 5, side=side, endpoint_upper=1.1641, float32_target=True) as (s, g):
                    name = g.mapping.channels[-1]
                    upper = g.mapping.limits[name]['upper']
                    next_up = float(np.nextafter(np.float32(upper), np.float32(np.inf)))
                    s.task.robot._hand_joints[name].set_drive_target(next_up)
                    with self.assertRaisesRegex(ValueError, 'Continuous commanded position exceeds'):
                        s.plan_path(None, None)
                    self.assertEqual(s.steps, 1475)
                    self.assertEqual(g.rows, [])
                    self.assertEqual(s.call_log, [])

    def test_other_primary_above_limit_is_not_an_endpoint_exception(self):
        with fixture(lambda _: [0.0] * 5, float32_target=True) as (s, g):
            name = g.mapping.channels[0]
            upper = g.mapping.limits[name]['upper']
            over = float(np.nextafter(np.float32(upper), np.float32(np.inf)))
            self.assertGreater(over, upper)
            s.task.robot._hand_joints[name].set_drive_target(over)
            with self.assertRaisesRegex(ValueError, 'Continuous commanded position exceeds'):
                s.plan_path(None, None)
            self.assertEqual(s.steps, 1475)
            self.assertEqual(g.rows, [])

    def test_nominal_negative_lower_float32_readback_is_not_permitted(self):
        with fixture(lambda _: [0.0] * 5, endpoint_upper=1.1641, float32_target=True) as (s, g):
            name = g.mapping.channels[-1]
            lower = -0.1
            rounded_lower = float(np.float32(lower))
            self.assertLess(rounded_lower, lower)
            (positions, velocities) = g.mapping.state(s.setting['close_sdk'])
            g.mapping.limits[name]['lower'] = lower
            g.nominal_q[-1] = lower
            positions[name] = rounded_lower
            raw_q = g.nominal_q.copy()
            raw_q[-1] = rounded_lower
            with self.assertRaisesRegex(ValueError, 'Continuous commanded position exceeds'):
                g._validate_drive_readback(positions, velocities, raw_q)

    def test_float32_upper_requires_same_nominal_boundary(self):
        with fixture(lambda _: [0.0] * 5, endpoint_upper=1.1641, float32_target=True) as (s, g):
            name = g.mapping.channels[-1]
            upper = g.mapping.limits[name]['upper']
            rounded = float(np.float32(upper))
            (positions, velocities) = g.mapping.state(s.setting['close_sdk'])
            positions[name] = rounded
            raw_q = g.nominal_q.copy()
            raw_q[-1] = rounded
            g.nominal_q[-1] = upper - 0.01
            with self.assertRaisesRegex(ValueError, 'Continuous commanded position exceeds'):
                g._validate_drive_readback(positions, velocities, raw_q)

    def test_no_contacts_never_promotes_history_or_uses_CMA(self):
        with fixture(lambda _: [0.0] * 5) as (s, g):
            old = copy.deepcopy(s.feedback['latched_q'])
            with self.assertRaisesRegex(RuntimeError, 'Carry started without actual'):
                s.plan_path(None, None)
            self.assertEqual(len(g.rows), MAX_EXTRA_STEPS)
            self.assertEqual(s.steps, 1975)
            self.assertEqual(s.call_log, [])
            self.assertEqual(s.feedback['latched_q'], old)
            self.assertEqual(s.feedback['latched'], [False, False, True, True, False])
            self.assertFalse(s.feedback['had_pair'])
            self.assertEqual(s.feedback['actual_pair_streak'], 0)
            self.assertEqual(s.feedback['lp_history'], [])
            self.assertEqual(g.events, [])
            self.assertEqual(s.stage['name'], 'close')
            self.assertTrue(all((row['phase']['name'] == 'close' for row in g.rows)))
            np.testing.assert_allclose(g.last_q, g.nominal_q, atol=1e-12)
            for row in g.rows:
                self.assertEqual(len(row['applied_command24']), 24)
                self.assertLessEqual(max(abs(np.array(row['contact_update']['applied_primary_target_fd_rad_s']))), 0.16 + 1e-12)
            clocks = [r['physics_step'] for r in s.sample_metadata]
            self.assertEqual(clocks, list(range(1485, 1976, 15)))
            self.assertTrue(all((abs(r['sim_time_after_s'] - r['sim_time_before_s'] - 0.004) < 1e-12 for r in g.rows)))
            self.assertGreater(g.rows[-1]['contact_update']['applied_primary_target_rad'][2], old[2])
            self.assertAlmostEqual(s.task.robot.get_hand_sdk('left')[4], 887.0)

    def test_middle_contact_without_thumb_cannot_pass(self):
        with fixture(lambda _: [0.0, 0.0, 0.08, 0.0, 0.0]) as (s, g):
            with self.assertRaises(RuntimeError):
                s.plan_path(None, None)
            self.assertFalse(s.feedback['had_pair'])
            self.assertFalse(g.pair_seen)
            self.assertEqual(s.call_log, [])

    def test_already_tighter_channel_is_held_without_opening_or_extra_closure(self):
        with fixture(lambda _: [0.0] * 5) as (s, g):
            n = g.mapping.channels[2]
            tighter = g.nominal_q[2] + 0.01
            s.task.robot._hand_joints[n].target = tighter
            s.task.robot._hand_joints[n].q = tighter
            with self.assertRaises(RuntimeError):
                s.plan_path(None, None)
            self.assertAlmostEqual(g.last_q[2], tighter)
            self.assertTrue(all((abs(r['contact_update']['applied_primary_target_rad'][2] - tighter) < 1e-12 for r in g.rows)))

    def test_parallel_real_impulses_and_LP_usable_cannot_fake_opposition(self):
        with fixture(lambda _: [0.0, 0.0, 0.1, 0.1, 0.1], parallel=True) as (s, g):
            with self.assertRaises(RuntimeError):
                s.plan_path(None, None)
            self.assertTrue(s.feedback['had_pair'])
            self.assertGreater(s.feedback['actual_pair_streak'], 50)
            self.assertTrue(s.feedback['lp_history'][-1]['allocation']['usable'])
            self.assertFalse(g.pair_seen)
            self.assertFalse(g.completed)
            self.assertEqual(s.call_log, [])

    def test_transient_actual_opposition_cannot_meet_stable_pair_gate(self):
        with fixture(lambda i: [0.0, 0.0, 0.1, 0.1, 0.1] if i % 30 < 15 else [0.0] * 5) as (s, g):
            with self.assertRaises(RuntimeError):
                s.plan_path(None, None)
            self.assertTrue(g.pair_seen)
            self.assertLess(g.streak, STABLE_PAIR_STEPS)
            self.assertEqual(s.feedback['lp_history'], [])
            self.assertEqual(s.call_log, [])

    def test_truthful_pair_recapture_precedes_original_CMA_and_feedback_handoff(self):
        with fixture(lambda i: [0.0] * 5 if i < 8 else [0.0, 0.0, 0.1, 0.1, 0.1]) as (s, g):
            s.plan_path(None, None)
            self.assertTrue(g.completed)
            self.assertEqual(s.call_log, ['original_CMA_carry_audit'])
            self.assertGreaterEqual(s.feedback['actual_pair_streak'], 50)
            self.assertGreaterEqual(g.streak, 50)
            self.assertLessEqual(len(g.rows), 500)
            recaptures = [e for e in g.events if e['event'] == 'actual_paired_measured_channel_recapture']
            self.assertEqual(sorted((e['channel_index'] for e in recaptures)), [2, 3, 4])
            for event in recaptures:
                self.assertGreater(event['actual_channel_normal_N'], 0.025)
                self.assertTrue(event['original_latched_boolean'])
            self.assertTrue(s.feedback['had_pair'])
            self.assertEqual(g.receipt['nominal_close_sdk6'], [650.0, 670.0, 610.0, 610.0, 887.0, 0.0])
            self.assertLessEqual(g.last_update['maximum_feedback_handoff_target_gap_rad'], 1e-10)
            self.assertLessEqual(g.last_update['maximum_feedback_handoff_drive_velocity_gap_rad_s'], 1e-10)
            self.assertAlmostEqual(s.setting['lp_min_main_force_N'], 1.15)
            index = s.feedback['index']
            steps = s.steps
            s.plan_path(None, None)
            self.assertEqual(s.steps, steps)
            self.assertEqual(s.feedback['index'], index)
            data = json.loads(g.path.read_text())
            self.assertTrue(data['this_receipt_is_not_full_episode_acceptance'])
            self.assertFalse(data['fake_contacts_or_feedback_counters_written'])

    def test_right_hand_and_direct_holding_use_same_guard_and_clock(self):
        with fixture(lambda _: [0.0, 0.0, 0.1, 0.1, 0.1], side='right') as (s, g):
            s.holding = True
            s.dense({'left_gripper': None, 'right_gripper': None})
            self.assertTrue(g.completed)
            self.assertTrue(s.holding)
            self.assertEqual(s.call_log, ['original_dense_or_holding'])
            self.assertTrue(all((row['phase']['terminal_hold'] is False for row in g.rows)))
            self.assertTrue(all((e['channel'].startswith('r_') for e in g.events if 'channel' in e)))

    def test_original_LP_failure_stays_failure(self):
        with fixture(lambda _: [0.0, 0.0, 0.1, 0.1, 0.1], usable_lp=False) as (s, g):
            with self.assertRaisesRegex(RuntimeError, 'No feasible current physical contact wrench'):
                s.plan_path(None, None)
            self.assertLess(len(g.rows), 500)
            self.assertFalse(g.completed)
            self.assertEqual(s.call_log, [])
            self.assertEqual(s.feedback['lp_min_main_force_N'], 1.15)

    def test_original_step_failure_and_partial_receipt_are_retained(self):
        with fixture(lambda _: [0.0] * 5) as (s, g):
            s.inject_guard_failure_step = 7
            with self.assertRaisesRegex(RuntimeError, 'CPU original-step guard'):
                s.plan_path(None, None)
            self.assertEqual(len(g.rows), 7)
            self.assertEqual(g.rows[-1]['original_step_error'], 'CPU original-step guard counterexample')
            self.assertEqual(s.call_log, [])
            self.assertFalse(g.active)

    def test_missing_original_telemetry_or_capture_clock_cannot_reach_CMA(self):
        with fixture(lambda _: [0.0] * 5) as (s, g):
            original_step = s.step

            def missing_telemetry():
                original_step()
                s.telemetry.pop()
            s.step = missing_telemetry
            with self.assertRaisesRegex(RuntimeError, 'telemetry was bypassed'):
                s.plan_path(None, None)
            self.assertEqual(s.call_log, [])
        with fixture(lambda _: [0.0] * 5) as (s, g):
            original_step = s.step

            def missing_capture():
                original_step()
                s.sample_metadata.clear()
            s.step = missing_capture
            with self.assertRaisesRegex(RuntimeError, 'capture clock was bypassed'):
                s.plan_path(None, None)
            self.assertEqual(len(g.rows), 10)
            self.assertEqual(s.call_log, [])

    def test_original_already_stable_pair_uses_no_extra_ticks_or_new_latches(self):
        with fixture(lambda _: [0.0, 0.0, 0.1, 0.1, 0.1]) as (s, g):
            for _ in range(50):
                s.step()
            before_q = np.array([j.target for j in s.task.robot._hand_joints.values()])
            before_latches = copy.deepcopy(s.feedback['latched_q'])
            before_steps = s.steps
            s.plan_path(None, None)
            self.assertEqual(s.steps, before_steps)
            self.assertEqual(len(g.rows), 0)
            self.assertEqual(g.events, [])
            self.assertEqual(s.feedback['latched_q'], before_latches)
            np.testing.assert_array_equal(before_q, [j.target for j in s.task.robot._hand_joints.values()])
            self.assertEqual(g.receipt['status'], 'original_stable_actual_pair_no_extra_steps_needed')

    def test_original_force_streak_with_only_current_opposition_needs_new_stability(self):
        with fixture(lambda _: [0.0, 0.0, 0.1, 0.1, 0.1], parallel=True) as (s, g):
            for _ in range(50):
                s.step()
            self.assertEqual(s.feedback['actual_pair_streak'], 50)
            s.task.scene.parallel = False
            s.plan_path(None, None)
            self.assertTrue(g.completed)
            self.assertGreaterEqual(len(g.rows), 50)
            self.assertEqual(g.receipt['status'], 'actual_pair_handed_back_to_original_feedback_LP')

    def test_existing_reference_reanchors_only_to_current_stable_real_pair(self):
        with fixture(lambda _: [0.0, 0.0, 0.1, 0.1, 0.1]) as (s, g):
            old = np.eye(4)
            old[0, 3] = 0.001
            (s.reference, s.carry_support_baseline) = (old.copy(), 0.011)
            s.plan_path(None, None)
            event = [e for e in g.events if e['event'] == 'actual_stable_pair_carry_reference_handoff'][0]
            np.testing.assert_array_equal(event['previous_reference'], old)
            np.testing.assert_array_equal(event['actual_current_reference'], s.grip()['relative_bottle_palm'])
            self.assertFalse(event['actor_pose_modified'])
            self.assertEqual(s.carry_support_baseline, 0.012)

    def test_guard_refuses_changed_clock_hardware_or_nominal_target(self):
        with tempfile.TemporaryDirectory(prefix='realman_acquisition_CPU_') as temp:
            s = Session(temp, lambda _: [0.0] * 5)
            mapping = s.task.robot._hand_maps['left'].mapping
            s.task.robot.communication_flag = True
            with self.assertRaisesRegex(RuntimeError, 'Physical IO forbidden'):
                GraspAcquisitionGuard(s, mapping, s.registration)
            s.task.robot.communication_flag = False
            s.task.scene.get_timestep = lambda : 0.005
            with self.assertRaisesRegex(RuntimeError, '4ms clock'):
                GraspAcquisitionGuard(s, mapping, s.registration)
            s.task.scene.get_timestep = lambda : 0.004
            s.spec['grasp_setting']['close_sdk'][4] = 850
            with self.assertRaisesRegex(RuntimeError, 'resolved sample_spec'):
                GraspAcquisitionGuard(s, mapping, s.registration)
if __name__ == '__main__':
    unittest.main(verbosity=2)
