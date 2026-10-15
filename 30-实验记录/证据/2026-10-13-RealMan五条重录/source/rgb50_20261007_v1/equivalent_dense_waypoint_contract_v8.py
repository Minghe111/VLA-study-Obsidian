"""Independent zero-step mocks for the new CPU flow, not native/physics proof."""
from pathlib import Path
import ast
import hashlib
import importlib
import json
import numpy as np

work = Path(__file__).resolve().parent
fixture = work / 'equivalent_native_planner_contract_dense_v7.py'
prefix = fixture.read_text(encoding='utf-8').split('results = []\n', 1)[0]
env = {'__file__': str(fixture), '__name__': '__private_waypoint_mock__'}
exec(compile(prefix, str(fixture) + ':private_mock_prefix', 'exec'), env)
helper = importlib.import_module('full_native_approach_equivalent_dense_waypoint_v8')
sources = [Path(helper.__file__), Path(helper.original.__file__), Path(helper.equivalent.__file__),
           Path(helper.dense_policy.__file__), work / 'expert_generation_v9_equivalent_dense_waypoint_v8.py',
           fixture, Path(__file__)]
before_hashes = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
old_direct = helper.original.plan_free_approach
old_planner = env['ompl'].OMPLPlanner
old_topp = env['carry'].official_topp_polyline
hand_value = [.2]
env['robot']._hand_joints['hand'].get_drive_target = lambda: np.asarray(hand_value)
direct_success = {'status': 'Success', 'position': np.zeros((2, 6)),
                  'planning_metadata': {'physical_state_unchanged': True}}
direct_failure = {'status': 'Fail', 'reason': 'Retained original direct rejection fixture',
                  'planning_metadata': {'physical_state_unchanged': True, 'immutable_failed_marker': 917}}


def direct(*args, **kwargs):
    env['calls']['direct'].append(kwargs)
    mode = env['mode']
    if mode == 'direct_success':
        return direct_success
    if mode == 'direct_unexpected_status':
        return {'status': 'Approximate solution'}
    if mode == 'direct_physical_q_write':
        env['physical'][0] += .1
    if mode == 'direct_physical_hand_target_write':
        hand_value[0] += .1
    return direct_failure


class Planner(env['Planner']):
    def plan(self, start, goals, **kwargs):
        status, path = super().plan(start, goals, **kwargs)
        if env['mode'] == 'second_goal_index':
            endpoint = np.asarray(goals[-1]).copy()
            path = np.vstack([start, (start + endpoint) / 2., endpoint])
        if env['mode'] == 'second_leg_unmatched' and len(env['calls']['RRT']) % 2 == 0:
            path[-1, 5] += .001
        if env['mode'] == 'second_leg_wrong_start' and len(env['calls']['RRT']) % 2 == 0:
            path[0, 1] += .001
        return status, path


def topp(task, path, side, *, time_step):
    result = old_topp(task, path, side, time_step=time_step)
    if env['mode'] == 'retimed_binding':
        position = result[0] + 5e-11
        return position, result[1], result[2], {'audit_vertices': position.tolist()}
    if env['mode'] == 'TOPP_exception':
        raise ValueError('TOPP exception fixture')
    return result


original_audit = env['Auditor'].audit


def audit(self, vertices, side, **kwargs):
    if env['mode'] in ('dense_binding', 'retimed_binding'):
        env['calls']['FMA'].append({'actual_input_vertices': np.asarray(vertices).copy(), **kwargs})
        refined = kwargs.get('vertex_step_m') == .0001
        return {'passed': refined, 'joint_limit_audit': {'passed': True},
            'unique_collision_pairs': 0, 'collisions_first_64': [],
            'metadata': {'native_collision_links': 93},
            'thresholds_m': {'bottle': .0005, 'table': 0., 'self': 0.},
            'sampled_minimum_gap': {'bottle': {'distance_m': .0008}, 'table': {'distance_m': .005}, 'self': {'distance_m': .002}},
            'continuous_gap_lower_bound': {'bottle': {'lower_bound_m': .0007 if refined else -.0001}, 'table': {'lower_bound_m': .004}, 'self': {'lower_bound_m': .001}}}
    if env['mode'] == 'FMA_exception':
        raise ValueError('FMA exception fixture')
    return original_audit(self, vertices, side, **kwargs)


helper.original.plan_free_approach = direct
env['ompl'].OMPLPlanner = Planner
env['carry'].official_topp_polyline = topp
env['Auditor'].audit = audit
cases = ('direct_success', 'fallback_success', 'second_goal_index', 'second_leg_unmatched',
         'second_leg_wrong_start', 'wrong_start', 'unmatched_endpoint', 'raw_limit_violation',
         'retimed_limit_violation', 'audit_reject', 'nonfinite_path', 'approximate_status',
         'endpoint_collision', 'start_collision', 'frozen_IK_changed', 'FK_not_equivalent',
         'world_exception', 'TOPP_exception', 'FMA_exception', 'direct_unexpected_status',
         'direct_physical_q_write', 'direct_physical_hand_target_write', 'dense_binding', 'retimed_binding')
successful = {'direct_success', 'fallback_success', 'second_goal_index', 'dense_binding', 'retimed_binding'}
exceptions = {'world_exception', 'TOPP_exception', 'FMA_exception', 'direct_unexpected_status',
              'direct_physical_q_write', 'direct_physical_hand_target_write'}
results = []
try:
    for mode in cases:
        env['mode'] = mode
        env['created'] = []
        env['physical'][:6] = 0.
        hand_value[0] = .2
        env['calls'] = {name: [] for name in ('RRT', 'TOPP', 'FMA', 'ACM', 'obstacles', 'direct')}
        env['calls']['IK_count'] = 0
        physical_before = env['physical'].copy()
        try:
            result = helper.plan_free_approach(env['task'], [0] * 7, 'right', ik_attempts=2, random_seed=100025)
            assert mode not in exceptions
            if mode == 'direct_success':
                assert result is direct_success and not env['created'] and not env['calls']['RRT']
            elif mode in successful:
                assert result['status'] == 'Success'
                metadata = result['planning_metadata']
                assert metadata['original_direct_result_before_waypoint'] is direct_failure
                assert direct_failure['planning_metadata']['immutable_failed_marker'] == 917
                assert metadata['physical_state_unchanged'] is True
                goal = np.asarray(metadata['goal_palm_world_matrix'])
                via = np.asarray(metadata['via_palm_world_matrix'])
                assert np.array_equal(goal[:3, :3], via[:3, :3])
                assert np.max(abs(via[:3, 3] - goal[:3, 3] - [0., 0., .08])) < 1e-15
                row = metadata['RRT_attempts'][-1]
                first, second = row['first_leg'], row['second_leg']
                assert first['numeric_endpoint_match']['passed'] and second['numeric_endpoint_match']['passed']
                assert row['raw_join_error_rad'] <= helper.equivalent.RAW_START_TOLERANCE_RAD
                raw = np.vstack([np.asarray(first['raw_path_arm_rad']), np.asarray(second['raw_path_arm_rad'])])
                assert np.array_equal(raw, np.asarray(row['raw_path_arm_rad']))
                assert np.array_equal(raw, env['calls']['TOPP'][-1])
                full = np.asarray(row['raw_path_full_qpos_rad'])
                assert full.shape[1] == 36 and np.array_equal(full[:, 6:], np.tile(physical_before[6:], (len(full), 1)))
                assert len(metadata['native_FK_link_names']) == 93
                assert all(len(call['goals']) == 2 for call in env['calls']['RRT'])
                if mode == 'second_goal_index':
                    assert metadata['selected_via_declared_goal_index'] == metadata['selected_goal_declared_goal_index'] == 1
                if mode in ('dense_binding', 'retimed_binding'):
                    actual = np.asarray(row['actual_TOPP_audit_vertices_arm_rad'])
                    assert len(env['calls']['FMA']) == 2
                    assert all(np.array_equal(call['actual_input_vertices'], actual) for call in env['calls']['FMA'])
                    assert row['strict_dense_certificate_refinement']['original_complete_default_grid_audit']['passed'] is False
                    assert row['strict_dense_certificate_refinement']['refined_complete_native_audit']['passed'] is True
                    if mode == 'retimed_binding':
                        assert not np.array_equal(raw, actual)
                else:
                    assert env['calls']['FMA'] == [{'min_bottle_clearance_m': .0005, 'min_self_clearance_m': 0., 'min_table_clearance_m': 0.}]
            else:
                assert result['status'] == 'Fail'
            if mode in ('second_leg_unmatched', 'second_leg_wrong_start', 'wrong_start', 'unmatched_endpoint',
                        'raw_limit_violation', 'nonfinite_path', 'approximate_status', 'endpoint_collision',
                        'start_collision', 'frozen_IK_changed', 'FK_not_equivalent'):
                assert not env['calls']['TOPP']
            if mode == 'retimed_limit_violation':
                assert not env['calls']['FMA']
            results.append({'case': mode, 'passed': True, 'RRT_calls': len(env['calls']['RRT']),
                            'TOPP_calls': len(env['calls']['TOPP']), 'FMA_calls': len(env['calls']['FMA'])})
        except (ValueError, RuntimeError) as error:
            assert mode in exceptions, (mode, str(error))
            if mode.startswith('direct_physical'):
                assert 'changed physical' in str(error)
            results.append({'case': mode, 'passed': True, 'exception_rejected': str(error)})
        assert len(env['calls']['direct']) == 1
        for model in env['created']:
            assert np.array_equal(model.q, env['original_aux_q'])
            assert np.array_equal(model.pose.m, env['original_base'])
            assert model.group == ['original_group']
        if not mode.startswith('direct_physical'):
            assert np.array_equal(env['physical'], physical_before) and hand_value[0] == .2
finally:
    helper.original.plan_free_approach = old_direct
    env['ompl'].OMPLPlanner = old_planner
    env['carry'].official_topp_polyline = old_topp
    env['Auditor'].audit = original_audit
source_tree = ast.parse(Path(helper.__file__).read_text(encoding='utf-8'))
for node in ast.walk(source_tree):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        assert node.func.attr not in ('step', 'set_pose', 'set_qvel', 'set_drive_target', 'set_drive_velocity_target')
        if node.func.attr in ('set_qpos', 'set_base_pose', 'set_move_group'):
            assert ast.unparse(node.func.value) == 'model'
overlay_tree = ast.parse((work / 'expert_generation_v9_equivalent_dense_waypoint_v8.py').read_text(encoding='utf-8'))
session = next(node for node in overlay_tree.body if isinstance(node, ast.ClassDef))
assert [node.name for node in session.body if isinstance(node, ast.FunctionDef)] == ['__init__']
assert not any('v11' in ast.unparse(node) for node in ast.walk(overlay_tree) if isinstance(node, (ast.Import, ast.ImportFrom)))
after_hashes = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
assert before_hashes == after_hashes
receipt = {'passed': True, 'scope': 'Zero-step synthetic mocked contracts only; no actual native IK/RRT/TOPP/FCL or physics success',
           'new_physics_steps': 0, 'GPU_initialized': False, 'physical_success_not_inferred': True,
           'original_direct_success_returned_unchanged': True, 'original_direct_failures_retained': True,
           'raw_rows_preserved_and_all_frozen_columns_exact': True, 'all_auxiliary_state_restored_including_exceptions': True,
           'original_raw_limits_and_numeric_endpoint_guards_retained': True,
           'full_actual_TOPP_vertices_bound_to_both_certificate_calls': True,
           'v9_physics_control_methods_inherited_without_overrides': True,
           'cases': results, 'source_hashes_before': before_hashes, 'source_hashes_after': after_hashes}
with (work / 'equivalent_dense_waypoint_contract_receipt_v8.json').open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
print(json.dumps(receipt, indent=2), flush=True)
