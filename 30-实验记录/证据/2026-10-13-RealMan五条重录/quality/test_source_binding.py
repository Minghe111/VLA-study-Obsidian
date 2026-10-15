"""Explicit synthetic CPU counterexamples, never publishable planner evidence.

Fixtures exist only in a TemporaryDirectory and are removed on completion.
They cannot pass the complete gate because the synthetic flag is required.
"""
import copy
import json
from pathlib import Path
import tempfile

import numpy as np

from motion_gate import (CONFIG, DENSE_BEFORE, DENSE_COMMAND, TIME_BEFORE,
                         load_json, sha, validate_planner_source_binding)


def fixture(root):
    raw = np.array([.1, .2, -.3, .4, -.5, -5.5])
    ref = raw.copy(); ref[-1] = .9
    chosen = raw.copy(); chosen[-1] += 2 * np.pi
    indices = [1, 3, 5, 7, 9, 11]
    def full(q):
        result = np.zeros(36); result[indices] = q; return result.tolist()
    limits = [[-np.pi, np.pi]] * 5 + [[-2 * np.pi, 2 * np.pi]]
    kind = ['revolute'] * 6
    proof = {'passed': True, 'native_FK_checked_links': 93}
    goalrows = [{'inside_exact_native_commanded_limits': True, 'native_FK_equivalent': True,
                 'arm_qpos_rad': q.tolist(), 'native_FK_checked_links': 93,
                 'native_FK_max_matrix_error': 1e-15, 'native_FK_matrix_tolerance': 1e-8} for q in (raw, chosen)]
    collision = [{'collision_count': 0, 'full_qpos_rad': full(q)} for q in (raw, chosen)]
    row = {'branch': 0, 'IK_attempt': 0, 'RRT_status': 'Exact solution',
           'original_IK_full_qpos_rad': full(raw), 'audit_passed': True, 'audit': {'passed': True},
           'near_numeric_goal_selection': {'chosen_arm_qpos_rad': chosen.tolist(),
                'original_legal_goal_count': 2, 'original_legal_goals_full_qpos_rad': [full(chosen), full(raw)],
                'RRT_goal_count': 1, 'far_equivalent_goal_fallback_allowed': False},
           'predeclared_native_free_goals_arm_rad': [chosen.tolist()], 'chosen_goal_target_FK_proof': proof,
           'raw_path_arm_rad': [ref.tolist(), chosen.tolist()],
           'retimed_commanded_position_arm_rad': [ref.tolist(), chosen.tolist()],
           'actual_TOPP_audit_vertices_arm_rad': [ref.tolist(), chosen.tolist()]}
    for name in ('numeric_endpoint_match', 'raw_joint_limit_audit', 'retimed_joint_limit_audit',
                 'retimed_numeric_endpoint_match', 'audited_vertices_numeric_endpoint_match',
                 'motion_quality_raw', 'motion_quality_retimed'):
        row[name] = {'passed': True}
    data = {'synthetic_component_fixture_only': True, 'status': 'Success', 'side': 'left', 'index': 0,
            'full_mesh_audit_passed': True,
            'planning_metadata': {'physical_state_unchanged': True, 'no_attached_objects': True,
                'selected_arm_indices': indices, 'initial_actual_full_qpos_rad': full(ref),
                'native_commanded_limits_rad': limits, 'native_commanded_URDF_types': kind,
                'selected_branch': 0, 'selected_numeric_goal_arm_qpos_rad': chosen.tolist(),
                'rerecord_planning_sim_time_s': 0., 'RRT_attempts': [row],
                'endpoint_attempts': [{'attempt': 0, 'declared_goal_collision_checks': collision,
                    'equivalent_goal_declaration': {'goal_attempts': goalrows,
                        'original_full_IK_qpos_rad': full(raw), 'commanded_native_limits_rad': limits,
                        'commanded_native_URDF_types': kind, 'native_FK_link_count': 93,
                        'frozen_joints_bitwise_unchanged': True}}]}}
    path = root / 'free_approach_0.json'; path.write_text(json.dumps(data))
    receipt = {'schema': 'nearest-periodic-branch-v1', 'synthetic_component_fixture_only': True,
               'source_slot_index': 17, 'future_original_commands_used': False,
               'events_from_actual_planner_receipts': True,
               'arm_joint_limits_rad': {'left': limits}, 'arm_joint_types': {'left': kind},
               'events': [{'arm': 'left', 'stage': 'free_approach_0', 'sim_time_s': 0.,
                   'reference_kind': 'measured', 'reference_q_rad': ref.tolist(),
                   'raw_ik_q_rad': raw.tolist(), 'selected_q_rad': chosen.tolist(),
                   'selected_RRT_branch': 0, 'actual_source_receipt_path': str(path),
                   'actual_source_receipt_sha256': sha(path), 'chosen_target_FK_proof': proof,
                   'original_full_native_collision_checks': collision,
                   'candidates': [{'q_rad': item['arm_qpos_rad'], 'native_FK_equivalent': True} for item in goalrows]}]}
    before = np.zeros((3, 24)); before[:, :6] = ref
    command = before.copy(); command[1, :6] = chosen
    arrays = {DENSE_BEFORE: before, DENSE_COMMAND: command, TIME_BEFORE: np.arange(3) * .004}
    numeric = root / 'data' / 'episode_0000000.hdf5'; numeric.parent.mkdir()
    numeric.write_text('SYNTHETIC COMPONENT PATH ONLY; NOT HDF5 OR ACTUAL DATA')
    return data, receipt, arrays, numeric


def main():
    cfg = load_json(CONFIG)
    meta = {'index': 17, 'arm': 'left', 'brand': 'Fanta', 'pose_class': 'upright'}
    checks = []
    with tempfile.TemporaryDirectory(prefix='SYNTHETIC-source-binding-', dir=Path(__file__).parent) as temp:
        data, receipt, arrays, numeric = fixture(Path(temp))
        def evaluate(r=receipt, a=arrays):
            return validate_planner_source_binding(r, a, meta, cfg, numeric)
        good = evaluate()
        assert not good['passed'] and good['events'][0]['passed'], good
        assert good['failures'] == ['synthetic_or_original_future_commands_are_not_actual_new_planning']
        checks.append('synthetic fixture stays blocked despite internally consistent source binding')
        for label, mutate in (
            ('wrong source SHA rejected', lambda r: r['events'][0].update(actual_source_receipt_sha256='0' * 64)),
            ('wrong original raw IK family rejected', lambda r: r['events'][0].update(raw_ik_q_rad=[0.] * 6)),
            ('candidate omitted from actual declaration rejected', lambda r: r['events'][0]['candidates'].pop()),
            ('planner time absent from physical clock rejected', lambda r: r['events'][0].update(sim_time_s=.006)),
            ('selected branch differs from source rejected', lambda r: r['events'][0].update(selected_RRT_branch=1)),
            ('joint limits differ from source rejected', lambda r: r['arm_joint_limits_rad']['left'][0].__setitem__(0, -2.)),
            ('collision receipt relabel rejected', lambda r: r['events'][0]['original_full_native_collision_checks'][0].update(collision_count=1)),
        ):
            changed = copy.deepcopy(receipt); mutate(changed)
            result = evaluate(changed)
            assert not result['events'][0]['passed'], (label, result)
            checks.append(label)
        wrong = copy.deepcopy(arrays); wrong[DENSE_BEFORE][0, 0] += .01
        assert not evaluate(a=wrong)['events'][0]['passed']
        checks.append('planner before q differs from actual numeric measurement rejected')
        wrong = copy.deepcopy(arrays); wrong[DENSE_COMMAND][1, 5] += 2 * np.pi
        assert not evaluate(a=wrong)['events'][0]['passed']
        checks.append('actually applied drive target differs from original TOPP path rejected')
        duplicate = copy.deepcopy(receipt); duplicate['events'] *= 2
        assert 'duplicate_actual_approach_source_events' in evaluate(duplicate)['failures']
        checks.append('duplicate accepted source events rejected')
    print(json.dumps({'passed': True, 'synthetic_component_only': True,
                      'actual_planner_or_physics_evidence_generated': False,
                      'temporary_fixtures_removed': True, 'checks': checks}, ensure_ascii=False))


if __name__ == '__main__':
    main()
