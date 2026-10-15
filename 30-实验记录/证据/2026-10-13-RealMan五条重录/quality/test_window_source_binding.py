"""CPU-only schema counterexamples; synthetic fixtures can never be published."""
import copy
import json
from pathlib import Path
import tempfile

import numpy as np

from motion_gate import CONFIG, WINDOW_REJECTION_REASON, load_json, sha, validate_planner_source_binding
from test_source_binding import fixture


def window_fixture(root):
    data, receipt, arrays, numeric = fixture(root)
    meta = data['planning_metadata']; indices = meta['selected_arm_indices']
    names = [f'joint_{i}' for i in range(36)]
    meta['selected_arm_joint_names'] = [names[i] for i in indices]
    limits = np.asarray([[-10., 10.]] * 36)
    limits[indices] = meta['native_commanded_limits_rad']
    # Keep a genuine float32-like measured underflow below hand lower zero.
    # The original FMA permits unchanged start/frozen deviations <=1e-4.
    limits[0] = [0., 1.]
    meta['initial_actual_full_qpos_rad'][0] = -7e-7
    row = meta['RRT_attempts'][0]
    endpoint = meta['endpoint_attempts'][0]
    declaration = endpoint['equivalent_goal_declaration']
    declaration['commanded_indices'] = indices
    for full in [row['original_IK_full_qpos_rad'], declaration['original_full_IK_qpos_rad']]:
        full[0] = -7e-7
    for full in row['near_numeric_goal_selection']['original_legal_goals_full_qpos_rad']:
        full[0] = -7e-7
    for i, item in enumerate(endpoint['declared_goal_collision_checks']):
        item['full_qpos_rad'][0] = -7e-7
        item['declared_goal_index'] = i
    outside = endpoint['declared_goal_collision_checks'][0]
    del outside['collision_count']; outside['rejection_reason'] = WINDOW_REJECTION_REASON
    selection = row['near_numeric_goal_selection']
    selection['original_legal_goal_count'] = 1
    selection['original_legal_goals_full_qpos_rad'] = [endpoint['declared_goal_collision_checks'][1]['full_qpos_rad']]
    current = meta['initial_actual_full_qpos_rad'][indices[-1]]
    bounds = [max(limits[indices[-1], 0], current - (np.pi - 1e-5)),
              min(limits[indices[-1], 1], current + (np.pi - 1e-5))]
    meta['auxiliary_numeric_joint_window'] = {
        'selected_joint': names[indices[-1]], 'selected_full_q_index': indices[-1],
        'current_q_rad': current, 'original_vendor_limits_rad': limits[indices[-1]].tolist(),
        'auxiliary_window_rad': bounds, 'auxiliary_width_rad': bounds[1] - bounds[0],
        'actual_full_vendor_limits_rad': limits.tolist(), 'actual_full_joint_names': names,
        'source_kinematic_urdf_sha256': '1' * 64, 'canonical_original_urdf_sha256': '2' * 64,
        'auxiliary_derived_urdf_sha256': '3' * 64, 'collision_shape_inventory': {'synthetic_mesh': 1},
        'only_selected_auxiliary_limit_changed': True,
        'source_model_limits_state_and_collision_poses_unchanged': True,
        'source_collision_geometry_and_local_shape_poses_cloned': True,
        'path_postprocessing_wrap_or_clip_allowed': False,
        'original_FMA_and_physical_limits_must_still_be_audited': True,
        'native_FK_checked_link_count': 93, 'native_full_FK_initial_matrix_max_error': 0.,
        'native_collision_link_initial_matrix_max_error': 0., 'original_physics_exemption_pair_count': 0,
        'actual_vendor_FMA_model_not_replaced': True, 'auxiliary_planning_pk_is_separate_from_auditor_pk': True}
    return data, receipt, arrays, numeric


def main():
    cfg = load_json(CONFIG)
    metadata = {'index': 17, 'arm': 'left', 'brand': 'Fanta', 'pose_class': 'upright'}
    results = []
    with tempfile.TemporaryDirectory(prefix='SYNTHETIC-window-source-', dir=Path(__file__).parent) as temp:
        data, receipt, arrays, numeric = window_fixture(Path(temp))
        source = Path(receipt['events'][0]['actual_source_receipt_path'])
        def evaluate(changed):
            source.write_text(json.dumps(changed))
            r = copy.deepcopy(receipt)
            r['events'][0]['actual_source_receipt_sha256'] = sha(source)
            r['events'][0]['original_full_native_collision_checks'] = changed['planning_metadata']['endpoint_attempts'][0]['declared_goal_collision_checks']
            return validate_planner_source_binding(r, arrays, metadata, cfg, numeric)
        good = evaluate(data)
        assert good['events'][0]['passed'], good
        assert good['events'][0]['explicit_proven_window_rejected_goal_count'] == 1
        assert not good['passed'] and good['failures'] == ['synthetic_or_original_future_commands_are_not_actual_new_planning']
        results.append('valid explicit outside-window rejection plus frozen -7e-7 measured underflow accepted internally; synthetic publish stays blocked')
        def win(d): return d['planning_metadata']['auxiliary_numeric_joint_window']
        def counts(d): return d['planning_metadata']['endpoint_attempts'][0]['declared_goal_collision_checks']
        def attempt(d): return d['planning_metadata']['RRT_attempts'][0]
        cases = [
            ('missing window proof rejected', lambda d: d['planning_metadata'].pop('auxiliary_numeric_joint_window')),
            ('incomplete full36 limit proof rejected', lambda d: win(d).pop('actual_full_vendor_limits_rad')),
            ('wrong explicit rejection reason rejected', lambda d: counts(d)[0].update(rejection_reason='Generic missing check')),
            ('inside-window goal without actual collision_count rejected', lambda d: counts(d)[1].pop('collision_count')),
            ('outside-window collision_count=0 fabrication rejected', lambda d: counts(d)[0].update(collision_count=0)),
            ('wrong selected joint index rejected', lambda d: win(d).update(selected_full_q_index=9)),
            ('wrong actual current reference rejected', lambda d: win(d).update(current_q_rad=.91)),
            ('wrong original vendor bounds rejected', lambda d: win(d)['original_vendor_limits_rad'].__setitem__(0, -6.)),
            ('wrong current-centered window rejected', lambda d: win(d)['auxiliary_window_rad'].__setitem__(0, -6.)),
            ('2pi-wide window rejected', lambda d: win(d).update(auxiliary_width_rad=2*np.pi)),
            ('missing native geometry preservation rejected', lambda d: win(d).update(source_collision_geometry_and_local_shape_poses_cloned=False)),
            ('92-link FK proof rejected', lambda d: win(d).update(native_FK_checked_link_count=92)),
            ('collision-link pose mismatch rejected', lambda d: win(d).update(native_collision_link_initial_matrix_max_error=1e-4)),
            ('original identity hash missing rejected', lambda d: win(d).pop('source_kinematic_urdf_sha256')),
            ('declaration goal coverage missing rejected', lambda d: counts(d).pop()),
            ('duplicate declaration index rejected', lambda d: counts(d)[1].update(declared_goal_index=0)),
            ('goal differs from actual FK declaration rejected', lambda d: counts(d)[0]['full_qpos_rad'].__setitem__(1, .101)),
            ('frozen goal differs from actual measured state rejected', lambda d: counts(d)[0]['full_qpos_rad'].__setitem__(0, 0.)),
            ('boolean collision count rejected', lambda d: counts(d)[1].update(collision_count=False)),
            ('original FMA failure remains rejected', lambda d: attempt(d)['audit'].update(passed=False)),
        ]
        for label, mutate in cases:
            changed = copy.deepcopy(data); mutate(changed)
            result = evaluate(changed)
            assert not result['events'][0]['passed'], (label, result)
            results.append(label)
        # Recompute every stored frozen value identically, but exceed the actual
        # original 1e-4 start/frozen measurement tolerance: still reject.
        changed = copy.deepcopy(data); meta = changed['planning_metadata']
        meta['initial_actual_full_qpos_rad'][0] = -2e-4
        meta['RRT_attempts'][0]['original_IK_full_qpos_rad'][0] = -2e-4
        meta['endpoint_attempts'][0]['equivalent_goal_declaration']['original_full_IK_qpos_rad'][0] = -2e-4
        for item in counts(changed): item['full_qpos_rad'][0] = -2e-4
        assert not evaluate(changed)['events'][0]['passed']
        results.append('unchanged frozen/start outside original1e-4 measurement tolerance rejected')
        for key in ('raw_path_arm_rad', 'retimed_commanded_position_arm_rad', 'actual_TOPP_audit_vertices_arm_rad'):
            changed = copy.deepcopy(data)
            path = attempt(changed)[key]
            middle = copy.deepcopy(path[0]); middle[-1] = -5.5
            path.insert(1, middle)
            result = evaluate(changed)
            assert key + '_inside_exact_auxiliary_numeric_window' in result['events'][0]['failures'], result
            results.append(key + ' numeric outside-window interior rejected')
    report = {'passed': True, 'synthetic_component_only': True, 'actual_new_planner_or_physics_evidence_generated': False,
              'temporary_fixtures_removed': True, 'schema_and_counterexample_checks': len(results), 'checks': results}
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
