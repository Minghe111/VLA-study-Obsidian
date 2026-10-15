"""Read-only compact summary of immutable explicit-goal raw planning evidence."""
from pathlib import Path
import json
import sys
import numpy as np

for argument in sys.argv[1:]:
    root = Path(argument)
    files = [root] if root.is_file() else sorted(root.rglob('free_approach_*.json'))
    for path in files:
        result = json.loads(path.read_text(encoding='utf-8'))
        metadata = result.get('planning_metadata', {})
        if 'explicit_bounded_equivalent_goals' not in metadata.get('backend', ''):
            continue
        indices = metadata['selected_arm_indices']
        names = metadata['selected_arm_joint_names']
        attempts = []
        for row in metadata.get('RRT_attempts', []):
            original = np.asarray(row['original_IK_full_qpos_rad'])[indices]
            matched = row.get('numeric_endpoint_match', {})
            endpoint = row.get('raw_end_arm_rad')
            literal_error = None if endpoint is None else float(np.max(abs(np.asarray(endpoint) - original)))
            native = row.get('audit', {})
            attempts.append({'branch': row['branch'], 'IK_attempt': row['IK_attempt'],
                'RRT_status': row.get('RRT_status'), 'elapsed_s': row.get('elapsed_s'),
                'declared_native_free_goal_count': len(row['predeclared_native_free_goals_arm_rad']),
                'raw_path_rows': row.get('raw_path_rows'),
                'numeric_match': matched, 'raw_end_error_to_original_IK_rad': literal_error,
                'original_literal_IK_endpoint_guard_would_pass_on_this_new_raw_path':
                    None if literal_error is None else bool(literal_error <= metadata['raw_end_tolerance_rad']),
                'raw_end_delta_from_original_IK_per_joint_rad':
                    None if endpoint is None else {name: float(value) for name, value in zip(names, np.asarray(endpoint) - original)},
                'raw_limits_passed': row.get('raw_joint_limit_audit', {}).get('passed'),
                'retimed_limits_passed': row.get('retimed_joint_limit_audit', {}).get('passed'),
                'audit_passed': row.get('audit_passed'), 'rejection_reason': row.get('rejection_reason'),
                'first_geometry_collisions': native.get('collisions_first_64', [])[:4]})
        declarations = [row['equivalent_goal_declaration'] for row in metadata.get('endpoint_attempts', []) if 'equivalent_goal_declaration' in row]
        checks = [goal for declaration in declarations for goal in declaration['goal_attempts']]
        print(json.dumps({'file': str(path.resolve()), 'status': result.get('status'),
            'reason': result.get('reason'), 'physical_state_unchanged': metadata.get('physical_state_unchanged'),
            'auxiliary_state_restored': metadata.get('auxiliary_state_restored_including_construction_exception'),
            'native_FK_links': len(metadata.get('native_FK_link_names', [])),
            'IK_attempt_count': len(metadata.get('endpoint_attempts', [])),
            'native_free_goal_family_count': metadata.get('native_free_goal_family_count'),
            'declared_goal_count_over_all_IK_attempts': sum(d['declared_native_FK_equivalent_goal_count'] for d in declarations),
            'native_FK_max_matrix_error_over_all_declared_checks': max((g.get('native_FK_max_matrix_error', 0.) for g in checks), default=None),
            'selected_branch': metadata.get('selected_branch'),
            'selected_declared_goal_index': metadata.get('selected_declared_goal_index'),
            'new_physics_or_historical_outcome_not_inferred_by_this_reader': True,
            'RRT_attempts': attempts}), flush=True)
