"""Read immutable native planning receipts without changing or admitting data."""
from pathlib import Path
import hashlib
import json
import sys


def geometry_summary(audit):
    return {key: audit.get(key) for key in
            ('passed', 'sample_count', 'collision_count', 'continuous_gap_lower_bound')}


for argument in sys.argv[1:]:
    root = Path(argument)
    paths = [root] if root.is_file() else sorted(root.rglob('free_approach_*.json'))
    for path in paths:
        try:
            content = path.read_bytes()
            result = json.loads(content)
        except (json.JSONDecodeError, OSError):
            print(json.dumps({'path': str(path.resolve()), 'write_in_progress_or_unreadable': True}), flush=True)
            continue
        metadata = result.get('planning_metadata', {})
        backend = metadata.get('backend', '')
        used = backend.endswith('dense_waypoint_v8')
        output = {'scope': 'Read-only CPU path evidence; no physics or formal success inferred',
                  'path': str(path.resolve()), 'sha256': hashlib.sha256(content).hexdigest(),
                  'status': result.get('status'), 'reason': result.get('reason'),
                  'backend': backend, 'waypoint_fallback_used': used,
                  'physical_state_unchanged': metadata.get('physical_state_unchanged'),
                  'auxiliary_state_restored': metadata.get('auxiliary_state_restored_including_construction_exception'),
                  'audit': geometry_summary(result.get('audit', {}))}
        if used:
            original = metadata.get('original_direct_result_before_waypoint', {})
            output.update(original_direct_status=original.get('status'),
                          original_direct_reason=original.get('reason'),
                          waypoint_world_z_m=metadata.get('waypoint_world_z_m'),
                          native_free_via_family_count=metadata.get('native_free_via_family_count'),
                          native_free_goal_family_count=metadata.get('native_free_goal_family_count'),
                          selected_via_branch=metadata.get('selected_via_branch'),
                          selected_goal_branch=metadata.get('selected_goal_branch'),
                          selected_via_declared_goal_index=metadata.get('selected_via_declared_goal_index'),
                          selected_goal_declared_goal_index=metadata.get('selected_goal_declared_goal_index'))
            attempts = []
            for branch in metadata.get('RRT_attempts', []):
                row = {key: branch.get(key) for key in
                       ('via_branch', 'goal_branch', 'raw_join_error_rad', 'raw_path_rows',
                        'retimed_rows', 'audit_passed', 'rejection_reason')}
                for name in ('first_leg', 'second_leg'):
                    leg = branch.get(name, {})
                    row[name] = {key: leg.get(key) for key in
                                 ('RRT_status', 'raw_path_rows', 'rejection_reason', 'numeric_endpoint_match')}
                row['raw_limits_passed'] = branch.get('raw_joint_limit_audit', {}).get('passed')
                row['retimed_limits_passed'] = branch.get('retimed_joint_limit_audit', {}).get('passed')
                row['geometry'] = geometry_summary(branch.get('audit', {}))
                refinement = branch.get('strict_dense_certificate_refinement', {})
                row['dense_refinement'] = {key: refinement.get(key) for key in (
                    'eligible_original_complete_collision_free_insufficient_certificate',
                    'refined_certificate_passed', 'finite_capacity_exhausted',
                    'unchanged_actual_TOPP_vertices', 'sample_truncation_allowed')}
                attempts.append(row)
            output['attempts'] = attempts
        print(json.dumps(output), flush=True)
