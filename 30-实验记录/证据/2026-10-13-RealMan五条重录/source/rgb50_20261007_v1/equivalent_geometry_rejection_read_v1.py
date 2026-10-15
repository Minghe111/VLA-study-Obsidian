"""Read-only sampled and continuous gap classification for saved native audits."""
from pathlib import Path
import json
import sys

for argument in sys.argv[1:]:
    for path in sorted(Path(argument).rglob('free_approach_*.json')):
        data = json.loads(path.read_text(encoding='utf-8'))
        rows = []
        for rrt in data.get('planning_metadata', {}).get('RRT_attempts', []):
            audit = rrt.get('audit')
            if audit is None:
                continue
            thresholds = audit.get('thresholds_m', {})
            certified = audit.get('continuous_gap_lower_bound', {})
            sampled = audit.get('sampled_minimum_gap', {})
            rows.append({'branch': rrt['branch'], 'audit_passed': audit['passed'],
                         'raw_predeclared_match_error_rad': rrt.get('numeric_endpoint_match', {}).get('raw_end_error_rad'),
                         'actual_FCL_collision_pair_count': audit.get('unique_collision_pairs'),
                         'failed_bound_categories': [key for key, value in certified.items() if value['lower_bound_m'] < thresholds[key]],
                         'continuous_gap_lower_bound': certified,
                         'sampled_minimum_gap': sampled,
                         'original_clearance_thresholds_m': thresholds,
                         'first_FCL_collisions': audit.get('collisions_first_64', [])[:4]})
        print(json.dumps({'file': str(path.resolve()), 'status': data.get('status'), 'branches': rows,
                          'no_acceptance_or_threshold_change_performed': True}), flush=True)
