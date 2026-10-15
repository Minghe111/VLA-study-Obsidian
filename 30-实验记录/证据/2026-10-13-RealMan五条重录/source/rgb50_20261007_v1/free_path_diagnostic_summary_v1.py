"""Compact read-only extraction of immutable native free-path rejection evidence."""
from pathlib import Path
import json,sys

for arg in sys.argv[1:]:
    root=Path(arg)
    paths=[root] if root.is_file() else sorted(root.rglob('free_approach_*.json'))
    for path in paths:
        result=json.loads(path.read_text(encoding='utf-8'))
        metadata=result.get('planning_metadata',{})
        endpoints=metadata.get('endpoint_attempts',[])
        rows=[]
        for entry in metadata.get('RRT_attempts',[]):
            audit=entry.get('audit',{})
            rows.append({**{k:entry.get(k) for k in
                ('branch','RRT_status','elapsed_s','raw_path_rows','raw_start_error_rad',
                 'raw_end_error_rad','retimed_rows','audit_passed','rejection_reason')},
                'retimed_joint_limits_passed':entry.get('retimed_joint_limit_audit',{}).get('passed'),
                'first_collisions':audit.get('collisions_first_64',[])[:5],
                'minimum_gap_evidence':audit.get('continuous_gap_lower_bound'),
                'collision_pair_sample_occurrences':audit.get('collision_pair_sample_occurrences')})
        print(json.dumps({'file':str(path.resolve()),'index':result.get('index'),
            'status':result.get('status'),'reason':result.get('reason'),
            'IK_attempts':len(endpoints),'IK_converged_count':sum(bool(r.get('IK_converged')) for r in endpoints),
            'within_limit_frozen_endpoint_checks':sum('collision_count' in r for r in endpoints),
            'native_free_endpoint_count':sum(r.get('collision_count')==0 for r in endpoints),
            'selected_branch':metadata.get('selected_branch'),
            'physical_state_unchanged':metadata.get('physical_state_unchanged'),
            'branches':rows}),flush=True)
