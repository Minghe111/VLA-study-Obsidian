"""Read-only compact direct/via path evidence, including auxiliary restoration."""
from pathlib import Path
import json,sys

def branches(metadata):
    rows=[]
    for entry in metadata.get('RRT_attempts',[]):
        audit=entry.get('audit',{})
        row={k:entry.get(k) for k in ('branch','via_branch','goal_branch','RRT_status',
            'elapsed_s','raw_path_rows','raw_end_error_rad','retimed_rows','audit_passed','rejection_reason') if k in entry}
        for key in ('first_leg','second_leg'):
            if key in entry:row[key]=entry[key]
        row.update(first_collisions=audit.get('collisions_first_64',[])[:3],
            minimum_gaps=audit.get('continuous_gap_lower_bound'))
        rows.append(row)
    return rows
for arg in sys.argv[1:]:
    path=Path(arg)
    result=json.loads(path.read_text(encoding='utf-8'));m=result.get('planning_metadata',{})
    original=m.get('original_direct_planning_metadata',{})
    print(json.dumps({'file':str(path.resolve()),'status':result.get('status'),'reason':result.get('reason'),
        'physical_state_unchanged':m.get('physical_state_unchanged'),
        'auxiliary_restored':m.get('auxiliary_state_restored_including_construction_exception'),
        'move_group_restored':m.get('original_auxiliary_move_group_restored'),
        'original_direct_status':m.get('original_direct_status'),
        'original_direct_branches':branches(original),
        'endpoints':{label:{'attempts':sum(e.get('target')==label for e in m.get('endpoint_attempts',[])),
            'native_free':sum(e.get('target')==label and e.get('collision_count')==0 for e in m.get('endpoint_attempts',[]))}
            for label in sorted({e.get('target','original') for e in m.get('endpoint_attempts',[])})},
        'waypoint_branches':branches(m)}),flush=True)
