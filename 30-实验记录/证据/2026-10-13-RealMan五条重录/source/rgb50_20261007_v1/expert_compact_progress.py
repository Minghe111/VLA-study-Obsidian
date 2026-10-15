"""Compact read-only inspection of own expert trials."""
import json,sys
from pathlib import Path
def pick(value,keys):
    return {k:value.get(k) for k in keys if k in value} if isinstance(value,dict) else value
for entry in sys.argv[1:]:
    root=Path(entry)
    if root.is_file() and root.suffix=='.jsonl':
        for line in root.read_text().splitlines():
            value=json.loads(line)
            print(json.dumps(pick(value,('trial','accepted','error','physical_steps','final_phase','final_relative_translation_drift_m','final_relative_rotation_drift_deg','final_bottle_bottom_gap_m','frozen_relative_motion_decomposition','final_contact_forces_N'))),flush=True)
        continue
    for path in sorted(root.rglob('expert_report.json')):
        r=json.loads(path.read_text())
        row={k:r.get(k) for k in ('accepted','physical_steps','error','expert_version','stage','active_arm','pre_retreat_grip','strict_placement_pair')}
        row.update(path=str(path),planning=[pick(c,('index','side','backend','status','samples','reason','target_pose','full_mesh_audit_passed')) for c in r.get('planning_calls',[])],selected=r.get('selected_carry_candidates'))
        for candidate in sorted(path.parent.glob('native_carry_*_result.json')):
            d=json.loads(candidate.read_text());row[candidate.stem]=pick(d,('status','reason','samples','index','candidate','physical_state_unchanged','scope'))
        for audit in sorted(path.parent.glob('mesh_path_*.json')):
            d=json.loads(audit.read_text());row[audit.stem]={k:d.get(k) for k in ('passed','failure_reason_counts','minimum_table_clearance_m','minimum_bottle_clearance_m','minimum_self_clearance_m')}
            row[audit.stem]['first_collisions']=d.get('collisions_first_64',[])[:3]
            row[audit.stem]['continuous_gap_lower_bound']=d.get('continuous_gap_lower_bound')
        print(json.dumps(row,default=str),flush=True)
