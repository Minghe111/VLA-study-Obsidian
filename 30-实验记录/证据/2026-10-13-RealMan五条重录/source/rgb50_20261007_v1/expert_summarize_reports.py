"""Read-only compact physical-trial report inspection."""
import json,sys
from pathlib import Path
for root in sys.argv[1:]:
    root=Path(root)
    for p in sorted(root.glob('*/expert_report.json')):
        r=json.loads(p.read_text());row={k:r.get(k) for k in ('accepted','result','physical_steps','error','active_arm')}
        row['trial']=p.parent.name;row['planning']=r.get('planning_calls')
        if r.get('pre_retreat_grip'):row['pre_retreat_grip']=r['pre_retreat_grip']
        for q in sorted(p.parent.glob('mesh_path_*.json')):
            a=json.loads(q.read_text());row[q.stem]={k:a.get(k) for k in ('passed','reason','violations','continuous_gap_lower_bound')}
        for q in sorted(p.parent.glob('carry_wrench_preflight_*.json')):
            a=json.loads(q.read_text());row[q.stem]={k:a.get(k) for k in ('passed','failure_count','failure_reason_counts')}
        print(json.dumps(row,default=str),flush=True)
