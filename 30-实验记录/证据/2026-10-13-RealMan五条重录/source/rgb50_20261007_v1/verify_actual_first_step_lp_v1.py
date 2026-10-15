"""Read actual LP history against actual phase steps; no control changes."""
from pathlib import Path
import json,sys
WORK=Path(__file__).resolve().parent
paths=[WORK/'smoke_worker_right_v8/right_coke_upright_slot25_v8',
       WORK/'smoke_worker_right_feedback_v9/right_coke_upright_slot25_gain6_v9']
rows=[]
for path in paths:
    report=json.loads((path/'expert_report.json').read_text())
    trace=json.loads((path/'physics_trace.json').read_text())
    first_close=next(row['step'] for row in trace if row['phase']['name']=='close')
    actual=[]
    for dense in sorted(set(row['phase']['dense_call'] for row in trace if row['phase']['name']=='after_grasp')):
        first=next(row for row in trace if row['phase']['dense_call']==dense)
        entries=[e for e in report['adaptive_force_allocations'] if e['phase']['dense_call']==dense]
        if not entries:raise ValueError('No LP entry in physically executed after_grasp segment')
        entry=entries[0]
        inferred_step=first_close+entry['step']-1
        actual.append({'dense_call':dense,'first_actual_physics_step':first['step'],
            'first_LP_control_step':entry['step'],'inferred_first_LP_physics_step':inferred_step,
            'first_LP_refresh_delay_physics_steps':inferred_step-first['step'],
            'first_LP_wrench_index':entry['wrench_index'],'first_LP_wrench_mode':entry['wrench_mode'],
            'first_LP_feasible':entry['allocation']['feasible']})
    rows.append({'trial':path.name,'actual_carry_segments':actual,
        'every_executed_segment_refreshed_first_step':all(e['first_LP_refresh_delay_physics_steps']==0 for e in actual)})
result={'scope':'Read-only measured actual LP/physics timeline; no controller, motion or acceptance changes',
        'hypothesis_of_missing_first_step_refresh_supported':False,
        'canonical_schedule_install_already_resets_last_lp_phase':True,'rows':rows}
assert all(row['every_executed_segment_refreshed_first_step'] for row in rows)
output=WORK/'actual_first_step_lp_verification_v1.json'
with output.open('x',encoding='utf-8') as stream:stream.write(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
