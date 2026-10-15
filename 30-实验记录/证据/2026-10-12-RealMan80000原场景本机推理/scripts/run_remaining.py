"""Sequential real episodes, sharing the single already-loaded local policy."""
from pathlib import Path
import os,sys,subprocess,json,time
ROOT=Path(__file__).resolve().parent.parent
IDENTITY=ROOT/'policy_runtime/service_v1/model_identity.json'
SHA='1d01a7b0902e34a432c15b6d8f4a83175253d7d98e5dad43ca6fe79dcb04d1db'
rows=[]
def compact(r):
    return {'index':r['index'],'native_task_success':r['native_task_success'],'model_query_count':r['model_query_count'],
            'actual_action_count':r['actual_action_count'],'actual_physics_steps':r['actual_physics_steps'],
            'physical_duration_s':r['physical_duration_s'],'wall_duration_s':r['wall_duration_s'],
            'air_grasp_1cm_1s':r['observation']['user_air_grasp_observed_1cm_1s'],
            'continuous_air_hold_s':r['observation']['longest_observed_air_grasp_s'],
            'terminal_upright_error_deg':r['observation']['terminal_upright_error_deg'],
            'first_native_success_sim_time_s':r['observation']['first_native_success_sim_time_s'],
            'maximum_bottom_gap_m':r['observation']['maximum_native_bottom_gap_m'],
            'interface_or_infrastructure_error':r['interface_or_infrastructure_error']}
rows.append(compact(json.loads((ROOT/'episodes/ep00_model80000/report.json').read_text())))
for pos,index in enumerate((38,30,31,17,7),start=2):
    out=ROOT/'episodes'/f'ep{index:02d}_model80000';log=ROOT/'episodes'/f'ep{index:02d}_model80000.log'
    args=[sys.executable,str(ROOT/'scripts/run_model.py'),'--index',str(index),'--episode-position',str(pos),
          '--identity',str(IDENTITY),'--identity-sha256',SHA,'--output',str(out)]
    print('START_EPISODE',index,pos,flush=True)
    with log.open('x') as output:
        result=subprocess.run(args,env=os.environ,stdout=output,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'Episode{index} failed infrastructure; inspect{log}')
    r=json.loads((out/'report.json').read_text());row=compact(r);rows.append(row)
    print('COMPLETED_EPISODE',json.dumps(row),flush=True)
    (ROOT/'partial_six_model_summary.json').write_text(json.dumps({'episodes':rows},indent=2)+'\n')
summary={'scope':'Six chosen training original scenes, checkpoint80000 actual closed-loop model inference; small in-distribution diagnostic, not full benchmark SR',
         'selected_indices':[0,38,30,31,17,7],'native_success_count':sum(r['native_task_success'] for r in rows),
         'air_grasp_1cm_1s_count':sum(r['air_grasp_1cm_1s'] for r in rows),'actual_model_queries':sum(r['model_query_count'] for r in rows),
         'actual_model_actions':sum(r['actual_action_count'] for r in rows),'actual_physics_steps':sum(r['actual_physics_steps'] for r in rows),
         'stop_on_success':False,'policy_full_horizon':16,'model_service_restarted_or_rng_reset_between_cases':False,
         'future_expert_actions_read_or_used':False,'episodes':rows}
(ROOT/'six_model_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('SIX_COMPLETED',json.dumps(summary),flush=True)
