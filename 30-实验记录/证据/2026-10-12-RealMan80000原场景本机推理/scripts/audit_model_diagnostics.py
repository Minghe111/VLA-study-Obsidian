"""Observe terminal hand/support/upright from newly recorded physics only."""
from pathlib import Path
import json,gzip,math,hashlib,argparse
import numpy as np
ROOT=Path(__file__).resolve().parent.parent
ap=argparse.ArgumentParser();ap.add_argument('--indices',nargs='+',type=int,default=[0,38,30,31,17,7]);args=ap.parse_args()
rows=[]
for i in args.indices:
    p=ROOT/'episodes'/f'ep{i:02d}_model80000';r=json.loads((p/'report.json').read_text())
    last=None;count=0
    with gzip.open(p/'actual_physical_trace.jsonl.gz','rt') as f:
        for line in f:last=line;count+=1
    terminal=json.loads(last);g=terminal['contact_observation']
    gap=float(terminal['native_bottom_gap_m']);held=bool(terminal['opposed_contact_observed'] and gap>=.01 and g['table_normal_N']<=1e-5)
    axis=np.asarray(terminal['bottle_world_pose'])[:3,1];upright=math.degrees(math.acos(np.clip(axis[2]/np.linalg.norm(axis),-1,1)))
    row={'index':i,'native_success':r['native_task_success'],'air_grasp_1cm_1s':r['observation']['user_air_grasp_observed_1cm_1s'],
         'longest_continuous_air_grasp_s':r['observation']['longest_observed_air_grasp_s'],
         'terminal_air_grasp':held,'terminal_opposed_hand_contact':bool(terminal['opposed_contact_observed']),
         'terminal_bottom_gap_m':gap,'terminal_table_normal_N':float(g['table_normal_N']),
         'terminal_upright_error_deg':upright,'diagnostic_user_completion':bool(r['observation']['user_air_grasp_observed_1cm_1s'] and held and upright<=10.),
         'actual_trace_rows':count,'actual_trace_sha256':hashlib.sha256((p/'actual_physical_trace.jsonl.gz').read_bytes()).hexdigest(),
         'report_sha256':hashlib.sha256((p/'report.json').read_bytes()).hexdigest()}
    assert count==6000 and r['actual_physics_steps']==6000 and r['actual_action_count']==400 and r['model_query_count']==25 and not r['interface_or_infrastructure_error']
    (p/'posthoc_diagnostic.json').write_text(json.dumps(row,indent=2)+'\n');rows.append(row)
summary={'scope':'Actual model physics posthoc observations only; not used for controls;6selected training starts, not full benchmarkSR',
         'completion_observation_thresholds':{'past_continuous_air_hold_min_s':1.,'terminal_bottom_gap_min_m':.01,'terminal_table_support_max_N':1e-5,'terminal_upright_error_max_deg':10.,'terminal_opposed_hand_contact_required':True},
         'thresholds_declared_before_remaining5_completed':True,
         'native_count':sum(r['native_success'] for r in rows),'air_grasp_1cm_1s_count':sum(r['air_grasp_1cm_1s'] for r in rows),
         'terminal_air_grasp_count':sum(r['terminal_air_grasp'] for r in rows),'diagnostic_user_completion_count':sum(r['diagnostic_user_completion'] for r in rows),'episodes':rows}
name='six_model_posthoc_diagnostics.json' if args.indices==[0,38,30,31,17,7] else 'partial_model_posthoc_diagnostics.json'
(ROOT/name).write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
