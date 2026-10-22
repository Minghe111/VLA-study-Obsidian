from pathlib import Path
import json,gzip,hashlib,collections,math
import numpy as np
R=Path('/home/admin123/liminghe/VLA-WAM-RL/30-实验记录/evidence/2026-10-22-labware-agent-pipeline');p=R/'robot_cap_task/left_contact_probes/left115_first_contact_v4_axisymmetric'
def load(n):return json.loads((p/n).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stream(path):
 d=json.JSONDecoder();b='';started=False
 with gzip.open(path,'rt') as f:
  while True:
   b=b.lstrip()
   if not started:
    if not b:b=f.read(1<<20);continue
    assert b[0]=='[';b=b[1:];started=True;continue
   b=b.lstrip(', \r\n\t')
   if b.startswith(']'):return
   try:o,i=d.raw_decode(b)
   except json.JSONDecodeError:
    extra=f.read(1<<20)
    if not extra:raise
    b+=extra;continue
   b=b[i:];yield o
manifest=load('snapshot_manifest.json');snapchecks={k:sha(v['snapshot'])==v['sha256'] for k,v in manifest.items()};prefixpath=Path(next(v['snapshot'] for k,v in manifest.items() if k.endswith('/left115_first_contact_v3_world_transit/transitions.json')));prefix=json.loads(prefixpath.read_text());rows=load('transitions.json');proposals=load('controller_proposals.json');preclose=load('actual_preclose_snapshot.json');terminal=load('actual_terminal_snapshot.json');res=load('result.json')
total=0;active=0;close=0;ctxcount=collections.Counter();tickfail=0;chainfail=0;requestfail=0;driveendmax=0.;previous=None;manifolds=[];firstpositive=None;maxforce=0.;zeropoints=0;positivepoints=0;sumimpulse=0.;closedfalseclose=0;releaseclose=0
for w in stream(p/'world_kinematics_pre_post.json.gz'):
 total+=1;c=w['context']
 if c['requested_command24'] is None:continue
 active+=1;n=c['active_tick_before_step'];idx=c['action_index'];ctxcount[idx]+=1;phase=c['phase'];isclose=phase=='left_first_contact_lowforce'
 if isclose:close+=1
 if n!=active-1:tickfail+=1
 if previous is not None:
  for key in ['body','cap']:
   if previous['post'][key]!=w['pre'][key]:chainfail+=1
  if previous['post']['tablet_world_pose_pq']!=w['pre']['tablet_world_pose_pq']:chainfail+=1
 if c['requested_command24']!=rows[idx]['action']:requestfail+=1
 if (n+1)%15==0:driveendmax=max(driveendmax,float(abs(np.array(c['actual_drive_command24'])-np.array(rows[idx]['action'])).max()))
 stepforce=0.
 for contact in w['contacts_post']:
  if 104 not in contact['body_actor_ids'] or not any(x.startswith('l_inspire_') for x in contact['body_names']):continue
  pointrows=[]
  for point in contact['points']:
   impulse=float(point['normal_impulse_abs_Ns']);sumimpulse+=impulse
   if impulse==0:zeropoints+=1
   else:positivepoints+=1
   pointrows.append(point);stepforce+=impulse/w['dt_s']
  manifolds.append({'active_tick':n+1,'assembly_step':w['physics_step'],'phase':phase,'action_index':idx,'body_names':contact['body_names'],'body_actor_ids':contact['body_actor_ids'],'points':pointrows})
 maxforce=max(maxforce,stepforce)
 if stepforce>0 and firstpositive is None:firstpositive={'active_tick':n+1,'assembly_step':w['physics_step'],'phase':phase,'normal_force_N':stepforce}
 s=w['post_assembly_state']
 if isclose:
  isclosed=(not s['released'] and abs(s['rotation_measured_rad'])<.15 and abs(s['lift_measured_m'])<.001 and s['radial_error_m']<.001 and s['axis_tilt_rad']<.02)
  closedfalseclose+=int(not isclosed);releaseclose+=int(s['released'])
 previous=w
# Compare physical snapshot, without changing either state.
b=np.array(preclose['cap_matrix']);a=np.array(terminal['cap_matrix']);reld=np.linalg.inv(b)@a;angle=lambda m:float(np.arccos(np.clip((np.trace(m)-1)/2,-1,1)));z0=b[:3,2]/np.linalg.norm(b[:3,2]);z1=a[:3,2]/np.linalg.norm(a[:3,2]);lastprop=np.array(proposals[-1]['sdk6']);sdkactual=np.array(terminal['state24'][6:12]);sdkdrive=np.array(terminal['actual_drive_target24'][6:12])
prefixexact=all(all(rows[i][k]==prefix[i][k] for k in ['phase','state','action','next_state','tick_start','tick_end']) for i in range(505))
o={'scope':'Read-only independent actual v4 output/4ms raw cap-left contact audit, no physics/source/threshold changes','input_sha256':{str(p/n):sha(p/n) for n in ['result.json','snapshot_manifest.json','transitions.json','world_kinematics_pre_post.json.gz','controller_proposals.json','actual_preclose_snapshot.json','actual_terminal_snapshot.json']},'sources':{'all22_snapshots_sha256_match':all(snapchecks.values()),'snapshot_count':len(snapchecks),'entry_sha256':manifest[str(R/'robot_cap_task/left_cap_contact_probe.py')]['sha256'],'controller_sha256':manifest[str(R/'robot_cap_task/cap_contact_controller.py')]['sha256']},'actual_execution':{'complete_targets':len(rows),'first505exact_state_action_nextstate_tick':prefixexact,'phase_target_counts':dict(collections.Counter(x['phase'] for x in rows)),'new_final_segment_targets':sum(x['phase']=='left_approach_via5' for x in rows),'actual_close_targets':sum(x['phase']=='left_first_contact_lowforce' for x in rows),'controller_proposal_count':len(proposals),'all_transitions_15_ticks':all(x['tick_end']-x['tick_start']==15 for x in rows),'state_tick_chain_exact':all(rows[i]['next_state']==rows[i+1]['state'] and rows[i]['tick_end']==rows[i+1]['tick_start'] for i in range(len(rows)-1)),'raw_total_rows':total,'active_rows':active,'closing_rows':close,'active_tick_failures':tickfail,'each_target_has15rawrows':all(ctxcount[i]==15 for i in range(len(rows))),'prepost_body_cap_and12pill_chain_mismatches':chainfail,'requested_action_failures':requestfail,'max_end_drive_minus_requested24':driveendmax,'preclose_active_tick':preclose['physics_tick'],'terminal_active_tick':terminal['physics_tick'],'terminal_state_equals_lastnextstate':terminal['state24']==rows[-1]['next_state'],'partial':load('partial_command.json')},'raw_cap_left_contact':{'actor_id':104,'all_active_matching_manifold_count':len(manifolds),'matching_manifold_rows_saved':'independent_v4_raw_left_cap_manifolds.json','closing_matching_manifold_count':sum(x['phase']=='left_first_contact_lowforce' for x in manifolds),'zero_normal_impulse_point_count':zeropoints,'positive_normal_impulse_point_count':positivepoints,'sum_normal_impulse_Ns':sumimpulse,'max_aggregated_normal_force_N':maxforce,'first_positive_normal_impulse_event':firstpositive,'first_matching_manifold':manifolds[0] if manifolds else None,'last_matching_manifold':manifolds[-1] if manifolds else None,'interpretation':'No cap-left contact manifold in the actual raw PhysX contact stream' if not manifolds else 'Manifolds exist; force/impulse counts above distinguish zero/weak-positive loads from the1mN event'},'control_endpoint':{'first_proposal_sdk6':proposals[0]['sdk6'],'last_proposal_sdk6':lastprop.tolist(),'actual_terminal_drive_sdk6_unclipped':sdkdrive.tolist(),'actual_terminal_measured_sdk6':sdkactual.tolist(),'last_proposal_latched':proposals[-1]['latched_index_thumb'],'last_proposal_filtered_index_thumb_force_N':proposals[-1]['filtered_index_thumb_force_N'],'last_proposal_ready':proposals[-1]['ready'],'scope':'Original screened range retained; metadata endpoint does not prove actual force-bearing contact'},'closing_cap_motion':{'cap_center_displacement_m':float(np.linalg.norm(a[:3,3]-b[:3,3])),'cap_world_rotation_delta_rad':angle(reld[:3,:3]),'cap_axis_angle_delta_rad':float(np.arccos(np.clip(z0@z1,-1,1))),'closing_ticks_closed_false':closedfalseclose,'closing_ticks_released':releaseclose,'terminal_cap_state':res['final_metrics']['cap_state']},'outcome':{'error':res['error'],'guard_failure':res['guard_failure'],'first_contact_observed':res['first_contact_observed'],'full_task_success':res['full_task_success'],'reason':'Finite160-boundary closing budget actually executed and exhausted; not an observer tolerance or planning rejection','limit':'Do not infer successful cap grip/unscrew/pour, nor blame large cap drift from the endpoint displacement alone'},'new_physical_trials':0,'live_source_modified':False}
assert all(snapchecks.values()) and prefixexact and len(rows)==684 and close==2400 and active==10260 and tickfail==chainfail==requestfail==0
(p/'independent_v4_raw_left_cap_manifolds.json').write_text(json.dumps(manifolds,indent=2)+'\n');(p/'independent_v4_audit.json').write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:o[k] for k in ['sources','actual_execution','raw_cap_left_contact','control_endpoint','closing_cap_motion']},indent=2))
