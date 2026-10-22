from pathlib import Path
import json,gzip,hashlib,collections,math
import numpy as np
ROOT=Path('/home/admin123/liminghe/VLA-WAM-RL/30-实验记录/evidence/2026-10-22-labware-agent-pipeline');p=ROOT/'robot_cap_task/left_contact_probes/left115_first_contact_v6_ordered_true_surface'
def load(n):return json.loads((p/n).read_text())
def sha(x):return hashlib.sha256(Path(x).read_bytes()).hexdigest()
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
manifest=load('snapshot_manifest.json');snapok={k:sha(v['snapshot'])==v['sha256'] for k,v in manifest.items()};prefixpath=Path(next(v['snapshot'] for k,v in manifest.items() if k.endswith('/transitions.json')));prefix=json.loads(prefixpath.read_text());rows=load('transitions.json');partial=load('partial_command.json');terminal=load('actual_terminal_snapshot.json');result=load('result.json');proposals=load('controller_proposals.json');positive=[];manifolds=0;zero=0;rawcount=0;active=0;counts=collections.Counter();chainfail=0;requestedfail=0;tickfail=0;enddriveerr=0.;previous=None;lastworld=None;dt=0.;maxleftbody=0.
for w in stream(p/'world_kinematics_pre_post.json.gz'):
 rawcount+=1;c=w['context'];requested=c['requested_command24']
 if requested is None:continue
 n=c['active_tick_before_step'];active+=1;idx=c['action_index'];counts[idx]+=1
 if n!=active-1:tickfail+=1
 expected=rows[idx]['action'] if idx<len(rows) else partial['requested_target24']
 if requested!=expected:requestedfail+=1
 if previous is not None:
  for key in ['body','cap']:
   if previous['post'][key]!=w['pre'][key]:chainfail+=1
  if previous['post']['tablet_world_pose_pq']!=w['pre']['tablet_world_pose_pq']:chainfail+=1
 if idx<len(rows) and(n+1)%15==0:enddriveerr=max(enddriveerr,float(abs(np.array(c['actual_drive_command24'])-np.array(expected)).max()))
 for contact in w['contacts_post']:
  ids=contact['body_actor_ids'];names=contact['body_names']
  if not any(name.startswith('l_inspire_') for name in names):continue
  if 103 in ids:maxleftbody=max(maxleftbody,sum(x['normal_impulse_abs_Ns'] for x in contact['points'])/w['dt_s'])
  if 104 not in ids:continue
  manifolds+=1
  for point in contact['points']:
   impulse=point['normal_impulse_abs_Ns']
   if impulse==0:zero+=1
   else:positive.append({'active_tick':n+1,'assembly_step':w['physics_step'],'phase':c['phase'],'action_index':idx,'dt_s':w['dt_s'],'body_names':names,'body_actor_ids':ids,'point':point,'post_cap_world_pose_pq':w['post']['cap']['world_pose_pq']})
 previous=w;lastworld=w;dt=w['dt_s']
separation_positive=[];separation_total=0
for row in stream(p/'left_cap_separation_trace.json.gz'):
 separation_total+=1
 for contact in row['contacts']:
  for point in contact['points']:
   normal=np.array(point['normal_raw_world']);imp=np.array(point['impulse_raw_world_Ns']);normalimp=abs(float(imp@normal))
   if normalimp>0:separation_positive.append({'physics_tick':row['physics_tick'],'phase':row['phase'],'actor_names':contact['actor_names'],'actor_ids':contact['actor_ids'],**point,'normal_impulse_abs_Ns':normalimp,'normal_force_N':normalimp/dt})
assert len(positive)==len(separation_positive)==1
point=positive[0];sp=separation_positive[0];rawnormal=np.array(sp['normal_raw_world']);rawimpulse=np.array(sp['impulse_raw_world_Ns']);normal=rawnormal if sp['actor_ids'][0]==104 else -rawnormal;normal/=np.linalg.norm(normal)
cap=np.array(terminal['cap_matrix']);cp=np.array(sp['position_world_m']);R=cap[:3,:3];cap_point=R.T@(cp-cap[:3,3]);cap_normal=R.T@normal;radial=cap_point.copy();radial[2]=0;radial/=np.linalg.norm(radial);tangent=np.array([-radial[1],radial[0],0.]);N=sp['normal_force_N'];orientedforce=normal*N;moment=np.cross(cp-cap[:3,3],orientedforce);momentcap=R.T@moment;impnormal=float(rawimpulse@rawnormal);tangentimp=rawimpulse-impnormal*rawnormal
prefixexact=all(all(rows[i][k]==prefix[i][k] for k in ['phase','state','action','next_state','tick_start','tick_end'])for i in range(684));lastcomplete=rows[-1]
o={'scope':'Finite read-only independent actual v6 first-load/normal-direction/partial audit; no newphysics, no newgate, no live sourcewrites','input_sha256':{str(p/n):sha(p/n) for n in ['result.json','snapshot_manifest.json','transitions.json','partial_command.json','actual_terminal_snapshot.json','world_kinematics_pre_post.json.gz','left_cap_separation_trace.json.gz']},'source':{'all_snapshots_sha256_match':all(snapok.values()),'snapshot_count':len(snapok),'entry':manifest[str(ROOT/'robot_cap_task/left_cap_contact_probe.py')]['sha256'],'controller':manifest[str(ROOT/'robot_cap_task/cap_contact_controller.py')]['sha256'],'observer':manifest[str(ROOT/'robot_cap_task/left_cap_contact_task.py')]['sha256']},'control_evidence':{'complete_targets':len(rows),'prefix684_state_action_nextstate_tick_exact':prefixexact,'continuation_complete_phases':dict(collections.Counter(x['phase'] for x in rows[684:])),'controller_proposal_count':len(proposals),'active_steps':active,'raw_total_rows':rawcount,'expected_active_steps':len(rows)*15+partial['executed_ticks'],'all_complete_rows15ticks':all(x['tick_end']-x['tick_start']==15 for x in rows),'each_complete_action15_raw_rows':all(counts[i]==15 for i in range(len(rows))),'partial_action_raw_rows':counts[len(rows)],'active_tick_failures':tickfail,'requested_action_failures':requestedfail,'prepost_bodycap12pill_chain_mismatches':chainfail,'complete_actualdrive_endpoint_max_error':enddriveerr,'lastcomplete_tick_end':lastcomplete['tick_end'],'partial_start':partial['start_tick'],'partial_end':partial['end_tick'],'partial_executed_ticks':partial['executed_ticks'],'partial_before_state_matches_lastnextstate':partial['before']['state24']==lastcomplete['next_state'],'partial_after_state_matches_terminal':partial['after']['state24']==terminal['state24'],'partial_actualdrive_matches_terminal':partial['actual_drive_target24_at_stop']==terminal['actual_drive_target24'],'last_world_context_drive_matches_partial_after':lastworld['context']['actual_drive_command24']==partial['actual_drive_target24_at_stop'],'partial_not_appended_as_full_transition':len(rows)==732 and terminal['physics_tick']==10981},'raw_contact':{'cap_entity_id':104,'active_matching_manifolds':manifolds,'zero_impulse_point_count':zero,'positive_normal_point_count':len(positive),'first_positive_point_world_trace':point,'same_positive_point_separation_trace':sp,'normal_force_N':N,'observed_event_force_N':result['first_contact']['left_contact']['forces_N'][4],'separation_trace_rows':separation_total,'left_body_max_force_from_raw_N':maxleftbody},'contact_geometry_at_actual_terminal':{'participant_order':sp['actor_names'],'raw_normal_world':rawnormal.tolist(),'raw_impulse_world_Ns':rawimpulse.tolist(),'code_cap_oriented_unit_normal_world':normal.tolist(),'contact_position_world_m':cp.tolist(),'contact_position_in_cap_frame_m':cap_point.tolist(),'code_cap_oriented_normal_in_cap_frame':cap_normal.tolist(),'normal_dot_outward_radial':float(cap_normal@radial),'normal_dot_azimuth_tangent':float(cap_normal@tangent),'normal_dot_cap_axis':float(cap_normal[2]),'code_oriented_normal_force_moment_cap_Nm':momentcap.tolist(),'raw_tangent_impulse_norm_Ns':float(np.linalg.norm(tangentimp)),'sign_boundary':'Cap-oriented normal uses existing observer participant-order conversion. Absolute impulse-on-body convention is not newly calibrated here; oriented moment is geometry calculation, not measured grasp/turn torque.'},'outcome':{'error':result['error'],'guard_failure':result['guard_failure'],'forbidden_left_terminal_N':result['final_metrics']['forbidden_left_contact_N'],'first_contact_observed':result['first_contact_observed'],'left_cap_opposed':result['final_metrics']['left_cap_contact']['opposed'],'cap_released':result['final_metrics']['cap_state']['released'],'cap_axial_lift_m':result['final_metrics']['cap_state']['lift_measured_m'],'cap_closed_observed':result['final_metrics']['cap_closed_observed'],'unscrew_verified':result['final_metrics']['unscrew_verified'],'pour_verified':result['final_metrics']['pour_verified'],'full_task_success':result['full_task_success'],'meaning':'Exactly one thumb-side normal load observed on thefirststep of a partial60ms command. Index-sidecap load/opposition, sustainedpinch andfrictionalturn were not demonstrated.'},'next_contact_evidence_needed':'Use actual normals/positions to acquire measured opposite index+thumb load and inspect tangential/friction capacity before attributing any cap rotation to the robot. Do not count historical right-body opposition as cap opposition.','new_physical_trials':0,'live_source_modified':False}
assert all(snapok.values()) and prefixexact and len(rows)==732 and active==10981 and tickfail==chainfail==requestedfail==0 and counts[len(rows)]==1
assert partial['start_tick']==10980 and partial['end_tick']==10981 and sp['physics_tick']==10981 and point['active_tick']==10981
(p/'independent_v6_contact_audit.json').write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:o[k] for k in ['source','control_evidence','raw_contact','contact_geometry_at_actual_terminal','outcome']},ensure_ascii=False,indent=2))
