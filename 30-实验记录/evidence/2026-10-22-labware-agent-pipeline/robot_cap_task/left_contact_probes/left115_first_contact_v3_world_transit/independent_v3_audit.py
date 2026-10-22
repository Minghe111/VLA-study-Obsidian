from pathlib import Path
import json,gzip,hashlib,collections,math
import numpy as np
ROOT=Path('/home/admin123/liminghe/VLA-WAM-RL/30-实验记录/evidence/2026-10-22-labware-agent-pipeline');p=ROOT/'robot_cap_task/left_contact_probes/left115_first_contact_v3_world_transit'
def load(n):return json.loads((p/n).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stream(path):
 d=json.JSONDecoder();b='';started=False;end=False
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
manifest=load('snapshot_manifest.json');snapok={k:sha(v['snapshot'])==v['sha256'] for k,v in manifest.items()};source=Path(next(v['snapshot'] for k,v in manifest.items() if k.endswith('/static_hold_probes/held115_static_reference_v1/transitions.json')));prefix=json.loads(source.read_text())
rows=load('transitions.json');result=load('result.json');terminal=load('actual_terminal_snapshot.json');prep=load('actual_preapproach_snapshot.json');plans=load('planning_reports.json');bindings=load('actual_cap_rebindings.json');firstcap=None;firstclosedfalse=None;firstclosedfalse_approach=None;prev=None;worldcount=0;activecount=0;phasecounts=collections.Counter();ctxcount=collections.Counter();tickfail=0;chainfail=0;requestedfail=0;drive_end_error=0.;leftcapmax=0.;leftbodymax=0.;rightcapmax=0.;maxdt=0.;laststate=None;commandvectors=0;frozenrightmax=0.;closed_false_count=0;released_count=0
for w in stream(p/'world_kinematics_pre_post.json.gz'):
 worldcount+=1;maxdt=max(maxdt,abs(w['dt_s']-.004));c=w['context'];a=c.get('requested_command24')
 if a is None:continue
 n=c['active_tick_before_step'];phase=c['phase'];idx=c['action_index'];activecount+=1;phasecounts[phase]+=1;ctxcount[idx]+=1
 if n!=activecount-1:tickfail+=1
 if prev is not None:
  for body in ['body','cap']:
   if prev['post'][body]!=w['pre'][body]:chainfail+=1
  if prev['post']['tablet_world_pose_pq']!=w['pre']['tablet_world_pose_pq']:chainfail+=1
 if a!=rows[idx]['action']:requestedfail+=1
 drive=np.array(c['actual_drive_command24']);assert drive.shape==(24,) and np.isfinite(drive).all();commandvectors+=1
 if (n+1)%15==0:drive_end_error=max(drive_end_error,float(abs(drive-np.array(rows[idx]['action'])).max()))
 if phase.startswith('left_'):frozenrightmax=max(frozenrightmax,float(abs(drive[12:18]-np.array(prep['actual_drive_target24'])[12:18]).max()))
 forces={'leftcap':0.,'leftbody':0.,'rightcap':0.}
 for contact in w['contacts_post']:
  names=contact['body_names'];ids=contact['body_actor_ids'];f=sum(x['normal_impulse_abs_Ns'] for x in contact['points'])/w['dt_s']
  if any(n.startswith('l_inspire_') for n in names):
   if 104 in ids:forces['leftcap']+=f
   if 103 in ids:forces['leftbody']+=f
  if any(n.startswith('r_inspire_') for n in names) and 104 in ids:forces['rightcap']+=f
 leftcapmax=max(leftcapmax,forces['leftcap']);leftbodymax=max(leftbodymax,forces['leftbody']);rightcapmax=max(rightcapmax,forces['rightcap'])
 if forces['leftcap']>.001 and firstcap is None:firstcap={'active_tick':n+1,'assembly_step':w['physics_step'],'phase':phase,'force_N':forces['leftcap']}
 s=w['post_assembly_state'];closed=(not s['released'] and abs(s['rotation_measured_rad'])<.15 and abs(s['lift_measured_m'])<.001 and s['radial_error_m']<.001 and s['axis_tilt_rad']<.02)
 if not closed:
  closed_false_count+=1
  info={'active_tick':n+1,'assembly_step':w['physics_step'],'phase':phase,'cap_state':s}
  if firstclosedfalse is None:firstclosedfalse=info
  if phase.startswith('left_approach') and firstclosedfalse_approach is None:firstclosedfalse_approach=info
 if s['released']:released_count+=1
 laststate=s;prev=w
boundaryforbidden={k:max((r['metrics'].get('forbidden_left_contact_N',{}).get(k,0.) for r in rows[294:]),default=0.) for k in ['left_body','left_table','left_right']}
check={'snapshot_count':len(snapok),'all_snapshot_sha256_match':all(snapok.values()),'entry_sha256':manifest[str(ROOT/'robot_cap_task/left_cap_contact_probe.py')]['sha256'],'complete_targets':len(rows),'prefix294_exact_action_state_nextstate_tick':all(all(rows[i][k]==prefix[i][k] for k in ['phase','state','action','next_state','tick_start','tick_end']) for i in range(294)),'preparation_targets':sum(r['phase']=='left_prepare_open400' for r in rows),'approach_targets':sum(r['phase'].startswith('left_approach') for r in rows),'approach_targets_per_via':dict(collections.Counter(r['phase'] for r in rows if r['phase'].startswith('left_approach'))),'all_transitions_15_ticks':all(r['tick_end']-r['tick_start']==15 for r in rows),'transition_tick_chain_exact':all(rows[i]['tick_end']==rows[i+1]['tick_start'] for i in range(len(rows)-1)),'transition_state_chain_exact':all(rows[i]['next_state']==rows[i+1]['state'] for i in range(len(rows)-1)),'terminal_tick':terminal['physics_tick'],'terminal_state_matches_lastnextstate':terminal['state24']==rows[-1]['next_state'],'partial':load('partial_command.json'),'controller_proposals':len(load('controller_proposals.json')),'planning_reports':[{k:r.get(k) for k in ['scoped_left_sampled_path_passed','unique_collision_pairs','audited_samples','joint_limit_audit','scope']} for r in plans],'completed_binding_via_indices':[r['via_index'] for r in bindings]}
# Do not copy bulky per-joint limit details into this compact audit.
for r in check['planning_reports']:r['joint_limit_audit_passed']=r.pop('joint_limit_audit')['passed']
o={'scope':'Finite read-only independent audit of actual terminal v3; no GPU, no live source writes, no new physical trial','source_sha256':{str(p/n):sha(p/n) for n in ['result.json','transitions.json','snapshot_manifest.json','world_kinematics_pre_post.json.gz','actual_terminal_snapshot.json','planning_reports.json']},'source_and_control_chain':check,'actual_4ms_trace':{'total_rows_including_settle':worldcount,'active_rows':activecount,'expected_active_rows':len(rows)*15,'active_tick_failures':tickfail,'each_action_has_15_kinematics_rows':all(ctxcount[i]==15 for i in range(len(rows))),'prepost_body_cap_and_12pill_chain_mismatches':chainfail,'requested_command_matches_transition_failures':requestedfail,'all_actual_drive24_finite':commandvectors==activecount,'max_end_actual_drive_minus_transition_target':drive_end_error,'right_arm_actual_drive_change_during_left_phases_max_rad':frozenrightmax,'dt_error_max_s':maxdt,'phase_tick_counts':dict(phasecounts)},'actual_contacts':{'raw_body_cap_contact_stream_left_cap_max_N':leftcapmax,'raw_body_cap_contact_stream_left_body_max_N':leftbodymax,'raw_body_cap_contact_stream_right_cap_max_N':rightcapmax,'first_actual_left_cap_over_1mN':firstcap,'persisted60ms_forbidden_left_max_N':boundaryforbidden,'guard_failure':result['guard_failure'],'per4ms_guard_source':'frozen labware_cap_task.py advance lines120-125 calls polymorphic _update_metrics every raw step; frozen left_cap_contact_task.py lines19-57 scans scene contacts and raises on forbidden force/release','raw_scope_boundary':'world contacts_post intentionally retains only body/cap participants; left-table and left-right zero-load/threshold crossings cannot be independently reconstructed from that filtered stream. Per4ms observer completed without guard exception; recorded60ms values are zero.'},'failure_boundary':{'error':result['error'],'failed_final_via_index':5,'completed_vias':[0,1,2,3,4],'final_via_target_executed':False,'final_via_report_present':False,'first_contact_observed':result['first_contact_observed'],'current_final_closed_observed':result['final_metrics']['cap_closed_observed'],'terminal_cap_state':laststate,'active_closed_false_ticks':closed_false_count,'first_active_closed_false':firstclosedfalse,'first_approach_closed_false':firstclosedfalse_approach,'active_released_ticks':released_count,'bodyhold_stage_success':result['final_metrics']['stage_body_hold_success'],'full_task_success':result['full_task_success'],'interpretation':'Actual five transit segments completed, but final single-seed IK returned failure before its target generation/execution and before contact controller construction. Cap remained attached; 1.319mm axial separation is above the original1mm closed criterion. Historical bodyhold stage success does not restore current closed=true or full success.'},'new_physical_trials':0,'live_source_modified':False}
assert all(snapok.values()) and check['prefix294_exact_action_state_nextstate_tick'] and check['transition_state_chain_exact'] and check['all_transitions_15_ticks']
assert activecount==505*15 and tickfail==0 and chainfail==0 and requestedfail==0 and leftcapmax==0 and check['controller_proposals']==0
(p/'independent_v3_audit.json').write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'chain':{k:v for k,v in check.items() if k!='planning_reports'},'4ms':o['actual_4ms_trace'],'contacts':o['actual_contacts'],'failure':{k:v for k,v in o['failure_boundary'].items() if k not in ['error','terminal_cap_state','first_active_closed_false','first_approach_closed_false']},'first_closed_false':firstclosedfalse},indent=2))
