import json,gzip,hashlib,math,importlib.util
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path('/home/admin123/liminghe/VLA-WAM-RL/30-实验记录/evidence/2026-10-22-labware-agent-pipeline/robot_cap_task')
P=ROOT/'static_hold_probes/held115_static_reference_v1'
OLD=ROOT/'closed_cap_pick_probes/derived115_closed_filled_pick_v3_observer_identity_replay'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(n):return json.loads((P/n).read_text())
def R(q):return Rotation.from_quat(np.r_[q[1:],q[0]]).as_matrix()
def load(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
geom=load(ROOT/'capsule_geometry.py','independent_capsule_geometry_audit')
spec=read('source_snapshot/a52359200359389e27dc20aa4cc98673b5fff8bd6d2b319337cfe474c7b3294d_physics_spec115.json')
bottle=spec['bottle'];pill=spec['tablets'];tol=spec['physics']['contact_offset_m'];n=bottle['wall_segments'];ri=bottle['inner_radius_m'];ro=bottle['outer_radius_m'];h=bottle['height_m'];v=[]
for i in range(n):
 a=2*math.pi*i/n;rot=Rotation.from_rotvec([0,0,a]).as_matrix()
 for x in [-(ro-ri)/2,(ro-ri)/2]:
  for y in [-ro*math.tan(math.pi/n)*1.03,ro*math.tan(math.pi/n)*1.03]:
   for z in [-h/2,h/2]:v.append(rot@np.array([x,y,z])+[(ri+ro)/2*math.cos(a),(ri+ro)/2*math.sin(a),h/2])
for a in np.arange(48)*2*math.pi/48:
 for z in [0,bottle['base_thickness_m']]:v.append([ro*math.cos(a),ro*math.sin(a),z])
v=np.asarray(v)
with gzip.open(P/'world_kinematics_pre_post.json.gz','rt') as f:data=json.load(f)
print('loaded',len(data),flush=True)
trans=read('transitions.json');old=json.loads((OLD/'transitions.json').read_text());snapshot=read('snapshot_manifest.json');terminal=read('actual_terminal_snapshot.json');sourcechecks={k:sha(x['snapshot'])==x['sha256'] for k,x in snapshot.items()}
prefix=[{'action_index':i,'state_exact':r['state']==old[i]['state'],'action_exact':r['action']==old[i]['action'],'next_state_exact':r['next_state']==old[i]['next_state'],'ticks_exact':(r['tick_start'],r['tick_end'])==(old[i]['tick_start'],old[i]['tick_end'])} for i,r in enumerate(trans[:269])]
rows=[];capsule_min={k:math.inf for k in ['radial_margin_m','floor_margin_m','rim_margin_m']};capsule_min_active=dict(capsule_min);capsule_issues=[];prepost_continuity=0.;drive_endpoint=0.;drive_interp=0.;requested_error=0.;indexgood=True;last_endpoint=None;start=None;support_ids=set();actor_pair_id_name_consistent=True
for i,row in enumerate(data):
 ctx=row['context'];active=ctx['actual_drive_command24'] is not None;d=row['dt_s'];b=row['post']['body']['world_pose_pq'];rb=R(b[3:]);s=row['post_assembly_state'];sid=ctx['support_actor_identity']['per_scene_id'];support_ids.add((sid,ctx['support_actor_identity']['name']))
 body_link_forces={};cap_link_forces={};force=thumb=fingers=capforce=0.;norms={'thumb':np.zeros(3),'fingers':np.zeros(3)};bodyname='derived115_closed_tablet_bottle_body';capname='derived115_closed_tablet_bottle_cap'
 for con in row['contacts_post']:
  names=con['body_names'];ids=con['body_actor_ids'];actor_pair_id_name_consistent &= all((id_==sid)==(name=='table') for id_,name in zip(ids,names))
  if bodyname in names:
   ix=names.index(bodyname);other=1-ix
   for pt in con['points']:
    f=abs(float(np.array(pt['impulse_raw_world_Ns'])@np.array(pt['normal_raw_world'])))/.004
    if ids[other]==sid:force+=f
    if names[other].startswith('r_inspire_'):
     body_link_forces[names[other]]=body_link_forces.get(names[other],0.)+f
     group='thumb' if 'thumb' in names[other] else 'fingers'
     if group=='thumb':thumb+=f
     else:fingers+=f
     norms[group]+=f*np.array(pt['normal_raw_world'])*(1 if ix==0 else -1)
  if capname in names:
   ix=names.index(capname);other=1-ix
   if names[other].startswith(('r_inspire_','l_inspire_')):
    cf=sum(abs(float(np.array(pt['impulse_raw_world_Ns'])@np.array(pt['normal_raw_world'])))/.004 for pt in con['points']);capforce+=cf;cap_link_forces[names[other]]=cap_link_forces.get(names[other],0.)+cf
 den=np.linalg.norm(norms['thumb'])*np.linalg.norm(norms['fingers']);dot=float(norms['thumb']@norms['fingers']/den) if den>1e-12 else 1.;opp=thumb>.05 and fingers>.05 and dot<-.3
 gap=float((v@rb.T)[:,2].min()+b[2]-.74);angle=float(np.degrees(np.arccos(np.clip(rb[2,2],-1,1))));closed=not s['released'] and abs(s['rotation_measured_rad'])<.15 and abs(s['lift_measured_m'])<.001 and s['radial_error_m']<.001 and s['axis_tilt_rad']<.02
 geo_counts={};center_count_post=0;local_min={k:math.inf for k in capsule_min}
 for phase in ['pre','post']:
  st=row[phase];bp=st['body']['world_pose_pq'];br=R(bp[3:]);insidezero=insidetol=0
  arr=st['tablet_world_pose_pq'];assert np.asarray(arr).shape==(12,7) and np.isfinite(arr).all()
  for j,pq in enumerate(arr):
   local=br.T@(np.array(pq[:3])-np.array(bp[:3]));axis=br.T@R(pq[3:])[:,0]
   q=geom.capsule_cylinder_margins(local,axis,half_length_m=pill['half_length_m'],capsule_radius_m=pill['radius_m'],inner_radius_m=bottle['inner_radius_m'],floor_z_m=bottle['base_thickness_m'],rim_z_m=bottle['height_m'],tolerance_m=tol)
   insidezero+=all(q[k]>=0 for k in capsule_min);insidetol+=q['full_geometry_inside']
   for k in capsule_min:
    capsule_min[k]=min(capsule_min[k],q[k]);local_min[k]=min(local_min[k],q[k])
    if active:capsule_min_active[k]=min(capsule_min_active[k],q[k])
   if phase=='post':center_count_post+=bool(np.linalg.norm(local[:2])<=bottle['inner_radius_m']+.001 and -.001<=local[2]<=spec['cap']['closed_center_z_m']+.001)
   if active and not q['full_geometry_inside'] and len(capsule_issues)<20:capsule_issues.append({'physics_step':row['physics_step'],'phase':phase,'tablet_index':j,'margins':q})
  geo_counts[phase+'_zero']=insidezero;geo_counts[phase+'_tol']=insidetol
 held=active and opp and gap>=.02 and force<1e-5 and angle<15
 rows.append({'physics_step':row['physics_step'],'active':active,'active_tick':ctx['active_tick_before_step']+1 if active else None,'support_N':force,'thumb_N':thumb,'fingers_N':fingers,'dot':dot,'gap_m':gap,'angle_deg':angle,'cap_hand_N':capforce,'body_link_forces_N':body_link_forces,'cap_link_forces_N':cap_link_forces,'closed':bool(closed),'held':bool(held),'online_equivalent_stage':bool(held and closed and center_count_post==12),'strict_full_capsule_stage':bool(held and closed and geo_counts['post_tol']==12),'zero_tolerance_full_capsule_stage':bool(held and closed and geo_counts['post_zero']==12),'full_geometry_counts':geo_counts,'min_margins_pre_post':local_min,'body_speed_m_s':float(np.linalg.norm(row['post']['body']['world_linear_velocity_m_s'])),'cap_state':s})
 if i:
  oldstate=data[i-1]['post'];newstate=row['pre'];prepost_continuity=max(prepost_continuity,max(np.max(abs(np.array(oldstate[k][f])-np.array(newstate[k][f]))) for k in ['body','cap'] for f in ['world_pose_pq','world_linear_velocity_m_s','world_angular_velocity_rad_s']),float(np.max(abs(np.array(oldstate['tablet_world_pose_pq'])-np.array(newstate['tablet_world_pose_pq'])))))
 if active:
  ix=ctx['action_index'];sub=ctx['active_tick_before_step']-trans[ix]['tick_start']+1;indexgood &= 0<=ix<len(trans) and 1<=sub<=15
  target=np.array(trans[ix]['action']);got=np.array(ctx['actual_drive_command24']);requested_error=max(requested_error,float(np.max(abs(np.array(ctx['requested_command24'])-target))))
  if sub==1:start=last_endpoint.copy() if last_endpoint is not None else None
  if start is not None:drive_interp=max(drive_interp,float(np.max(abs(got-(start+(target-start)*sub/15)))))
  if sub==15:drive_endpoint=max(drive_endpoint,float(np.max(abs(got-target))));last_endpoint=got.copy()
 del row['contacts_post']
def windows(key):
 out=[];a=None
 for i,r in enumerate(rows+[{}]):
  if r.get(key) and a is None:a=i
  if not r.get(key) and a is not None:
   z=i-1;out.append({'first_assembly_step':rows[a]['physics_step'],'last_assembly_step':rows[z]['physics_step'],'first_active_tick':rows[a]['active_tick'],'last_active_tick':rows[z]['active_tick'],'ticks':z-a+1,'nominal_duration_s':(z-a+1)*.004,'actual_duration_s':sum(x['dt_s'] for x in data[a:z+1])});a=None
 return sorted(out,key=lambda x:x['ticks'],reverse=True)
active=[r for r in rows if r['active']];result=read('result.json');out={'date':'2026-10-22','scope':'Independent CPU read-only replay/actual24drive/4ms support/wholecapsule geometry audit; bodyhold substage only','physics_steps':len(data),'active_steps':len(active),'targets':len(trans),'source_sha256':{str(P/n):sha(P/n) for n in ['world_kinematics_pre_post.json.gz','transitions.json','result.json','feedback_shadow_checks.json','actual_terminal_snapshot.json','source_manifest.json','snapshot_manifest.json']},'source_snapshot_all_sha_match':all(sourcechecks.values()),'source_snapshot_check':sourcechecks,'capsule_geometry_source_sha256':sha(ROOT/'capsule_geometry.py'),'source_prefix_comparison':{'source':str(OLD/'transitions.json'),'source_sha256':sha(OLD/'transitions.json'),'all_state_action_nextstate_ticks_exact':all(all(v for k,v in r.items() if k!='action_index') for r in prefix),'checked_targets':len(prefix)},'clock_control':{'physics_steps_contiguous':all(r['physics_step']==i+1 for i,r in enumerate(rows)),'active_ticks_contiguous':all(r['active_tick']==i+1 for i,r in enumerate(active)),'all_transitions15steps':all(r['tick_end']-r['tick_start']==15 for r in trans),'all_states_actions_finite24':all(len(r[k])==24 and np.isfinite(r[k]).all() for r in trans for k in ['state','action','next_state']),'state_chain_max_abs':float(max(np.max(abs(np.array(a['next_state'])-np.array(b['state']))) for a,b in zip(trans,trans[1:]))),'context_action_index_alignment':bool(indexgood),'requested_command_vs_row_max_abs':requested_error,'endpoint_drive_vs_action_max_abs':drive_endpoint,'internal_drive_vs_linear_clock_max_abs_after_first_action':drive_interp,'all_logged_pre_equal_prior_post_max_abs':prepost_continuity,'first_action_start_drive_not_separately_logged':True},'support_identity':{'identities':list(support_ids),'all_name_ID_pairs_consistent':bool(actor_pair_id_name_consistent),'table_force_zero_for_all_bodyheld_steps':all(r['support_N']==0 for r in active if r['held'])},'geometry_contract':{'method':'actualcapsule=line segment±half_length localX Minkowski sphere; original capsule_cylinder_margins called on each actualpre/post worldpq in actualbody frame','actual12_pose_per_step':True,'capsule_half_length_m':pill['half_length_m'],'capsule_radius_m':pill['radius_m'],'body_inner_radius_m':bottle['inner_radius_m'],'floor_m':bottle['base_thickness_m'],'rim_m':bottle['height_m'],'primary_report_tolerance_m':tol,'tolerance_source':'frozen physics_spec115.physics.contact_offset_m; zero-tolerance results also reported; no online threshold modified','raw_min_margins_all_steps':capsule_min,'raw_min_margins_active':capsule_min_active,'active_pre_or_post_full_geometry_violation_samples':sum(r['full_geometry_counts']['pre_tol']!=12 or r['full_geometry_counts']['post_tol']!=12 for r in active),'first_violations':capsule_issues,'active_min_pre_zero_inside_count':min(r['full_geometry_counts']['pre_zero'] for r in active),'active_min_post_zero_inside_count':min(r['full_geometry_counts']['post_zero'] for r in active),'active_min_pre_tol_inside_count':min(r['full_geometry_counts']['pre_tol'] for r in active),'active_min_post_tol_inside_count':min(r['full_geometry_counts']['post_tol'] for r in active)},'bodyheld_windows':windows('held')[:5],'online_equivalent_stage_windows':windows('online_equivalent_stage')[:5],'full_capsule_stage_windows':windows('strict_full_capsule_stage')[:5],'zero_tolerance_stage_windows':windows('zero_tolerance_full_capsule_stage')[:5],'last_actual_state':rows[-1],'reported_online_success':result['final_metrics']['success'],'reported_stage_seconds':result['final_metrics']['closed_fill_hold_seconds'],'online_observer_uses_centers_only':True,'terminal_phase':data[-1]['context']['phase'],'terminal_speed_m_s':rows[-1]['body_speed_m_s'],'terminal_static_reference_proven':False,'cap_hand_force_max_N':max(r['cap_hand_N'] for r in active),'new_physical_trials_run_by_auditor':0,'full_unscrew_pour_verified':False,'dataset_collected_by_this_run':False,'original_receipts_modified':False}
(P/'independent_static_hold_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:out[k] for k in ['physics_steps','active_steps','source_prefix_comparison','clock_control','geometry_contract','bodyheld_windows','online_equivalent_stage_windows','full_capsule_stage_windows','zero_tolerance_stage_windows','terminal_speed_m_s','cap_hand_force_max_N']},ensure_ascii=False,indent=2),flush=True)

static=[r for r in rows if r['active'] and r['active_tick']>4035]
fixed=read('moving_prefix_terminal_snapshot.json')['actual_drive_target24'];static_raw=[r for r in data if r['context']['actual_drive_command24'] is not None and r['context']['active_tick_before_step']>=4035]
armerr=max(float(np.max(abs(np.array(r['context']['actual_drive_command24'])[12:18]-np.array(fixed)[12:18]))) for r in static_raw)
allfixed=max(float(np.max(abs(np.array(r['context']['actual_drive_command24'])[:18]-np.array(fixed)[:18]))) for r in static_raw)
snapshots=read('static_reference_observations.json');initial=read('moving_prefix_terminal_snapshot.json')
def drift(samples):
 ts=[np.linalg.inv(x['body_matrix'])@np.array(x['right_palm_matrix']) for x in samples];origin=ts[0]
 return {'first_active_tick':samples[0]['physics_tick'],'last_active_tick':samples[-1]['physics_tick'],'actual_point_span_s':(samples[-1]['physics_tick']-samples[0]['physics_tick'])*.004,'max_palm_in_body_translation_from_first_m':max(float(np.linalg.norm(t[:3,3]-origin[:3,3])) for t in ts),'max_palm_in_body_rotation_from_first_deg':max(float(np.degrees(np.arccos(np.clip((np.trace(origin[:3,:3].T@t[:3,:3])-1)/2,-1,1)))) for t in ts)}
firstcap=next((r for r in static if r['cap_hand_N']>1e-6),None)
out['static_continuation']={'actual_steps':len(static),'actual_duration_s':sum(r['dt_s'] for r in static_raw),'right_arm_drive_max_delta_rad':armerr,'left12_and_rightarm6_drive_max_delta':allfixed,'right_hand_feedback_varies_intentionally':True,'all_center_closed':all(r['closed'] for r in static),'table_force_max_N':max(r['support_N'] for r in static),'opposed_every_step':all(r['thumb_N']>.05 and r['fingers_N']>.05 and r['dot']<-.3 for r in static),'cap_hand_force_min_max_N':[min(r['cap_hand_N'] for r in static),max(r['cap_hand_N'] for r in static)],'first_cap_hand_contact':firstcap,'final_body_link_forces_N':static[-1]['body_link_forces_N'],'final_cap_link_forces_N':static[-1]['cap_link_forces_N'],'body_speed_min_max_m_s':[min(r['body_speed_m_s'] for r in static),max(r['body_speed_m_s'] for r in static)],'cap_lift_min_max_m':[min(r['cap_state']['lift_measured_m'] for r in static),max(r['cap_state']['lift_measured_m'] for r in static)],'cap_theta_min_max_rad':[min(r['cap_state']['rotation_measured_rad'] for r in static),max(r['cap_state']['rotation_measured_rad'] for r in static)],'cap_release_any':any(r['cap_state']['released'] for r in static),'geometry_pre_post_min_count_at_frozen0_4mm':min(min(r['full_geometry_counts']['pre_tol'],r['full_geometry_counts']['post_tol']) for r in static),'body_palm_drift_whole1_5s':drift([initial]+snapshots),'body_palm_drift_original_last10points':drift(snapshots[-10:]),'body_palm_drift_actual_last0_6s':drift(snapshots[-11:]),'interpretation':'Actual drives held fixed for arm; geometry/contact vary over time. Nonzero velocity and lightcap contact are observations, not extrafulltask rejection thresholds.'}
# Metrics are final-only false in workflow, independently check all recorded boundaries.
out['all_logged_transition_final_success_false']=all(not r['metrics']['success'] and not r['metrics']['full_unscrew_pour_success'] for r in trans)
out['terminal_independent_readback']={'state24_matches_last_nextstate':terminal['state24']==trans[-1]['next_state'],'q36_matches_last_fullstate':terminal['q36']==read('measured_full_state_trace.json')[-1]['q36'],'all_terminal_tabletpq_matches_last_fullstate':terminal['tablet_world_pose_pq']==read('measured_full_state_trace.json')[-1]['tablet_world_pose_pq'],'actual_final_drive_matches_last_action':bool(np.max(abs(np.array(terminal['actual_drive_target24'])-np.array(trans[-1]['action'])))<=1e-4)}
(P/'independent_static_hold_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'static':out['static_continuation'],'all_finalfalse':out['all_logged_transition_final_success_false']},ensure_ascii=False,indent=2))
