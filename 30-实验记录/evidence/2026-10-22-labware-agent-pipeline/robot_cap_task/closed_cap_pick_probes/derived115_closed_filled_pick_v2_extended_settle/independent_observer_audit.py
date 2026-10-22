import json,gzip,math,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
P=Path('/home/admin123/liminghe/VLA-WAM-RL/30-实验记录/evidence/2026-10-22-labware-agent-pipeline/robot_cap_task/closed_cap_pick_probes/derived115_closed_filled_pick_v2_extended_settle')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(n):return json.loads((P/n).read_text())
def quatR(q):return Rotation.from_quat(np.r_[q[1:],q[0]]).as_matrix()
def quatmul(q,p):
 w,x,y,z=q;v,a,b,c=p
 return np.array([w*v-x*a-y*b-z*c,w*a+x*v+y*c-z*b,w*b-x*c+y*v+z*a,w*c+x*b-y*a+z*v])
def rate(s):
 b=s['body'];c=s['cap'];qb=np.array(b['world_pose_pq'][3:],float);qc=np.array(c['world_pose_pq'][3:],float);qb/=np.linalg.norm(qb);qc/=np.linalg.norm(qc)
 q=quatmul(qb*np.array([1,-1,-1,-1]),qc)
 w=quatR(qb).T@(np.array(c['world_angular_velocity_rad_s'])-np.array(b['world_angular_velocity_rad_s']))
 qw,qx,qy,qz=q
 return w[2]+(w[0]*(qw*qy+qz*qx)+w[1]*(-qw*qx+qz*qy))/(qw*qw+qz*qz),q,w
# Reconstruct exactly the hollow wall boxes and original48-sided bottom vertices.
spec=read('source_snapshot/a52359200359389e27dc20aa4cc98673b5fff8bd6d2b319337cfe474c7b3294d_physics_spec115.json')
cfg=spec['bottle'];ri=cfg['inner_radius_m'];ro=cfg['outer_radius_m'];h=cfg['height_m'];n=cfg['wall_segments'];r=(ri+ro)/2;ht=ro*math.tan(math.pi/n)*1.03
vs=[]
for i in range(n):
 a=2*math.pi*i/n;R=Rotation.from_rotvec([0,0,a]).as_matrix()
 for x in [-(ro-ri)/2,(ro-ri)/2]:
  for y in [-ht,ht]:
   for z in [-h/2,h/2]:vs.append(R@np.array([x,y,z])+[r*math.cos(a),r*math.sin(a),h/2])
for a in np.arange(48)*2*math.pi/48:
 for z in [0,cfg['base_thickness_m']]:vs.append([ro*math.cos(a),ro*math.sin(a),z])
vs=np.asarray(vs)
with gzip.open(P/'world_kinematics_pre_post.json.gz','rt') as f:data=json.load(f)
N=len(data);dense=[];pre_rates=[];post_rates=[];pose_delta=[];pre_q=[];post_q=[];dt=[];continuity=[];other_names={};bodyname='derived115_closed_tablet_bottle_body';capname='derived115_closed_tablet_bottle_cap';bad_others={};actives=[]
for i,row in enumerate(data):
 d=row['dt_s'];dt.append(d);s=row['post_assembly_state'];b=row['post']['body'];mat=quatR(np.array(b['world_pose_pq'][3:]));gap=float((vs@mat.T)[:,2].min()+b['world_pose_pq'][2]-.74);angle=float(np.degrees(np.arccos(np.clip(mat[2,2],-1,1))))
 body_links={};cap_links={};thumb=fingers=actual=buggy=0.;normal={'thumb':np.zeros(3),'fingers':np.zeros(3)};capforce=0.;bad={}
 for con in row['contacts_post']:
  names=con['body_names']
  if bodyname in names:
   first=names[0]==bodyname;other=names[1] if first else names[0];other_names[other]=other_names.get(other,0)+1
   for pt in con['points']:
    f=abs(float(np.asarray(pt['impulse_raw_world_Ns'])@np.asarray(pt['normal_raw_world'])))/.004
    if other=='table':actual+=f
    if 'table' in other.lower():
     buggy+=f
     if other!='table':bad[other]=bad.get(other,0.)+f
    if other.startswith('r_inspire_'):
     body_links[other]=body_links.get(other,0.)+f
     group='thumb' if 'thumb' in other else 'fingers'
     if group=='thumb':thumb+=f
     else:fingers+=f
     normal[group]+=f*np.array(pt['normal_raw_world'])*(1 if first else -1)
  if capname in names:
   other=names[1] if names[0]==capname else names[0]
   if other.startswith(('l_inspire_','r_inspire_')):
    cf=sum(abs(float(np.asarray(pt['impulse_raw_world_Ns'])@np.asarray(pt['normal_raw_world'])))/.004 for pt in con['points']);capforce+=cf;cap_links[other]=cap_links.get(other,0.)+cf
 denom=np.linalg.norm(normal['thumb'])*np.linalg.norm(normal['fingers']);dot=float(normal['thumb']@normal['fingers']/denom) if denom>1e-12 else 1.;opp=thumb>.05 and fingers>.05 and dot<-.3
 held=opp and gap>=.02 and actual<1e-5 and angle<15
 oldheld=opp and gap>=.02 and buggy<1e-5 and angle<15
 closed=not s['released'] and abs(s['rotation_measured_rad'])<.15 and abs(s['lift_measured_m'])<.001 and s['radial_error_m']<.001 and s['axis_tilt_rad']<.02
 active=row['context']['actual_drive_command24'] is not None
 rec={'physics_step':row['physics_step'],'active_tick':row['context']['active_tick_before_step']+1 if active else None,'time_s':s['time_s'],'phase':row['context']['phase'],'active':active,'actual_support_N':actual,'legacy_substring_support_N':buggy,'false_support_by_actor_N':bad,'thumb_N':thumb,'fingers_N':fingers,'opposing_normals_dot':dot,'opposed':opp,'bottom_gap_m':gap,'upright_deg':angle,'bodyheld':bool(held and active),'legacy_bodyheld':bool(oldheld and active),'closed':bool(closed),'closedheld':bool(held and closed and active),'cap_hand_N':capforce,'body_contact_by_link_N':body_links,'cap_contact_by_link_N':cap_links,'cap_state':s}
 dense.append(rec)
 for key,val in bad.items():
  q=bad_others.setdefault(key,{'ticks':0,'force_nonzero_ticks':0,'max_N':0.});q['ticks']+=1;q['force_nonzero_ticks']+=int(val>0);q['max_N']=max(q['max_N'],val)
 pr,pq,pw=rate(row['pre']);qr,qq,qw=rate(row['post']);pre_rates.append(pr);post_rates.append(qr);pre_q.append(pq);post_q.append(qq)
 pa=2*np.arctan2(pq[3],pq[0]);qa=2*np.arctan2(qq[3],qq[0]);pose_delta.append((qa-pa+math.pi)%(2*math.pi)-math.pi)
 if i:
  prior=data[i-1]['post'];now=row['pre']
  continuity.append(max(np.max(np.abs(np.array(prior[key][field])-np.array(now[key][field]))) for key in ['body','cap'] for field in ['world_pose_pq','world_linear_velocity_m_s','world_angular_velocity_rad_s']))
 del row['contacts_post']
print('dense_loaded',N,flush=True)
def ranges(key):
 out=[];start=None
 for i,r in enumerate(dense+[{}]):
  valid=bool(r.get(key))
  if valid and start is None:start=i
  if not valid and start is not None:
   end=i-1;out.append({'start':dense[start]['physics_step'],'end':dense[end]['physics_step'],'first_active_tick':dense[start]['active_tick'],'last_active_tick':dense[end]['active_tick'],'samples':end-start+1,'elapsed_integrated_s':float(np.sum(dt[start:end+1])),'legacy_fixed4ms_s':(end-start+1)*.004});start=None
 return sorted(out,key=lambda r:r['samples'],reverse=True)
# Validate reconstructed dense metrics against saved60ms metrics.
obs=read('assembly_observations.json');by={r['physics_step']:r for r in dense};errs={k:0. for k in ['gap_m','angle_deg','thumb_N','fingers_N','table_N','normal_dot']}
for x in obs:
 r=by[x['cap_state']['physics_step']];m=x['body_hold_metrics']
 for k,v in [('gap_m',r['bottom_gap_m']-m['bottom_gap_m']),('angle_deg',r['upright_deg']-m['upright_angle_deg']),('thumb_N',r['thumb_N']-m['thumb_contact_N']),('fingers_N',r['fingers_N']-m['finger_contact_N']),('table_N',r['legacy_substring_support_N']-m['table_contact_N']),('normal_dot',r['opposing_normals_dot']-m['opposing_normals_dot'])]:errs[k]=max(errs[k],abs(v))
pre_rates=np.array(pre_rates);post_rates=np.array(post_rates);pose_delta=np.array(pose_delta);dt=np.array(dt)
def integration(mask):
 mask=np.asarray(mask,bool);target=float(pose_delta[mask].sum());out={'samples':int(mask.sum()),'pose_delta_theta_rad':target}
 for label,pred in [('pre_left',pre_rates*dt),('post_right',post_rates*dt),('trapezoid',(pre_rates+post_rates)*.5*dt)]:
  e=pose_delta-pred;cum=np.cumsum(e[mask]);out[label]={'integral_rad':float(pred[mask].sum()),'pose_minus_integral_rad':float(e[mask].sum()),'max_abs_step_residual_rad':float(abs(e[mask]).max()),'rms_step_residual_rad':float(np.sqrt(np.mean(e[mask]**2))),'max_abs_cumulative_residual_rad':float(abs(cum).max())}
 return out
active=np.array([r['active'] for r in dense]);capcontact=np.array([r['cap_hand_N']>1e-6 for r in dense]);firstbad=next((r for r in dense if r['active'] and not r['closed']),None)
firsthold=next((r for r in dense if r['bodyheld']),None)
mask_all=np.ones(N,bool)
firstcap=next((r for r in dense if r['active'] and r['cap_hand_N']>1e-6),None)
trans=read('transitions.json');drive_error=0.;drive_endpoint_error=0.;drive_interpolation_error=0.;requested_error=0.;action_index_valid=True;last_endpoint=None;start_drive=None
for row in data:
 if row['context']['actual_drive_command24'] is not None:
  ix=row['context']['action_index'];action_index_valid=action_index_valid and 0<=ix<len(trans) and trans[ix]['tick_start']<=row['context']['active_tick_before_step']<trans[ix]['tick_end']
  if ix<len(trans):
   got=np.array(row['context']['actual_drive_command24']);target=np.array(trans[ix]['action']);sub=row['context']['active_tick_before_step']-trans[ix]['tick_start']+1
   drive_error=max(drive_error,float(np.max(abs(got-target))));requested_error=max(requested_error,float(np.max(abs(np.array(row['context']['requested_command24'])-target))))
   if sub==1:start_drive=last_endpoint.copy() if last_endpoint is not None else None
   if start_drive is not None:drive_interpolation_error=max(drive_interpolation_error,float(np.max(abs(got-(start_drive+(target-start_drive)*(sub/15))))))
   if sub==15:drive_endpoint_error=max(drive_endpoint_error,float(np.max(abs(got-target))));last_endpoint=got.copy()
summary={'date':'2026-10-22','scope':'CPU read-only offline115 robot evidence audit; original failed run unchanged; no new simulation','original_result_success':read('result.json')['success'],'source_inputs_sha256':{n:sha(P/n) for n in ['world_kinematics_pre_post.json.gz','assembly_trace.json.gz','assembly_observations.json','body_physics_trace.json','result.json','source_manifest.json','snapshot_manifest.json']},'audit_script_sha256':sha(__file__),'samples':N,'physics_steps_contiguous':all(r['physics_step']==i+1 for i,r in enumerate(dense)),'active_steps':int(active.sum()),'initialization_steps':int((~active).sum()),'active_tick_sequence_valid':all(r['active_tick']==i+1 for i,r in enumerate([x for x in dense if x['active']])),'actual_dt_s_min_max':[float(min(dt)),float(max(dt))],'record_pre_equals_previous_post_max_abs_all_fields':float(max(continuity)),'table_support_bug':{'source':'labware_task.py:222, frozenSHA22cd86c3d0836633ad7911e10f999fe366f99b66d9b883a626fe91d05711fa3e','predicate':"'table' in other.lower()",'actual_support_actor_name':'table','misclassified_other_actors':bad_others,'last_state':dense[-1],'reconstructed_metric_max_abs_error_at60ms':errs},'dense_bodyhold_windows_top5':ranges('bodyheld')[:5],'dense_legacy_bodyhold_windows_top5':ranges('legacy_bodyheld')[:5],'dense_closed_plus_bodyhold_windows_top5':ranges('closedheld')[:5],'first_dense_bodyheld':firsthold,'first_dense_cap_hand_contact':firstcap,'first_dense_cap_contact_over005N':next((r for r in dense if r['active'] and r['cap_hand_N']>.05),None),'body_contact_at_firstcap':firstcap['body_contact_by_link_N'] if firstcap else None,'control_readback':{'context_action_index_alignment':action_index_valid,'all_substeps_vs_endpoint_max_abs_diagnostic_only':drive_error,'actual_endpoint_drive_vs_saved_action_max_abs':drive_endpoint_error,'actual_interpolated_drive_vs_clock_prediction_max_abs_actions1through426':drive_interpolation_error,'first_action_start_drive_separately_logged':False,'requested_command24_vs_saved_action_max_abs':requested_error,'all_rows24_finite':all(np.isfinite(np.array(x[k])).all() and len(x[k])==24 for x in trans for k in ['state','action','next_state']),'all_rows15ticks':all(x['tick_end']-x['tick_start']==15 for x in trans),'measured_state_chain_max_abs':float(max(np.max(abs(np.array(a['next_state'])-np.array(b['state']))) for a,b in zip(trans,trans[1:]))),'source_note':'independently recomputed from full logged drive context and transitions; interior4ms targets interpolate and must not be compared to60ms endpoint as equality'},'first_active_closed_false':firstbad,'closed_false_total_active_steps':sum(r['active'] and not r['closed'] for r in dense),'cap_release_any':any(r['cap_state']['released'] for r in dense),'tablet_observation_boundary':{'samples':len(obs),'cadence_s':.06,'all_samples_center_count12':all(x['tablet_centers_inside']==12 for x in obs),'full4ms_tablet_pose_available':False,'full_capsule_shape_containment_proven':False,'dense_closedfilled_success_cannot_be_claimed_from_only60ms_centers':True},'twist_velocity':{'formula':'q=conj(qb)*qc,w=Rb^T(wc-wb);theta_dot=wz+[wx(qw*qy+qz*qx)+wy(-qw*qx+qz*qy)]/(qw^2+qz^2)','quaternions_normalized':True,'whole_run':integration(mask_all),'active':integration(active),'cap_contact_steps':integration(capcontact),'bodyheld_steps':integration(np.array([r['bodyheld'] for r in dense])),'raw_body_z_projection_pre_vs_twist_rate_max_abs':float(max(abs(pre_rates-np.array([r['pre']['relative_omega_body_z_rad_s'] for r in data])))),'raw_body_z_projection_post_vs_twist_rate_max_abs':float(max(abs(post_rates-np.array([r['post']['relative_omega_body_z_rad_s'] for r in data]))))},'source_snapshot_sha_passed':all(sha(v['snapshot'])==v['sha256'] for v in read('snapshot_manifest.json').values()),'new_physical_trials':0,'original_success_flags_modified':False,'full_unscrew_pour_or_dataset_success':False}
(P/'independent_observer_audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:summary[k] for k in ['samples','active_steps','initialization_steps','dense_bodyhold_windows_top5','dense_closed_plus_bodyhold_windows_top5','closed_false_total_active_steps','twist_velocity']},ensure_ascii=False,indent=2))
