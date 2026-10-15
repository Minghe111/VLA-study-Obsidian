from pathlib import Path
import ast,base64
import hashlib,json,os,sys,time,traceback,gc
import numpy as np
from scipy.spatial.transform import Rotation
W=Path('/bigdata2/liminghe/VLA-benchmark/deploy/realman_gen4/rgb50_20261007_v1');ROOT=Path('/bigdata2/liminghe/VLA-benchmark/RoboTwin');LEGACY=W.parent/'adjust_bottle_20261002'
TRIAL=W/'batch_candidates/v9_gpu1_grasp_refinement_attempt120/slot0000027_candidate_attempt120_v9_fanta_skin1mm'
OUT=W/'independent_fanta27_wake_initialization_diagnosis_20261007_v1'
assert not OUT.exists();OUT.mkdir()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(name,data):(OUT/name).write_text(json.dumps(data,indent=2,ensure_ascii=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x)))
source_names=[W/'generation_runner_v2.py',W/'expert_generation_v9_pathdiag.py',W/'expert_generation_v9.py',W/'expert_generation.py',W/'geometry_overlay.py',ROOT/'envs/_base_task.py',ROOT/'envs/robot/planner.py',ROOT/'integrations/realman_rm65b_gen4/native_robot.py',LEGACY/'try_coupled_side_entry_v29.py',TRIAL/'scene_sample.json',TRIAL/'expert_report.json']
pins={str(p):sha(p) for p in source_names}
scene_source=json.loads((TRIAL/'scene_sample.json').read_text());spec=scene_source['sample_spec'];write('input_spec.json',spec)
sys.path[:0]=[str(ROOT),str(LEGACY),str(W)];os.chdir(ROOT);os.environ['RGB50_EXPERT_MODULE']='expert_generation_v9_pathdiag';os.environ.setdefault('ROBOTWIN_RT_DENOISER','none')
import generation_runner_v2 as collector
print('WAKE_DIAGNOSTIC_START',json.dumps({'pid':os.getpid(),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'output':str(OUT),'scope':'One original full-native initializer, no expert/episode/RGB. Wake native bottle once, 2000 ordinary physics steps.'}),flush=True)
task=original_initializer=None;records=[];wake_calls=0;post_wake_steps=0;started=time.time()
try:
 task,scene,original_initializer=collector.prepare_task(spec,OUT,need_planner=True,backend='full_native')
 assert not task.robot.communication_flag
 comp=next(c for c in task.bottle.actor.get_components() if hasattr(c,'collision_shapes'))
 assert hasattr(comp,'wake_up') and hasattr(comp,'is_sleeping') and hasattr(comp,'get_sleep_threshold')
 dt=float(task.scene.get_timestep());assert abs(dt-.004)<1e-9
 M0=np.asarray(task.bottle.get_pose().to_transformation_matrix(),float);q0=np.asarray(task.robot.left_entity.get_qpos(),float)
 api_receipt={n:hasattr(comp,n) for n in ['is_sleeping','wake_up','get_sleep_threshold','linear_velocity','angular_velocity','get_linear_velocity','get_angular_velocity']}
 def attr_read(name,alternate=None):
  k=name if hasattr(comp,name) else alternate
  assert k is not None and hasattr(comp,k),name
  v=getattr(comp,k);return v() if callable(v) else v
 def snapshot(step,stage):
  M=np.asarray(task.bottle.get_pose().to_transformation_matrix(),float);Q=np.asarray(task.robot.left_entity.get_qpos(),float)
  rel=Rotation.from_matrix(M0[:3,:3].T@M[:3,:3]);qq=rel.as_quat();twist=float(np.rad2deg(2*np.arctan2(qq[1],qq[3])));twist=(twist+180)%360-180
  contacts=[];max_non_table=0.;support=0.;bottle_name=task.bottle.actor.name
  for c in task.scene.get_contacts():
   names=[b.entity.name for b in c.bodies]
   if bottle_name not in names:continue
   pts=[]
   for p in c.points:
    imp=np.asarray(p.impulse,float);normal=np.asarray(p.normal,float);norm=float(np.linalg.norm(imp));sep=float(p.separation)
    pts.append({'position':np.asarray(p.position,float).tolist(),'normal':normal.tolist(),'separation_m':sep,'impulse_Ns':imp.tolist(),'impulse_norm_Ns':norm})
    if 'table' in names:support+=abs(float(np.dot(imp,normal)))/dt
    else:max_non_table=max(max_non_table,norm)
   contacts.append({'names':names,'points':pts})
  return {'post_wake_step':step,'stage':stage,'post_wake_time_s':step*dt,'bottle_matrix':M.tolist(),'native_is_sleeping':bool(attr_read('is_sleeping')),'native_sleep_threshold':float(comp.get_sleep_threshold()),'native_linear_velocity_m_s':np.asarray(attr_read('linear_velocity','get_linear_velocity'),float).tolist(),'native_angular_velocity_rad_s':np.asarray(attr_read('angular_velocity','get_angular_velocity'),float).tolist(),'diagnostic_relative_rotation_deg':float(np.rad2deg(rel.magnitude())),'diagnostic_relative_localY_axial_twist_deg':twist,'origin_xyz_delta_m':(M[:3,3]-M0[:3,3]).tolist(),'max_absolute_q_change_rad':float(np.abs(Q-q0).max()),'table_support_absolute_normal_N':support,'max_non_table_point_impulse_norm_Ns':max_non_table,'bottle_native_contacts':contacts}
 before=snapshot(0,'post_original_setup_before_native_wake')
 write('before_wake_native_state.json',before)
 comp.wake_up();wake_calls+=1
 immediate=snapshot(0,'immediately_after_one_native_wake_before_scene_step');records.append(immediate)
 for step in range(1,2001):
  task.scene.step();post_wake_steps+=1;records.append(snapshot(step,'ordinary_physics_after_native_wake'))
  if step%500==0:print('WAKE_DIAGNOSTIC_PROGRESS',step,json.dumps({k:records[-1][k] for k in ['native_is_sleeping','diagnostic_relative_rotation_deg','diagnostic_relative_localY_axial_twist_deg','native_angular_velocity_rad_s']}),flush=True)
 write('native_state_trace.json',records)
 current_pins={str(p):sha(p) for p in source_names};assert current_pins==pins,'Diagnostic source changed'
 def compact(row):return {k:v for k,v in row.items() if k!='bottle_native_contacts'}
 first_change=next((compact(x) for x in records[1:] if not np.array_equal(np.asarray(x['bottle_matrix']),M0)),None)
 first_sleep=next((compact(x) for x in records[1:] if x['native_is_sleeping']),None)
 max_rot=max(records,key=lambda x:x['diagnostic_relative_rotation_deg']);last200=records[-200:]
 receipt={'scope':'Independent numerical native wake state experiment; one exact original Fanta27 full-native setup, no expert/arm task/RGB or episode acceptance. Original failed source untouched.', 'source_hashes_before_and_after':pins,'generator_sha256':sha(__file__),'pid':os.getpid(),'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'component_class':str(type(comp)),'read_api_available':api_receipt,'original_initialization_by_current_source':{'delay4_physics_steps':1200,'check_stable_physics_steps':2500,'steps_after_bottle_creation':3700,'nominal_simulated_s_after_bottle_creation':14.8,'actual_scene_dt_s':dt,'qualification':'This original setup step count is source/fake-counter verified, not a runtime pre-record counter.'},'wake_calls':wake_calls,'ordinary_post_wake_steps':post_wake_steps,'ordinary_post_wake_seconds':post_wake_steps*dt,'before_wake':compact(before),'immediate_after_wake':compact(immediate),'first_bottle_matrix_change':first_change,'first_natural_sleep_after_wake':first_sleep,'max_relative_rotation':compact(max_rot),'final':compact(records[-1]),'last200_sample_summary':{'all_sleeping':all(x['native_is_sleeping'] for x in last200),'max_native_angular_speed_rad_s':max(float(np.linalg.norm(x['native_angular_velocity_rad_s'])) for x in last200),'max_native_linear_speed_m_s':max(float(np.linalg.norm(x['native_linear_velocity_m_s'])) for x in last200),'twist_peak_to_peak_deg':float(np.ptp([x['diagnostic_relative_localY_axial_twist_deg'] for x in last200])),'rotation_peak_to_peak_deg':float(np.ptp([x['diagnostic_relative_rotation_deg'] for x in last200]))},'max_non_table_impulse_norm_all_post_wake_Ns':max(x['max_non_table_point_impulse_norm_Ns'] for x in records),'initial_matrix_max_difference_from_old27_scene':float(np.abs(M0-np.asarray(scene_source['initial_pose'])).max()),'pose_velocity_q_or_sleep_threshold_writes_after_original_setup':0,'post_setup_only_native_write':'one bottle PhysxRigidDynamicComponent.wake_up call','renderer_RGB_observation_calls_after_setup':0,'no_success_claim':True,'counterfactual_actual_contact_generated_wakeup_not_proven':True,'elapsed_wall_s':time.time()-started,'production_files_changed':False,'old_trial_bytes_or_labels_changed':False}
 write('diagnostic_receipt.json',receipt)
 print('WAKE_DIAGNOSTIC_DONE',json.dumps({'receipt':str(OUT/'diagnostic_receipt.json'),'sha256':sha(OUT/'diagnostic_receipt.json'),'before_sleeping':before['native_is_sleeping'],'max_rotation_deg':max_rot['diagnostic_relative_rotation_deg'],'final_rotation_deg':records[-1]['diagnostic_relative_rotation_deg'],'final_sleeping':records[-1]['native_is_sleeping'],'final_angular_velocity':records[-1]['native_angular_velocity_rad_s'],'first_sleep_step':None if first_sleep is None else first_sleep['post_wake_step']}),flush=True)
except BaseException as e:
 write('diagnostic_error.json',{'error':str(e),'traceback':traceback.format_exc(),'wake_calls':wake_calls,'post_wake_steps':post_wake_steps,'sources':pins,'records_count':len(records),'new_episode_success_claim':False});raise
finally:
 if task is not None:
  try:task.close_env(clear_cache=True)
  except BaseException:pass
 if original_initializer is not None:
  from integrations.realman_rm65b_gen4.native_robot import RealManRobot
  RealManRobot._initialize_hands=original_initializer
 task=None;gc.collect()