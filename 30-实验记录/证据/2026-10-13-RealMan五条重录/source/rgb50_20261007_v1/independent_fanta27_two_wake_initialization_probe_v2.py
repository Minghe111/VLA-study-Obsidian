from pathlib import Path
import hashlib,json,os,sys,time,traceback,gc,subprocess
import numpy as np
from scipy.spatial.transform import Rotation
W=Path('/bigdata2/liminghe/VLA-benchmark/deploy/realman_gen4/rgb50_20261007_v1');ROOT=Path('/bigdata2/liminghe/VLA-benchmark/RoboTwin');LEGACY=W.parent/'adjust_bottle_20261002'
TRIAL=W/'batch_candidates/v9_gpu1_grasp_refinement_attempt120/slot0000027_candidate_attempt120_v9_fanta_skin1mm';OUT=W/'independent_fanta27_two_wake_initialization_diagnosis_20261007_v2'
assert not OUT.exists();OUT.mkdir()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(name,a):(OUT/name).write_text(json.dumps(a,indent=2,ensure_ascii=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x)))
source_names=[W/'generation_runner_v2.py',W/'expert_generation_v9_pathdiag.py',W/'expert_generation_v9.py',W/'expert_generation.py',W/'geometry_overlay.py',ROOT/'envs/_base_task.py',ROOT/'envs/robot/planner.py',ROOT/'envs/utils/actor_utils.py',ROOT/'integrations/realman_rm65b_gen4/native_robot.py',LEGACY/'try_coupled_side_entry_v29.py',TRIAL/'scene_sample.json',TRIAL/'expert_report.json',W/'independent_fanta27_wake_initialization_probe_v1.py',W/'independent_fanta27_wake_initialization_diagnosis_20261007_v1/diagnostic_receipt.json',W/'independent_fanta27_wake_initialization_diagnosis_20261007_v1/native_state_trace.json']
pins={str(p):sha(p) for p in source_names};old_scene=json.loads((TRIAL/'scene_sample.json').read_text());spec=old_scene['sample_spec'];write('input_spec.json',spec)
sys.path[:0]=[str(ROOT),str(LEGACY),str(W)];os.chdir(ROOT);os.environ['RGB50_EXPERT_MODULE']='expert_generation_v9_pathdiag';os.environ.setdefault('ROBOTWIN_RT_DENOISER','none')
import generation_runner_v2 as collector
print('TWO_WAKE_DIAGNOSTIC_START',json.dumps({'pid':os.getpid(),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'scope':'Fresh original Fanta27 setup, exactly two native bottle wakes, 2000 normal steps each, no expert/task/RGB.'}),flush=True)
task=original_initializer=None;wake_calls=post_wake_steps=0;rounds=[];started=time.time()
try:
 task,scene,original_initializer=collector.prepare_task(spec,OUT,need_planner=True,backend='full_native');assert not task.robot.communication_flag
 comp=next(c for c in task.bottle.actor.get_components() if hasattr(c,'collision_shapes'));dt=float(task.scene.get_timestep());assert abs(dt-.004)<1e-9
 gpu_lines=subprocess.run(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.splitlines();gpu_map={line.split(',')[1].strip():int(line.split(',')[0]) for line in gpu_lines}
 app_lines=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.splitlines();own=[line for line in app_lines if line.split(',')[0].strip()==str(os.getpid())];assert own and all(gpu_map[line.split(',')[1].strip()]==7 for line in own),'Own diagnostic GPU differs from authorized7'
 initial_M=np.asarray(task.bottle.get_pose().to_transformation_matrix(),float);initial_q=np.asarray(task.robot.left_entity.get_qpos(),float);bottle_name=task.bottle.actor.name
 def snap(step,stage,round_index,M0,FP0):
  M=np.asarray(task.bottle.get_pose().to_transformation_matrix(),float);fp=np.asarray(task.bottle.get_functional_point(0,ret='matrix'),float)[:3,3];Q=np.asarray(task.robot.left_entity.get_qpos(),float);rel=Rotation.from_matrix(M0[:3,:3].T@M[:3,:3]);qq=rel.as_quat();twist=(float(np.rad2deg(2*np.arctan2(qq[1],qq[3])))+180)%360-180;contacts=[];max_non_table=support=0.
  for c in task.scene.get_contacts():
   names=[b.entity.name for b in c.bodies]
   if bottle_name not in names:continue
   pts=[]
   for p in c.points:
    imp=np.asarray(p.impulse,float);normal=np.asarray(p.normal,float);norm=float(np.linalg.norm(imp));pts.append({'position':np.asarray(p.position,float).tolist(),'normal':normal.tolist(),'separation_m':float(p.separation),'impulse_Ns':imp.tolist(),'impulse_norm_Ns':norm})
    if 'table' in names:support+=abs(float(np.dot(imp,normal)))/dt
    else:max_non_table=max(max_non_table,norm)
   contacts.append({'names':names,'points':pts})
  return {'round':round_index,'round_post_wake_step':step,'stage':stage,'round_post_wake_time_s':step*dt,'bottle_matrix':M.tolist(),'functional_point0_world_m':fp.tolist(),'FP0_delta_from_round_initial_m':float(np.linalg.norm(fp-FP0)),'native_is_sleeping':bool(comp.is_sleeping),'native_sleep_threshold':float(comp.get_sleep_threshold()),'native_linear_velocity_m_s':np.asarray(comp.linear_velocity,float).tolist(),'native_angular_velocity_rad_s':np.asarray(comp.angular_velocity,float).tolist(),'diagnostic_relative_rotation_deg':float(np.rad2deg(rel.magnitude())),'diagnostic_relative_localY_axial_twist_deg':twist,'origin_xyz_delta_m':(M[:3,3]-M0[:3,3]).tolist(),'max_absolute_q_change_from_original_rad':float(np.abs(Q-initial_q).max()),'table_support_absolute_normal_N':support,'max_non_table_point_impulse_norm_Ns':max_non_table,'bottle_native_contacts':contacts}
 def compact(row):return {k:v for k,v in row.items() if k!='bottle_native_contacts'}
 for round_index in (1,2):
  M0=np.asarray(task.bottle.get_pose().to_transformation_matrix(),float);FP0=np.asarray(task.bottle.get_functional_point(0,ret='matrix'),float)[:3,3];before=snap(0,'before_native_wake',round_index,M0,FP0);write('round'+str(round_index)+'_before_native_wake.json',before)
  comp.wake_up();wake_calls+=1;rows=[snap(0,'after_native_wake_before_scene_step',round_index,M0,FP0)]
  for step in range(1,2001):
   task.scene.step();post_wake_steps+=1;rows.append(snap(step,'ordinary_physics_after_native_wake',round_index,M0,FP0))
   if step%500==0:print('TWO_WAKE_DIAGNOSTIC_PROGRESS',round_index,step,json.dumps({k:rows[-1][k] for k in ['native_is_sleeping','diagnostic_relative_rotation_deg','FP0_delta_from_round_initial_m','native_angular_velocity_rad_s']}),flush=True)
  write('round'+str(round_index)+'_native_state_trace.json',rows);last200=rows[-200:];peak=max(rows,key=lambda a:a['diagnostic_relative_rotation_deg']);peak_fp=max(rows,key=lambda a:a['FP0_delta_from_round_initial_m']);first_change=next((compact(a) for a in rows[1:] if not np.array_equal(a['bottle_matrix'],M0)),None);first_sleep=next((compact(a) for a in rows[1:] if a['native_is_sleeping']),None)
  rounds.append({'round':round_index,'round_reference':'Native pose/FP read immediately before that round wake; rotation and FPdelta refer only to this round baseline.','wake_calls_this_round':1,'ordinary_physics_steps_this_round':2000,'seconds':2000*dt,'before_wake':compact(before),'after_wake_before_first_step':compact(rows[0]),'first_matrix_change':first_change,'first_natural_sleep':first_sleep,'max_relative_rotation':compact(peak),'max_FP0_delta':compact(peak_fp),'final':compact(rows[-1]),'max_non_table_point_impulse_norm_Ns':max(a['max_non_table_point_impulse_norm_Ns'] for a in rows),'last200':{'all_native_sleeping':all(a['native_is_sleeping'] for a in last200),'max_linear_speed_m_s':max(float(np.linalg.norm(a['native_linear_velocity_m_s'])) for a in last200),'max_angular_speed_rad_s':max(float(np.linalg.norm(a['native_angular_velocity_rad_s'])) for a in last200),'twist_peak_to_peak_deg':float(np.ptp([a['diagnostic_relative_localY_axial_twist_deg'] for a in last200])),'FP0_delta_peak_to_peak_m':float(np.ptp([a['FP0_delta_from_round_initial_m'] for a in last200]))}})
 assert wake_calls==2 and post_wake_steps==4000;assert {str(p):sha(p) for p in source_names}==pins,'Source or v1 bytes changed'
 receipt={'scope':'Fresh original setup with two-round native wake diagnostic, not dataset/episode/acceptance. No expert, arm commands or RGB. Originalfailed27/v1 evidence unchanged.', 'source_hashes_before_and_after':pins,'generator_sha256':sha(__file__),'pid':os.getpid(),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'actual_own_gpu_evidence':own,'gpu_uuid_to_index':gpu_map,'initial_matrix_max_difference_from_old27_scene':float(np.abs(initial_M-np.asarray(old_scene['initial_pose'])).max()),'actual_scene_dt_s':dt,'bottle_collision_shape_count':len(comp.collision_shapes),'original_setup_steps_by_source_after_bottle_creation':3700,'wake_calls':wake_calls,'post_original_setup_physical_steps':post_wake_steps,'post_original_setup_seconds':post_wake_steps*dt,'rounds':rounds,'scope_exclusions':{'sleep_threshold_or_material_changed':False,'post_setup_actor_pose_velocity_or_robot_q_writes':0,'only_native_post_setup_writes':'Exactly2 bottle component.wake_up() calls','expert_created':False,'task_or_gripper_commands_after_setup':False,'render_RGB_observation_calls_after_setup':0,'episode_success_claim':False,'acceptance_criteria_changed':False,'original_failed_label_changed':False},'interpretation_limit':'Native manual wake sufficiency/round repeatability can be measured in this one frozen sampled scene only; historical zero-impulse contacts causing wake, other scenes and actual grasp success remain unproved.','elapsed_wall_s':time.time()-started}
 write('two_wake_diagnostic_receipt.json',receipt);print('TWO_WAKE_DIAGNOSTIC_DONE',json.dumps({'receipt':str(OUT/'two_wake_diagnostic_receipt.json'),'sha256':sha(OUT/'two_wake_diagnostic_receipt.json'),'rounds':[{'round':a['round'],'max_rotation_deg':a['max_relative_rotation']['diagnostic_relative_rotation_deg'],'max_FPdelta_m':a['max_FP0_delta']['FP0_delta_from_round_initial_m'],'final_rotation_deg':a['final']['diagnostic_relative_rotation_deg'],'first_sleep_step':None if a['first_natural_sleep'] is None else a['first_natural_sleep']['round_post_wake_step'],'last200':a['last200']} for a in rounds]}),flush=True)
except BaseException as e:
 write('diagnostic_error.json',{'error':str(e),'traceback':traceback.format_exc(),'wake_calls':wake_calls,'post_wake_steps':post_wake_steps,'finished_rounds':rounds,'sources':pins,'new_episode_success_claim':False});raise
finally:
 if task is not None:
  try:task.close_env(clear_cache=True)
  except BaseException:pass
 if original_initializer is not None:
  from integrations.realman_rm65b_gen4.native_robot import RealManRobot
  RealManRobot._initialize_hands=original_initializer
 task=None;gc.collect()