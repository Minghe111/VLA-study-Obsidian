"""Bounded first-episode experiment using native RoboTwin expert/planning/recording.
No physical IO. Keep canonical model/config and task source unchanged.
"""
from pathlib import Path
import os,sys,json,math,hashlib,copy,traceback,argparse,time
import xml.etree.ElementTree as ET
import faulthandler
faulthandler.enable();faulthandler.dump_traceback_later(150,repeat=True)
os.environ.setdefault('ROBOTWIN_RT_DENOISER','none')
ROOT=Path('/bigdata2/liminghe/VLA-benchmark/RoboTwin');sys.path.insert(0,str(ROOT));os.chdir(ROOT)
import numpy as np
import yaml
import sapien.core as sapien
from PIL import Image
from transforms3d.quaternions import mat2quat
from integrations.realman_rm65b_gen4.verify_native_support import load_args
import importlib
ap=argparse.ArgumentParser();ap.add_argument('--task',default='adjust_bottle',choices=['adjust_bottle']);ap.add_argument('--seed',type=int,default=0)
ap.add_argument('--align-table',action='store_true');ap.add_argument('--collect',action='store_true');ap.add_argument('--label',default='official')
opt=ap.parse_args();BASE=Path(__file__).resolve().parent
OUT=BASE/f'{opt.task}_{opt.label}_seed{opt.seed}'
if OUT.exists() and any(OUT.iterdir()):
 raise SystemExit(f'Refusing to overwrite an existing attempt: {OUT}; choose a new label.')
OUT.mkdir(exist_ok=True)
args,model=load_args('realman_rm65b_gen4_head',opt.task,OUT)
args.update(seed=opt.seed,save_freq=15,language_num=1)
ready=json.loads((BASE/'ready_home.json').read_text(encoding='utf-8'))
protected=[model/'robot.urdf',model/'robot_sapien.urdf',model/'config.yml',ROOT/f'envs/{opt.task}.py',ROOT/'envs/_base_task.py',ROOT/'integrations/realman_rm65b_gen4/native_robot.py']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
hashes={str(p):sha(p) for p in protected}
# A constant legal mechanical head pose preserves the native 36-DOF arm/hand contract.
# This is a posed simulation derivative, not a changed mechanical limit/optical transform.
source=ET.parse(model/'robot.urdf').getroot();posed=ET.parse(model/'robot_sapien.urdf').getroot()
head_source=source.find("joint[@name='head_joint2']");down=float(head_source.find('limit').get('lower'))
assert down==-.419 and head_source.find('axis').get('xyz')=='1 0 0'
joint=posed.find("joint[@name='head_joint2']");assert joint.get('type')=='fixed'
assert joint.find('origin').get('rpy')=='0 0 0'
joint.find('origin').set('rpy',f'{down} 0 0')
for mesh in posed.findall('.//mesh'):
 p=(model/mesh.get('filename')).resolve();assert p.exists();mesh.set('filename',str(p))
ET.indent(posed);posed_file=OUT/'robot_head_down_fixed.urdf'
ET.ElementTree(posed).write(posed_file,encoding='utf-8',xml_declaration=True)
for side in ['left','right']:
 cfg=copy.deepcopy(args[side+'_embodiment_config']);cfg['urdf_path']=str(posed_file)
 cfg['homestate']=[ready['left_joint_rad'],ready['right_joint_rad']]
 if opt.align_table:cfg['planner_table_pose']={'frame':'world','pose':[0,0,.715,1,0,0,0],'dims':[1.2,.7,.05]}
 assert cfg['head_camera_mount']['optical_preset']=='vendor_zero_view'
 args[side+'_embodiment_config']=cfg
(OUT/'run_config.yml').write_text(yaml.safe_dump(args,sort_keys=False),encoding='utf-8')
report={'task':opt.task,'seed':opt.seed,'hardware_io':False,'source_hashes':hashes,'attempts':[],
 'head_pose':{'yaw_rad':0,'pitch_rad':down,'mode':'constant legal mechanical pose in derived URDF','extra_optical_pitch_deg':0,'calibrated':False},
 'representation':'native 14D arm6+synergy per side; measured 24D dexterous extension kept separately',
 'planner_table_geometry':args['left_embodiment_config']['planner_table_pose'],
 'official_task_methods_unmodified':True,'ready_home':ready,'status_note':'runtime validation required; no success assumed',
 'created_at':time.strftime('%Y-%m-%d %H:%M:%S'),'result':'running'}
def write():
 (OUT/'attempt_report.json').write_text(json.dumps(report,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x)),encoding='utf-8')
def instrument(task,phase):
 calls=[];report['attempts'].append({'phase':phase,'planning_calls':calls})
 for side in ['left','right']:
  for suffix in ['plan_path','plan_multi_path']:
   name=side+'_'+suffix;original=getattr(task.robot,name)
   def wrapper(*a,_original=original,_name=name,**kw):
    t=time.monotonic();result=_original(*a,**kw)
    entry={'method':_name,'elapsed_s':time.monotonic()-t,'status':result.get('status'),'target':a[0] if a else None}
    if 'position' in result:entry['path_shape']=[np.asarray(p).shape for p in result['position']] if _name.endswith('multi_path') else np.asarray(result['position']).shape
    calls.append(entry);write();print('PLAN',_name,entry['status'],entry.get('path_shape'),flush=True)
    return result
   setattr(task.robot,name,wrapper)
 return report['attempts'][-1]
def snapshot(task,prefix):
 obs=task.get_obs();Image.fromarray(obs['observation']['head_camera']['rgb']).save(OUT/f'{prefix}_head.png')
 camera=task.scene.add_camera(prefix+'_external',960,720,math.radians(55),.005,10)
 eye=np.array([1.35,.28,1.65]);target=np.array([0,-.36,.9]);f=target-eye;f/=np.linalg.norm(f);left=np.cross([0,0,1.],f);left/=np.linalg.norm(left)
 camera.entity.set_pose(sapien.Pose(eye,mat2quat(np.stack([f,left,np.cross(f,left)],axis=1))))
 task._update_render();camera.take_picture()
 Image.fromarray((np.clip(camera.get_picture('Color')[...,:3],0,1)*255).astype(np.uint8)).save(OUT/f'{prefix}_external.png')
 return {'joint_action':obs['joint_action']['vector'].tolist(),'state24':obs['dexterous_state']['state24'],
         'camera_pose':task.cameras.head_camera.entity.get_pose().to_transformation_matrix().tolist()}
def audit_ready_home(task):
 measured=np.asarray(task.robot.get_state24(measured=True),dtype=float)
 camera=task.cameras.head_camera.entity.get_pose().to_transformation_matrix()
 links={x.get_name():x for x in task.robot.left_entity.get_links()}
 audit={}
 for side,start,prefix in [('left',0,'l'),('right',12,'r')]:
  error=float(np.max(np.abs(measured[start:start+6]-np.asarray(ready[side+'_joint_rad']))))
  palm=links[prefix+'_inspire_base_link'].get_pose().to_transformation_matrix()
  tcp=palm[:3,3]+.18*palm[:3,2]
  local=camera[:3,:3].T @ (tcp-camera[:3,3])
  tan_v=math.tan(math.radians(37)/2)
  in_frustum=bool(local[0]>0 and abs(local[1])<local[0]*tan_v*320/240
                  and abs(local[2])<local[0]*tan_v)
  audit[side]={'max_joint_error_rad':error,'nominal_tcp_in_frustum':in_frustum,
               'nominal_tcp_world_m':tcp.tolist(),'occlusion_and_full_hand_extent_checked':False}
 report['ready_home_audit']=audit;write()
 if not all(v['max_joint_error_rad']<.025 and v['nominal_tcp_in_frustum'] for v in audit.values()):
  raise RuntimeError('Candidate ready posture has not settled or is outside the unchanged head view')

def bottle_state(task):
 contacts=[]
 for contact in task.scene.get_contacts():
  names=[b.entity.name for b in contact.bodies]
  if '001_bottle' not in names:continue
  other=names[1] if names[0]=='001_bottle' else names[0]
  if other in task.robot.gripper_name:
   impulses=[float(np.linalg.norm(p.impulse)) for p in contact.points]
   contacts.append({'hand_link':other,'point_count':len(contact.points),'active_point_count':sum(x>1e-7 for x in impulses),'impulse_norm_sum':sum(impulses)})
 pose=task.bottle.get_pose().to_transformation_matrix();prefix='l_' if task.qpose_tag==0 else 'r_'
 return {'model_id':int(task.model_id),'qpose_tag':int(task.qpose_tag),'pose':pose.tolist(),
         'functional_point0':np.asarray(task.bottle.get_functional_point(0)).tolist(),
         'official_success':bool(task.check_success()),'hand_contacts':contacts,
         'upright_angle_deg':float(np.degrees(np.arccos(np.clip(pose[2,1],-1,1)))),
         'selected_hand_active_points':sum(c['active_point_count'] for c in contacts if c['hand_link'].startswith(prefix))}

def hold_current_targets(task, duration_s):
 controls={'left_arm':None,'right_arm':None,'left_gripper':None,'right_gripper':None}
 steps=int(round(duration_s/task.scene.get_timestep()))
 for side in ('left','right'):
  q=np.array([j.get_drive_target()[0] for j in getattr(task.robot,side+'_arm_joints')])
  controls[side+'_arm']={'position':np.tile(q,(steps,1)),'velocity':np.zeros((steps,6))}
 task.take_dense_action(controls)

def run_official_episode(task, phase):
 # Record the ready posture before the unchanged official expert starts.
 hold_current_targets(task,.8)
 before=bottle_state(task)
 info=task.play_once()
 immediate=bottle_state(task)
 # Extra validation only: a bottle flying through the target is not a grasp.
 if task.plan_success:
  hold_current_targets(task,1.0)
 after=bottle_state(task)
 drift=float(np.linalg.norm(np.asarray(after['functional_point0'][:3])-np.asarray(immediate['functional_point0'][:3])))
 accepted=bool(task.plan_success and immediate['official_success'] and after['official_success']
               and before['selected_hand_active_points']==0 and after['selected_hand_active_points']>0
               and immediate['upright_angle_deg']<15 and after['upright_angle_deg']<15 and drift<.02)
 audit={'phase':phase,'before':before,'immediate':immediate,'after_hold':after,
        'accepted':accepted,'hold_validation_s':1.0,'hold_drift_m':drift,'contact_requirement':'selected-hand contact impulse above 1e-7 Ns, upright under 15 degrees, hold drift under 20 mm'}
 report.setdefault('bottle_attempts',[]).append(audit)
 info['realman_bottle_audit']=audit
 write()
 return info,accepted

try:
 Task=getattr(importlib.import_module('envs.'+opt.task),opt.task)
 task=Task();print('SETUP START',opt.task,opt.seed,flush=True);task.setup_demo(**args);print('SETUP DONE',flush=True)
 report['initial']=snapshot(task,'initial');audit_ready_home(task)
 report['initial_bottle']=bottle_state(task)
 phase=instrument(task,'plan');t=time.monotonic();info,contact_pass=run_official_episode(task,'plan')
 phase.update(plan_success=bool(task.plan_success),task_success=bool(task.check_success()),contact_and_hold_success=contact_pass,elapsed_s=time.monotonic()-t)
 legal,excess=task.planned_joints_legal();phase.update(planned_joints_legal=bool(legal),joint_limit_excess_rad=float(excess))
 report['plan_scene_info']=info;report['final']=snapshot(task,'final');print('TASK_RESULT',json.dumps({k:v for k,v in phase.items() if k!='planning_calls'}),flush=True)
 if not (phase['plan_success'] and phase['task_success'] and phase['contact_and_hold_success'] and legal):
  report['result']='task_failed';write();sys.exit(2)
 task.save_traj_data(0);(OUT/'seed.txt').write_text(str(opt.seed)+'\n')
 if not opt.collect:
  report['result']='plan_and_task_passed_not_collected';write();sys.exit(0)
 task.close_env();args.update(need_plan=False,save_data=True)
 print('REPLAY SETUP',flush=True);task.setup_demo(**args)
 traj=task.load_tran_data(0);args.update(traj);task.set_path_lst(args)
 step=[0];original_step=task.scene.step
 def timed_step():
  original_step();step[0]+=1
 task.scene.step=timed_step
 original_obs=task.get_obs
 def timed_obs():
  obs=original_obs();obs['recording_clock']={'physics_step':step[0],'sim_time_s':step[0]*task.scene.get_timestep()};obs['object_audit']=bottle_state(task);return obs
 task.get_obs=timed_obs
 t=time.monotonic();info,contact_pass=run_official_episode(task,'replay');replay_success=bool(task.check_success() and contact_pass)
 report['replay']={'task_success':replay_success,'plan_success':bool(task.plan_success),'frames':task.FRAME_IDX,'elapsed_s':time.monotonic()-t}
 if not replay_success:report['result']='replay_failed';write();sys.exit(3)
 from scripts.collect_data import build_episode_instructions
 descriptions=build_episode_instructions(args,info,0)
 (OUT/'scene_info.json').write_text(json.dumps({'episode_0':info},indent=2),encoding='utf-8')
 task.merge_pkl_to_hdf5_video(instructions=descriptions['seen'])
 # Preserve measured dexterous information omitted by the installed generic exporter.
 import pickle,h5py
 files=sorted(Path(task.folder_path['cache']).glob('*.pkl'),key=lambda p:int(p.stem))
 samples=[pickle.loads(p.read_bytes()) for p in files];hdf=OUT/'data/episode_0000000.hdf5'
 with h5py.File(hdf,'a') as f:
  g=f.create_group('realman_dexterous');g.attrs['sdk_mapping_calibrated']=False
  g.attrs['ordering']='Larm6 rad, Lhand6 model-SDK, Rarm6 rad, Rhand6 model-SDK'
  g.create_dataset('measured_state24',data=np.asarray([s['dexterous_state']['state24'] for s in samples])[:-1])
  g.create_dataset('next_command24',data=np.asarray([s['dexterous_state']['command24'] for s in samples])[1:])
  g.create_dataset('sample_time_s',data=np.asarray([s['recording_clock']['sim_time_s'] for s in samples])[:-1])
  g.create_dataset('next_sample_time_s',data=np.asarray([s['recording_clock']['sim_time_s'] for s in samples])[1:])
  g.attrs['timing_note']='Use sample times; native save_freq is physics-step interval and includes duplicate action boundary frames.'
  n=len(samples)-1;assert f['vision/cam_head/colors'].shape[0]==n and f['state/left_arm_joint_states'].shape==(n,6)
  assert g['measured_state24'].shape==(n,24)
  report['dataset']={'hdf5':str(hdf),'video':str(OUT/'video/episode_0000000.mp4'),'pairs':n,'recorded_frames':len(samples),'cameras':list(f['vision']),'physical_step_s':float(task.scene.get_timestep()),'save_interval_steps':args['save_freq'],'sha256':None}
 report['dataset']['sha256']=sha(hdf)
 report['result']='one_episode_collected';report['instructions']=descriptions
 print('DATASET',report['dataset'],flush=True)
except Exception:
 report['result']='exception';report['traceback']=traceback.format_exc();print(report['traceback'],flush=True);raise
finally:
 report['canonical_sources_unchanged']=all(sha(p)==hashes[str(p)] for p in protected);write()
 print('FINISHED',report['result'],flush=True)