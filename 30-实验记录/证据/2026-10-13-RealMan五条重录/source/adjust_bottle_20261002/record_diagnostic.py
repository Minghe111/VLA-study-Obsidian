if 'accepted_grasp_setting' in globals():raise RuntimeError('A successful candidate exists; collect it instead of rendering a failed diagnostic')
from types import MethodType
from envs.robot.robot import Robot
from envs._base_task import Base_Task
import subprocess,shutil
OUT=BASE/'official_adjust_bottle_diagnostic_seed7';OUT.mkdir(exist_ok=False)
grasp_setting={'pitch_deg':30,'hand_y':.085,'hand_z':.17,'open_sdk':[1000,1000,1000,1000,1000,0],'close_sdk':[400]*4+[500,0],'hand_stiffness':.1,'hand_damping':.01}
report={'result':'running','task':'adjust_bottle','seed':7,'attempts':[],'grasp_setting':grasp_setting,'hardware_io':False,'source_hashes':{str(p):sha(p) for p in protected},'ready_home':copy.deepcopy(ready),'purpose':'Actual failed official-task diagnostic, not an accepted dataset episode','head_pose':{'pitch_rad':-.419,'extra_optical_pitch_deg':0,'preset':'vendor_zero_view'}}
for name in ['set_gripper','set_arm_joints']:
 if name in task.robot.__dict__:del task.robot.__dict__[name]
for side in ['left','right']:
 args[side+'_embodiment_config'].update(hand_synergy={'open':grasp_setting['open_sdk'],'close':grasp_setting['close_sdk']},hand_joint_stiffness=.1,hand_joint_damping=.01)
 task.robot._hand_config[side].update(args[side+'_embodiment_config'])
args.update(seed=7,save_path=str(OUT),save_data=False,need_plan=True)
for name in ['move','take_dense_action']:
 if name in task.__dict__:del task.__dict__[name]
task.close_env();task.setup_demo(**args)
for side in ['left','right']:
 for suffix in ['plan_path','plan_multi_path']:
  name=side+'_'+suffix;setattr(task.robot,name,MethodType(getattr(Robot,name),task.robot))
task.choose_grasp_pose=MethodType(horizontal_grasp,task);audit_ready_home(task)
(OUT/'run_config.yml').write_text(yaml.safe_dump(args,sort_keys=False))
external=task.scene.add_camera('diagnostic_external',640,480,math.radians(52),.005,10)
eye=np.array([-.85,.16,1.40]);target=np.array([-.025,-.22,.89]);f=target-eye;f/=np.linalg.norm(f);l=np.cross([0,0,1.],f);l/=np.linalg.norm(l)
external.entity.set_pose(sapien.Pose(eye,mat2quat(np.stack([f,l,np.cross(f,l)],axis=1))))
ffmpeg=shutil.which('ffmpeg');assert ffmpeg
writers={}
for name,w,h in [('head',320,240),('external',640,480)]:
 writers[name]=subprocess.Popen([ffmpeg,'-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r','25','-i','-','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p',str(OUT/f'failed_diagnostic_{name}_realtime.mp4')],stdin=subprocess.PIPE)
steps=[0];frames=[];stages=[];base_step=task.scene.step

def capture_frame(label=None):
 task._update_render()
 for name,camera in [('head',task.cameras.head_camera),('external',external)]:
  camera.take_picture();rgb=(np.clip(camera.get_picture('Color')[...,:3],0,1)*255).astype(np.uint8)
  writers[name].stdin.write(rgb.tobytes())
  if label:Image.fromarray(rgb).save(OUT/f'{label}_{name}.png')
 frames.append(steps[0]*task.scene.get_timestep())
def video_step():
 base_step();steps[0]+=1
 if steps[0]%10==0:capture_frame()
task.scene.step=video_step

def video_stage(self,*a,**k):
 result=Base_Task.move(self,*a,**k);m=bottle_state(self);m['sim_time_s']=steps[0]*self.scene.get_timestep();stages.append(m)
 # The labelled still is captured without adding an extra video frame.
 self._update_render()
 for name,camera in [('head',self.cameras.head_camera),('external',external)]:
  camera.take_picture();Image.fromarray((np.clip(camera.get_picture('Color')[...,:3],0,1)*255).astype(np.uint8)).save(OUT/f'stage{len(stages)}_{name}.png')
 print('VIDEO_STAGE',len(stages),m['official_success'],m['selected_hand_contact_points'],flush=True)
 return result
task.move=MethodType(video_stage,task);report['initial_bottle']=bottle_state(task);phase=instrument(task,'plan')
try:
 capture_frame('initial');info,extra=run_official_episode(task,'plan')
 if not task.plan_success:hold_current_targets(task,1.0)
 legal,excess=task.planned_joints_legal();report['final_bottle']=bottle_state(task)
 phase.update(plan_success=bool(task.plan_success),task_success=bool(task.check_success()),planned_joints_legal=bool(legal),joint_limit_excess_rad=float(excess))
 report['result']='physics_success_not_collected' if extra and legal and report['final_bottle']['upright_angle_deg']<15 else 'failed_diagnostic_only'
 report['stages']=stages
 if report['result']=='physics_success_not_collected':accepted_grasp_setting=copy.deepcopy(grasp_setting);accepted_seed=7;task.save_traj_data(0)
except Exception:
 report['result']='exception';report['traceback']=traceback.format_exc();print(report['traceback'],flush=True)
finally:
 for writer in writers.values():writer.stdin.close();assert writer.wait()==0
 report['video']={'fps':25,'frames':len(frames),'source_time_s':frames[-1] if frames else 0,'head':'failed_diagnostic_head_realtime.mp4','external':'failed_diagnostic_external_realtime.mp4','source':'actual simulation frames every 10 physics steps; planning wall-clock excluded; no interpolation'}
 report['canonical_sources_unchanged']=all(sha(p)==report['source_hashes'][str(p)] for p in protected);write();(OUT/'video_timestamps.json').write_text(json.dumps(frames))
 print('VIDEO_FINISHED',report['result'],report['video'],flush=True)