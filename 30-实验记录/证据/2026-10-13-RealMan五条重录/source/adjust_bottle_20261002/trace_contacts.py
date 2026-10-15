from types import MethodType
from envs._base_task import Base_Task
from envs.robot.robot import Robot
OUT=BASE/'grasp_contact_trace_seed1';OUT.mkdir(exist_ok=False)
report={'result':'running','attempts':[],'source_hashes':{str(p):sha(p) for p in protected}}
args.update(seed=1,save_path=str(OUT),save_data=False,need_plan=True)
task.close_env();task.setup_demo(**args)
for key in ['move','take_dense_action']:
 if key in task.__dict__:del task.__dict__[key]
for side in ['left','right']:
 for suffix in ['plan_path','plan_multi_path']:
  name=side+'_'+suffix;setattr(task.robot,name,MethodType(getattr(Robot,name),task.robot))
grasp_setting={'hand_y':.065,'hand_z':.18}
task.choose_grasp_pose=MethodType(hand_bottle_grasp,task)
trace=[]
def trace_state():
 d=bottle_state(task);palm=next(l for l in task.robot.left_entity.get_links() if l.name=='l_inspire_base_link').get_pose().to_transformation_matrix()
 d['bottle_in_palm']=(np.linalg.inv(palm)@np.asarray(d['pose'])).tolist()
 d['hand_sdk_measured']=task.robot.get_hand_sdk('left');d['hand_sdk_target']=task.robot.get_hand_sdk('left',False)
 d['palm']=palm.tolist();d['ee']=task.robot.get_left_ee_pose()
 return d
def closeup(label):
 cam=task.scene.add_camera(label,800,600,math.radians(50),.005,10)
 eye=np.array([.35,-.02,1.1]);target=np.array([-.05,-.15,.80]);f=target-eye;f/=np.linalg.norm(f);l=np.cross([0,0,1],f);l/=np.linalg.norm(l)
 cam.entity.set_pose(sapien.Pose(eye,mat2quat(np.stack([f,l,np.cross(f,l)],axis=1))))
 task._update_render();cam.take_picture();Image.fromarray((np.clip(cam.get_picture('Color')[...,:3],0,1)*255).astype(np.uint8)).save(OUT/(label+'.png'))
def trace_dense(self,control_seq,save_freq=-1):
 keys=[k for k,v in control_seq.items() if v is not None]
 before=trace_state();result=Base_Task.take_dense_action(self,control_seq,save_freq)
 after=trace_state();trace.append({'controls':keys,'before':before,'after':after})
 n=len(trace);closeup(f'action{n:02d}');print('DENSE_TRACE',n,keys,'bottle',after['pose'],'sdk',after['hand_sdk_measured'],'target',after['hand_sdk_target'],flush=True)
 (OUT/'trace.json').write_text(json.dumps(trace,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x)))
 return result
task.take_dense_action=MethodType(trace_dense,task)
report['initial']=trace_state();report['grasp_pose']=hand_bottle_grasp(task,task.bottle,'left')
phase=instrument(task,'contact_trace');task.play_once();report['result']='diagnostic_completed';write()