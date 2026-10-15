from types import MethodType
from envs.robot.robot import Robot
from integrations.realman_rm65b_gen4.native_robot import RealManRobot
from envs._base_task import Base_Task

def install_contact_limit(task, setting):
 robot=task.robot;dt=task.scene.get_timestep();feedback={'active':False,'latched':np.zeros(5,dtype=bool),'target':np.asarray(robot.get_hand_sdk('left',False),dtype=float),'scalar':1.0,'forces':np.zeros(5),'samples':[]}
 def contact_forces():
  f=np.zeros(5);tokens=['little','ring','middle','index','thumb']
  for c in task.scene.get_contacts():
   names=[b.entity.name for b in c.bodies]
   if '001_bottle' not in names:continue
   other=names[1] if names[0]=='001_bottle' else names[0]
   if not other.startswith('l_inspire_'):continue
   for i,token in enumerate(tokens):
    if token in other:f[i]+=sum(abs(float(np.dot(p.impulse,p.normal))) for p in c.points)/dt;break
  return f
 def regulate(nominal):
  f=contact_forces();measured=np.asarray(robot.get_hand_sdk('left'));output=np.asarray(nominal,dtype=float).copy()
  for i in range(5):
   if f[i]>.08 and not feedback['latched'][i]:feedback['latched'][i]=True;feedback['target'][i]=measured[i]-5
   if feedback['latched'][i]:
    desired=setting['finger_force_N']*(3 if i==4 else 1)
    delta=float(np.clip((desired-f[i])*150*dt,-1.,1.))
    feedback['target'][i]=np.clip(feedback['target'][i]-delta,nominal[i],1000)
    output[i]=feedback['target'][i]
  output[5]=0;feedback['forces']=f;robot.set_hand_angle('left',np.rint(output).astype(int).tolist());robot.left_gripper_val=feedback['scalar']
 def adaptive_gripper(self,value,arm_tag,gripper_eps=.1):
  if str(arm_tag)!='left' or value>.999:
   if str(arm_tag)=='left':feedback['active']=False;feedback['latched'][:]=False;feedback['scalar']=float(value)
   return RealManRobot.set_gripper(self,value,arm_tag,gripper_eps)
  requested=float(np.clip(value,0,1));feedback['scalar']=requested;feedback['active']=True
  nominal=self._synergy['left']['close']+requested*(self._synergy['left']['open']-self._synergy['left']['close']);regulate(nominal)
 def adaptive_arm(self,position,velocity,arm_tag):
  Robot.set_arm_joints(self,position,velocity,arm_tag)
  if str(arm_tag)=='left' and feedback['active']:
   nominal=self._synergy['left']['close']+feedback['scalar']*(self._synergy['left']['open']-self._synergy['left']['close']);regulate(nominal)
 robot.set_gripper=MethodType(adaptive_gripper,robot);robot.set_arm_joints=MethodType(adaptive_arm,robot)
 return feedback

for trial_index,(y,z,force) in enumerate([(.085,.17,.25),(.095,.18,.25),(.075,.175,.25),(.085,.17,.5)]):
 OUT=BASE/f'contact_limited_{trial_index:02d}_seed7';OUT.mkdir(exist_ok=False)
 grasp_setting={'pitch_deg':30,'hand_y':y,'hand_z':z,'open_sdk':[1000,1000,1000,1000,1000,0],'close_sdk':[400]*4+[500,0],'hand_stiffness':1.,'hand_damping':.05,'finger_force_N':force,'tactile_source':'PhysX bottle contact normal impulses; simulation only'}
 report={'result':'running','task':'adjust_bottle','seed':7,'attempts':[],'grasp_setting':grasp_setting,'hardware_io':False,'source_hashes':{str(p):sha(p) for p in protected},'ready_home':copy.deepcopy(ready),'fail_fast_diagnostics':True}
 for name in ['set_gripper','set_arm_joints']:
  if name in task.robot.__dict__:del task.robot.__dict__[name]
 for side in ['left','right']:
  args[side+'_embodiment_config'].update(hand_synergy={'open':grasp_setting['open_sdk'],'close':grasp_setting['close_sdk']},hand_joint_stiffness=1.,hand_joint_damping=.05)
  task.robot._hand_config[side].update(args[side+'_embodiment_config'])
 args.update(seed=7,save_path=str(OUT),save_data=False,need_plan=True)
 for name in ['move','take_dense_action']:
  if name in task.__dict__:del task.__dict__[name]
 task.close_env();task.setup_demo(**args)
 for side in ['left','right']:
  for suffix in ['plan_path','plan_multi_path']:
   name=side+'_'+suffix;setattr(task.robot,name,MethodType(getattr(Robot,name),task.robot))
 task.choose_grasp_pose=MethodType(horizontal_grasp,task);stages=[];feedback=install_contact_limit(task,grasp_setting)
 def limit_stage(self,*a,**k):
  result=Base_Task.move(self,*a,**k);m=bottle_state(self);m['sdk']=self.robot.get_hand_sdk('left');m['forces']=feedback['forces'].tolist();m['latched']=feedback['latched'].tolist();stages.append(m)
  print('LIMIT_STAGE',len(stages),'fpz',m['functional_point0'][2],'contact',m['selected_hand_contact_points'],'force',m['forces'],flush=True)
  if len(stages)==2 and m['selected_hand_contact_points']==0:self.plan_success=False
  return result
 task.move=MethodType(limit_stage,task);report['initial_bottle']=bottle_state(task);phase=instrument(task,'plan')
 try:
  info,extra=run_official_episode(task,'plan');legal,excess=task.planned_joints_legal();report['final_bottle']=bottle_state(task)
  phase.update(plan_success=bool(task.plan_success),task_success=bool(task.check_success()),planned_joints_legal=bool(legal),joint_limit_excess_rad=float(excess))
  report['result']='physics_success' if extra and legal and report['final_bottle']['upright_angle_deg']<15 else 'physics_failed'
 except Exception:
  report['result']='exception';report['traceback']=traceback.format_exc()
 report['stages']=stages;write();print('LIMIT_RESULT',trial_index,report['result'],flush=True)
 if report['result']=='physics_success':
  accepted_grasp_setting=copy.deepcopy(grasp_setting);accepted_seed=7;task.save_traj_data(0);snapshot(task,'final');break