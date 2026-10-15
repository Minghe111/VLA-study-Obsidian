from types import MethodType
from envs.robot.robot import Robot
from envs._base_task import Base_Task
if 'accepted_grasp_setting' in globals():
 print('SKIP_SOFT_ALREADY_SUCCESS',flush=True)
else:
 for trial_index,(stiffness,finger,thumb,y,z) in enumerate([(1.,400,250,.065,.165),(1.,350,250,.075,.15),(2.,500,300,.055,.18),(.5,400,250,.075,.15)]):
  OUT=BASE/f'opposed_{trial_index:02d}_seed7';OUT.mkdir(exist_ok=False)
  grasp_setting={'pitch_deg':30,'hand_y':y,'hand_z':z,'open_sdk':[1000,1000,1000,1000,1000,0],'close_sdk':[finger]*4+[thumb,0],'hand_stiffness':stiffness,'hand_damping':.05}
  report={'result':'running','task':'adjust_bottle','seed':7,'attempts':[],'grasp_setting':grasp_setting,'hardware_io':False,'source_hashes':{str(p):sha(p) for p in protected},'ready_home':copy.deepcopy(ready)}
  for side in ['left','right']:
   args[side+'_embodiment_config'].update(hand_synergy={'open':grasp_setting['open_sdk'],'close':grasp_setting['close_sdk']},hand_joint_stiffness=stiffness,hand_joint_damping=.05)
   task.robot._hand_config[side].update(args[side+'_embodiment_config'])
  args.update(seed=7,save_path=str(OUT),save_data=False,need_plan=True)
  if 'move' in task.__dict__:del task.__dict__['move']
  task.close_env();task.setup_demo(**args)
  for side in ['left','right']:
   for suffix in ['plan_path','plan_multi_path']:
    name=side+'_'+suffix;setattr(task.robot,name,MethodType(getattr(Robot,name),task.robot))
  task.choose_grasp_pose=MethodType(horizontal_grasp,task);stages=[]
  def soft_trace(self,*a,**k):
   result=Base_Task.move(self,*a,**k);m=bottle_state(self);m['sdk']=self.robot.get_hand_sdk('left');m['targets']=self.robot.get_hand_sdk('left',False);stages.append(m)
   print('SOFT_STAGE',len(stages),'z',m['pose'][2][3],'contacts',m['selected_hand_contact_points'],'sdk',m['sdk'],flush=True)
   return result
  task.move=MethodType(soft_trace,task);report['initial_bottle']=bottle_state(task);phase=instrument(task,'plan')
  try:
   info,extra=run_official_episode(task,'plan');legal,excess=task.planned_joints_legal()
   phase.update(plan_success=bool(task.plan_success),task_success=bool(task.check_success()),planned_joints_legal=bool(legal),joint_limit_excess_rad=float(excess))
   report['final_bottle']=bottle_state(task);report['result']='physics_success' if extra and legal and report['final_bottle']['upright_angle_deg']<15 else 'physics_failed'
  except Exception:
   report['result']='exception';report['traceback']=traceback.format_exc()
  report['stages']=stages;report['canonical_sources_unchanged']=all(sha(p)==report['source_hashes'][str(p)] for p in protected);write()
  print('SOFT_RESULT',trial_index,report['result'],flush=True)
  if report['result']=='physics_success':
   accepted_grasp_setting=copy.deepcopy(grasp_setting);accepted_seed=7;task.save_traj_data(0);snapshot(task,'final');break