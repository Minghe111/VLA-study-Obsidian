from types import MethodType
from envs._base_task import Base_Task
from envs.robot.robot import Robot
# Preshape and bounded power-grasp trials. Official task source/objects untouched.
for key in ['move','take_dense_action']:
 if key in task.__dict__:del task.__dict__[key]
for side in ['left','right']:
 args[side+'_embodiment_config']['hand_synergy']={'open':[1000,1000,1000,1000,1000,1000],'close':[250,250,250,250,150,1000]}
 task.robot._hand_config[side]['hand_synergy']=args[side+'_embodiment_config']['hand_synergy']
seed13=None
for seed_candidate in range(2,8):
 args.update(seed=seed_candidate,save_data=False,need_plan=True)
 task.close_env();task.setup_demo(**args)
 print('OFFICIAL_SEED',seed_candidate,task.model_id,task.qpose_tag,flush=True)
 if task.model_id==13 and task.qpose_tag==0:seed13=seed_candidate;break
if seed13 is None:raise RuntimeError('No left model13 seed in bounded search')
for trial_index,(hand_y,hand_z,close_finger) in enumerate([(.055,.16,250),(.055,.18,250),(.065,.17,350),(.045,.16,150),(.055,.15,350),(.065,.15,250)]):
 OUT=BASE/f'preshape_{trial_index:02d}_seed{seed13}';OUT.mkdir(exist_ok=False)
 grasp_setting={'hand_y':hand_y,'hand_z':hand_z,'contact_id':1,'roll_flip':True,'open_sdk':[1000,1000,1000,1000,1000,1000],'close_sdk':[close_finger]*4+[150,1000]}
 report={'result':'running','task':'adjust_bottle','seed':seed13,'attempts':[],'grasp_setting':grasp_setting,'hardware_io':False,'source_hashes':{str(p):sha(p) for p in protected},'ready_home':copy.deepcopy(ready)}
 for side in ['left','right']:
  args[side+'_embodiment_config']['hand_synergy']={'open':grasp_setting['open_sdk'],'close':grasp_setting['close_sdk']}
  task.robot._hand_config[side]['hand_synergy']=args[side+'_embodiment_config']['hand_synergy']
 args.update(seed=seed13,save_path=str(OUT),save_data=False,need_plan=True)
 task.close_env();task.setup_demo(**args)
 for side in ['left','right']:
  for suffix in ['plan_path','plan_multi_path']:
   name=side+'_'+suffix;setattr(task.robot,name,MethodType(getattr(Robot,name),task.robot))
 task.choose_grasp_pose=MethodType(hand_bottle_grasp,task)
 stages=[]
 def short_trace(self,*a,**k):
  result=Base_Task.move(self,*a,**k);metric=bottle_state(self);stages.append(metric)
  print('GRASP_STAGE',len(stages),'plan',self.plan_success,'bottle',np.asarray(metric['pose'])[:3,3].tolist(),'contact',metric['selected_hand_contact_points'],flush=True)
  return result
 task.move=MethodType(short_trace,task)
 report['initial_bottle']=bottle_state(task)
 phase=instrument(task,'plan')
 try:
  info,extra=run_official_episode(task,'plan');legal,excess=task.planned_joints_legal()
  phase.update(plan_success=bool(task.plan_success),task_success=bool(task.check_success()),planned_joints_legal=bool(legal),joint_limit_excess_rad=float(excess))
  report['final_bottle']=bottle_state(task)
  report['result']='physics_success' if extra and legal and report['final_bottle']['upright_angle_deg']<15 else 'physics_failed'
 except Exception:
  report['result']='exception';report['traceback']=traceback.format_exc()
 report['stages']=stages;report['canonical_sources_unchanged']=all(sha(p)==report['source_hashes'][str(p)] for p in protected);write()
 print('PRESHAPE_RESULT',trial_index,report['result'],flush=True)
 if report['result']=='physics_success':
  accepted_grasp_setting=copy.deepcopy(grasp_setting);accepted_seed=seed13
  task.save_traj_data(0);snapshot(task,'final');break