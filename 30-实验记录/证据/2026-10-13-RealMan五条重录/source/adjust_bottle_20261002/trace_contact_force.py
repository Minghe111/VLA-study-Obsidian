from types import MethodType
from envs.robot.robot import Robot
from envs._base_task import Base_Task
for trial_index,(stiffness,y,z) in enumerate([(.03,.095,.18),(.1,.095,.18),(.3,.095,.18),(.6,.095,.18),(.1,.085,.17),(.3,.085,.17)]):
 OUT=BASE/f'contact_force_{trial_index:02d}_seed7';OUT.mkdir(exist_ok=False)
 grasp_setting={'pitch_deg':30,'hand_y':y,'hand_z':z,'open_sdk':[1000,1000,1000,1000,1000,0],'close_sdk':[400]*4+[500,0],'hand_stiffness':stiffness,'hand_damping':.01}
 report={'result':'running','task':'adjust_bottle','seed':7,'attempts':[],'grasp_setting':grasp_setting,'hardware_io':False,'source_hashes':{str(p):sha(p) for p in protected},'ready_home':copy.deepcopy(ready),'fail_fast_diagnostics':True}
 for side in ['left','right']:
  args[side+'_embodiment_config'].update(hand_synergy={'open':grasp_setting['open_sdk'],'close':grasp_setting['close_sdk']},hand_joint_stiffness=stiffness,hand_joint_damping=.01)
  task.robot._hand_config[side].update(args[side+'_embodiment_config'])
 args.update(seed=7,save_path=str(OUT),save_data=False,need_plan=True)
 for name in ['move','take_dense_action']:
  if name in task.__dict__:del task.__dict__[name]
 task.close_env();task.setup_demo(**args)
 for side in ['left','right']:
  for suffix in ['plan_path','plan_multi_path']:
   name=side+'_'+suffix;setattr(task.robot,name,MethodType(getattr(Robot,name),task.robot))
 task.choose_grasp_pose=MethodType(horizontal_grasp,task);stages=[];physics_trace=[];step_count=[0];active_control=['hold']
 for c in task.bottle.actor.get_components():
  if isinstance(c,sapien.physx.PhysxRigidDynamicComponent):report['object_physics']={'mass':c.mass,'inertia':c.inertia.tolist()}
 original_step=task.scene.step
 def force_step():
  original_step();step_count[0]+=1
  if step_count[0]%25:return
  contacts=[]
  for c in task.scene.get_contacts():
   names=[b.entity.name for b in c.bodies]
   if '001_bottle' not in names:continue
   points=[{'position':p.position.tolist(),'normal':p.normal.tolist(),'impulse':p.impulse.tolist(),'separation':float(p.separation)} for p in c.points]
   contacts.append({'bodies':names,'points':points})
  physics_trace.append({'step':step_count[0],'control':active_control[:],'bottle':task.bottle.get_pose().to_transformation_matrix().tolist(),'sdk':task.robot.get_hand_sdk('left'),'target':task.robot.get_hand_sdk('left',False),'contacts':contacts})
 task.scene.step=force_step
 def force_dense(self,control_seq,save_freq=-1):
  active_control[:]=[k for k,v in control_seq.items() if v is not None];return Base_Task.take_dense_action(self,control_seq,save_freq)
 task.take_dense_action=MethodType(force_dense,task)
 def force_stage(self,*a,**k):
  r=Base_Task.move(self,*a,**k);m=bottle_state(self);m['sdk']=self.robot.get_hand_sdk('left');stages.append(m)
  print('FORCE_STAGE',len(stages),'fpz',m['functional_point0'][2],'contacts',m['selected_hand_contact_points'],'sdk',m['sdk'],flush=True)
  if len(stages)==2 and m['selected_hand_contact_points']==0:self.plan_success=False
  return r
 task.move=MethodType(force_stage,task);report['initial_bottle']=bottle_state(task);phase=instrument(task,'plan')
 try:
  info,extra=run_official_episode(task,'plan');legal,excess=task.planned_joints_legal()
  phase.update(plan_success=bool(task.plan_success),task_success=bool(task.check_success()),planned_joints_legal=bool(legal),joint_limit_excess_rad=float(excess))
  report['final_bottle']=bottle_state(task);report['result']='physics_success' if extra and legal and report['final_bottle']['upright_angle_deg']<15 else 'physics_failed'
 except Exception:
  report['result']='exception';report['traceback']=traceback.format_exc()
 report['stages']=stages;write();(OUT/'physics_trace.json').write_text(json.dumps(physics_trace,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x)))
 print('FORCE_RESULT',trial_index,report['result'],flush=True)
 if report['result']=='physics_success':
  accepted_grasp_setting=copy.deepcopy(grasp_setting);accepted_seed=7;task.save_traj_data(0);snapshot(task,'final');break