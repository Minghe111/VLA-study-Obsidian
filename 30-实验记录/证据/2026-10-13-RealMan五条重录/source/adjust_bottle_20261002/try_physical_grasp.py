from types import MethodType
from envs.utils import ArmTag
from envs.robot.robot import Robot
from envs._base_task import Base_Task
original_bottle_state=bottle_state

def audited_bottle_state(task):
 d=original_bottle_state(task)
 R=np.asarray(d['pose'])[:3,:3]
 d['upright_angle_deg']=float(np.degrees(np.arccos(np.clip(R[2,1],-1,1))))
 d['selected_side']='left' if task.qpose_tag==0 else 'right'
 prefix='l_' if task.qpose_tag==0 else 'r_'
 d['selected_hand_contact_points']=sum(c['point_count'] for c in d['hand_contacts'] if c['hand_link'].startswith(prefix))
 return d
bottle_state=audited_bottle_state

def hand_bottle_grasp(self,actor,arm_tag,pre_dis=.1,target_dis=0,contact_point_id=None):
 T=actor.get_contact_point(1,'matrix')@np.array([[0,0,1,0],[-1,0,0,0],[0,-1,0,0],[0,0,0,1]],dtype=float)
 R=T[:3,:3]@np.diag([1.,-1.,-1.]);center=T[:3,3]
 # Same native EE reference, with grasp center in palm +Y/+Z.
 grasp=center-.12*R[:,0]+R@np.array([.18-grasp_setting['hand_z'],-grasp_setting['hand_y'],0.])
 pre=grasp-pre_dis*R[:,0];grasp=grasp-target_dis*R[:,0]
 quat=t3d.quaternions.mat2quat(R).tolist()
 return pre.tolist()+quat,grasp.tolist()+quat

for trial_index,(hand_y,hand_z) in enumerate([(.065,.18),(.055,.18),(.075,.165),(.065,.165)]):
 OUT=BASE/f'physical_grasp_{trial_index:02d}_seed1';OUT.mkdir(exist_ok=False)
 grasp_setting={'hand_y':hand_y,'hand_z':hand_z,'contact_id':1,'roll_flip':True}
 report={'result':'running','task':'adjust_bottle','seed':1,'attempts':[], 'grasp_setting':grasp_setting,'hardware_io':False,
         'source_hashes':{str(p):sha(p) for p in protected},'ready_home':copy.deepcopy(ready)}
 args.update(seed=1,save_path=str(OUT),save_data=False,need_plan=True)
 task.close_env();task.setup_demo(**args)
 for side in ['left','right']:
  for suffix in ['plan_path','plan_multi_path']:
   name=side+'_'+suffix;setattr(task.robot,name,MethodType(getattr(Robot,name),task.robot))
 task.choose_grasp_pose=MethodType(hand_bottle_grasp,task)
 if hasattr(task,'move') and 'move' in task.__dict__:del task.__dict__['move']
 stages=[]
 def traced_move(self,*a,**k):
  result=Base_Task.move(self,*a,**k)
  metric=bottle_state(self);stages.append(metric)
  print('STAGE',len(stages),'plan',self.plan_success,'state',metric,flush=True)
  snapshot(self,f'stage_{len(stages)}')
  return result
 task.move=MethodType(traced_move,task)
 report['initial']=snapshot(task,'initial');report['initial_bottle']=bottle_state(task);audit_ready_home(task)
 phase=instrument(task,'plan');info,extra=run_official_episode(task,'plan')
 legal,excess=task.planned_joints_legal()
 phase.update(plan_success=bool(task.plan_success),task_success=bool(task.check_success()),planned_joints_legal=bool(legal),joint_limit_excess_rad=float(excess))
 report['stages']=stages;report['final']=snapshot(task,'final');report['final_bottle']=bottle_state(task)
 report['result']='physics_success' if extra and legal and report['final_bottle']['upright_angle_deg']<15 else 'physics_failed'
 report['canonical_sources_unchanged']=all(sha(p)==report['source_hashes'][str(p)] for p in protected);write()
 print('TRIAL_RESULT',trial_index,grasp_setting,report['result'],flush=True)
 if report['result']=='physics_success':
  accepted_grasp_setting=grasp_setting;break