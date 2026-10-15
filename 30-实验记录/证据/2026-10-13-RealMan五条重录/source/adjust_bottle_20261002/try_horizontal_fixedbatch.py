from curobo.types.math import Pose as CuroboPose
from curobo.types.state import JointState
from envs.robot.robot import Robot
from envs._base_task import Base_Task
from types import MethodType
seed13=7
OUT=BASE/'horizontal_geometry_seed7';OUT.mkdir(exist_ok=True)
report={'result':'running','attempts':[]};args.update(seed=7,save_path=str(OUT),save_data=False,need_plan=True)
if 'move' in task.__dict__:del task.__dict__['move']
task.close_env();task.setup_demo(**args)
def horizontal_grasp(self,actor,arm_tag,pre_dis=.1,target_dis=0,contact_point_id=None):
 M=actor.get_pose().to_transformation_matrix();px=M[:3,1].copy()
 if px[0]<0:px=-px
 px/=np.linalg.norm(px)
 angle=math.radians(grasp_setting['pitch_deg']);pz=np.array([0,math.cos(angle),-math.sin(angle)])
 pz-=px*np.dot(px,pz);pz/=np.linalg.norm(pz);py=np.cross(pz,px)
 R=np.stack([pz,py,-px],axis=1)
 center=actor.get_contact_point(0,'matrix')[:3,3]
 grasp=center-(grasp_setting['hand_z']-.06)*pz-grasp_setting['hand_y']*py
 return [*(grasp-pre_dis*pz),*mat2quat(R)],[*(grasp-target_dis*pz),*mat2quat(R)]
metadata=[];positions=[];quaternions=[]
mg=task.robot.left_planner.motion_gen;origin=task.robot.left_entity_origion_pose;rootR=t3d.quaternions.quat2mat(origin.q)
for pitch_deg in [0,15,30,45,60]:
 for hand_y in [.045,.055,.065,.075]:
  for hand_z in [.12,.135,.15,.165,.18]:
   grasp_setting={'pitch_deg':pitch_deg,'hand_y':hand_y,'hand_z':hand_z}
   pre,g=horizontal_grasp(task,task.bottle,'left');R=t3d.quaternions.quat2mat(g[3:]);palm=np.asarray(g[:3])-.06*R[:,0];Rp=R@np.array([[0,0,1],[0,1,0],[-1,0,0]])
   metadata.append(dict(grasp_setting));positions.append(rootR.T@(palm-origin.p));quaternions.append(mat2quat(rootR.T@Rp))
batch_ok=[];batch_q=[]
for start in range(0,len(metadata),96):
 ix=np.minimum(np.arange(start,start+96),len(metadata)-1);count=min(96,len(metadata)-start)
 res=kin_ik.solve_batch(CuroboPose(position=ta.to_device(np.asarray(positions)[ix]),quaternion=ta.to_device(np.asarray(quaternions)[ix])),return_seeds=4)
 batch_ok.append(res.success.detach().cpu().numpy().reshape(96,4)[:count]);batch_q.append(res.solution.detach().cpu().numpy().reshape(96,4,6)[:count])
ok=np.concatenate(batch_ok);qs=np.concatenate(batch_q)
valid=np.zeros(ok.size,dtype=bool);ix=np.flatnonzero(ok.ravel())
if len(ix):valid[ix]=mg.ik_solver.check_constraints(JointState.from_position(ta.to_device(qs.reshape(-1,6)[ix]))).feasible.detach().cpu().numpy().ravel()
valid=valid.reshape(ok.shape);indices=np.flatnonzero(valid.any(axis=1));candidates=[metadata[i] for i in indices]
candidates.sort(key=lambda x:(abs(x['pitch_deg']-15),abs(x['hand_z']-.15),abs(x['hand_y']-.055)))
(OUT/'candidates.json').write_text(json.dumps({'kinematic':int(ok.sum()),'valid':int(valid.sum()),'candidates':candidates},indent=2));print('HORIZONTAL_CANDIDATES',len(candidates),candidates[:12],flush=True)
for trial_index,geometry in enumerate(candidates[:10]):
 OUT=BASE/f'horizontal_{trial_index:02d}_seed7';OUT.mkdir(exist_ok=False)
 grasp_setting=dict(geometry);grasp_setting.update(open_sdk=[1000,1000,1000,1000,1000,1000],close_sdk=[250,250,250,250,150,1000])
 report={'result':'running','task':'adjust_bottle','seed':7,'attempts':[],'grasp_setting':grasp_setting,'hardware_io':False,'source_hashes':{str(p):sha(p) for p in protected},'ready_home':copy.deepcopy(ready)}
 for side in ['left','right']:
  args[side+'_embodiment_config']['hand_synergy']={'open':grasp_setting['open_sdk'],'close':grasp_setting['close_sdk']}
  task.robot._hand_config[side]['hand_synergy']=args[side+'_embodiment_config']['hand_synergy']
 args.update(seed=7,save_path=str(OUT),save_data=False,need_plan=True)
 task.close_env();task.setup_demo(**args)
 for side in ['left','right']:
  for suffix in ['plan_path','plan_multi_path']:
   name=side+'_'+suffix;setattr(task.robot,name,MethodType(getattr(Robot,name),task.robot))
 task.choose_grasp_pose=MethodType(horizontal_grasp,task);stages=[]
 task.move=MethodType(short_trace,task);report['initial_bottle']=bottle_state(task);phase=instrument(task,'plan')
 try:
  info,extra=run_official_episode(task,'plan');legal,excess=task.planned_joints_legal()
  phase.update(plan_success=bool(task.plan_success),task_success=bool(task.check_success()),planned_joints_legal=bool(legal),joint_limit_excess_rad=float(excess))
  report['final_bottle']=bottle_state(task);report['result']='physics_success' if extra and legal and report['final_bottle']['upright_angle_deg']<15 else 'physics_failed'
 except Exception:
  report['result']='exception';report['traceback']=traceback.format_exc()
 report['stages']=stages;report['canonical_sources_unchanged']=all(sha(p)==report['source_hashes'][str(p)] for p in protected);write()
 print('HORIZONTAL_RESULT',trial_index,grasp_setting,report['result'],flush=True)
 if report['result']=='physics_success':
  accepted_grasp_setting=copy.deepcopy(grasp_setting);accepted_seed=7;task.save_traj_data(0);snapshot(task,'final');break