from curobo.wrap.reacher.ik_solver import IKSolver,IKSolverConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose as CuroboPose
from curobo.types.state import JointState
from envs.utils import ArmTag
import transforms3d as t3d
# New diagnostics are separate from the official baseline attempt.
OUT=BASE/'grasp_geometry_roll_seed1';OUT.mkdir(exist_ok=True)
report={'result':'diagnostic','seed':1,'official_task_source_unchanged':True}
side='left' if task.qpose_tag==0 else 'right'
planner=getattr(task.robot,side+'_planner');mg=planner.motion_gen
cfg=yaml.safe_load(Path(planner.yml_path).read_text())['robot_cfg']
ta=TensorDeviceType()
kin_ik=IKSolver(IKSolverConfig.load_from_robot_config(cfg,None,tensor_args=ta,num_seeds=32,
 position_threshold=.003,rotation_threshold=.03,self_collision_check=False,self_collision_opt=False,
 use_cuda_graph=True,seed=20261002))
origin=getattr(task.robot,side+'_entity_origion_pose');rootR=t3d.quaternions.quat2mat(origin.q)
A=np.array([[0,0,1,0],[-1,0,0,0],[0,-1,0,0],[0,0,0,1]],dtype=float)
metadata=[];positions=[];quaternions=[]
for contact_id,_ in task.bottle.iter_contact_points():
 T=task.bottle.get_contact_point(contact_id,'matrix')@A
 center=T[:3,3];quat=t3d.quaternions.mat2quat(T[:3,:3]);native=[*(center-.12*T[:3,0]),*quat]
 variants=task.robot.create_target_pose_list(native,task.bottle.get_contact_point(contact_id,'list'),ArmTag(side))
 for variant,p in enumerate(variants):
  R=t3d.quaternions.quat2mat(p[3:])@np.diag([1.,-1.,-1.]);p=list(p[:3])+t3d.quaternions.mat2quat(R).tolist()
  for hand_y in [.045,.055,.065,.075]:
   for hand_z in [.15,.165,.18,.195]:
    ee=np.array(p[:3])+R@np.array([.18-hand_z,-hand_y,0])
    palm=ee-.06*R[:,0];Rpalm=R@np.array([[0,0,1],[0,1,0],[-1,0,0]])
    positions.append(rootR.T@(palm-origin.p));quaternions.append(t3d.quaternions.mat2quat(rootR.T@Rpalm))
    metadata.append({'contact_id':contact_id,'variant':variant,'hand_y':hand_y,'hand_z':hand_z,'task_pose':[*ee.tolist(),*p[3:]],'center':center.tolist()})
positions=np.asarray(positions);quaternions=np.asarray(quaternions);n=len(metadata);B=96
allq=[];ok=[];err=[]
for start in range(0,n,B):
 ix=np.minimum(np.arange(start,start+B),n-1);count=min(B,n-start)
 res=kin_ik.solve_batch(CuroboPose(position=ta.to_device(positions[ix]),quaternion=ta.to_device(quaternions[ix])),return_seeds=4)
 allq.append(res.solution.detach().cpu().numpy().reshape(B,4,6)[:count]);ok.append(res.success.detach().cpu().numpy().reshape(B,4)[:count]);err.append(res.position_error.detach().cpu().numpy().reshape(B,4)[:count])
allq=np.concatenate(allq);ok=np.concatenate(ok);err=np.concatenate(err)
valid=np.zeros(ok.size,dtype=bool);flatq=allq.reshape(-1,6);ix=np.flatnonzero(ok.ravel())
for start in range(0,len(ix),128):
 j=ix[start:start+128];con=mg.ik_solver.check_constraints(JointState.from_position(ta.to_device(flatq[j])))
 valid[j]=con.feasible.detach().cpu().numpy().ravel()
valid=valid.reshape(ok.shape)
qnow=np.array([task.robot._measured_positions()[j.name] for j in getattr(task.robot,side+'_arm_joints')])
choices=[]
for i,j in np.argwhere(valid):
 item=dict(metadata[i]);item.update(pose_index=int(i),solution_index=int(j),q=allq[i,j].tolist(),q_distance=float(np.linalg.norm(allq[i,j]-qnow)))
 choices.append(item)
choices.sort(key=lambda x:(x['variant']>2,x['q_distance']))
np.savez_compressed(OUT/'grasp_candidates.npz',q=allq,kinematic_ok=ok,endpoint_ok=valid,position_error=err)
(OUT/'grasp_candidates.json').write_text(json.dumps({'side':side,'bottle':bottle_state(task),'poses':n,'kinematic':int(ok.sum()),'collision_free':int(valid.sum()),'choices':choices},indent=2))
print('SCAN_RESULT',side,n,'kinematic',int(ok.sum()),'collision_free',int(valid.sum()),flush=True)
for item in choices[:12]:print('CANDIDATE',item,flush=True)