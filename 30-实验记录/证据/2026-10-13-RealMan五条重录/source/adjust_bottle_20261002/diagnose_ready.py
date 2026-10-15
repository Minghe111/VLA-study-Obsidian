import torch
from envs.robot.robot import Robot
from curobo.types.state import JointState
from transforms3d.quaternions import quat2mat
print('CURRENT',bottle_state(task),flush=True)
for side in ['left','right']:
 mg=getattr(task.robot,side+'_planner').motion_gen
 q=np.array([task.robot._measured_positions()[j.name] for j in getattr(task.robot,side+'_arm_joints')])
 c=mg.ik_solver.check_constraints(JointState.from_position(mg.tensor_args.to_device(q).view(1,6)))
 print('START_CONSTRAINT',side,c.feasible.tolist(),c.constraint.tolist(),flush=True)
new_ready={}
for side,dx in [('left',-.035),('right',.035)]:
 goal=np.asarray(getattr(task.robot,'get_'+side+'_ee_pose')());goal[:3]+=[dx,0,.09]
 r=getattr(Robot,side+'_plan_path')(task.robot,goal.tolist())
 print('READY_PLAN',side,r['status'],flush=True)
 if r['status']!='Success':raise RuntimeError('ready IK/path failed')
 new_ready[side+'_joint_rad']=np.asarray(r['position'][-1]).tolist()
ready.update(new_ready);ready['validation']='native planned elevated ready pose; reset contact check follows'
(BASE/'ready_home_elevated.json').write_text(json.dumps(ready,indent=2))
for side in ['left','right']:args[side+'_embodiment_config']['homestate']=[ready['left_joint_rad'],ready['right_joint_rad']]
args.update(save_data=False,need_plan=True,seed=1)
task.close_env();task.setup_demo(**args)
print('RESET_DONE',bottle_state(task),flush=True)
snapshot(task,'elevated_ready_seed1');audit_ready_home(task)
print('NATIVE_SNAPSHOT',getattr(task.bottle,'config',{}).get('extents'),flush=True)