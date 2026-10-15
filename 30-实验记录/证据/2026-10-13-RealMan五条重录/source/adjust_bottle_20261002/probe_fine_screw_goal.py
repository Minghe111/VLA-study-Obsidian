from preflight_place_goals import ReadOnlyActorProxy
initial=np.asarray(next(x for x in json.loads((OUT/'physics_trace.json').read_text()) if x['phase']['name']=='close')['bottle']['pose'])
proxy=ReadOnlyActorProxy(task.bottle,initial);pre,g=side_entry_grasp(task,proxy,'left')
solver=task.robot.left_mplib_planner.planner;pgoal=task.robot._trans_from_gripper_to_endlink(g,arm_tag='left');startq=task.robot.left_entity.get_qpos().copy();saved_model=solver.robot.get_qpos().copy()
try:
 res=solver.plan_screw(goal_pose=pgoal,current_qpos=startq.copy(),time_step=.004,qpos_step=.005)
 a={'status':res['status'],'qpos_step':.005}
 if res['status']=='Success':
  last=startq.copy();armids=[task.robot._joint_index[n] for n in task.robot.left_arm_joints_name];last[armids]=res['position'][-1]
  solver.pinocchio_model.compute_forward_kinematics(last);fk=solver.pinocchio_model.get_link_pose(solver.link_name_2_idx[solver.move_group]);world=solver.robot.get_base_pose()*fk
  A=world.to_transformation_matrix();B=pgoal.to_transformation_matrix();a.update(position_error_m=float(np.linalg.norm(A[:3,3]-B[:3,3])),rotation_error_deg=float(np.degrees(np.arccos(np.clip((np.trace(A[:3,:3].T@B[:3,:3])-1)/2,-1,1)))),shape=list(res['position'].shape),goal_world=B.tolist(),endpoint_world=A.tolist())
finally:solver.robot.set_qpos(saved_model,True)
(BASE/'fine_screw_goal_preflight_20261003.json').write_text(json.dumps(a,default=str,indent=2));print('FINE_SCREW_GOAL_FK',a,flush=True)
