import full_native_carry_planner_v20 as ncp19
import preflight_place_goals as ppg19
snapshot19=ppg19._snapshot(task)
goal19=list(task.robot.get_left_ee_pose());goal19[2]+=.1
reference19=np.asarray(grip_measure(task)['relative_bottle_palm'])
result19=ncp19.plan_carry_to_task_pose(task,goal19,'left',palm_to_bottle=reference19,time_step=.004,planning_time_s=10,ik_attempts=24,support_departure=True)
ppg19._assert_snapshot(task,snapshot19)
(BASE/'closed_lift_v20_readonly.json').write_text(json.dumps(result19,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x),indent=2))
print('CLOSED_LIFT_V20_READONLY',result19['status'],result19.get('reason'),flush=True)
