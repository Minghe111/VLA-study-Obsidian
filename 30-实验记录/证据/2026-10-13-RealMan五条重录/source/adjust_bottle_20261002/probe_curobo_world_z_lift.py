from preflight_place_goals import _snapshot,_assert_snapshot
import carry_mesh_path_audit as cma
before=_snapshot(task);pose=np.asarray(task.robot.get_left_ee_pose(),float);goal=pose.copy();goal[2]+=.1
res=Robot.left_plan_path(task.robot,goal.tolist())
probe={'planner_status':res.get('status'),'task_goal':goal.tolist(),'physical_state_unchanged':False}
if res['status']=='Success':
 np.savez(BASE/'curobo_world_z_lift_probe_20261003.npz',position=res['position'],velocity=res['velocity'])
 probe['audit']=cma.audit_carry_path(task,res,'left',palm_to_bottle=carry_reference[0],min_table_clearance_m=0,min_self_clearance_m=0)
_assert_snapshot(task,before);probe['physical_state_unchanged']=True
(BASE/'curobo_world_z_lift_probe_20261003.json').write_text(json.dumps(probe,indent=2,default=str))
print('CUROBO_WORLD_LIFT_PROBE',probe.get('planner_status'),probe.get('audit',{}).get('passed'),probe.get('audit',{}).get('collisions_first_64'),flush=True)
