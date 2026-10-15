trace=json.loads((OUT/'physics_trace.json').read_text());first=next(x for x in trace if x['phase']['name']=='close');hist=np.asarray(first['bottle']['pose'])
palm=task.robot.left_entity.find_link_by_name('l_inspire_base_link').entity.pose.to_transformation_matrix();local=np.linalg.inv(palm)@hist
fp_local=np.linalg.inv(task.bottle.get_pose().to_transformation_matrix())@task.bottle.get_contact_point(0,'matrix')
a={'palm_world':palm.tolist(),'historical_bottle_pose':hist.tolist(),'bottle_local_p':local[:3,3].tolist(),'historical_contact0_center_local':(np.linalg.inv(palm)@hist@fp_local)[:3,3].tolist(),'task_ee':task.robot.get_left_ee_pose()}
(BASE/'b085_actual_palm_probe_20261003.json').write_text(json.dumps(a,default=str,indent=2));print('B085_ACTUAL_PALM',a,flush=True)
snapshot(task,'failure_debug')
from preflight_place_goals import preflight_place_goals,ReadOnlyActorProxy
proxy=ReadOnlyActorProxy(task.bottle,hist);pre,g=side_entry_grasp(task,proxy,'left')
r=preflight_place_goals(task,g,bottle_at_grasp_pose=hist,run_planning=True,planner_modes=('robot_path','curobo_ik'))
(BASE/'b085_place_preflight_20261003.json').write_text(json.dumps(r,default=str,indent=2));print('B085_PLACE_PREFLIGHT',r,flush=True)
