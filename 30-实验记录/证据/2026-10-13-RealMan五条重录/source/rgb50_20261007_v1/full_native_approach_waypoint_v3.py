"""Contact-free CPU path candidate via goal palm +world-Z 80 mm.

The original direct planner is tried without changing its predicates. On failure
only, two native RRT legs are concatenated and passed to the same installed
official TOPP and complete native-mesh gate. No physical-state setters or steps.
This is a candidate, not a grasp/physics success certificate.
"""
import time
import numpy as np
import full_native_approach_diagnostics_v2 as original

WAYPOINT_WORLD_Z_M = .08


def _physical_snapshot(task):
    robot=task.robot
    return {'q':np.asarray(robot.left_entity.get_qpos()).copy(),
            'v':np.asarray(robot.left_entity.get_qvel()).copy(),
            'bottle':task.bottle.get_pose().to_transformation_matrix().copy(),
            'hand_targets':np.asarray([robot._hand_joints[n].get_drive_target()[0]
                                      for n in sorted(robot._hand_joints)])}


def _assert_unchanged(task,before):
    after=_physical_snapshot(task)
    if not all(np.array_equal(before[k],after[k]) for k in before):
        raise RuntimeError('CPU waypoint planning changed physical state/hand targets')


def plan_free_approach(task,target_pose,side,*,time_step=.004,random_seed=0,ik_attempts=24):
    before=_physical_snapshot(task)
    direct=original.plan_free_approach(task,target_pose,side,time_step=time_step,
                                     random_seed=random_seed,ik_attempts=ik_attempts)
    _assert_unchanged(task,before)
    if direct.get('status')=='Success':
        return direct
    import full_mesh_path_audit as fma
    from full_native_carry_planner_v31 import official_topp_polyline
    from mplib import PlanningWorld
    from mplib.pymp.planning.ompl import OMPLPlanner

    audit=fma.FullMeshPathAuditor(task);model=audit.model
    old_pose=model.get_base_pose();old_q=model.get_qpos().copy()
    base=audit.initial_sim_q.copy();names=audit.sim_joints
    indices=[names.index(n) for n in getattr(task.robot,side+'_arm_joints_name')]
    palm=('l_' if side=='left' else 'r_')+'inspire_base_link'
    model.set_move_group(palm);pk=audit.pk;palm_index=list(pk.get_link_names()).index(palm)
    root=task.robot.left_entity.root_pose.to_transformation_matrix();inverse=np.linalg.inv(root)
    goal_world=task.robot._trans_from_gripper_to_endlink(target_pose,arm_tag=side).to_transformation_matrix()
    via_world=goal_world.copy();via_world[2,3]+=WAYPOINT_WORLD_Z_M
    goal=inverse@goal_world;via=inverse@via_world
    model.set_base_pose(audit.Pose());model.set_qpos(base,full=True)
    obstacles=[]
    for name,source in audit.static.items():
        shapes=[audit.fcl.CollisionObject(s.get_collision_geometry()) for s in source.shapes]
        obstacles.append(audit.fcl.FCLObject('realman_'+name,
            audit.Pose(inverse@source.pose.to_transformation_matrix()),shapes,
            [audit.Pose(p) for p in source.shape_poses]))
    world=PlanningWorld([model],obstacles);acm=world.get_allowed_collision_matrix()
    for pair in audit.allowed:
        a,b=sorted(pair);acm.set_entry(a,b,True)
    limits=np.concatenate([np.asarray(j.limits).reshape(-1,2) for j in audit.art.active_joints])
    mask=np.ones(len(base),dtype=bool);mask[indices]=False
    metadata={'backend':'full_native_free_hand_MPLib_goal_height_waypoint',
        'physical_state_unchanged':False,'no_attached_objects':True,
        'original_direct_status':direct.get('status'),
        'original_direct_reason':direct.get('reason'),
        'original_direct_planning_metadata':direct.get('planning_metadata',{}),
        'waypoint_semantics':'Original hardware palm target +world Z .08m, identical rotation; original target remains endpoint',
        'waypoint_world_z_m':WAYPOINT_WORLD_Z_M,'goal_palm_world_matrix':goal_world.tolist(),
        'via_palm_world_matrix':via_world.tolist(),
        'initial_actual_full_qpos_rad':before['q'].tolist(),
        'initial_native_bottle_matrix':before['bottle'].tolist(),
        'endpoint_attempts':[],'RRT_attempts':[],
        'physical_success_not_inferred_from_CPU_path':True}

    def reset_model(q):
        model.set_base_pose(audit.Pose());model.set_qpos(q,full=True)
        world.set_qpos_all(q[indices])

    def endpoints(target,label,offset):
        rng=np.random.default_rng(random_seed+offset);candidates=[]
        for attempt in range(ik_attempts):
            seed=base.copy()
            if attempt:seed[indices]=rng.uniform(limits[indices,0],limits[indices,1])
            q,ok,error=pk.compute_IK_CLIK(palm_index,audit.Pose(target),seed,mask.tolist(),
                eps=1e-7,max_iter=2000,dt=.1,damp=1e-8)
            q=np.asarray(q);row={'target':label,'attempt':attempt,'IK_converged':bool(ok)}
            if ok and np.isfinite(q).all() and np.all(q[indices]>=limits[indices,0]-1e-5) and np.all(q[indices]<=limits[indices,1]+1e-5) and np.max(abs(q[mask]-base[mask]))<1e-8:
                reset_model(q);collisions=world.check_collision();row['collision_count']=len(collisions)
                if not collisions and not any(np.linalg.norm(q[indices]-c[indices])<1e-3 for c in candidates):
                    candidates.append(q.copy())
            metadata['endpoint_attempts'].append(row)
        return sorted(candidates,key=lambda q:np.linalg.norm(q[indices]-base[indices]))

    def leg(planner,start,end,label):
        reset_model(start);stamp=time.time()
        status,path=planner.plan(start[indices],[end[indices]],time=10.,range=.05,simplify=True,verbose=False)
        path=np.asarray(path)
        row={'leg':label,'RRT_status':str(status),'elapsed_s':time.time()-stamp,
             'raw_path_rows':len(path)}
        if str(status).rstrip('.')!='Exact solution':
            row['rejection_reason']='Original exact RRT status predicate';return None,row
        if path.ndim!=2 or path.shape[1]!=len(indices) or len(path)<2 or not np.isfinite(path).all():
            row['rejection_reason']='Original finite >=2-row six-arm-joint path requirement';return None,row
        row.update(raw_start_error_rad=float(np.max(abs(path[0]-start[indices]))),
                   raw_end_error_rad=float(np.max(abs(path[-1]-end[indices]))))
        if row['raw_start_error_rad']>1e-7 or row['raw_end_error_rad']>1e-6:
            row['rejection_reason']='Original raw path start/end predicate';return None,row
        return path,row

    try:
        reset_model(base);start_collisions=world.check_collision()
        if start_collisions:
            metadata['start_collision_count']=len(start_collisions)
            return {'status':'Fail','reason':'Full native waypoint approach start collides','planning_metadata':metadata}
        vias=endpoints(via,'via_goal_world_z_plus_80mm',13331)
        goals=endpoints(goal,'original_goal',26663)
        planner=OMPLPlanner(world)
        for via_branch,qvia in enumerate(vias[:2]):
            near_goals=sorted(goals,key=lambda q:np.linalg.norm(q[indices]-qvia[indices]))[:3]
            for goal_branch,qgoal in enumerate(near_goals):
                row={'via_branch':via_branch,'goal_branch':goal_branch}
                metadata['RRT_attempts'].append(row)
                first,first_result=leg(planner,base,qvia,'start_to_raised_goal')
                row['first_leg']=first_result
                if first is None:continue
                join=base.copy();join[indices]=first[-1]
                second,second_result=leg(planner,join,qgoal,'raised_goal_to_original_goal')
                row['second_leg']=second_result
                if second is None:continue
                row['raw_join_error_rad']=float(np.max(abs(first[-1]-second[0])))
                if row['raw_join_error_rad']>1e-7:
                    row['rejection_reason']='Original raw start predicate at concatenation';continue
                path=np.vstack([first,second])
                position,velocity,times,retime=official_topp_polyline(task,path,side,time_step=time_step)
                full=np.tile(base,(len(position),1));full[:,indices]=position
                check=fma.joint_limit_audit(np.vstack([base,full]),limits,names,indices)
                row.update(raw_path_rows=len(path),retimed_rows=len(position),retimed_joint_limit_audit=check)
                if not check['passed']:
                    row['rejection_reason']='Original retimed full native joint-limit predicate';continue
                model.set_base_pose(old_pose);model.set_qpos(old_q,full=True)
                geometry=audit.audit(retime['audit_vertices'],side,min_bottle_clearance_m=.0005,
                                     min_self_clearance_m=0.,min_table_clearance_m=0.)
                row.update(audit_passed=geometry['passed'],audit=geometry)
                if not geometry['passed']:
                    row['rejection_reason']='Original complete native mesh clearance predicate';continue
                metadata.update(selected_via_branch=via_branch,selected_goal_branch=goal_branch,
                    retiming=retime,selected_raw_joint_polyline=path.tolist(),
                    original_goal_endpoint_qpos_rad=qgoal[indices].tolist(),
                    native_no_collision_via_qpos_rad=qvia[indices].tolist())
                return {'status':'Success','position':position,'velocity':velocity,'time':times,
                    'duration':float(times[-1]),'audit':geometry,'planning_metadata':metadata}
        return {'status':'Fail','reason':'No raised-goal native path passed original exact endpoint/TOPP/full mesh predicates',
                'planning_metadata':metadata}
    finally:
        model.set_base_pose(old_pose);model.set_qpos(old_q,full=True)
        _assert_unchanged(task,before);metadata['physical_state_unchanged']=True
