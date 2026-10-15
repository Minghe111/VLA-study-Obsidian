"""Full native free-hand approach: official CPU IK/RRT/TOPP and every physical mesh.
No grasp attachment, simulation step, physical pose/qpos/drive writes.
"""
import numpy as np,time

def plan_free_approach(task,target_pose,side,*,time_step=.004,random_seed=0,ik_attempts=24):
    import full_mesh_path_audit as fma
    from full_native_carry_planner_v31 import official_topp_polyline
    from mplib import PlanningWorld
    from mplib.pymp.planning.ompl import OMPLPlanner
    before_q=np.asarray(task.robot.left_entity.get_qpos()).copy()
    before_v=np.asarray(task.robot.left_entity.get_qvel()).copy()
    before_b=task.bottle.get_pose().to_transformation_matrix().copy()
    audit=fma.FullMeshPathAuditor(task);model=audit.model;old_pose=model.get_base_pose();old_q=model.get_qpos().copy()
    base=audit.initial_sim_q.copy();names=audit.sim_joints
    indices=[names.index(n) for n in getattr(task.robot,side+'_arm_joints_name')]
    palm_name=('l_' if side=='left' else 'r_')+'inspire_base_link'
    model.set_move_group(palm_name);pk=audit.pk;palm_index=list(pk.get_link_names()).index(palm_name)
    root=task.robot.left_entity.root_pose.to_transformation_matrix();inverse=np.linalg.inv(root)
    goal=inverse@task.robot._trans_from_gripper_to_endlink(target_pose,arm_tag=side).to_transformation_matrix()
    model.set_base_pose(audit.Pose());model.set_qpos(base,full=True)
    obstacles=[]
    for name,source in audit.static.items():
        shapes=[audit.fcl.CollisionObject(s.get_collision_geometry()) for s in source.shapes]
        obstacles.append(audit.fcl.FCLObject('realman_'+name,audit.Pose(inverse@source.pose.to_transformation_matrix()),shapes,[audit.Pose(p) for p in source.shape_poses]))
    world=PlanningWorld([model],obstacles);acm=world.get_allowed_collision_matrix()
    for pair in audit.allowed:
        a,b=sorted(pair);acm.set_entry(a,b,True)
    limits=np.concatenate([np.asarray(j.limits).reshape(-1,2) for j in audit.art.active_joints])
    mask=np.ones(len(base),dtype=bool);mask[indices]=False;rng=np.random.default_rng(random_seed)
    metadata={'backend':'full_native_free_hand_MPLib','physical_state_unchanged':False,'endpoint_attempts':[],'RRT_attempts':[],'no_attached_objects':True};candidates=[]
    try:
        world.set_qpos_all(base[indices])
        start_collisions=world.check_collision()
        if start_collisions:return {'status':'Fail','reason':'Full native approach start collides','planning_metadata':metadata}
        for attempt in range(ik_attempts):
            seed=base.copy()
            if attempt:seed[indices]=rng.uniform(limits[indices,0],limits[indices,1])
            q,ok,error=pk.compute_IK_CLIK(palm_index,audit.Pose(goal),seed,mask.tolist(),eps=1e-7,max_iter=2000,dt=.1,damp=1e-8)
            q=np.asarray(q);row={'attempt':attempt,'IK_converged':bool(ok)}
            if ok and np.all(q[indices]>=limits[indices,0]-1e-5) and np.all(q[indices]<=limits[indices,1]+1e-5) and np.max(abs(q[mask]-base[mask]))<1e-8:
                model.set_qpos(q,full=True);world.set_qpos_all(q[indices]);collisions=world.check_collision()
                row['collision_count']=len(collisions)
                if not collisions and not any(np.linalg.norm(q[indices]-c[indices])<1e-3 for c in candidates):candidates.append(q.copy())
            metadata['endpoint_attempts'].append(row)
        candidates.sort(key=lambda q:np.linalg.norm(q[indices]-base[indices]))
        planner=OMPLPlanner(world)
        for branch,q in enumerate(candidates[:6]):
            model.set_qpos(base,full=True);world.set_qpos_all(base[indices]);stamp=time.time()
            status,path=planner.plan(base[indices],[q[indices]],time=10.,range=.05,simplify=True,verbose=False)
            row={'branch':branch,'RRT_status':str(status),'elapsed_s':time.time()-stamp};metadata['RRT_attempts'].append(row)
            if str(status).rstrip('.')!='Exact solution':continue
            path=np.asarray(path)
            if len(path)<2 or np.max(abs(path[0]-base[indices]))>1e-7 or np.max(abs(path[-1]-q[indices]))>1e-6:continue
            position,velocity,times,retime=official_topp_polyline(task,path,side,time_step=time_step)
            full=np.tile(base,(len(position),1));full[:,indices]=position
            limit_check=fma.joint_limit_audit(np.vstack([base,full]),limits,names,indices)
            if not limit_check['passed']:continue
            model.set_base_pose(old_pose);model.set_qpos(old_q,full=True)
            geometry=audit.audit(retime['audit_vertices'],side,min_bottle_clearance_m=.0005,min_self_clearance_m=0.,min_table_clearance_m=0.)
            row.update(audit_passed=geometry['passed'],audit=geometry)
            if geometry['passed']:
                metadata.update(selected_branch=branch,retiming=retime)
                return {'status':'Success','position':position,'velocity':velocity,'time':times,'duration':float(times[-1]),'audit':geometry,'planning_metadata':metadata}
            model.set_base_pose(audit.Pose());model.set_qpos(base,full=True)
        return {'status':'Fail','reason':'No exact free-hand path passed IK/RRT/TOPP/full mesh audit','planning_metadata':metadata}
    finally:
        model.set_base_pose(old_pose);model.set_qpos(old_q,full=True)
        if not np.array_equal(before_q,task.robot.left_entity.get_qpos()) or not np.array_equal(before_v,task.robot.left_entity.get_qvel()) or not np.array_equal(before_b,task.bottle.get_pose().to_transformation_matrix()):raise RuntimeError('CPU approach planning changed physical scene')
        metadata['physical_state_unchanged']=True