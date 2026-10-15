"""v21: Cartesian first WORLD lift; exact returned TOPP command-polyline audit.
Actual PhysX meshes, native self ACM and exact real grasp touch links are retained.
No SAPIEN pose/qpos/drive writes, scene steps, hardware IO or GPU initialization.
"""
from __future__ import annotations
import math
import time
import numpy as np
import carry_mesh_path_audit as cma


def _pose_error(goal,current):
    goal=cma._matrix(goal);current=cma._matrix(current)
    r=goal[:3,:3]@current[:3,:3].T;u,_,vt=np.linalg.svd(r)
    return float(np.linalg.norm(goal[:3,3]-current[:3,3])),float(math.acos(float(np.clip((np.trace(u@vt)-1)/2,-1,1))))


def retime_polyline(path,*,time_step=.004,max_joint_velocity=.15,max_joint_acceleration=.3,min_duration_s=0.):
    """Piecewise quintic rest-to-rest; stays on EVERY original joint line segment.
    Unlike cubic joint splines, this cannot cut RRT corners into an obstacle.
    All internal waypoints have zero velocity AND acceleration; C2 continuity.
    """
    p=np.asarray(path,dtype=float)
    if p.ndim!=2 or len(p)<2 or not np.isfinite(p).all():raise ValueError('Expected finite >=2 row joint polyline')
    for x in (time_step,max_joint_velocity,max_joint_acceleration):
        if not math.isfinite(x) or x<=0:raise ValueError('Invalid time/velocity/acceleration bound')
    if not math.isfinite(min_duration_s) or min_duration_s<0:raise ValueError('Invalid minimum duration')
    # Remove only exact duplicates; no smoothing/angle wrapping/position clipping.
    p=p[np.r_[True,np.any(p[1:]!=p[:-1],axis=1)]]
    if len(p)==1:
        n=max(2,int(math.ceil(min_duration_s/time_step))+1)
        return np.tile(p,(n,1)),np.zeros((n,p.shape[1])),np.arange(n)*time_step,{'duration_s':(n-1)*time_step,'waypoints':p.tolist()}
    span=np.max(np.abs(np.diff(p,axis=0)),axis=1)
    durations=np.maximum(np.maximum(1.875*span/max_joint_velocity,np.sqrt((10/math.sqrt(3))*span/max_joint_acceleration)),time_step)
    durations*=max(1.,min_duration_s/float(durations.sum()))
    positions=[];velocities=[];times=[];offset=0.;end_indices=[]
    for index,(q0,q1,duration) in enumerate(zip(p[:-1],p[1:],durations)):
        steps=max(1,int(math.ceil(float(duration)/time_step)));duration=steps*time_step
        u=np.arange(steps+1,dtype=float)/steps;v=np.minimum(u,1-u)
        small=10*v**3-15*v**4+6*v**5;ramp=np.where(u<=.5,small,1-small)
        derivative=30*u**2*(1-u)**2/duration
        pos=q0+ramp[:,None]*(q1-q0);vel=derivative[:,None]*(q1-q0)
        pos[0]=q0;pos[-1]=q1;vel[0]=0.;vel[-1]=0.
        keep=slice(0,None) if index==0 else slice(1,None)
        positions.append(pos[keep]);velocities.append(vel[keep]);times.append(offset+np.arange(steps+1)[keep]*time_step)
        offset+=duration;end_indices.append(sum(len(a) for a in positions)-1)
    pos=np.vstack(positions);vel=np.vstack(velocities);t=np.concatenate(times)
    return pos,vel,t,{'duration_s':float(offset),'waypoints':p.tolist(),'quintic_segment_end_indices':end_indices,
        'peak_command_velocity_rad_s':float(np.abs(vel).max()),'curve':'Piecewise exact joint lines, monotonic quintic timing, all waypoint velocity/acceleration zero.'}



def official_topp_polyline(task,path,side='left',*,time_step=.004):
    """Actual installed official TOPP on each exact two-endpoint RRT segment.

    Two-waypoint TOPPRA SplineInterpolator is certified affine by its actual
    scipy polynomial coefficients. This avoids a multi-waypoint cubic shortcut.
    Audit vertices are formed from the ACTUAL returned TOPP endpoints; ALL
    returned samples are verified on those lines, monotonic, finite and bounded.
    Native linspace timestamps and physical 250Hz row clock are both retained.
    No canonical planner limits, source, physical scene or robot state is changed.
    """
    import inspect,hashlib
    p=np.asarray(path,dtype=float)
    if p.ndim!=2 or len(p)<2 or not np.isfinite(p).all():raise ValueError('Invalid official TOPP polyline')
    if side not in ('left','right') or not math.isfinite(time_step) or time_step<=0:raise ValueError('Invalid side/time_step')
    dt=float(task.scene.get_timestep())
    if not math.isclose(dt,time_step,abs_tol=1e-8,rel_tol=0):raise ValueError('TOPP step differs from actual physical timestep')
    wrapper=getattr(task.robot,side+'_mplib_planner');planner=wrapper.planner;native=wrapper.TOPP
    code=getattr(native,'__func__',None)
    if code is None or code.__module__!='mplib.planner':raise ValueError('Expected actual installed official MPlib TOPP binding')
    if getattr(planner.TOPP,'__func__',None) is not code:raise ValueError('TOPP alias differs from actual official planner method')
    velocities=np.asarray(planner.joint_vel_limits,dtype=float);accelerations=np.asarray(planner.joint_acc_limits,dtype=float)
    if velocities.shape!=(p.shape[1],) or accelerations.shape!=velocities.shape or not np.isfinite(velocities).all() or not np.isfinite(accelerations).all() or np.any(velocities<=0) or np.any(accelerations<=0):raise ValueError('Invalid actual official TOPP limits')
    source=inspect.getsource(code)
    if 'ta.SplineInterpolator(ss, path)' not in source or 'np.linspace(0, 1, N_samples)' not in source:raise ValueError('Installed TOPP geometry implementation changed; require review')
    ta=code.__globals__['ta'];p=p[np.r_[True,np.any(p[1:]!=p[:-1],axis=1)]]
    positions=[];speeds=[];accels=[];times=[];segments=[];vertices=[];offset=0.;physical_rows=0
    if len(p)==1:
        return p.copy(),np.zeros_like(p),np.array([0.]),{'method':'No-motion duplicate endpoint; no timing inflation',
            'duration_s':0.,'dense_duration_s':dt,'dense_step_s':dt,'audit_vertices':p.tolist(),'segments':[],
            'official_joint_velocity_limits_rad_s':velocities.tolist(),'official_joint_acceleration_limits_rad_s2':accelerations.tolist()}
    for number,(q0,q1) in enumerate(zip(p[:-1],p[1:])):
        pair=np.vstack([q0,q1]);spline=ta.SplineInterpolator(np.linspace(0,1,2),pair)
        coefficient=np.asarray(spline.cspl.c,dtype=float)
        if coefficient.shape!=(4,1,p.shape[1]) or not np.isfinite(coefficient).all() or np.max(np.abs(coefficient[:2]))>1e-12 or not np.allclose(coefficient[2,0],q1-q0,atol=1e-12,rtol=0) or not np.allclose(coefficient[3,0],q0,atol=1e-12,rtol=0):raise ValueError('Two-point actual native spline is not the claimed exact joint line')
        native_time,pos,vel,acc,duration=native(pair,step=time_step)
        native_time=np.asarray(native_time,dtype=float);pos=np.asarray(pos,dtype=float);vel=np.asarray(vel,dtype=float);acc=np.asarray(acc,dtype=float);duration=float(duration)
        count=len(native_time)
        if count<2 or pos.shape!=(count,p.shape[1]) or vel.shape!=pos.shape or acc.shape!=pos.shape or not all(np.isfinite(a).all() for a in (native_time,pos,vel,acc)) or not math.isfinite(duration) or duration<=0:raise ValueError('Official TOPP returned invalid/too-short trajectory')
        if abs(native_time[0])>1e-12 or abs(native_time[-1]-duration)>1e-10 or np.any(np.diff(native_time)<=0):raise ValueError('Official TOPP time grid invalid')
        if np.max(abs(pos[0]-q0))>1e-8 or np.max(abs(pos[-1]-q1))>1e-8:raise ValueError('Official TOPP endpoint mismatch')
        if np.max(abs(vel[[0,-1]]))>1e-6:raise ValueError('Official segmented TOPP does not end at rest')
        if np.any(np.max(np.abs(vel),axis=0)>velocities+1e-5) or np.any(np.max(np.abs(acc),axis=0)>accelerations+1e-5):raise ValueError('Official TOPP returned velocity/acceleration exceeds its actual planner limits')
        delta=q1-q0;largest=int(np.argmax(abs(delta)));u=(pos[:,largest]-q0[largest])/delta[largest]
        error=float(np.max(abs(pos-(q0+u[:,None]*delta))))
        if error>1e-9 or np.min(u)<-1e-9 or np.max(u)>1+1e-9 or np.min(np.diff(u))<-1e-10:raise ValueError('Native TOPP samples shortcut/reverse the certified two-point spline')
        keep=slice(None) if number==0 else slice(1,None)
        actual_ratio=float(np.max(np.diff(native_time))/dt)
        fd=np.diff(pos,axis=0)/dt;velocity_target_acc=np.diff(vel,axis=0)/dt
        if np.any(np.max(abs(fd),axis=0)>velocities*actual_ratio+1e-5) or np.any(np.max(abs(velocity_target_acc),axis=0)>accelerations*actual_ratio+1e-5):raise ValueError('Native target samples inconsistent with recorded official linspace time grid')
        before=physical_rows;physical_rows+=len(pos[keep]);end_index=physical_rows-1
        segments.append({'index':number,'official_native_duration_s':duration,'official_native_times_s':native_time.tolist(),
            'native_sample_count':count,'kept_dense_sample_count':len(pos[keep]),'dense_start_index':before,'dense_end_index':end_index,
            'official_native_grid_min_s':float(np.min(np.diff(native_time))),'official_native_grid_max_s':float(np.max(np.diff(native_time))),
            'native_grid_over_physical_dt_ratio':actual_ratio,'sample_line_error_rad':error,'two_point_spline_high_order_coeff_max':float(np.max(abs(coefficient[:2]))),
            'native_peak_velocity_rad_s':np.max(abs(vel),axis=0).tolist(),'native_peak_acceleration_rad_s2':np.max(abs(acc),axis=0).tolist(),
            'physical_clock_command_fd_peak_rad_s':np.max(abs(fd),axis=0).tolist(),
            'physical_clock_velocity_target_acc_peak_rad_s2':np.max(abs(velocity_target_acc),axis=0).tolist()})
        positions.append(pos[keep]);speeds.append(vel[keep]);accels.append(acc[keep]);times.append(offset+native_time[keep]);offset+=duration
        if not vertices:vertices.append(pos[0].copy())
        vertices.append(pos[-1].copy())
    position=np.vstack(positions);velocity=np.vstack(speeds);timing=np.concatenate(times)
    metadata={'method':'Actual official MPlib TOPP, separately on each two-endpoint native RRT segment; no minimum duration or global speed override',
        'official_TOPP_source_sha256':hashlib.sha256(source.encode()).hexdigest(),'duration_s':offset,
        'dense_step_s':dt,'dense_rows':len(position),'dense_duration_s':len(position)*dt,'dense_last_row_time_s':(len(position)-1)*dt,
        'native_vs_dense_timing_note':'Installed TOPP uses linspace(T,int(T/step)); returned native times are preserved. Official take_dense_action executes one physical dt per row. Wrench must use physical dt, not native time.',
        'official_joint_velocity_limits_rad_s':velocities.tolist(),'official_joint_acceleration_limits_rad_s2':accelerations.tolist(),
        'segments':segments,'audit_vertices':np.asarray(vertices).tolist(),'waypoints':np.asarray(vertices).tolist(),
        'peak_command_velocity_rad_s':float(abs(velocity).max()),
        'curve':'Each actual native two-point spline is certified affine. Returned samples are collinear, monotonic and match endpoints. Fresh continuous mesh audit covers precisely these actual native curves; no old certificate reused.'}
    return position,velocity,timing,metadata

def resolve_bottle_table_bound(task,path,side,palm_to_bottle,audit,audit_options=None,*,certificate_function=None):
    """Resolve ONLY conservative table ambiguity; every actual collision rejects.
    Preserve original negative bound and attach an independently derived strict
    certificate. Other full-mesh classes and joint limits cannot be overridden.
    """
    if audit['passed']:return audit
    thresholds=audit['thresholds_m'];bounds=audit['continuous_gap_lower_bound']
    eligible=(audit['joint_limit_audit']['passed'] and not audit['unique_collision_pairs'] and
        not audit['collisions_first_64'] and 'bottle_table' in bounds and
        all(value['lower_bound_m']>=thresholds[name] for name,value in bounds.items() if name!='bottle_table'))
    if not eligible:return audit
    if certificate_function is None:
        from bottle_table_clearance_certificate import certify_bottle_table_path
        certificate_function=certify_bottle_table_path
    options=dict(audit_options or {})
    certificate=certificate_function(task,path,side,palm_to_bottle=palm_to_bottle,
        clearance_m=thresholds['bottle_table'],vertex_step_m=options.get('vertex_step_m',.001))
    combined=dict(audit);combined['basic_lipschitz_audit_passed']=audit['passed']
    combined['basic_lipschitz_continuous_gap_lower_bound']=dict(bounds)
    combined['bottle_table_refinement_certificate']=certificate
    if certificate['passed']:
        expected_ref=np.asarray(audit['metadata']['rigid_carry_reference'])
        actual_ref=np.asarray(certificate['metadata']['rigid_carry_reference'])
        if not np.allclose(expected_ref,actual_ref,atol=1e-8,rtol=0):raise RuntimeError('Table certificate reference differs from full audit')
        import hashlib
        if certificate['command_path_sha256']!=hashlib.sha256(np.asarray(path,dtype=np.float64).tobytes()).hexdigest():
            raise RuntimeError('Table certificate command path differs from full audit')
        combined['passed']=True;combined['continuous_gap_lower_bound']=dict(bounds)
        combined['continuous_gap_lower_bound']['bottle_table']={**certificate['continuous_lower_bound'],
            'method':'Strict recursive real-FCL nearest-endpoint Lipschitz bound; original basic negative bound retained separately.'}
        combined['scope']+=' Table bound resolved only by attached strict recursive continuous certificate for identical path/reference.'
    return combined

def _collision_summary(world):
    result=[]
    for collision in world.check_collision():
        result.append({k:str(getattr(collision,k)) for k in ('link_name1','link_name2','object_name1','object_name2') if hasattr(collision,k)})
    return result


def _support_start_pair_only(collisions):
    return all(set((c.get('link_name1'),c.get('link_name2')))=={'held_bottle','realman_table'} for c in collisions)


def cartesian_world_lift_knots(start,goal,*,position_tolerance_m=5e-5,rotation_tolerance_rad=2e-4):
    start=cma._matrix(start);goal=cma._matrix(goal);expected=start.copy();expected[:3,3]+=[0,0,.1]
    pe,re=_pose_error(expected,goal)
    if pe>position_tolerance_m or re>rotation_tolerance_rad:raise ValueError('Cartesian lift requires official WORLD z+.1m fixed orientation')
    knots=np.tile(start,(21,1,1));knots[:,2,3]+=.1*np.linspace(0,1,21)
    return knots


def official_topp_command_polyline(task,path,side='left',*,time_step=.004):
    """One actual installed official TOPP call; certify RETURNED command polyline.

    No claim covers TOPP's internal continuous multi-knot cubic spline. Official
    dense execution sends these returned targets once per physical dt, and fresh
    mesh/support audits cover linear connections of exactly these command rows.
    The full executed physical trace must independently pass actual-state gates.
    """
    import inspect,hashlib
    p=np.asarray(path,dtype=float)
    if p.ndim!=2 or len(p)<2 or not np.isfinite(p).all() or np.max(np.abs(np.diff(p,axis=0)))<1e-10:raise ValueError('Invalid moving TOPP knot path')
    if side not in ('left','right') or not math.isfinite(time_step) or time_step<=0:raise ValueError('Invalid side/time_step')
    dt=float(task.scene.get_timestep())
    if not math.isclose(dt,time_step,abs_tol=1e-8,rel_tol=0):raise ValueError('TOPP step differs from actual physical timestep')
    wrapper=getattr(task.robot,side+'_mplib_planner');planner=wrapper.planner;native=wrapper.TOPP;code=getattr(native,'__func__',None)
    if code is None or code.__module__!='mplib.planner' or getattr(planner.TOPP,'__func__',None) is not code:raise ValueError('Expected actual installed official TOPP binding')
    vl=np.asarray(planner.joint_vel_limits,dtype=float);al=np.asarray(planner.joint_acc_limits,dtype=float)
    if vl.shape!=(p.shape[1],) or al.shape!=vl.shape or not np.isfinite(vl).all() or not np.isfinite(al).all() or np.any(vl<=0) or np.any(al<=0):raise ValueError('Invalid official planner limits')
    source=inspect.getsource(code)
    if 'ta.SplineInterpolator(ss, path)' not in source or 'np.linspace(0, 1, N_samples)' not in source:raise ValueError('Installed TOPP implementation changed')
    ts,pos,vel,acc,T=native(p,step=time_step)
    ts=np.asarray(ts,dtype=float);pos=np.asarray(pos,dtype=float);vel=np.asarray(vel,dtype=float);acc=np.asarray(acc,dtype=float);T=float(T)
    if len(ts)<2 or pos.shape!=(len(ts),p.shape[1]) or vel.shape!=pos.shape or acc.shape!=pos.shape or not all(np.isfinite(x).all() for x in (ts,pos,vel,acc)) or not math.isfinite(T) or T<=0:raise ValueError('Invalid actual TOPP trajectory')
    if abs(ts[0])>1e-12 or abs(ts[-1]-T)>1e-10 or np.any(np.diff(ts)<=0):raise ValueError('Invalid native TOPP time grid')
    if np.max(np.abs(pos[0]-p[0]))>1e-8 or np.max(np.abs(pos[-1]-p[-1]))>1e-8:raise ValueError('TOPP endpoints differ from frozen native IK knots')
    if np.max(np.abs(vel[[0,-1]]))>1e-6:raise ValueError('TOPP whole lift endpoints are not stationary')
    if np.any(np.max(np.abs(vel),axis=0)>vl+1e-5) or np.any(np.max(np.abs(acc),axis=0)>al+1e-5):raise ValueError('Native velocity/acceleration limits exceeded')
    ratio=float(np.max(np.diff(ts))/dt);fd=np.diff(pos,axis=0)/dt;va=np.diff(vel,axis=0)/dt
    if np.any(np.max(np.abs(fd),axis=0)>vl*ratio+1e-5) or np.any(np.max(np.abs(va),axis=0)>al*ratio+1e-5):raise ValueError('Actual target FD exceeds official native grid-scaled limits')
    meta={'method':'ONE actual installed official MPlib TOPP on all frozen full-native Cartesian IK knots',
        'official_TOPP_source_sha256':hashlib.sha256(source.encode()).hexdigest(),
        'input_knot_count':len(p),'input_arm_knots_rad':p.tolist(),'input_knots_sha256':hashlib.sha256(p.tobytes()).hexdigest(),
        'duration_s':T,'dense_step_s':dt,'dense_rows':len(pos),'dense_duration_s':len(pos)*dt,
        'official_native_times_s':ts.tolist(),'official_native_grid_min_s':float(np.diff(ts).min()),
        'official_native_grid_max_s':float(np.diff(ts).max()),'native_grid_over_physical_dt_ratio':ratio,
        'official_joint_velocity_limits_rad_s':vl.tolist(),'official_joint_acceleration_limits_rad_s2':al.tolist(),
        'native_peak_velocity_rad_s':np.max(np.abs(vel),axis=0).tolist(),'native_peak_acceleration_rad_s2':np.max(np.abs(acc),axis=0).tolist(),
        'physical_clock_command_fd_peak_rad_s':np.max(np.abs(fd),axis=0).tolist(),
        'physical_clock_velocity_target_acc_peak_rad_s2':np.max(np.abs(va),axis=0).tolist(),
        'audit_vertices':pos.tolist(),'returned_commands_sha256':hashlib.sha256(pos.tobytes()).hexdigest(),
        'command_curve_scope':'Fresh audit of exact returned command-target joint polyline; does NOT certify internal TOPP continuous cubic or physical tracking',
        'curve':'Linear joint connections of the EXACT returned 250Hz dense command targets; full physical actual-state audit is separately required',
        'native_vs_dense_timing_note':'Native TOPP linspace grid is preserved; actual dense executes one scene dt per row. Wrench uses the actual dt; no artificial pauses or duration floor.',
        'internal_knot_rest_stops_forced':False,'additional_hold_seconds':0.}
    return pos,vel,ts,meta


def plan_carry_to_task_pose(task,task_pose,side='left',*,palm_to_bottle=None,time_step=.004,
                            max_joint_velocity=1.,max_joint_acceleration=1.,min_duration_s=0.,
                            planning_time_s=10.,ik_attempts=24,rrt_range_rad=.05,max_goal_branches=6,
                            candidate_arm_q=None,position_tolerance_m=5e-5,rotation_tolerance_rad=2e-4,
                            random_seed=7,audit_options=None,endpoint_only=False,official_TOPP=True,support_departure=False):
    """Same task pose, full-native masked IK endpoints then RRT and complete audit.
    Goal IK is never clipped or wrapped. Noncommanded actual joints stay frozen.
    Supplied candidate_arm_q entries are checked like every IK branch, not trusted.
    Return Success only after full carry continuous mesh audit passes.
    """
    if side not in ('left','right'):raise ValueError('side must be left/right')
    if not isinstance(support_departure,bool):raise ValueError('support_departure must be explicit bool')
    if support_departure and (endpoint_only or not official_TOPP):raise ValueError('Initial support departure requires full trajectory and actual official TOPP')
    if not isinstance(ik_attempts,int) or ik_attempts<1 or not isinstance(max_goal_branches,int) or max_goal_branches<1:raise ValueError('Invalid branch budget')
    if not math.isfinite(planning_time_s) or planning_time_s<=0 or not math.isfinite(rrt_range_rad) or rrt_range_rad<=0:raise ValueError('Invalid RRT budget')
    if position_tolerance_m<=0 or rotation_tolerance_rad<=0:raise ValueError('Invalid endpoint tolerances')
    for value in (time_step,max_joint_velocity,max_joint_acceleration):
        if not math.isfinite(value) or value<=0:raise ValueError('Invalid retiming options')
    if not math.isfinite(min_duration_s) or min_duration_s<0:raise ValueError('Invalid minimum duration')
    from mplib import PlanningWorld
    from mplib.pymp.planning.ompl import OMPLPlanner
    import preflight_place_goals as ppg
    before=ppg._snapshot(task);started=time.monotonic()
    auditor=cma.CarryMeshPathAuditor(task,side,palm_to_bottle=palm_to_bottle)
    model=auditor.model;oldq=cma.fma._array(model.get_qpos()).copy();old_base_pose=auditor.Pose(model.get_base_pose());world=None
    meta={'scope':'Independent full-native CPU IK/RRT, planning-only attached collision object, physical scene unchanged.',
          'task_goal_pose_world':ppg._vector(task_pose),'ik_attempts_requested':ik_attempts,
          'endpoint_position_tolerance_m':position_tolerance_m,'endpoint_rotation_tolerance_rad':rotation_tolerance_rad}
    try:
        palm_goal=cma._matrix(task.robot._trans_from_gripper_to_endlink(list(map(float,ppg._vector(task_pose))),arm_tag=side))
        root=cma._matrix(auditor.art.root_pose);goal_base=np.linalg.inv(root)@palm_goal
        base=auditor.initial_sim_q.copy();names=auditor.sim_joints
        arm_names=list(getattr(task.robot,side+'_arm_joints_name'));indices=[names.index(n) for n in arm_names]
        frozen=[i for i in range(len(base)) if i not in indices]
        model.set_move_group(auditor.palm_name)
        if list(model.get_move_group_joint_indices())!=indices or list(model.get_move_group_joint_names())!=arm_names:
            raise ValueError('Full-native move-group ordering does not match commanded arm joints')
        # Installed AttachedBody incorrectly ignores nonidentity model.base_pose.
        # Put ALL auxiliary planning geometry in robot BASE coordinates. The
        # full mesh auditor remains a separate WORLD-frame validation phase.
        inverse_root=np.linalg.inv(root);model.set_base_pose(auditor.Pose());model.set_qpos(base,full=True)
        source_table=auditor.static['table'];source_bottle=auditor.static['bottle']
        def clone_obstacle(source,name,pose):
            # Geometry is immutable/shared, but CollisionObject transforms are
            # independent: base-frame planning MUST NOT alter FMA world shapes.
            shapes=[auditor.fcl.CollisionObject(shape.get_collision_geometry()) for shape in source.shapes]
            locals=[auditor.Pose(local) for local in source.shape_poses]
            return auditor.fcl.FCLObject(name,auditor.Pose(pose),shapes,locals)
        table_object=clone_obstacle(source_table,'realman_table',inverse_root@cma._matrix(source_table.pose))
        bottle_object=clone_obstacle(source_bottle,'held_bottle',inverse_root@cma._matrix(source_bottle.pose))
        if table_object.name in auditor.objects or bottle_object.name in auditor.objects:raise ValueError('Obstacle name conflicts with robot link')
        table_before=cma._matrix(table_object.pose)
        table_shape_before=[cma._matrix(shape.get_pose()) for shape in table_object.shapes]
        source_table_shape_before=[cma._matrix(shape.get_pose()) for shape in source_table.shapes]
        def assert_table_unchanged():
            if not np.allclose(cma._matrix(table_object.pose),table_before,atol=1e-10,rtol=0):raise RuntimeError('Planning table global pose changed')
            for shape,old in zip(table_object.shapes,table_shape_before):
                if not np.allclose(cma._matrix(shape.get_pose()),old,atol=1e-10,rtol=0):raise RuntimeError('Planning table shape moved')
            for shape,old in zip(source_table.shapes,source_table_shape_before):
                if not np.allclose(cma._matrix(shape.get_pose()),old,atol=1e-10,rtol=0):raise RuntimeError('FMA world table changed through shared CollisionObject')
        world=PlanningWorld([model],[table_object,bottle_object]);acm=world.get_allowed_collision_matrix()
        registry=list(world.get_object_names())
        if set(registry)!={'realman_table','held_bottle'}:raise RuntimeError('Invalid distinct world registry names')
        for pair in auditor.allowed:
            a,b=sorted(pair);acm.set_entry(a,b,True)
        palm_index=list(auditor.pk.get_link_names()).index(auditor.palm_name)
        user_palm_index=list(model.get_user_link_names()).index(auditor.palm_name)
        if user_palm_index!=palm_index:raise RuntimeError('PK/user attachment indices differ; explicit diagnostic needed')
        world.attach_object('held_bottle',model.get_name(),user_palm_index,auditor.Pose(auditor.reference),sorted(auditor.allowed_bottle_hand))
        world.set_qpos_all(base[indices]);attached=world.get_attached_object('held_bottle');attached.update_pose()
        anchor_base=cma._matrix(attached.get_attached_link_global_pose())
        bottle_base=cma._matrix(attached.get_global_pose())
        actual_palm=auditor.initial_scene_poses[auditor.palm_name];actual_bottle=cma._matrix(task.bottle.get_pose())
        anchor_error=_pose_error(actual_palm,root@anchor_base)
        bottle_error=_pose_error(actual_bottle,root@bottle_base)
        predicted_error=_pose_error(actual_palm@auditor.reference,root@bottle_base)
        if anchor_error[0]>2e-5 or anchor_error[1]>2e-4:raise RuntimeError('Attachment anchor does not match actual palm after BASE->WORLD transform: '+str(anchor_error))
        if bottle_error[0]>2e-5 or bottle_error[1]>2e-4:raise RuntimeError('Attachment bottle does not match actual actor after BASE->WORLD transform: '+str(bottle_error))
        shape_errors=[]
        attached_object=attached.get_object()
        for shape,local in zip(attached_object.shapes,attached_object.shape_poses):
            shape_errors.append(_pose_error(bottle_base@cma._matrix(local),cma._matrix(shape.get_pose())))
        if any(pe>2e-5 or re>2e-4 for pe,re in shape_errors):raise RuntimeError('Attached collision shape frame mismatch')
        assert_table_unchanged()
        if not world.is_object_attached('held_bottle') or world.is_object_attached('realman_table'):raise RuntimeError('Wrong auxiliary attachment')
        meta.update(planning_coordinate_frame='Robot BASE, ALL geometry consistently transformed from unchanged scene WORLD',
            planning_model_base_pose_identity=True,world_from_planning_base=root.tolist(),planning_object_names=registry,
            only_planning_attached_object='held_bottle',independent_obstacle_collision_objects=True,table_world_pose_unchanged=True,
            attachment_frame_validation={'anchor_world_error_m_rad':list(anchor_error),'bottle_actor_world_error_m_rad':list(bottle_error),
                'bottle_predicted_world_error_m_rad':list(predicted_error),'collision_shape_pose_errors_m_rad':[list(x) for x in shape_errors],
                'anchor_matrix_base':anchor_base.tolist(),'bottle_matrix_base':bottle_base.tolist()},
            actual_start_arm_joint_rad=base[indices].tolist(),goal_palm_matrix_world=palm_goal.tolist(),
            goal_palm_matrix_base=goal_base.tolist(),move_group_joint_names=arm_names,
            frozen_joint_names=[names[i] for i in frozen],native_self_exception_count=len(auditor.allowed),
            actual_bottle_touch_links=sorted(auditor.allowed_bottle_hand),start_collisions=_collision_summary(world),full_mesh_metadata=auditor.metadata)
        if support_departure:
            if not _support_start_pair_only(meta['start_collisions']):
                return {'status':'Fail','reason':'Support departure permits ONLY the initial held_bottle/realman_table pair','planning_metadata':meta}
            expected=actual_palm.copy();expected[:3,3]+=[0,0,.1]
            pe,re=_pose_error(expected,palm_goal)
            if pe>position_tolerance_m or re>rotation_tolerance_rad:
                return {'status':'Fail','reason':'Support departure target must be the first official WORLD z+.1m fixed-orientation lift','planning_metadata':meta}
            vertices=auditor.bottle_vertices@actual_bottle[:3,:3].T+actual_bottle[:3,3]
            signed_baseline=float(vertices[:,2].min()-auditor.static_boxes['table'][1,2])
            if signed_baseline < -.0005 or signed_baseline>=.001:
                return {'status':'Fail','reason':'Initial actual signed support depth outside [-.5mm,1mm)','planning_metadata':meta}
            meta['support_departure']={'enabled':True,'planning_only_allowed_pair':['held_bottle','realman_table'],
                'actual_start_signed_gap_m':signed_baseline,'max_initial_penetration_m':.0005,'max_extra_depth_m':.000020,
                'release_gap_m':.001,'physical_table_collisions_unchanged':True,
                'qualification':'Every candidate requires unchanged full CMA raw evidence plus independent continuous signed support-exit certificate'}
            acm.set_entry('held_bottle','realman_table',True)
            if _collision_summary(world):raise RuntimeError('Other start collisions remained after the sole planning support pair')
        elif meta['start_collisions']:return {'status':'Fail','reason':'Full-native start state collides','planning_metadata':meta}
        limits=np.concatenate([cma.fma._array(j.limits).reshape(-1,2) for j in auditor.art.active_joints])
        mask=np.ones(len(base),dtype=bool);mask[indices]=False;rng=np.random.default_rng(random_seed)
        candidates=[];endpoint_reports=[]
        def inspect_candidate(q,label,ik_success=True,ik_error=None):
            q=np.asarray(q,dtype=float).copy();entry={'label':label,'ik_success':bool(ik_success)}
            if ik_error is not None:entry['ik_error_vector']=np.asarray(ik_error).tolist()
            if q.shape!=base.shape or not np.isfinite(q).all():entry['reason']='Invalid joint vector';endpoint_reports.append(entry);return
            entry['arm_joint_rad']=q[indices].tolist()
            if not ik_success:entry['reason']='CLIK did not converge';endpoint_reports.append(entry);return
            if frozen and np.max(np.abs(q[frozen]-base[frozen]))>1e-8:entry['reason']='IK changed frozen joints';endpoint_reports.append(entry);return
            check=cma.fma.joint_limit_audit(np.vstack([base,q]),limits,names,indices)
            entry['joint_limit_audit']=check
            if not check['passed']:entry['reason']='Strict joint limit rejection';endpoint_reports.append(entry);return
            world.set_qpos_all(q[indices]);assert_table_unchanged();auditor.pk.compute_forward_kinematics(q)
            endpoint=root@cma._matrix(auditor.pk.get_link_pose(palm_index));pe,re=_pose_error(palm_goal,endpoint)
            entry.update(position_error_m=pe,rotation_error_rad=re,collisions=_collision_summary(world))
            if pe>position_tolerance_m or re>rotation_tolerance_rad:entry['reason']='Endpoint FK mismatch'
            elif entry['collisions']:entry['reason']='Full-native endpoint collision'
            elif any(np.linalg.norm(q[indices]-old[indices])<1e-3 for old in candidates):entry['reason']='Duplicate endpoint branch'
            else:
                candidates.append(q);entry['reason']='Accepted free endpoint'
                entry['joint_distance_from_start_rad']=float(np.linalg.norm(q[indices]-base[indices]))
                entry['wrist6_delta_rad']=float(q[indices[-1]]-base[indices[-1]])
            endpoint_reports.append(entry)
        support_path=None
        if support_departure:
            try:
                cartesian_knots=cartesian_world_lift_knots(actual_palm,palm_goal,
                    position_tolerance_m=position_tolerance_m,rotation_tolerance_rad=rotation_tolerance_rad)
                meta['cartesian_world_knots']=cartesian_knots.tolist()
                meta['cartesian_ik_scope']='21 WORLD vertical poses, fixed orientation/XY; previous full native IK seed; all 30 other actual DOFs frozen'
                meta['frozen_actual_qpos_rad']=base.tolist()
                meta['prior_actual_palm_world']=actual_palm.tolist()
                meta['prior_native_FK_palm_world']=(root@anchor_base).tolist()
                solved=[base.copy()];seed=base.copy();knot_reports=[]
                for knot_index,knot_world in enumerate(cartesian_knots[1:],1):
                    knot_base=inverse_root@knot_world
                    q,ok,error=auditor.pk.compute_IK_CLIK(palm_index,auditor.Pose(knot_base),seed,mask.tolist(),eps=1e-7,max_iter=2000,dt=.1,damp=1e-8)
                    q=np.asarray(q,dtype=float);entry={'knot_index':knot_index,'IK_success':bool(ok),'IK_error':np.asarray(error).tolist()}
                    knot_reports.append(entry);meta['cartesian_ik_knots']=knot_reports
                    if q.shape!=base.shape or not np.isfinite(q).all() or not ok:raise ValueError('Full native Cartesian CLIK failed at knot '+str(knot_index))
                    if frozen and np.max(np.abs(q[frozen]-base[frozen]))>1e-8:raise ValueError('Cartesian CLIK changed frozen actual joints')
                    check=cma.fma.joint_limit_audit(np.vstack([base,q]),limits,names,indices);entry['joint_limit_audit']=check
                    if not check['passed']:raise ValueError('Cartesian knot joint limits rejected')
                    world.set_qpos_all(q[indices]);assert_table_unchanged();auditor.pk.compute_forward_kinematics(q)
                    actual_knot=root@cma._matrix(auditor.pk.get_link_pose(palm_index));pe,re=_pose_error(knot_world,actual_knot)
                    entry.update(position_error_m=pe,rotation_error_rad=re,arm_joint_rad=q[indices].tolist(),collisions=_collision_summary(world))
                    if pe>position_tolerance_m or re>rotation_tolerance_rad:raise ValueError('Cartesian knot actual native FK rejected')
                    if entry['collisions']:raise ValueError('Cartesian knot non-support full-native collision')
                    solved.append(q.copy());seed=q.copy()
                support_path=np.asarray(solved)[:,indices];candidates.append(solved[-1]);endpoint_reports=knot_reports
                meta['cartesian_full_native_knots_rad']=np.asarray(solved).tolist()
            except (ValueError,RuntimeError,AssertionError) as error:
                return {'status':'Fail','reason':'Cartesian support lift construction rejected: '+str(error),'planning_metadata':meta}
        else:
            if candidate_arm_q is not None:
                supplied=np.asarray(candidate_arm_q,dtype=float)
                if supplied.ndim==1:supplied=supplied[None,:]
                if supplied.ndim!=2 or supplied.shape[1]!=len(indices) or not np.isfinite(supplied).all():raise ValueError('Invalid supplied candidate arm joints')
                for i,qarm in enumerate(supplied):q=base.copy();q[indices]=qarm;inspect_candidate(q,'supplied_'+str(i))
            for attempt in range(ik_attempts):
                seed=base.copy()
                if attempt:
                    if attempt<ik_attempts//2:
                        seed[indices]+=rng.normal(0,.15+.07*attempt,len(indices))
                        # Rejection-resample an out-of-bounds SEED; command goals are never clipped.
                        for i in indices:
                            if seed[i]<limits[i,0] or seed[i]>limits[i,1]:seed[i]=rng.uniform(limits[i,0],limits[i,1])
                    else:seed[indices]=rng.uniform(limits[indices,0],limits[indices,1])
                q,ok,error=auditor.pk.compute_IK_CLIK(palm_index,auditor.Pose(goal_base),seed,mask.tolist(),eps=1e-7,max_iter=2000,dt=.1,damp=1e-8)
                inspect_candidate(q,'CLIK_'+str(attempt),ok,error)
        meta.update(endpoint_candidates=endpoint_reports,accepted_free_endpoints=len(candidates))
        if not candidates:return {'status':'Fail','reason':'No free exact endpoint among finite IK/candidate search; this is not a proof of global unreachability.','planning_metadata':meta}
        if endpoint_only:
            return {'status':'EndpointSuccess','planning_metadata':meta,
                    'free_goal_arm_joint_rad':[q[indices].tolist() for q in candidates]}
        candidates.sort(key=lambda q:float(np.linalg.norm(q[indices]-base[indices]))+.15*abs(float(q[indices[-1]]-base[indices[-1]])))
        planner=OMPLPlanner(world);attempts=[]
        for branch,goal in enumerate(candidates[:max_goal_branches]):
            model.set_base_pose(auditor.Pose());model.set_qpos(base,full=True);world.set_qpos_all(base[indices])
            stamp=time.monotonic()
            if support_departure:
                status='Exact solution';path=support_path.copy()
            else:
                status,path=planner.plan(base[indices],[goal[indices]],time=float(planning_time_s),range=float(rrt_range_rad),simplify=True,verbose=False)
            assert_table_unchanged()
            detail={'branch':branch,'goal_arm_joint_rad':goal[indices].tolist(),
                    'RRT_status':None if support_departure else str(status),
                    'RRT_elapsed_s':None if support_departure else time.monotonic()-stamp,
                    'planning_algorithm':'Full native 21-knot Cartesian CLIK' if support_departure else 'Original strict full native OMPL RRT'}
            attempts.append(detail)
            if str(status).rstrip('.')!='Exact solution':continue
            path=np.asarray(path,dtype=float)
            if path.ndim!=2 or path.shape[1]!=len(indices) or len(path)<2 or not np.isfinite(path).all():
                detail['reason']='Invalid OMPL path';continue
            if np.max(np.abs(path[0]-base[indices]))>1e-7 or np.max(np.abs(path[-1]-goal[indices]))>1e-6:
                detail['reason']='OMPL endpoint differs from exact checked q';continue
            command_full=np.tile(base,(len(path),1));command_full[:,indices]=path
            check=cma.fma.joint_limit_audit(np.vstack([base,command_full]),limits,names,indices)
            if not check['passed']:detail['reason']='RRT strict commanded joint limits rejected';detail['joint_limit_audit']=check;continue
            try:
                if official_TOPP:
                    if min_duration_s!=0:raise ValueError('Official TOPP mode forbids artificial minimum duration')
                    position,velocity,timing,retime=(official_topp_command_polyline if support_departure else official_topp_polyline)(task,path,side,time_step=time_step)
                    geometry=np.asarray(retime['audit_vertices'],dtype=float)
                    retime['unused_external_velocity_acceleration_arguments']=[max_joint_velocity,max_joint_acceleration]
                else:
                    position,velocity,timing,retime=retime_polyline(path,time_step=time_step,max_joint_velocity=max_joint_velocity,
                        max_joint_acceleration=max_joint_acceleration,min_duration_s=min_duration_s)
                    geometry=path
            except (ValueError,RuntimeError,AssertionError) as error:
                detail['reason']='Retiming rejected: '+str(error);continue
            # Audit all actual returned joint limits before any geometry proof.
            retimed_full=np.tile(base,(len(position),1));retimed_full[:,indices]=position
            returned_limits=cma.fma.joint_limit_audit(np.vstack([base,retimed_full]),limits,names,indices)
            if not returned_limits['passed']:
                detail.update(reason='Actual TOPP command joint limits rejected',joint_limit_audit=returned_limits);continue
            # First support lift audits EXACT returned command-target rows as a
            # joint polyline; it makes no internal global cubic-curve claim. Later
            # strict carry retains the original certified two-point native curves.
            opts=dict(audit_options or {});model.set_base_pose(old_base_pose);model.set_qpos(oldq,full=True)
            audit=auditor.audit(geometry,side,**opts)
            if support_departure:
                from support_departure_certificate_v20 import certify_support_departure
                support=certify_support_departure(task,geometry,side,palm_to_bottle=auditor.reference,
                    audit_options=opts,
                    endpoint_translation_tolerance_m=position_tolerance_m,
                    endpoint_rotation_tolerance_rad=rotation_tolerance_rad)
                detail['support_departure_combined_audit']=support
                # Preserve raw ordinary audit. Its bottle_table failure is not
                # re-labelled as strict clearance; an explicit support proof replaces it.
                audit={**audit,'passed':bool(support['passed']),
                    'raw_ordinary_audit_passed':audit['passed'],
                    'support_departure_combined_audit':support,
                    'support_departure_certificate':support['signed_support_departure_certificate'],
                    'physical_signed_height_guard':support['physical_signed_height_guard'],
                    'scope':'Initial first WORLD upward support phase ONLY. Ordinary bottle_table evidence retained; support certificate replaces only that category. All other full meshes unchanged.'}
            else:
                audit=resolve_bottle_table_bound(task,geometry,side,auditor.reference,audit,opts)
            detail.update(polyline_waypoints=len(path),audit_passed=audit['passed'],retiming=retime,
                audit_collision_pairs=audit['collisions_first_64'],audit_continuous_lower_bound=audit['continuous_gap_lower_bound'])
            if not audit['passed']:continue
            meta.update(RRT_attempts=attempts,selected_branch=branch,elapsed_s=time.monotonic()-started,
                audit_curve_relation=retime.get('curve','Actual native TOPP curve proof'),
                geometric_polyline=path.tolist(),retiming=retime,physical_state_unchanged=True)
            return {'status':'Success','position':position,'velocity':velocity,'time':timing,
                    'duration':float(timing[-1]),'audit':audit,'planning_metadata':meta}
        meta.update(RRT_attempts=attempts,elapsed_s=time.monotonic()-started)
        return {'status':'Fail','reason':('Cartesian first-lift returned command polyline failed complete audits' if support_departure else 'No RRT branch passed complete continuous carry audit.'),'planning_metadata':meta}
    finally:
        model.set_base_pose(old_base_pose);model.set_qpos(oldq,full=True)
        source=auditor.static['bottle']
        auditor.static['bottle']=auditor.fcl.FCLObject(source.name,auditor.original_bottle_pose,source.shapes,source.shape_poses)
        ppg._assert_snapshot(task,before)
        meta['physical_state_unchanged']=True


def run_cpu_tests():
    p=np.array([[0.,0.],[.12,-.08],[-.04,.16],[.03,.05]])
    q,v,t,meta=retime_polyline(p,time_step=.004,max_joint_velocity=.15,max_joint_acceleration=.3,min_duration_s=8.)
    assert t[-1]>=8. and np.max(abs(v))<=.15+1e-12
    assert np.allclose(np.diff(t),.004,atol=1e-12)
    assert np.array_equal(q[0],p[0]) and np.array_equal(q[-1],p[-1])
    ends=[0,*meta['quintic_segment_end_indices']]
    for segment,(a,b) in enumerate(zip(ends[:-1],ends[1:])):
        qs=q[a:b+1];delta=p[segment+1]-p[segment]
        u=(qs[:,0]-p[segment,0])/delta[0]
        assert np.all(u>=-1e-14) and np.all(u<=1+1e-14) and np.all(np.diff(u)>=-1e-14)
        assert np.max(abs(qs-(p[segment]+u[:,None]*delta)))<1e-14
        assert np.array_equal(v[a],np.zeros(2)) and np.array_equal(v[b],np.zeros(2))
    assert np.max(abs(np.diff(v,axis=0)/.004))<=.3+1e-3
    q,v,t,_=retime_polyline([[.1,.2],[.1,.2]],min_duration_s=.1)
    assert np.all(q==[.1,.2]) and np.all(v==0) and t[-1]>=.1
    for fn in (lambda:retime_polyline([[0],[float('nan')]]),lambda:retime_polyline([[0],[1]],time_step=0),lambda:retime_polyline([[0],[1]],max_joint_velocity=-1)):
        try:fn()
        except ValueError:pass
        else:raise AssertionError('Invalid profile accepted')
    carry=cma.run_cpu_tests()
    return {'passed':True,'checks':['polyline_geometry_exactly_preserved','monotonic_quintic_progress',
        'waypoint_zero_velocity_C2','velocity_acceleration_limits','minimum_duration_uniform_004step',
        'stationary_duplicate_path','3_invalid_profile_inputs_rejected'],
        'carry_formula_tests_passed':carry['passed'],'GPU':False,'SAPIEN_scene_created':False}


if __name__=='__main__':
    import json
    print(json.dumps(run_cpu_tests(),indent=2))

