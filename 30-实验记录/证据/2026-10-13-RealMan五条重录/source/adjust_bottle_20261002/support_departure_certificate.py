"""Independent CPU certificate for ONLY the first WORLD vertical support exit.

No physics steps, SAPIEN writes, collision masks, attachments or hardware IO.
The ordinary full carry mesh audit remains unmodified and is retained as raw
failed evidence when bottle/table starts in support. Only that category is
replaced by a signed-height proof; every other category must pass unchanged.
The proof concerns the same linear joint polyline, not arbitrary spline timing.
"""
from __future__ import annotations
import hashlib
import math
import numpy as np


def _matrix(m):
    m=np.asarray(m,dtype=float)
    if m.shape!=(4,4) or not np.isfinite(m).all() or not np.allclose(m[3],[0,0,0,1],atol=1e-9):
        raise ValueError('Invalid finite homogeneous pose')
    if not np.allclose(m[:3,:3].T@m[:3,:3],np.eye(3),atol=1e-7) or np.linalg.det(m[:3,:3])<.999999:
        raise ValueError('Invalid rotation')
    return m


def _angle(a,b):
    return math.acos(float(np.clip((np.trace(a@b.T)-1)/2,-1,1)))


def certify_signed_support_polyline(full_path,value_function,upward_derivative_function,
        bottle_coefficients,palm_coefficients,ancestor_mask,*,is_first_world_lift,
        lift_height_m=.1,max_initial_penetration_m=.0005,max_extra_depth_m=.000020,
        release_gap_m=.001,path_xy_tolerance_m=.002,path_rotation_tolerance_rad=math.radians(.5),
        endpoint_translation_tolerance_m=.000020,endpoint_rotation_tolerance_rad=.0002,
        vertex_step_m=.001,max_evaluations=50000,max_depth=26,measured_baseline_signed_gap_m=None,require_monotonic_world_z=False):
    """Certify a signed floor and upward motion on the complete joint polyline.

    value_function(q) returns signed_gap_m, palm_pose_world and
    bottle_table_collision. The derivative callback returns the EXACT derivative
    of world palm z along q(u)=q0+u*dq, at the supplied q and dq.
    All active ancestors must be revolute: the global second derivative bound
    is 2*(palm_coefficients @ abs(dq))*sum(abs(dq[ancestors])).
    This bounds the sum of individual and mixed rotation derivatives; refining
    each interval records an upward derivative bound; only the explicit optional
    monotonic flag uses this as an additional acceptance condition.
    """
    p=np.asarray(full_path,dtype=float);bc=np.asarray(bottle_coefficients,dtype=float)
    pc=np.asarray(palm_coefficients,dtype=float);mask=np.asarray(ancestor_mask,dtype=bool)
    if p.ndim!=2 or len(p)<2 or not np.isfinite(p).all() or bc.shape!=(p.shape[1],) or pc.shape!=bc.shape or mask.shape!=bc.shape:
        raise ValueError('Invalid path or full DOF coefficient shapes')
    if not np.isfinite(bc).all() or not np.isfinite(pc).all() or np.any(bc<0) or np.any(pc<0):raise ValueError('Invalid motion coefficients')
    if is_first_world_lift is not True:raise ValueError('Restricted to explicitly identified first WORLD lift')
    if not isinstance(require_monotonic_world_z,bool):raise ValueError('Monotonic audit flag must be explicit bool')
    vals=[lift_height_m,max_initial_penetration_m,max_extra_depth_m,release_gap_m,path_xy_tolerance_m,
          path_rotation_tolerance_rad,endpoint_translation_tolerance_m,endpoint_rotation_tolerance_rad,vertex_step_m]
    if any(not math.isfinite(float(x)) or x<=0 for x in vals):raise ValueError('Invalid positive finite support budgets')
    if not math.isclose(lift_height_m,.1,abs_tol=1e-12,rel_tol=0) or max_initial_penetration_m>.0005 or max_extra_depth_m>.000020 or release_gap_m<.001 or vertex_step_m>.001 or path_xy_tolerance_m>.002 or path_rotation_tolerance_rad>math.radians(.5) or endpoint_translation_tolerance_m>5e-5 or endpoint_rotation_tolerance_rad>.0002:
        raise ValueError('Cannot widen official first-lift, penetration, release or mesh budgets')
    if isinstance(max_evaluations,bool) or not isinstance(max_evaluations,int) or max_evaluations<3 or isinstance(max_depth,bool) or not isinstance(max_depth,int) or max_depth<0:
        raise ValueError('Invalid refinement budget')
    cache={};accepted=[];release=None;worst_lower=math.inf;worst_actual=math.inf;unresolved=[]
    def value(q):
        key=q.tobytes()
        if key not in cache:
            if len(cache)>=max_evaluations:raise RuntimeError('Support certificate evaluation budget exhausted')
            v=value_function(q);g=float(v['signed_gap_m']);pose=_matrix(v['palm_pose_world'])
            if not math.isfinite(g):raise ValueError('Nonfinite actual signed gap')
            cache[key]={'signed_gap_m':g,'palm_pose_world':pose,'collision':bool(v['bottle_table_collision'])}
        return cache[key]
    start=value(p[0]);end=value(p[-1]);baseline=start['signed_gap_m'] if measured_baseline_signed_gap_m is None else float(measured_baseline_signed_gap_m);P0=start['palm_pose_world'];P1=end['palm_pose_world']
    if not math.isfinite(baseline):raise ValueError('Nonfinite measured native baseline')
    base={'passed':False,'baseline_signed_gap_m':baseline,'minimum_permitted_signed_gap_m':baseline-max_extra_depth_m,
          'max_initial_penetration_m':max_initial_penetration_m,'max_extra_depth_m':max_extra_depth_m,'release_gap_m':release_gap_m,
          'predicted_start_signed_gap_m':start['signed_gap_m'],'predicted_start_vs_measured_signed_gap_error_m':start['signed_gap_m']-baseline,
          'path_xy_tolerance_m':path_xy_tolerance_m,'path_rotation_tolerance_rad':path_rotation_tolerance_rad,
          'monotonic_world_z_required':require_monotonic_world_z,
          'path_rows_with_start':len(p),'full_path_sha256':hashlib.sha256(p.tobytes()).hexdigest()}
    if abs(start['signed_gap_m']-baseline)>2e-5:return {**base,'reason':'Predicted start floor disagrees with actual native baseline'}
    if baseline < -max_initial_penetration_m or baseline>=release_gap_m:return {**base,'reason':'Initial state outside bounded support-contact phase'}
    expected=P0[:3,3]+[0,0,lift_height_m]
    ep_error=float(np.linalg.norm(P1[:3,3]-expected));er_error=_angle(P1[:3,:3],P0[:3,:3])
    base.update(endpoint_translation_error_m=ep_error,endpoint_rotation_error_rad=er_error)
    if ep_error>endpoint_translation_tolerance_m or er_error>endpoint_rotation_tolerance_rad:return {**base,'reason':'Endpoint is not the official WORLD z+.1m fixed-orientation lift'}
    floor=baseline-max_extra_depth_m
    try:
        for row,(a,b) in enumerate(zip(p[:-1],p[1:])):
            stack=[(a.copy(),b.copy(),0.,1.,0)]
            while stack:
                q0,q1,s0,s1,depth=stack.pop();v0=value(q0);v1=value(q1)
                g0,g1=v0['signed_gap_m'],v1['signed_gap_m'];worst_actual=min(worst_actual,g0,g1)
                released=release is not None;threshold=1e-12 if released else floor
                if min(g0,g1)<threshold:return {**base,'reason':'Signed table depth/released clearance violated','at_interval':row,'fraction':[s0,s1],'signed_gaps_m':[g0,g1],'after_release':released}
                if released and (v0['collision'] or v1['collision']):return {**base,'reason':'True bottle/table collision after release','at_interval':row}
                dq=q1-q0;B=float(bc@np.abs(dq));PB=float(pc@np.abs(dq));AB=float(np.abs(dq[mask]).sum())
                lower=min(g0,g1)-B/2
                xy=max(float(np.linalg.norm(v['palm_pose_world'][:2,3]-P0[:2,3])) for v in (v0,v1))+PB/2
                rot=max(_angle(v['palm_pose_world'][:3,:3],P0[:3,:3]) for v in (v0,v1))+AB/2
                mid=(q0+q1)/2;d=float(upward_derivative_function(mid,dq))
                if not math.isfinite(d):raise ValueError('Nonfinite exact upward derivative')
                derivative_lower=d-PB*AB
                if B<=vertex_step_m and lower>=threshold and xy<=path_xy_tolerance_m and rot<=path_rotation_tolerance_rad and (not require_monotonic_world_z or derivative_lower>=-1e-12):
                    worst_lower=min(worst_lower,lower)
                    entry={'path_interval':row,'fraction_start':s0,'fraction_end':s1,'depth':depth,'signed_gap_lower_bound_m':lower,
                           'bottle_vertex_displacement_bound_m':B,'upward_derivative_lower_bound_m':derivative_lower,
                           'palm_xy_upper_bound_m':xy,'palm_rotation_upper_bound_rad':rot,'after_release':released}
                    accepted.append(entry)
                    if release is None and g1>=release_gap_m:
                        release={'path_interval':row,'fraction':s1,'signed_gap_m':g1,'qpos_rad':q1.tolist(),'accepted_leaf_index':len(accepted)-1}
                    continue
                if depth>=max_depth or np.array_equal(mid,q0) or np.array_equal(mid,q1):
                    unresolved.append({'path_interval':row,'fraction':[s0,s1],'depth':depth,'gap_lower_m':lower,
                                       'upward_derivative_lower_m':derivative_lower,'xy_upper_m':xy,'rotation_upper_rad':rot})
                    raise RuntimeError('Unresolved support exit; refusing incomplete proof')
                sm=(s0+s1)/2
                stack.append((mid,q1,sm,s1,depth+1));stack.append((q0,mid,s0,sm,depth+1))
    except RuntimeError as e:return {**base,'reason':str(e),'evaluations':len(cache),'unresolved_intervals':unresolved,'release':release}
    return {**base,'passed':release is not None and end['signed_gap_m']>=release_gap_m and not end['collision'],
            'reason':'Certified first WORLD upward support exit' if release is not None else 'No certified support release',
            'evaluations':len(cache),'accepted_intervals':len(accepted),'release':release,'sampled_minimum_signed_gap_m':worst_actual,
            'continuous_minimum_signed_gap_lower_bound_m':worst_lower,'certified_intervals':accepted,
            'proof':'Actual native vertex floor is 1-Lipschitz under attached displacement. Each linear joint interval uses nearest-endpoint B/2 floor; exact midpoint world-z derivative minus PB*sum(abs(dq)) records the upward derivative bound (optional monotonic gate). After the first >=1mm endpoint every remaining interval requires positive height and no true table collision.',
            'physical_scope':'Geometry only. Real table collisions, force support, command FD and actual slip guards must remain enabled.'}


def _other_categories_pass(raw):
    required=('bottle_robot','self','robot_table')
    events=raw['collisions_first_64']
    table_events=[e for e in events if e['category']=='bottle_table']
    return (bool(raw['joint_limit_audit']['passed']) and
            raw['unique_collision_pairs']==len(table_events) and
            all(raw['continuous_gap_lower_bound'][n]['lower_bound_m']>=raw['thresholds_m'][n] for n in required))


def certify_first_world_support_departure(task,path,side='left',*,palm_to_bottle=None,is_first_world_lift,
        audit_options=None,**certificate_options):
    """Read-only existing-task adapter. Never deploy or execute this path.

    raw FullMesh audit is preserved; its bottle_table failure is only replaced
    by this independent floor proof. All untouched categories must pass.
    The caller must prove the retimed trajectory is this same joint polyline.
    """
    import carry_mesh_path_audit as cma
    import preflight_place_goals as ppg
    from sapien.physx import PhysxRigidBaseComponent
    before=ppg._snapshot(task)
    a=cma.CarryMeshPathAuditor(task,side,palm_to_bottle=palm_to_bottle,
        max_reference_position_error_m=2e-5,max_reference_rotation_error_rad=2e-4)
    raw=cma.fma._array(path['position'] if isinstance(path,dict) else path)
    names=list(getattr(task.robot,side+'_arm_joints_name'));indices=[a.sim_joints.index(n) for n in names]
    if raw.ndim!=2 or not len(raw) or not np.isfinite(raw).all():raise ValueError('Invalid command path')
    if raw.shape[1]==len(indices):full=np.tile(a.initial_sim_q,(len(raw),1));full[:,indices]=raw
    elif raw.shape[1]==len(a.initial_sim_q):
        full=raw.copy();frozen=[i for i in range(full.shape[1]) if i not in indices]
        if np.max(np.abs(full[:,frozen]-a.initial_sim_q[frozen]))>1e-12:raise ValueError('First support lift must freeze hands and other arm in geometry proof')
    else:raise ValueError('Invalid arm/full path width')
    full=np.vstack([a.initial_sim_q,full])
    actors=[x for x in task.scene.get_all_actors() if x.name=='table']
    comp=actors[0].find_component_by_type(PhysxRigidBaseComponent)
    if len(comp.collision_shapes)!=1 or not hasattr(comp.collision_shapes[0],'half_size'):raise ValueError('Certificate requires the actual single official tabletop box')
    table_rotation=cma._matrix(comp.entity.pose)[:3,:3]@cma._matrix(comp.collision_shapes[0].local_pose)[:3,:3]
    if not np.allclose(np.abs(table_rotation),np.eye(3),atol=1e-7):raise ValueError('Table is not WORLD-axis-aligned')
    table_top=float(a.static_boxes['table'][1,2])
    if abs(table_top-.74)>1e-6:raise ValueError('Requires actual official horizontal table top z=.74m')
    actual_bottle=cma._matrix(task.bottle.get_pose())
    actual_vertices=a.bottle_vertices@actual_bottle[:3,:3].T+actual_bottle[:3,3]
    measured_baseline=float(actual_vertices[:,2].min()-table_top)
    table_bounds=a.static_boxes['table']
    actual_footprint_margin=float(np.minimum(actual_vertices[:,:2]-table_bounds[0,:2],table_bounds[1,:2]-actual_vertices[:,:2]).min())
    in_palm=a.bottle_vertices@a.reference[:3,:3].T+a.reference[:3,3]
    required_footprint_margin=float(certificate_options.get('path_xy_tolerance_m',.002)+
        np.linalg.norm(in_palm,axis=1).max()*certificate_options.get('path_rotation_tolerance_rad',math.radians(.5))+4e-5)
    if actual_footprint_margin<required_footprint_margin:raise ValueError('Insufficient tabletop footprint margin for continuous lateral/rotation budget')
    pc,ancestors=cma.attached_motion_coefficients(a.urdf_tree,a.palm_name,a.sim_joints,np.zeros((1,3)))
    mask=np.array([i in ancestors for i in range(len(pc))])
    joints={j.get('name'):j for j in a.urdf_tree.findall('joint')};pklinks=list(a.pk.get_link_names())
    root=cma._matrix(a.art.root_pose)
    def evaluate(q):
        a.model.set_qpos(q,full=True)
        palm=cma._matrix(a.objects[a.palm_name].pose);bp=palm@a.reference
        vertices=a.bottle_vertices@bp[:3,:3].T+bp[:3,3]
        # An infinite top-plane certificate is conservative, but this particular
        # support scope requires the complete bottle to remain over the desk.
        bounds=a.static_boxes['table']
        if np.any(vertices[:,:2]<bounds[0,:2]) or np.any(vertices[:,:2]>bounds[1,:2]):raise ValueError('Bottle leaves the actual supporting tabletop footprint')
        source=a.static['bottle'];bottle=a.fcl.FCLObject(source.name,a.Pose(bp),source.shapes,source.shape_poses)
        _,collision=a._distance(bottle,a.static['table'])
        return {'signed_gap_m':float(vertices[:,2].min()-table_top),'palm_pose_world':palm,'bottle_table_collision':collision}
    def upward(q,dq):
        a.pk.compute_forward_kinematics(q)
        palm=root@cma._matrix(a.pk.get_link_pose(pklinks.index(a.palm_name)));d=np.zeros(3)
        for i in ancestors:
            if not dq[i]:continue
            j=joints[a.sim_joints[i]];child=j.find('child').get('link');axis=j.find('axis')
            v=np.fromstring(axis.get('xyz','1 0 0'),sep=' ') if axis is not None else np.array([1.,0,0])
            if v.shape!=(3,) or not np.isfinite(v).all() or abs(np.linalg.norm(v)-1)>1e-6:raise ValueError('Requires true normalized native revolute axes')
            pose=root@cma._matrix(a.pk.get_link_pose(pklinks.index(child)))
            d+=np.cross(pose[:3,:3]@v,palm[:3,3]-pose[:3,3])*dq[i]
        return float(d[2])
    try:
        raw_mesh=a.audit(path,side,**dict(audit_options or {}))
        signed=certify_signed_support_polyline(full,evaluate,upward,a.bottle_coefficients,pc,mask,
                is_first_world_lift=is_first_world_lift,measured_baseline_signed_gap_m=measured_baseline,**certificate_options)
        result={'passed':bool(_other_categories_pass(raw_mesh) and signed['passed']),
                'scope':'ONLY first official WORLD z+.1m support exit; no physical collision exemption',
                'raw_unchanged_carry_mesh_audit':raw_mesh,'other_full_mesh_categories_passed':bool(_other_categories_pass(raw_mesh)),
                'signed_support_departure_certificate':signed,'table_top_world_z_m':table_top,
                'actual_native_bottle_collision_shapes':len(a.static['bottle'].shapes),
                'actual_native_collision_vertices':len(a.bottle_vertices),'metadata':a.metadata,
                'physical_signed_height_guard':{'baseline_signed_gap_m':measured_baseline,'support_floor_signed_gap_m':measured_baseline-.000020,
                    'max_initial_penetration_m':.0005,'release_gap_m':.001,'after_release_strict_clearance_m':0.,
                    'table_top_world_z_m':table_top,'table_world_bounds_m':a.static_boxes['table'].tolist(),
                    'bottle_collision_vertices_local_m':a.bottle_vertices.tolist(),
                    'actual_tabletop_footprint_margin_m':actual_footprint_margin,'required_continuous_footprint_margin_m':required_footprint_margin,
                    'geometry_source':'Actual current cooked native PhysX collision vertices and shape local poses/scales',
                    'physical_table_collisions_enabled':True},
                'raw_command_path_sha256':hashlib.sha256(raw.tobytes()).hexdigest(),
                'physical_state_unchanged':True,'timing_scope':'Identical linear joint polyline only; caller must certify TOPP curve relation.'}
        return result
    finally:
        a.model.set_qpos(a.old_model_q,full=True)
        source=a.static['bottle'];a.static['bottle']=a.fcl.FCLObject(source.name,a.original_bottle_pose,source.shapes,source.shape_poses)
        ppg._assert_snapshot(task,before)


def certify_support_departure(task,path,side='left',*,palm_to_bottle=None,audit_options=None,**certificate_options):
    """Public first-lift-only adapter. The caller identifies the first lift."""
    from pathlib import Path
    r=certify_first_world_support_departure(task,path,side,palm_to_bottle=palm_to_bottle,
        is_first_world_lift=True,audit_options=audit_options,**certificate_options)
    c=r['signed_support_departure_certificate'];g=r['physical_signed_height_guard']
    r.update(baseline_signed_gap_m=g['baseline_signed_gap_m'],floor_signed_gap_m=g['support_floor_signed_gap_m'],
             table_top_z=g['table_top_world_z_m'],minimum_signed_gap_m=c.get('sampled_minimum_signed_gap_m'),
             release=c.get('release'),source_hashes={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
             for n in ['support_departure_certificate.py','carry_mesh_path_audit.py','full_mesh_path_audit.py','preflight_place_goals.py']})
    return r
