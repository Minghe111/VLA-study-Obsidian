"""Recorded 36-DOF actual-state audit; independent copies, never scene.step.
Runtime contact readers do not modify the scene. FCL geometry analysis requires
an accepted trace and validates the recorded active-joint order and direct grasp
reference. No hand follower state is inferred and no canonical masks are changed.
"""
from __future__ import annotations
import hashlib,json,time
from pathlib import Path
import numpy as np
from observed_trace_audit import _matrix,_relative_metrics,_load_trace,_phase


def _point_record(point):
    impulse=np.asarray(point.impulse,dtype=float);normal=np.asarray(point.normal,dtype=float)
    if not np.isfinite(impulse).all() or not np.isfinite(normal).all():raise RuntimeError('Nonfinite physical contact')
    return {'normal_impulse_Ns':abs(float(np.dot(impulse,normal))),
            'impulse_Ns':impulse.tolist(),'normal_world':normal.tolist(),
            'point_world_m':np.asarray(point.position,dtype=float).tolist(),
            'separation_m':float(point.separation) if getattr(point,'separation',None) is not None else None}


def collect_actual_self_contacts(task,robot_link_names,allowed,*,normal_impulse_threshold_Ns=1e-7):
    """All nonexempt native link contacts, including zero-normal proximity points.
    A contact pair becomes active only when its summed TRUE normal impulse is
    greater than the requested threshold. Tangential impulse alone is not normal.
    Existing fixed/adjacent/native exclusions are supplied by FullMeshPathAuditor.
    """
    if normal_impulse_threshold_Ns<0:raise ValueError('Negative contact impulse threshold')
    records=[]
    for contact in task.scene.get_contacts():
        names=[b.entity.name for b in contact.bodies]
        if len(names)!=2 or not all(n in robot_link_names for n in names):continue
        if frozenset(names) in allowed:continue
        points=[_point_record(p) for p in contact.points]
        value=sum(p['normal_impulse_Ns'] for p in points)
        records.append({'links':names,'normal_impulse_Ns':value,'active':value>normal_impulse_threshold_Ns,
                        'normal_impulse_threshold_Ns':normal_impulse_threshold_Ns,'points':points})
    return records


def collect_actual_bottle_contacts(task):
    """Exact per-link normal impulses permit per-frame grasp touch exclusions."""
    name=task.bottle.get_name() if hasattr(task.bottle,'get_name') else task.bottle.actor.name
    records=[]
    for contact in task.scene.get_contacts():
        names=[b.entity.name for b in contact.bodies]
        if name not in names:continue
        points=[_point_record(p) for p in contact.points]
        other=names[1] if names[0]==name else names[0]
        records.append({'other_link':other,'normal_impulse_Ns':sum(p['normal_impulse_Ns'] for p in points),
                        'points':points})
    return records


def _touch_links(row,eligible,threshold,dt=.004):
    if _phase(row) not in ('close','close_hold','after_grasp'):return set()
    if 'bottle_actual_contacts' in row:
        return {c['other_link'] for c in row['bottle_actual_contacts']
                if c['other_link'] in eligible and float(c['normal_impulse_Ns'])>threshold}
    if 'contact_forces' in row:
        # Candidate v11 records true normal force per physical contact point;
        # force * actual scene dt is equivalent recorded normal impulse.
        return {c['link'] for c in row['contact_forces']
                if c['link'] in eligible and float(c['normal_force_N'])*dt>threshold}
    raise ValueError('Full audit requires exact recorded bottle normal contacts')


def audit_accepted_actual_trace(task,trace_path,*,report_path=None,output_path=None,
                               normal_impulse_threshold_Ns=1e-7,grasp_normal_threshold_Ns=1e-10,
                               release_clearance_m=.001,require_full_temporal_coverage=True):
    """Every recorded actual 36q frame: full self/table and actual bottle geometry.

    Only exact selected-hand links with recorded real normal impulse may touch
    the bottle. Initial/close bottle-table support is reported as expected;
    after the bottle clears the table by release_clearance_m, any later geometric
    intersection fails. Final bottle-table clearance must be positive. This is
    discrete geometry coverage, not a swept-volume or historical-force replay.
    """
    from full_mesh_path_audit import FullMeshPathAuditor,_pose_matrix,_aabb_gap,_link_vertices
    from preflight_place_goals import _snapshot,_assert_snapshot
    started=time.monotonic();trace_path=Path(trace_path)
    report_path=Path(report_path) if report_path else trace_path.with_name('attempt_report.json')
    trace,report,anchor,after,reference=_load_trace(trace_path,report_path)
    if report.get('accepted') is not True or report.get('result') not in ('physics_success','one_strictly_validated_episode_collected'):
        raise ValueError('Postaccepted full audit requires a strictly accepted completed episode')
    if report.get('canonical_sources_unchanged') is not True:raise ValueError('Missing canonical-source acceptance')
    if getattr(task.robot,'communication_flag',False):raise ValueError('Simulation-only read-only audit')
    if int(task.model_id)!=int(trace[anchor]['bottle']['model_id']) or int(task.qpose_tag)!=int(trace[anchor]['bottle']['qpose_tag']):
        raise ValueError('Current scene model differs from recorded geometry')
    sources={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in report['source_hashes']}
    if sources!=report['source_hashes']:raise ValueError('Current canonical sources differ from accepted trace')
    steps=np.array([int(r['step']) for r in trace]);full_coverage=bool(steps[0]==1 and np.all(np.diff(steps)==1))
    if require_full_temporal_coverage and not full_coverage:raise ValueError('Full audit requires every physical step beginning at1')
    physical=_snapshot(task);auditor=None
    try:
        auditor=FullMeshPathAuditor(task,check_table=True);names=auditor.sim_joints
        if report.get('full_qpos_joint_names')!=names:raise ValueError('Recorded full qpos active-joint order missing/different')
        qpos=np.array([r['full_qpos_actual_rad'] for r in trace],dtype=float)
        if qpos.shape!=(len(trace),36) or len(names)!=36 or not np.isfinite(qpos).all():raise ValueError('Expected all finite actual36q')
        for side in ('left','right'):
            idx=[names.index(n) for n in getattr(task.robot,side+'_arm_joints_name')]
            if not np.allclose(qpos[:,idx],[r['arm_qpos'][side] for r in trace],atol=1e-6,rtol=0):
                raise ValueError('Actual36q disagrees with separately recorded armq')
        side='left' if int(task.qpose_tag)==0 else 'right';prefix='l_' if side=='left' else 'r_'
        palm=prefix+'inspire_base_link';eligible={n for n in auditor.objects if n.startswith(prefix+'inspire_')}
        source_bottle=auditor.static['bottle'];bottle_vertices,_=_link_vertices(next(c for c in task.bottle.actor.get_components() if hasattr(c,'collision_shapes')))
        collider_names=list(auditor.objects);index={n:i for i,n in enumerate(collider_names)}
        pair_first=np.array([index[a] for a,b in auditor.self_pairs]);pair_second=np.array([index[b] for a,b in auditor.self_pairs])
        table=auditor.static['table'];table_box=auditor.static_boxes['table']
        auditor.model.set_qpos(qpos[anchor],full=True)
        check=_relative_metrics(_pose_matrix(auditor.objects[palm].pose),_matrix(trace[anchor]['bottle']['pose']),reference)
        if check['relative_translation_drift_m']>2e-5 or check['relative_rotation_drift_deg']>.01:
            raise ValueError('Full FK differs from historical direct grasp reference')
        limits=np.concatenate([np.asarray(j.limits).reshape(-1,2) for j in auditor.art.active_joints])
        max_excess=float(np.maximum(np.maximum(limits[:,0]-qpos,qpos-limits[:,1]),0).max())
        rows=[];events={};first_release=None;table_recontacts=[];true_self_count=0;max_drift=0.;max_angle=0.
        def event(category,pair,step,index):
            key=(category,*pair)
            record=events.setdefault(key,{'category':category,'pair':list(pair),'count':0,'first_step':step,'last_step':step,'first_trace_index':index})
            record['count']+=1;record['last_step']=step
        for i,(row,q) in enumerate(zip(trace,qpos)):
            auditor.model.set_qpos(q,full=True);boxes={}
            for n,obj in auditor.objects.items():
                t=_pose_matrix(obj.pose);v=auditor.vertices[n]@t[:3,:3].T+t[:3,3];boxes[n]=np.stack([v.min(0),v.max(0)])
            bottle_pose=_matrix(row['bottle']['pose']);v=bottle_vertices@bottle_pose[:3,:3].T+bottle_pose[:3,3];bottle_box=np.stack([v.min(0),v.max(0)])
            bottle=auditor.fcl.FCLObject(source_bottle.name,auditor.Pose(bottle_pose),source_bottle.shapes,source_bottle.shape_poses)
            step=int(row['step']);self_hits=[];robot_table=[];unexpected_bottle=[];touch=_touch_links(row,eligible,grasp_normal_threshold_Ns,float(task.scene.get_timestep()))
            # Vectorized broad phase retains EVERY native pair, reducing Python
            # work while exact FCL narrow phase covers every possible overlap.
            bb=np.array([boxes[n] for n in collider_names]);a=bb[pair_first];b=bb[pair_second]
            candidate_pairs=np.where(np.all((a[:,0]<=b[:,1])&(b[:,0]<=a[:,1]),axis=1))[0]
            for k in candidate_pairs:
                a,b=auditor.self_pairs[int(k)]
                hit=bool(auditor.fcl.collide(auditor.objects[a],auditor.objects[b],auditor.collision_request).is_collision())
                if hit:self_hits.append([a,b]);event('self',(a,b),step,i)
            candidate_table=np.where(np.all((bb[:,0]<=table_box[1])&(table_box[0]<=bb[:,1]),axis=1))[0]
            for k in candidate_table:
                n=collider_names[int(k)];hit=bool(auditor.fcl.collide(auditor.objects[n],table,auditor.collision_request).is_collision())
                if hit:robot_table.append(n);event('robot_table',(n,'table'),step,i)
            candidate_bottle=np.where(np.all((bb[:,0]<=bottle_box[1])&(bottle_box[0]<=bb[:,1]),axis=1))[0]
            for k in candidate_bottle:
                n=collider_names[int(k)]
                if n in touch:continue
                hit=bool(auditor.fcl.collide(auditor.objects[n],bottle,auditor.collision_request).is_collision())
                if hit:unexpected_bottle.append(n);event('bottle_robot_non_touch',('bottle',n),step,i)
            gap,table_hit=auditor._distance(bottle,table)
            if i>anchor and first_release is None and gap>=release_clearance_m:first_release={'step':step,'trace_index':i,'distance_m':gap}
            if first_release is not None and table_hit:table_recontacts.append({'step':step,'trace_index':i});event('bottle_table_after_release',('bottle','table'),step,i)
            if 'actual_self_contacts' not in row:raise ValueError('Missing recorded actual self normal contact audit')
            active_self=[c for c in row['actual_self_contacts'] if float(c['normal_impulse_Ns'])>normal_impulse_threshold_Ns]
            true_self_count+=len(active_self)
            for c in active_self:event('historical_true_self_normal',tuple(c.get('links',c.get('names',[]))),step,i)
            rec={'trace_index':i,'step':step,'phase':row['phase'],'self_collisions':self_hits,'robot_table_collisions':robot_table,
                 'bottle_non_touch_collisions':unexpected_bottle,'allowed_exact_grasp_touch_links':sorted(touch),
                 'bottle_table_geometry':{'distance_m':gap,'collision':table_hit},'actual_self_contacts':row['actual_self_contacts']}
            if i>=anchor:
                m=_relative_metrics(_pose_matrix(auditor.objects[palm].pose),bottle_pose,reference)
                rec.update(relative_translation_drift_m=m['relative_translation_drift_m'],relative_rotation_drift_deg=m['relative_rotation_drift_deg'])
                max_drift=max(max_drift,m['relative_translation_drift_m']);max_angle=max(max_angle,m['relative_rotation_drift_deg'])
            rows.append(rec)
        physical_steps=report.get('physical_steps')
        complete_to_endpoint=bool(physical_steps is None or int(physical_steps)==int(steps[-1]))
        passed=bool(not events and first_release is not None and rows[-1]['bottle_table_geometry']['distance_m']>0 and
                    max_excess<=1e-4 and max_drift<=.002 and max_angle<=2 and complete_to_endpoint and (full_coverage or not require_full_temporal_coverage))
        result={'passed':passed,'scope':'Every recorded actual36q state, full native geometry; exact real-normal selected-hand touch exclusions only. Discrete samples, no swept-volume claim. Initial bottle/table support is reported; any overlap after1mm table release fails.',
                'trace_sha256':hashlib.sha256(trace_path.read_bytes()).hexdigest(),'report_sha256':hashlib.sha256(report_path.read_bytes()).hexdigest(),
                'source_hashes':sources,'full_qpos_joint_names':names,'native_mesh_metadata':auditor.metadata,
                'recorded_samples':len(rows),'full_temporal_coverage':full_coverage,'complete_to_reported_endpoint':complete_to_endpoint,
                'actual_self_normal_impulse_threshold_Ns':normal_impulse_threshold_Ns,'grasp_touch_normal_impulse_threshold_Ns':grasp_normal_threshold_Ns,
                'historical_reference_validation':check,'max_measured_joint_limit_excess_rad':max_excess,
                'max_post_grasp_relative_translation_drift_m':max_drift,'max_post_grasp_relative_rotation_drift_deg':max_angle,
                'bottle_table_first_release':first_release,'bottle_table_release_clearance_m':release_clearance_m,
                'bottle_table_recontacts':table_recontacts,'collision_events':list(events.values()),'historical_actual_self_active_count':true_self_count,
                'samples':rows,'physical_scene_unchanged':False,'elapsed_s':time.monotonic()-started}
    finally:
        if auditor is not None:auditor.model.set_qpos(auditor.old_model_q,full=True)
        _assert_snapshot(task,physical)
        if sources!={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources}:raise RuntimeError('Canonical source changed during audit')
    result['physical_scene_unchanged']=True
    if output_path is not None:Path(output_path).write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    return result
