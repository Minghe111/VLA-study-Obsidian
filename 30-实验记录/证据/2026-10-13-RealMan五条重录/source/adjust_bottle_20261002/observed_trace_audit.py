"""Read-only FK and collision audit of recorded *actual* arm/bottle states.

No scene.step, SAPIEN qpos/drive writes, planner calls, queues, or hardware IO.
Only an independent FullMeshPathAuditor model changes qpos. Finger states are
not reconstructed: nonhand geometry and the palm are exactly arm-dependent.
Call analyze_observed_trace(task, trace_path, report_path=..., output_path=...).
Geometry intersection establishes overlap, not historical contact impulse or
causality. Historical fixed head/base/table poses were not recorded; canonical
hashes, current model identity, and direct historical palm-relative validation
are checked before interpreting the trace.
"""
from __future__ import annotations
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np


def _matrix(value):
    a=np.asarray(value,dtype=np.float64)
    if a.shape!=(4,4) or not np.isfinite(a).all():raise ValueError('Invalid recorded pose')
    if not np.allclose(a[3],[0,0,0,1],atol=1e-8):raise ValueError('Invalid recorded homogeneous transform')
    if not np.allclose(a[:3,:3].T@a[:3,:3],np.eye(3),atol=2e-5):raise ValueError('Invalid recorded rotation')
    return a.copy()


def _angle_deg(first,second):
    r=first[:3,:3]@second[:3,:3].T
    u,_,vt=np.linalg.svd(r);r=u@vt
    return float(np.degrees(np.arccos(np.clip((np.trace(r)-1)/2,-1,1))))


def _relative_metrics(palm,bottle,reference):
    actual=np.linalg.inv(palm)@bottle
    predicted=palm@reference
    return {'relative_bottle_palm':actual.tolist(),
            'relative_translation_delta_palm_m':(actual[:3,3]-reference[:3,3]).tolist(),
            'relative_translation_drift_m':float(np.linalg.norm(actual[:3,3]-reference[:3,3])),
            'relative_rotation_drift_deg':_angle_deg(actual,reference),
            'no_slip_predicted_bottle_world_pose':predicted.tolist(),
            'actual_minus_no_slip_bottle_world_m':(bottle[:3,3]-predicted[:3,3]).tolist()}


def _phase(row):return str(row.get('phase',{}).get('name','unknown'))


def _load_trace(trace_path,report_path):
    trace_path=Path(trace_path);report_path=Path(report_path)
    trace=json.loads(trace_path.read_text(encoding='utf-8'))
    report=json.loads(report_path.read_text(encoding='utf-8'))
    if not isinstance(trace,list) or not trace:raise ValueError('Expected nonempty trace list')
    steps=np.array([int(x['step']) for x in trace])
    if np.any(np.diff(steps)<=0):raise ValueError('Recorded steps must strictly increase')
    hold=[i for i,x in enumerate(trace) if _phase(x)=='close_hold']
    if not hold:raise ValueError('No historical close_hold reference')
    anchor=hold[-1]
    after=[i for i,x in enumerate(trace) if i>anchor and _phase(x)=='after_grasp']
    if not after:raise ValueError('No recorded retreat/carry actual states')
    reference=_matrix(report['pre_retreat_grip']['actual']['relative_bottle_palm'])
    if report['pre_retreat_grip'].get('passed') is not True:raise ValueError('Historical pre-retreat grip failed')
    for x in trace:
        for side in ('left','right'):
            q=np.asarray(x['arm_qpos'][side],dtype=float)
            if q.shape!=(6,) or not np.isfinite(q).all():raise ValueError('Invalid actual six-arm state')
        _matrix(x['bottle']['pose'])
    return trace,report,anchor,after,reference


def analyze_observed_trace(task,trace_path,*,report_path=None,output_path=None,
                           reference_tolerance_m=2e-5,reference_tolerance_deg=.01):
    """All recorded samples, actual FK, complete nonhand pair geometry.

    Copies meshes from the current scene, which must use the same historical
    bottle model, canonical sources and palm/base setup. No temporal interpolation
    is claimed; approach is recorded every ten steps, close/retreat every step.
    The reference validation rejects changed palm/root calibration. Unknown hand
    follower states are excluded, rather than synthesized into 'exact' geometry.
    """
    from full_mesh_path_audit import FullMeshPathAuditor,_pose_matrix,_aabb_gap
    from preflight_place_goals import _snapshot,_assert_snapshot
    started=time.monotonic();trace_path=Path(trace_path)
    report_path=Path(report_path) if report_path is not None else trace_path.with_name('attempt_report.json')
    trace,historical,anchor,after,reference=_load_trace(trace_path,report_path)
    side='left' if int(trace[anchor]['bottle']['qpose_tag'])==0 else 'right'
    if bool(getattr(task.robot,'communication_flag',False)):raise ValueError('Simulation-only audit')
    if int(task.model_id)!=int(trace[anchor]['bottle']['model_id']) or int(task.qpose_tag)!=int(trace[anchor]['bottle']['qpose_tag']):
        raise ValueError('Current scene bottle model/qpose_tag differs from historical scene')
    hashes={str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in historical['source_hashes']}
    if hashes!=historical['source_hashes']:raise ValueError('Current canonical sources differ from historical episode')
    physical_before=_snapshot(task);auditor=None
    try:
        auditor=FullMeshPathAuditor(task,check_table=True)
        names=auditor.sim_joints
        arm_indices={s:[names.index(n) for n in getattr(task.robot,s+'_arm_joints_name')] for s in ('left','right')}
        arm_dof=set(sum(arm_indices.values(),[]))
        palm_name=('l_' if side=='left' else 'r_')+'inspire_base_link'
        if palm_name not in auditor.objects:raise ValueError('Palm collider missing')
        # Include fixed force sensor descendants only if independent of all hand DOFs.
        exact_links={n for n,c in auditor.motion_coefficients.items()
                     if not any(abs(c[i])>1e-12 for i in range(len(c)) if i not in arm_dof)}
        # Inspire palm collision itself is arm-dependent, but no unknown finger links.
        if palm_name not in exact_links:raise ValueError('Palm unexpectedly depends on unrecorded DOF')
        nonhand_links={n for n in exact_links if '_inspire_' not in n}
        self_pairs=[(a,b) for a,b in auditor.self_pairs if a in nonhand_links and b in nonhand_links]
        watch_pairs=[(a,b) for a,b in self_pairs if 'body_base_link' in (a,b)
                     and any(n in ('l_link4','l_link5','r_link4','r_link5') for n in (a,b))]
        if not watch_pairs:raise ValueError('Expected complete body/arm watch pairs missing')
        source_bottle=auditor.static['bottle']
        q=auditor.initial_sim_q.copy()
        def fk_for(row):
            current=q.copy()
            for s,idx in arm_indices.items():current[idx]=np.asarray(row['arm_qpos'][s],dtype=float)
            auditor.model.set_qpos(current,full=True)
            return _pose_matrix(auditor.objects[palm_name].pose)
        anchor_palm=fk_for(trace[anchor]);anchor_bottle=_matrix(trace[anchor]['bottle']['pose'])
        anchor_metrics=_relative_metrics(anchor_palm,anchor_bottle,reference)
        if (anchor_metrics['relative_translation_drift_m']>reference_tolerance_m or
            anchor_metrics['relative_rotation_drift_deg']>reference_tolerance_deg):
            raise ValueError('Historical direct grip reference disagrees with reconstructed actual FK: '+str(anchor_metrics))
        rows=[];pair_stats={};watch_min={};table_min=None;max_drift=None;max_rotation=None
        intervals=[]
        for i,row in enumerate(trace):
            palm=fk_for(row);bottle_pose=_matrix(row['bottle']['pose'])
            record={'trace_index':i,'step':int(row['step']),'phase':row['phase'],
                    'palm_world_pose':palm.tolist(),'bottle_world_pose':bottle_pose.tolist()}
            if i>=anchor:
                record.update(_relative_metrics(palm,bottle_pose,reference))
                if max_drift is None or record['relative_translation_drift_m']>max_drift['value']:
                    max_drift={'step':record['step'],'trace_index':i,'value':record['relative_translation_drift_m']}
                if max_rotation is None or record['relative_rotation_drift_deg']>max_rotation['value']:
                    max_rotation={'step':record['step'],'trace_index':i,'value':record['relative_rotation_drift_deg']}
            boxes={}
            for n in nonhand_links:
                t=_pose_matrix(auditor.objects[n].pose);v=auditor.vertices[n]@t[:3,:3].T+t[:3,3]
                boxes[n]=np.stack([v.min(0),v.max(0)])
            collisions=[];watch={}
            for a,b in self_pairs:
                # AABB separation proves no overlap; full FCL for any candidate overlap.
                overlap_possible=_aabb_gap(boxes[a],boxes[b])<=0.
                watched=(a,b) in watch_pairs
                if not overlap_possible and not watched:continue
                gap,hit=auditor._distance(auditor.objects[a],auditor.objects[b])
                key=a+'/'+b
                if watched:
                    watch[key]={'distance_m':gap,'collision':hit}
                    if key not in watch_min or gap<watch_min[key]['distance_m']:
                        watch_min[key]={'distance_m':gap,'step':record['step'],'trace_index':i,'collision':hit}
                if hit:
                    collisions.append([a,b])
                    stats=pair_stats.setdefault(key,{'pair':[a,b],'count':0,'first_step':record['step'],'last_step':record['step'],'phase_counts':{}})
                    stats['count']+=1;stats['last_step']=record['step'];ph=_phase(row)
                    stats['phase_counts'][ph]=stats['phase_counts'].get(ph,0)+1
            bottle=auditor.fcl.FCLObject(source_bottle.name,auditor.Pose(bottle_pose),source_bottle.shapes,source_bottle.shape_poses)
            gap,hit=auditor._distance(bottle,auditor.static['table'])
            record.update(nonhand_self_collisions=collisions,body_arm_watch=watch,
                          bottle_table_geometry={'distance_m':gap,'collision':hit})
            if table_min is None or gap<table_min['distance_m']:
                table_min={'distance_m':gap,'collision':hit,'step':record['step'],'trace_index':i}
            if i>=anchor:
                impulse=sum(float(x.get('impulse_norm_sum',0)) for x in row['bottle'].get('hand_contacts',[]))
                record['historical_hand_impulse_norm_sum_Ns']=impulse
                record['historical_capture_actual_pair_streak']=row.get('capture_actual_pair_streak')
            rows.append(record)
        endpoint=rows[after[-1]];anchor_row=rows[anchor]
        result={'analysis':'actual recorded dual-arm FK and bottle poses; no commanded-path proxy',
                'trace_path':str(trace_path),'trace_sha256':hashlib.sha256(trace_path.read_bytes()).hexdigest(),
                'historical_report_sha256':hashlib.sha256(report_path.read_bytes()).hexdigest(),
                'historical_result':historical.get('result'),'current_model_id':int(task.model_id),
                'current_qpose_tag':int(task.qpose_tag),'side':side,'palm_link':palm_name,
                'source_hashes':hashes,'canonical_sources_match_historical':True,
                'auxiliary_mesh_metadata':auditor.metadata,'native_root_world_pose':_pose_matrix(auditor.art.root_pose).tolist(),
                'audited_recorded_samples':len(rows),'self_geometry_exact_nonhand_links':sorted(nonhand_links),
                'self_pairs_checked':len(self_pairs),'excluded_hand_dof_reason':'Recorded SDK masters omit actual follower states; palm/nonhand FK does not depend on them.',
                'historical_reference_validation':anchor_metrics,
                'anchor':anchor_row,'retreat_endpoint':endpoint,
                'retreat_recorded_rows':len(after),'retreat_step_gaps':sorted(set(int(trace[b]['step'])-int(trace[a]['step']) for a,b in zip([anchor]+after,after))),
                'max_post_grasp_translation_drift':max_drift,'max_post_grasp_rotation_drift':max_rotation,
                'bottle_world_displacement_m':(np.asarray(endpoint['bottle_world_pose'])[:3,3]-anchor_bottle[:3,3]).tolist(),
                'palm_world_displacement_m':(np.asarray(endpoint['palm_world_pose'])[:3,3]-anchor_palm[:3,3]).tolist(),
                'nonhand_self_collision_pairs':list(pair_stats.values()),'body_arm_watch_minima':watch_min,
                'bottle_table_minimum_geometry_gap':table_min,'samples':rows,
                'physical_scene_unchanged':False,'elapsed_s':time.monotonic()-started,
                'scope':'All recorded discrete states. No scene stepping, interpolated-path clearance claim, collision-mask changes, historical contact-force reconstruction, or causal attribution. Current fixed base/head/table are assumed unchanged from setup; palm/root exact alignment is checked against historical direct grip reference.'}
    finally:
        if auditor is not None:auditor.model.set_qpos(auditor.old_model_q,full=True)
        _assert_snapshot(task,physical_before)
        current_hashes={str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in hashes}
        if current_hashes!=hashes:raise RuntimeError('Canonical sources changed during read-only audit')
    result['physical_scene_unchanged']=True
    if output_path is not None:Path(output_path).write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    return result
