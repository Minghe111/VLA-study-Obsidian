"""Independent evidence-chain checks for v13/v14 recorded geometry.

No controller is imported. Only boundary-used records additionally rebuild one
independent frozen CPU event scene through dataset50_boundary_cpu_v2.
"""
from __future__ import annotations
import math
from pathlib import Path
import re
import numpy as np
from dataset50_audit import read_json, require, sha

HELPER_SHA = 'e40149ba15aff34de64a4e6d3cdcda700766934f077f47cab1888c1b36335020'
CONSTRUCTOR_SHA = 'e5d13b16ef6aff9ad9624978e9c269a1bd4b86b4c526542bb03acaa9c139f490'
META_KEYS = ('full_dof','collision_links','collision_shapes','geometry_types','bottle_shapes',
    'native_exemption_pairs','engine_adjacent_pairs','engine_adjacent_additional_pairs',
    'effective_exemption_pairs','checked_self_pairs','urdf_kinematic_export_sha256')


def finite(x, label, *, nonnegative=False):
    require(type(x) in (int,float) and math.isfinite(x) and (not nonnegative or x >= 0), 'Invalid finite scalar: '+label)
    return float(x)


def exact_number(a,b,label, *, atol=1e-15):
    a,b=finite(a,label),finite(b,label)
    require(math.isclose(a,b,rel_tol=1e-12,abs_tol=atol),'Numerical evidence disagrees: '+label)


def digest(x,label):
    require(isinstance(x,str) and re.fullmatch('[0-9a-f]{64}',x) is not None,'Missing SHA256: '+label)
    return x


def rebuild_events(samples, threshold, release):
    merged={}
    def add(category,pair,row):
        key=(category,*pair)
        e=merged.setdefault(key,{'category':category,'pair':list(pair),'count':0,
            'first_step':row['step'],'last_step':row['step'],'first_trace_index':row['trace_index']})
        e['count']+=1;e['last_step']=row['step']
    for row in samples:
        for pair in row['self_collisions']: add('self',pair,row)
        for link in row['robot_table_collisions']: add('robot_table',(link,'table'),row)
        for link in row['bottle_non_touch_collisions']: add('bottle_robot_non_touch',('bottle',link),row)
        if release is not None and row['trace_index']>=release['trace_index'] and row['bottle_table_geometry']['collision']:
            add('bottle_table_after_release',('bottle','table'),row)
        for contact in row['actual_self_contacts']:
            if finite(contact['normal_impulse_Ns'],'self impulse',nonnegative=True)>threshold:
                add('historical_true_self_normal',contact.get('links',contact.get('names',[])),row)
    return sorted(merged.values(),key=lambda e:(e['category'],e['pair']))


def check_full_geometry(g,trace,report):
    rows=g['samples'];n=len(trace)
    require(n>0 and len(rows)==g['recorded_samples']==report['physical_steps'],'Full geometry coverage differs')
    require(g['full_temporal_coverage'] is True and g['complete_to_reported_endpoint'] is True
        and g['physical_scene_unchanged'] is True,'Incomplete/changed geometry record')
    require(g['full_qpos_joint_names']==report['full_qpos_joint_names'],'Geometry joint ordering differs')
    require(g['source_hashes']==report['source_hashes'],'Geometry/expert source registry differs')
    threshold=finite(g['actual_self_normal_impulse_threshold_Ns'],'self threshold',nonnegative=True)
    require(threshold==1e-7 and g['grasp_touch_normal_impulse_threshold_Ns']==1e-10,'Original contact thresholds changed')
    for i,(r,s) in enumerate(zip(trace,rows)):
        require(r['step']==s['step']==i+1 and s['trace_index']==i,'Geometry frame/step differs')
        require(r['phase']==s['phase'] and r['actual_self_contacts']==s['actual_self_contacts'],'Geometry phase/native contacts differ')
        require(type(r['bottle']['model_id']) is int and r['bottle']['model_id']==report['sample_spec']['model_id'],'All-trace bottle model differs')
        require(type(r['bottle']['qpose_tag']) is int and r['bottle']['qpose_tag']==(0 if report['active_arm']=='left' else 1),'All-trace side tag differs')
        q=np.asarray(r['full_qpos_actual_rad'],float)
        require(q.shape==(36,) and np.isfinite(q).all(),'Nonfinite/non36 event state')
        for side,prefix in [('left','l_'),('right','r_')]:
            ids=[g['full_qpos_joint_names'].index(prefix+'joint'+str(j)) for j in range(1,7)]
            require(np.allclose(q[ids],r['arm_qpos'][side],atol=1e-6,rtol=0),'Actual36/arm state disagrees')
    release=g['bottle_table_first_release']
    if release is not None:
        ix=release['trace_index'];require(type(ix) is int and 0<=ix<n,'Invalid release index')
        require(release['step']==rows[ix]['step'],'Release frame differs')
        exact_number(release['distance_m'],rows[ix]['bottle_table_geometry']['distance_m'],'Release gap')
        require(release['distance_m']>=g['bottle_table_release_clearance_m']==.001,'Table release threshold changed')
    events=rebuild_events(rows,threshold,release)
    require(events==sorted(g['collision_events'],key=lambda e:(e['category'],e['pair'])),'Original collision events do not reconstruct')
    recontacts=[{'step':s['step'],'trace_index':s['trace_index']} for s in rows
        if release is not None and s['trace_index']>=release['trace_index'] and s['bottle_table_geometry']['collision']]
    require(recontacts==g['bottle_table_recontacts'],'Table recontact evidence differs')
    true_self=sum(sum(finite(c['normal_impulse_Ns'],'native self',nonnegative=True)>threshold
        for c in s['actual_self_contacts']) for s in rows)
    require(true_self==g['historical_actual_self_active_count'],'Native self event count differs')
    drift=max([finite(s['relative_translation_drift_m'],'relative drift',nonnegative=True) for s in rows if 'relative_translation_drift_m' in s] or [0.])
    angle=max([finite(s['relative_rotation_drift_deg'],'relative angle',nonnegative=True) for s in rows if 'relative_rotation_drift_deg' in s] or [0.])
    exact_number(drift,g['max_post_grasp_relative_translation_drift_m'],'Full-trace maximum drift')
    exact_number(angle,g['max_post_grasp_relative_rotation_drift_deg'],'Full-trace maximum rotation')
    excess=finite(g['max_measured_joint_limit_excess_rad'],'joint excess',nonnegative=True)
    baseline=release is not None and finite(rows[-1]['bottle_table_geometry']['distance_m'],'final gap')>0 and excess<=1e-4 and drift<=.002 and angle<=2
    require(g['passed'] is bool(baseline and not events),'Original acceptance does not reconstruct')
    return baseline,events


def native_event(previous,current,link,side,threshold):
    prefix='l_' if side=='left' else 'r_'
    require(re.fullmatch(prefix+'inspire_'+side+r'_(thumb|index|middle)_[1-4]',link) is not None,'Boundary uses palm/passive/inactive-arm link')
    require(current['step']==previous['step']+1,'Boundary lacks immediate previous frame')
    for r in [previous,current]:
        p=r['phase'];require(p['name']=='close' and p['gripper'] is True and type(p['dense_call']) is int,'Boundary outside native close')
    require(previous['phase']['dense_call']==current['phase']['dense_call'],'Boundary crosses close calls')
    require(all(previous['bottle'][k]==current['bottle'][k] for k in ['model_id','qpose_tag']),'Boundary changes bottle identity')
    totals=[];current_points=[]
    for r in [previous,current]:
        contacts=[c for c in r['bottle_actual_contacts'] if c['other_link']==link]
        require(contacts,'Missing exact native contact')
        all_points=[]
        for c in contacts:
            points=c['points'];require(isinstance(points,list) and points,'Missing native contact points')
            normal=[]
            for p in points:
                impulse=np.asarray(p['impulse_Ns'],float);axis=np.asarray(p['normal_world'],float);point=np.asarray(p['point_world_m'],float)
                require(all(v.shape==(3,) and np.isfinite(v).all() for v in [impulse,axis,point]),'Invalid native vectors')
                value=finite(p['normal_impulse_Ns'],'point normal',nonnegative=True)
                finite(p['separation_m'],'native separation')
                require(math.isclose(abs(float(impulse@axis)),value,rel_tol=1e-10,abs_tol=1e-18),'Point normal does not match native impulse')
                normal.append(value)
            require(math.isclose(sum(normal),finite(c['normal_impulse_Ns'],'contact normal',nonnegative=True),rel_tol=1e-12,abs_tol=1e-18),'Contact normal aggregate differs')
            all_points+=points
        totals.append(sum(p['normal_impulse_Ns'] for p in all_points));current_points=all_points
    require(totals[0]>threshold and all(p['normal_impulse_Ns']==0 for p in current_points),'Not a real positive-to-zero load transition')
    require(all(p['separation_m']>=0 for p in current_points),'Negative native separation')
    return {'previous_step':previous['step'],'current_step':current['step'],'dense_call':current['phase']['dense_call'],
        'previous_normal_impulse_Ns':totals[0],'current_normal_impulse_Ns':totals[1],'current_native_points':len(current_points),
        'minimum_native_separation_m':min(p['separation_m'] for p in current_points),'maximum_native_separation_m':max(p['separation_m'] for p in current_points)}


def validate_depth(d,row,link,metadata):
    for flag in ['all_shape_pairs_evaluated','contact_lists_untruncated','original_boolean_collision_reproduced','collision_count_positive']:
        require(d.get(flag) is True,'Missing full depth evidence: '+flag)
    v=d['event_specific_vertex_certificate'];require(v['event_fk_audit']['passed'] is True and v['event_fk_audit']['checked_links']==metadata['collision_links'],'Event FK proof incomplete');require(v['event_step']==row['step'] and v['link']==link,'Depth event identity differs')
    receipt=v['scene_receipt']
    for key,value in {'cpu_only_scene':True,'render_system_count':0,'scene_steps':0,'snapshot_initial_qpos_writes_before_audits':1,
        'audit_sapien_qpos_writes':0,'snapshot_initial_qpos_exact':True,'hardware_io':False,'new_physics_episodes':0,
        'event_step':row['step'],'changed_initialization_values':2,'constructor_source_sha256':CONSTRUCTOR_SHA}.items():
        require(receipt.get(key)==value,'Event CPU initialization receipt differs: '+key)
    require(receipt['initialization_state']=='exact_recorded_event_qpos_and_bottle_pose','Event initial state identity missing')
    require(v['event_physical_scene_unchanged'] is True,'Event CPU physical scene changed')
    for k in ['finger_collision_content_sha256','bottle_collision_content_sha256']:digest(v[k],k)
    require(finite(v['event_auxiliary_to_depth_auxiliary_vertex_error_m'],'event/depth FK error',nonnegative=True)<=1e-12,'Event FK differs from depth FK')
    terms=['native_finger_matrix_to_fcl_vertex_error_m','recorded_to_native_bottle_vertex_error_m','native_bottle_matrix_to_fcl_vertex_error_m']
    conversion=sum(finite(v[k],k,nonnegative=True) for k in terms)
    budget=conversion+finite(v['native_to_auxiliary_finger_vertex_error_m'],'event FK error',nonnegative=True)
    exact_number(conversion,v['native_signed_gap_conversion_budget_m'],'Pure conversion budget')
    exact_number(budget,v['numerical_vertex_budget_m'],'Event vertex budget')
    exact_number(budget,d['numerical_vertex_budget_m'],'Depth budget')
    probes=v['native_fk_signed_gap_probes'];require([p['gjk_tolerance'] for p in probes]==[1e-6,1e-9,1e-12],'Missing three signed-gap probes')
    for p in probes:require(finite(p['signed_distance_m'],'native signed gap')>=-conversion,'Native signed gap exceeds PURE conversion budget')
    native=v['native_all_shape_pair_probes'];pairs=d['shape_pair_depths'];bcount=metadata['bottle_shapes']
    fcount=len(metadata['geometry_types'][link]);expected={(i,j) for i in range(fcount) for j in range(bcount)}
    require({(p['finger_shape_index'],p['bottle_shape_index']) for p in pairs}==expected and len(pairs)==len(expected),'Aux shape-pair coverage differs')
    require(d['shape_pairs_evaluated']==len(expected),'Declared shape-pair count differs')
    require({(p['finger_shape_index'],p['bottle_shape_index'],p['gjk_tolerance']) for p in native}=={(i,j,t) for i,j in expected for t in [1e-6,1e-9,1e-12]}
        and len(native)==3*len(expected),'Native shape-pair coverage differs')
    for p in native:
        require(type(p['collision']) is bool,'Unknown native collision')
        require(finite(p['signed_distance_m'],'native pair gap')>=-conversion,'Native pair gap exceeds PURE conversion budget')
        depths=p['penetration_depths_m'];require(isinstance(depths,list) and len(depths)<512 and (not p['collision'] or depths),'Native depth missing/truncated')
        require(all(finite(x,'native depth',nonnegative=True)<=conversion for x in depths),'Native depth exceeds PURE conversion budget')
    maximum=0.;hits=0
    for pair in pairs:
        ps=pair['probes'];require([p['gjk_tolerance'] for p in ps]==[1e-6,1e-9,1e-12],'Missing aux precision probes')
        require(all(type(p['collision']) is bool and p['collision']==ps[0]['collision'] for p in ps),'Aux collision unstable')
        hits+=int(ps[0]['collision'])
        for p in ps:
            ds=p['penetration_depths_m'];require(p['contact_count']==len(ds)<512 and (not p['collision'] or ds),'Aux depths unknown/truncated')
            for x in ds:maximum=max(maximum,finite(x,'aux depth',nonnegative=True))
    exact_number(maximum,d['measured_penetration_depth_m'],'Enumerated maximum depth')
    require(hits>0 and hits==d['shape_pairs_colliding'] and maximum<=budget,'Aux penetration exceeds measured event budget')
    require(v['native_signed_gap_within_conversion_resolution'] is True,'Signed-gap outcome disagrees')
    actual_nonnegative=all(p['signed_distance_m']>=0 and not p['native_fcl_collision'] for p in probes)
    require(v['native_signed_gap_nonnegative'] is actual_nonnegative,'Native nonnegative claim disagrees')
    return {'native_signed_gap_min_m':min(p['signed_distance_m'] for p in probes),'pure_conversion_budget_m':conversion,
        'aux_penetration_depth_m':maximum,'event_vertex_budget_m':budget,'shape_pairs':len(expected)}


def audit_geometry_chain(directory, *, cpu_verify=True):
    directory=Path(directory).resolve();report=read_json(directory/'expert_report.json')
    require(report.get('accepted') is True and report.get('result')=='physics_success' and report.get('hold_passed') is True,'Expert not accepted physics success')
    scene=read_json(directory/'scene_sample.json');require({k:v for k,v in report['sample_spec'].items() if k!='grasp_setting'}=={k:v for k,v in scene['sample_spec'].items() if k!='grasp_setting'},'Scene/expert full sample identity differs');trace=read_json(directory/'physics_trace.json')
    final_path=directory/'actual36_geometry_audit.json';g=read_json(final_path)
    require(g['passed'] is True,'Final geometry not accepted')
    require(g['trace_sha256']==sha(directory/'physics_trace.json'),'Final trace hash mismatch')
    embedded=report.get('actual36_geometry_audit');require(isinstance(embedded,dict),'Report lacks embedded geometry proof')
    require(embedded=={k:v for k,v in g.items() if k!='samples'},'Report embedded geometry differs')
    is_v14=g.get('audit_version')=='closure_boundary_v14'
    require('audit_version' not in g or is_v14,'Unknown geometry audit version')
    if not is_v14:
        require(not any(g.get(k) for k in ['retained_original_v13_audit','resolved_closure_boundary_events','sample_overlay']),'Unversioned boundary evidence')
        baseline,events=check_full_geometry(g,trace,report)
        require(baseline and not events,'Canonical geometry fails independently reconstructed checks')
        return {'boundary_used':False,'geometry_audit_sha256':sha(final_path),'original_v13_passed':True,
            'geometry_samples':len(trace),'independent_cpu_event_reconstruction':False}
    require(report.get('actual36_geometry_audit_sha256')==sha(final_path),'v14 final certificate SHA not bound in expert report')
    retained=g['retained_original_v13_audit'];p=Path(retained['path']);p=p if p.is_absolute() else directory/p;p=p.resolve()
    require(p.is_relative_to(directory) and p!=final_path and p.is_file() and not p.is_symlink(),'Retained original must be a separate file in same trial')
    require(sha(p)==digest(retained['sha256'],'retained original'),'Retained original audit SHA mismatch')
    old=read_json(p);require(old['trace_sha256']==g['trace_sha256'],'Retained trace SHA differs')
    pre=g['retained_pre_actual_audit_report'];prepath=Path(pre['path']);prepath=prepath if prepath.is_absolute() else directory/prepath;prepath=prepath.resolve()
    require(prepath.is_relative_to(directory) and prepath not in [p,final_path,directory/'expert_report.json'] and not prepath.is_symlink(),'Pre-audit report must be separately retained in same trial')
    require(pre['original_bytes_unmodified'] is True and pre['matches_original_report_sha256'] is True,'Pre-audit original bytes not retained')
    require(sha(prepath)==pre['sha256']==old['report_sha256']==g['report_sha256'],'Pre-audit report SHA binding differs')
    original_report=read_json(prepath)
    require(original_report.get('accepted') is True and original_report.get('result')=='physics_success' and original_report.get('hold_passed') is True and original_report.get('hardware_io') is False,'Retained pre-audit physics report not originally successful simulation')
    for key in ['sample_spec','active_arm','source_hashes','physical_steps','full_qpos_joint_names']:
        require(original_report[key]==report[key],'Pre-audit report identity differs: '+key)
    baseline,events=check_full_geometry(old,trace,report)
    require(baseline and g['original_v13_passed'] is old['passed'],'Noncollision baseline/original pass evidence differs')
    resolved=g['resolved_closure_boundary_events'];used=bool(resolved)
    require(g['reviewed_original_non_touch_events']==resolved,'Review list contains other/non-resolved boundary events')
    require(g['boundary_used'] is used and g['boundary_resolved_event_count']==len(resolved) and report['boundary_resolved_event_count']==len(resolved),'Explicit boundary use/count differs')
    require(report.get('boundary_used',used) is used,'Expert boundary_used claim differs')
    if not used:
        require(old['passed'] is True and not events,'False old audit cannot pass without boundary proof')
    else:
        require(old['passed'] is False and len(resolved)==1,'Boundary original failed or unique event evidence differs')
        occurrences=[(i,l) for i,s in enumerate(old['samples']) for l in s['bottle_non_touch_collisions']]
        require(len(occurrences)==1 and len(events)==1 and events[0]['category']=='bottle_robot_non_touch' and events[0]['count']==1,'Boundary is not unique pair/frame across whole trial')
        ev=resolved[0];ix,link=occurrences[0];row=trace[ix]
        require(ix>0 and ev['trace_index']==ix and ev['step']==row['step'] and ev['link']==link and ev['resolved'] is True,'Resolved event identity differs')
        evidence=native_event(trace[ix-1],row,link,report['active_arm'],old['grasp_touch_normal_impulse_threshold_Ns'])
        declared=ev['native_evidence'];require(declared.get('eligible') is True and declared.get('link')==link,'Native evidence identity differs')
        for k,v in evidence.items():require(declared.get(k)==v,'Native contact evidence differs: '+k)
        numbers=validate_depth(ev['depth_certificate'],row,link,old['native_mesh_metadata'])
    helper=Path(g['boundary_helper_path']);require(helper.is_file() and sha(helper)==g['boundary_helper_sha256']==HELPER_SHA,'Frozen v14 helper source differs')
    constructor=helper.parent/'benchmark_geometry_cache.py';require(sha(constructor)==CONSTRUCTOR_SHA,'Frozen CPU constructor changed')
    identity=g['frozen_trial_identity']
    require(identity=={'run_config_sha256':sha(directory/'run_config.yml'),'scene_sample_sha256':sha(directory/'scene_sample.json'),
        'selected_side':report['active_arm'],'model_id':scene['sample_spec']['model_id'],'qpose_tag':0 if report['active_arm']=='left' else 1},'Frozen config/model/arm identity differs')
    require(g['source_hashes']==old['source_hashes']==report['source_hashes'],'Boundary source registries differ')
    require(all(Path(p).is_file() and sha(p)==v for p,v in g['source_hashes'].items()),'Boundary current source pin mismatch')
    identity_checks=g['current_geometry_identity_checks'];require(set(identity_checks)==set(META_KEYS) and all(v is True for v in identity_checks.values()),'Native geometry identity evidence missing')
    require(g['original_noncollision_checks_unchanged'] is True and g['baseline_all_other_checks_passed'] is True,'Other geometry checks changed')
    require(g['original_v13_collision_events']==old['collision_events'],'Original collision events modified')
    mutable={'samples','passed','collision_events','audit_version','original_v13_passed','boundary_resolution_policy','original_v13_collision_events',
        'resolved_closure_boundary_events','reviewed_original_non_touch_events','original_noncollision_checks_unchanged','baseline_all_other_checks_passed',
        'current_geometry_identity_checks','cpu_review_fk_audit','frozen_trial_identity','boundary_helper_sha256','boundary_helper_path','acceptance_scope',
        'retained_original_v13_audit','physical_snapshot_receipt','original_report_recorded_sha256','current_rejection_report_sha256','current_rejection_report_unmodified',
        'samples_retained_by_reference','sample_overlay','unchanged_sample_count','original_audit_unmodified'}
    require(all(k in g and g[k]==v for k,v in old.items() if k not in mutable),'Unrelated original geometry fields changed')
    rows=list(old['samples'])
    if used:
        new=dict(rows[ix]);new['bottle_non_touch_collisions']=[];new['reconciled_zero_load_closure_boundary_links']=[link];rows[ix]=new
    if 'samples' in g:require(g['samples']==rows,'Final sample changes outside unique boundary')
    else:require(g.get('sample_overlay')==([rows[ix]] if used else []) and g.get('unchanged_sample_count')==len(trace)-int(used),'Retained sample overlay differs')
    require(g['collision_events']==rebuild_events(rows,old['actual_self_normal_impulse_threshold_Ns'],old['bottle_table_first_release'])==[],'Remaining collision events exist')
    receipt={'boundary_used':used,'geometry_audit_sha256':sha(final_path),'retained_original_audit_sha256':sha(p),
        'retained_original_audit_file':str(p.relative_to(directory)),'original_v13_passed':old['passed'],'geometry_samples':len(trace),
        'boundary_helper_sha256':HELPER_SHA,'cpu_constructor_sha256':CONSTRUCTOR_SHA,'unchanged_sample_count':len(trace)-int(used),
        'independent_cpu_event_reconstruction':False}
    if used:
        receipt.update(event_step=row['step'],event_link=link,**numbers)
        require(cpu_verify,'Boundary requires independent zero-step CPU shape and numerical reconstruction')
        from dataset50_boundary_cpu_v2 import verify_cpu_event
        receipt['independent_cpu_event_proof']=verify_cpu_event(directory,row,link,old['native_mesh_metadata'],ev['depth_certificate'])
        receipt['independent_cpu_event_reconstruction']=True
    return receipt