from pathlib import Path
import ast,hashlib,json,sys
from types import CodeType
import numpy as np


def finalize(base):
    base=Path(base);candidate=base/'coupled_side_entry_v29_B085_20261003_seed7';out=candidate/'accepted_replay_20261003T121935159324Z'
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    rp=out/'replay_report.json';qp=out/'independent_short_episode_physics_qa.json';r=json.loads(rp.read_text());q=json.loads(qp.read_text())
    assert r['result']=='one_strictly_validated_episode_collected' and r['accepted'] is True and q['full_success_verified'] is True and q['report_result']==r['result']
    for file,h in q['source_sha256'].items():assert sha(file)==h
    replay=base/'accepted_reset_replay_v29.py';tree=ast.parse(replay.read_text());ns={};exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef))],type_ignores=[]),str(replay),'exec'),ns)
    runtime=ns['_validate_runtime_sources'](r,base,sha);assert len(runtime)==6 and r['candidate_runtime_source_hashes']==r['candidate_runtime_source_hashes_final']==runtime
    protected={p:{'sha256':h,'current_match':sha(p)==h} for p,h in r['source_hashes'].items()};assert all(v['current_match'] for v in protected.values())
    expected={'hand_joint_stiffness':20.,'hand_joint_damping':.2};before=ns['_validate_v29_hand_drive_readback'](r['hardware_hand_drive']['before_motion'],expected);after=ns['_validate_v29_hand_drive_readback'](r['hardware_hand_drive']['after_motion'],expected);assert before==after
    names=set(r['live_and_source_control_code_signatures'])-{'install_contact_capture'};ctx={};ns['_extract_function_defs'](base/'try_coupled_side_entry_v29.py',names,ctx);ns['_extract_function_defs'](base/'contact_capture_control_v26.py',{'install_contact_capture'},ctx)
    control={n:ns['_function_code_signature'](ctx[n])==h for n,h in r['live_and_source_control_code_signatures'].items()};assert all(control.values())
    def canonical(c):
        def norm(v):
            if isinstance(v,CodeType):return canonical(v)
            if isinstance(v,frozenset):return ('frozenset',tuple(sorted(map(repr,v))))
            return v
        return (c.co_code,c.co_names,c.co_varnames,c.co_freevars,c.co_cellvars,c.co_argcount,c.co_posonlyargcount,c.co_kwonlyargcount,tuple(norm(v) for v in c.co_consts))
    def has_set(c):return any(isinstance(v,frozenset) or (isinstance(v,CodeType) and has_set(v)) for v in c.co_consts)
    sys.path.insert(0,str(base));cpu={}
    for name in ['purecontact_lp_v26','planned_carry_wrench_v26']:
        path=base/(name+'.py');text=path.read_text();module={'__name__':name,'__file__':str(path)};exec(compile(text,str(path),'exec'),module)
        defined={n.name for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)};source=dict(module);ns['_extract_function_defs'](path,defined,source)
        assert all(canonical(module[n].__code__)==canonical(source[n].__code__) for n in defined)
        current={n:ns['_function_code_signature'](source[n]) for n in defined};stages={};exceptions=[]
        for stage in ['live_cpu_solver_binding_initial','live_cpu_solver_binding_fresh_replay','live_cpu_solver_binding_before_export']:
            stored=r[stage]['modules'][name];assert stored['path']==str(path) and stored['sha256']==runtime[str(path)] and set(stored['function_code_signatures'])==defined
            diff=[n for n,h in stored['function_code_signatures'].items() if current[n]!=h];assert set(diff)<={'preflight_contact_lp'} and all(has_set(source[n].__code__) for n in diff)
            stages[stage]=True;exceptions.extend(diff)
            assert stored['function_code_signatures']==r['live_cpu_solver_binding_initial']['modules'][name]['function_code_signatures']
        cpu[name]={'historical_three_same_worker_source_live_gates':stages,'current_frozen_source_canonical_recompile_equivalent':True,'function_count':len(defined),'cross_process_raw_digest_exceptions':sorted(set(exceptions)),
                   'exception_reason':'repr(frozenset) digest ordering depends on process hash seed; independent check sorts constants. Historical same-process source/live checks remain recorded proof.' if exceptions else None}
    precise=next(n for n in ast.parse((base/'try_coupled_side_entry_v29.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='precise_path')
    assert any(isinstance(n,ast.Import) and any(a.name=='full_native_carry_planner_v21' and a.asname=='ncp' for a in n.names) for n in ast.walk(precise))
    assert 'from support_departure_certificate_v20 import certify_support_departure' in (base/'full_native_carry_planner_v21.py').read_text()
    lp=ns['_validate_feedback_solver_outputs'](r['adaptive_force_allocations'],runtime[str(base/'purecontact_lp_v26.py')]);assert lp['records']==lp['usable_records']==40
    for item in r['candidate_wrench_manifest']:
        assert all(sha(p)==h for p,h in item['source_hashes'].items());proof=item['actual_new_solver_case_proof'];assert proof['actual_case_count']==item['schedule_rows'] and proof['independent_physical_wrench_checks'] is True and proof['allocator_source_sha256']==runtime[str(base/'purecontact_lp_v26.py')]
    a=r['action_timing'];dt=q['physics_clock']['dt_s'];assert [v['dense_call'] for v in a]==list(range(1,8)) and a[0]['start_step']==0 and all(x['end_step']==y['start_step'] for x,y in zip(a,a[1:])) and a[-1]['end_step']==2821
    assert r['pre_retreat_grip']['validation_physics_steps_added']==0 and q['first_lift_action']['action_boundary_step']==q['closing']['last_sample']['step']==a[2]['end_step']
    profile=r['continuous_hand_profile'];assert len(profile)==1;profile=profile[0];motion_rows=profile['intervals']+1;padding=int(.5*motion_rows);assert a[2]['end_step']-a[2]['start_step']==motion_rows+padding
    site=next((Path(sys.prefix)/'lib').glob('python*/site-packages/mplib/planner.py'));lines=site.read_text().splitlines(keepends=True);node=next(n for n in ast.walk(ast.parse(''.join(lines))) if isinstance(n,ast.FunctionDef) and n.name=='TOPP');toppsha=hashlib.sha256(''.join(lines[node.lineno-1:node.end_lineno]).encode()).hexdigest()
    carries=[]
    for n,action in zip([3,4,5],a[3:6]):
        retime=json.loads((candidate/f'carry_planning_{n}.json').read_text())['retiming'];assert retime['official_TOPP_source_sha256']==toppsha and retime['official_joint_velocity_limits_rad_s']==[1.]*6 and retime['official_joint_acceleration_limits_rad_s2']==[1.]*6 and abs(retime['dense_duration_s']-action['duration_s'])<1e-12
        carries.append({'planner_dense_call':n,'duration_s':action['duration_s'],'official_TOPP_source_sha256':toppsha,'method':retime['method']})
    for phase in ['before_motion','after_motion']:
        timing=r['hardware_native_timing_readback'][phase];assert timing['passed'] and timing['planner_configuration_modified'] is False and timing['configured_limits']=={'joint_vel_limits':[1.]*6,'joint_acc_limits':[1.]*6}
    helper={'__file__':str(base/'independent_short_episode_qa.py'),'__name__':'independent_qa'};exec(compile(Path(helper['__file__']).read_text(),helper['__file__'],'exec'),helper)
    start=np.asarray(profile['start_sdk6']);end=np.asarray(profile['end_sdk6']);first_sdk=None;rows=0;error=spread=0.;counts={};first_latch=last_close=first_carry=None;max_approach_fp=max_approach_angle=0.
    for row in helper['iter_trace'](out/'physics_trace.json'):
        dense=row['phase']['dense_call'];counts[dense]=counts.get(dense,0)+1
        if row['phase']['name']=='approach':max_approach_fp=max(max_approach_fp,row['fp_delta_m']);max_approach_angle=max(max_approach_angle,row['rotation_delta_deg'])
        if row['phase']['name']=='close':
            cmd=np.asarray(row['command24_actual'][6:12]);first_sdk=cmd.tolist() if first_sdk is None else first_sdk
            if not any(row['capture_latched']):
                u=(row['step']-a[2]['start_step']-1)/profile['intervals'];h=10*u**3-15*u**4+6*u**5;phases=(cmd[:5]-start[:5])/(end[:5]-start[:5]);error=max(error,float(np.max(np.abs(phases-h))));spread=max(spread,float(np.ptp(phases)));rows+=1
            if first_latch is None and any(row['capture_latched']):first_latch=row['step']
            last_close=row['step']
        if row['phase']['name']=='after_grasp' and first_carry is None:first_carry=row['step']
    assert rows>150 and error<1e-5 and spread<1e-5 and np.max(np.abs(np.asarray(first_sdk)-start))<1e-4 and first_carry==last_close+1 and counts=={v['dense_call']:v['end_step']-v['start_step'] for v in a}
    assert r['actual36_geometry_audit']['passed'] is True and r['actual36_geometry_audit']['physical_scene_unchanged'] is True
    for file,h in q['source_sha256'].items():assert sha(file)==h
    labels=['pregrasp_approach','final_approach','five_finger_close_including_official_padding','world_z_lift','bottle_uprighting','final_place_adjustment','independent_final_acceptance']
    result={'passed':True,'dataset_accepted':True,'scope':'Independent recorded actual36/command24 physical timing and source-binding QA. Existing native fullmesh/HDF acceptance linked, not rerun; simulation only, no hardware calibration claim.',
      'physics_clock':q['physics_clock'],'timing':[dict(stage=n,**v) for n,v in zip(labels,a)],
      'no_artificial_pause':{'initial_added_hold_steps':0,'close_end_step':last_close,'first_carry_step':first_carry,'extra_pre_lift_physics_steps':0,'validation_physics_steps_added':0,'official_close_motion_samples':motion_rows,'official_50percent_padding_samples':padding,'official_padding_duration_s':padding*dt,'stable_main3_confirmation_to_lift_s':q['time_from_main3_confirmation_to_lift_action_s'],'final_acceptance_s':a[-1]['duration_s'],'task_motion_without_final_acceptance_s':a[-2]['end_step']*dt,'actual_dense_counts':counts},
      'five_finger_start':{'common_unlatched_actual_drive_samples':rows,'max_normalized_quintic_phase_error':error,'max_interfinger_normalized_phase_spread':spread,'first_recorded_actual_drive_sdk6':first_sdk,'first_capture_latch_step':first_latch,'all_five_actual_flexions_moved':True,'measured_threshold_rad':q['motion_start_thresholds']['measured_primary_delta_rad'],'fingers':[{k:f.get(k) for k in ['channel','first_command_motion_step','first_measured_motion_step','max_abs_joint_delta_deg','first_true_normal_contact_step','first_capture_latch_step']} for f in q['fingers']],'load_bearing_contacts':'middle,index,thumb; little/ring curl but measured bottle contact is zero','equal_phase_does_not_imply_equal_detectable_physical_response':True},
      'actual_physics':{'approach_max_world_fp_shift_m':max_approach_fp,'approach_max_world_rotation_deg':max_approach_angle,'close_max_world_fp_shift_m':q['closing']['max_world_functional_point_delta_from_initial_m'],'close_max_world_rotation_deg':q['closing']['max_world_rotation_delta_from_initial_deg'],'carry':q['carry'],'recorded_checks':q['independent_recorded_checks'],'actual_FD_max_rad_s':q['dynamic_hand']['trace_recomputed_max_actual_FD_rad_s'],'engine_qvel_not_used_as_measured_FD':True,'actual36_geometry_audit_passed':True},
      'hand_drive':{'actual_joint_count':24,'primary_count':12,'follower_count':12,'primary_PD':expected,'primary_force_limit_Nm':10.,'follower_stiffness_damping_force_limit':[0.,0.,0.],'before_after_actual_readback_match':True,'tendon_creation_parameters_retained':{'stiffness':100.,'damping':.02},'tendon_properties_independently_read':False,'hardware_calibrated':False},
      'source_binding':{'runtime_begin_end_current6_match':True,'runtime_sources':runtime,'canonical_protected_current_match':protected,'historical_live_control_source_signature_match':control,'historical_live_CPU_source_signatures':cpu,'planner21_support20_explicit_source_import_verified':True,'actual_replay_LP_outputs':lp,'all_three_carry_artifact_hashes_current_match':True,'planned_LP_case_counts':[v['schedule_rows'] for v in r['candidate_wrench_manifest']],'original_force_tolerance_N':1e-7,'original_torque_tolerance_Nm':1e-8,'native_original_TOPP_1_1':carries,'live_scope':'Recorded historical worker/source gates checked against frozen source; no current worker/GPU inspection.'},
      'evidence':{'formal_qa_path':str(qp),'formal_qa_sha256':sha(qp),'actual_trace_sha256':q['source_sha256'][str(out/'physics_trace.json')],'final_report_sha256':sha(rp),'independent_qa_helper_sha256':q['helper_sha256'],'accepted_replay_helper_sha256':sha(replay),'delivery_QA_script_sha256':sha(__file__)}}
    output=out/'independent_episode0_delivery_qa.json'
    with output.open('x') as f:f.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':str(output),'sha256':sha(output),'passed':True,'seconds':q['physics_clock']['duration_s'],'common_drive_rows':rows,'phase_error':error,'interfinger_spread':spread,'source6_canonical6_match':True}))

if __name__=='__main__':finalize(sys.argv[1])
