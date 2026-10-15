"""Synthetic logical tamper tests; never create a training episode.

Fixture state/geometry is copied from a historical failed record for predicate
coverage only. It is explicitly synthetic, lacks HDF/images, and cannot pass the
real dataset entrypoint. Original source trial remains unchanged.
"""
from pathlib import Path
import argparse,copy,json,sys,time
import numpy as np
from dataset50_audit import read_json,sha
import dataset50_boundary_v2 as boundary


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    src=a.root/'batch_candidates/left_coke_attempt00/slot0000008_attempt00'
    originals={n:sha(src/n) for n in ['expert_report.json','physics_trace.json','actual36_geometry_audit.json','scene_sample.json','run_config.yml']}
    trace=read_json(src/'physics_trace.json');old=read_json(src/'actual36_geometry_audit.json');cert=read_json(a.root/'closure_boundary_slot8_v14_certificate_final.json')
    event=cert['resolved_closure_boundary_events'][0];ix=event['trace_index'];link=event['link'];depth=event['depth_certificate'];tests=[]
    def run(name,fn,*,reject=False):
        try:
            fn();passed=not reject;error=None
        except (ValueError,KeyError,TypeError,AssertionError) as exc:
            passed=reject;error=str(exc)
        tests.append({'name':name,'passed':passed,'expected_rejection':reject,'error':error})
        print(name,passed,error,flush=True)
    def native_case(change):
        p,c=copy.deepcopy(trace[ix-1]),copy.deepcopy(trace[ix]);target=[link,'left'];change(p,c,target)
        return boundary.native_event(p,c,*target,old['grasp_touch_normal_impulse_threshold_Ns'])
    run('actual_native_transition_baseline',lambda:native_case(lambda p,c,t:None))
    run('approach_rejected',lambda:native_case(lambda p,c,t:c['phase'].update(name='approach')),reject=True)
    run('gripper_false_rejected',lambda:native_case(lambda p,c,t:c['phase'].update(gripper=False)),reject=True)
    run('different_close_call_rejected',lambda:native_case(lambda p,c,t:c['phase'].update(dense_call=4)),reject=True)
    run('nonadjacent_frame_rejected',lambda:native_case(lambda p,c,t:c.update(step=c['step']+1)),reject=True)
    run('different_model_rejected',lambda:native_case(lambda p,c,t:c['bottle'].update(model_id=16)),reject=True)
    run('different_side_tag_rejected',lambda:native_case(lambda p,c,t:c['bottle'].update(qpose_tag=1)),reject=True)
    for label in ['little','ring']:
        run('passive_'+label+'_rejected',lambda label=label:native_case(lambda p,c,t:t.__setitem__(0,link.replace('thumb',label))),reject=True)
    run('palm_rejected',lambda:native_case(lambda p,c,t:t.__setitem__(0,'l_inspire_palm_force_sensor')),reject=True)
    run('inactive_arm_rejected',lambda:native_case(lambda p,c,t:t.__setitem__(1,'right')),reject=True)
    run('missing_exact_points_rejected',lambda:native_case(lambda p,c,t:c.update(bottle_actual_contacts=[])),reject=True)
    def zero_prev(p,c,t):
        for contact in p['bottle_actual_contacts']:
            if contact['other_link']==link:
                contact['normal_impulse_Ns']=0
                for x in contact['points']:x.update(normal_impulse_Ns=0,impulse_Ns=[0,0,0])
    run('no_prior_real_normal_rejected',lambda:native_case(zero_prev),reject=True)
    def current_load(p,c,t):
        contact=next(x for x in c['bottle_actual_contacts'] if x['other_link']==link);point=contact['points'][0];normal=np.asarray(point['normal_world'])
        point['normal_impulse_Ns']=1e-8;point['impulse_Ns']=(normal*(1e-8/(normal@normal))).tolist();contact['normal_impulse_Ns']=1e-8
    run('current_positive_load_rejected',lambda:native_case(current_load),reject=True)
    def negative_sep(p,c,t):next(x for x in c['bottle_actual_contacts'] if x['other_link']==link)['points'][0]['separation_m']=-1e-12
    run('negative_native_separation_rejected',lambda:native_case(negative_sep),reject=True)
    run('real_depth_enumeration_baseline',lambda:boundary.validate_depth(depth,trace[ix],link,old['native_mesh_metadata']))
    def depth_case(change):
        d=copy.deepcopy(depth);change(d);return boundary.validate_depth(d,trace[ix],link,old['native_mesh_metadata'])
    run('depth_nan_rejected',lambda:depth_case(lambda d:d.update(measured_penetration_depth_m=float('nan'))),reject=True)
    run('depth_inf_budget_rejected',lambda:depth_case(lambda d:d.update(numerical_vertex_budget_m=float('inf'))),reject=True)
    run('missing_shape_pair_rejected',lambda:depth_case(lambda d:d['shape_pair_depths'].pop()),reject=True)
    run('missing_precision_probe_rejected',lambda:depth_case(lambda d:d['event_specific_vertex_certificate']['native_fk_signed_gap_probes'].pop()),reject=True)
    run('truncated_contacts_rejected',lambda:depth_case(lambda d:d['shape_pair_depths'][4]['probes'][0].update(contact_count=512)),reject=True)
    run('native_signed_gap_beyond_pure_conversion_rejected',lambda:depth_case(lambda d:d['event_specific_vertex_certificate']['native_fk_signed_gap_probes'][0].update(signed_distance_m=-1e-6)),reject=True)
    run('false_native_nonnegative_claim_rejected',lambda:depth_case(lambda d:d['event_specific_vertex_certificate'].update(native_signed_gap_nonnegative=True)),reject=True)
    run('event_depth_fk_mismatch_rejected',lambda:depth_case(lambda d:d['event_specific_vertex_certificate'].update(event_auxiliary_to_depth_auxiliary_vertex_error_m=1e-8)),reject=True)
    run('event_depth_fk_negative_rejected',lambda:depth_case(lambda d:d['event_specific_vertex_certificate'].update(event_auxiliary_to_depth_auxiliary_vertex_error_m=-1e-8)),reject=True)
    run('initial_budget_substitution_rejected',lambda:depth_case(lambda d:d.update(numerical_vertex_budget_m=d['measured_initial_fk_translation_budget_m']+d['rotation_vertex_budget_m'])),reject=True)
    # A 3-frame explicitly synthetic v9 packet; NOT the original complete physical trace.
    mini_trace=copy.deepcopy([trace[ix-1],trace[ix],trace[-1]])
    mini_old={k:copy.deepcopy(v) for k,v in old.items() if k!='samples'}
    mini_old['samples']=copy.deepcopy([old['samples'][ix-1],old['samples'][ix],old['samples'][-1]])
    pix=1
    for i,(r,g) in enumerate(zip(mini_trace,mini_old['samples'])):
        r['step']=g['step']=i+1;g['trace_index']=i
    mini_old['recorded_samples']=3
    mini_old['bottle_table_first_release']={'step':3,'trace_index':2,'distance_m':mini_old['samples'][-1]['bottle_table_geometry']['distance_m']}
    mini_old['collision_events']=[{'category':'bottle_robot_non_touch','pair':['bottle',link],'count':1,'first_step':2,'last_step':2,'first_trace_index':1}]
    mini_old['max_post_grasp_relative_translation_drift_m']=max([g['relative_translation_drift_m'] for g in mini_old['samples'] if 'relative_translation_drift_m' in g] or [0.])
    mini_old['max_post_grasp_relative_rotation_drift_deg']=max([g['relative_rotation_drift_deg'] for g in mini_old['samples'] if 'relative_rotation_drift_deg' in g] or [0.])
    mini_cert=copy.deepcopy(cert);mini_cert['recorded_samples']=3;mini_cert['original_v13_collision_events']=copy.deepcopy(mini_old['collision_events'])
    for k in ['bottle_table_first_release','max_post_grasp_relative_translation_drift_m','max_post_grasp_relative_rotation_drift_deg']:
        mini_cert[k]=copy.deepcopy(mini_old[k])
    for key in ['resolved_closure_boundary_events','reviewed_original_non_touch_events']:
        ev=mini_cert[key][0];ev.update(step=2,trace_index=1);ev['native_evidence'].update(previous_step=1,current_step=2)
        vv=ev['depth_certificate']['event_specific_vertex_certificate'];vv['event_step']=vv['scene_receipt']['event_step']=2
    mini_cert['sample_overlay'][0].update(step=2,trace_index=1);mini_cert['unchanged_sample_count']=2
    # All physical-source hashes remain frozen; packet files below are fixture-only.
    fixture=a.out/'synthetic_protocol_fixture_NOT_DATASET';fixture.mkdir()
    def packet(change=lambda o,g,r,t,s:None):
        o,g,r,t,s=copy.deepcopy(mini_old),copy.deepcopy(mini_cert),copy.deepcopy(read_json(src/'expert_report.json')),copy.deepcopy(mini_trace),copy.deepcopy(read_json(src/'scene_sample.json'))
        r.update(accepted=True,result="physics_success",hold_passed=True,physical_steps=3,synthetic_fixture_only=True,test_scope='Synthetic v9 proof protocol only; not historical slot8 or a training episode');r.pop('error',None)
        g.update(boundary_used=True,boundary_resolved_event_count=1);r.update(boundary_used=True,boundary_resolved_event_count=1)
        g['samples']=copy.deepcopy(o['samples']);g['samples'][pix]=copy.deepcopy(mini_cert['sample_overlay'][0])
        change(o,g,r,t,s)
        g['reviewed_original_non_touch_events']=copy.deepcopy(g['resolved_closure_boundary_events'])
        def put(name,value):
            p=fixture/name;p.write_text(json.dumps(value,indent=2)+'\n');return p
        tp=put('physics_trace.json',t);sp=put('scene_sample.json',s);(fixture/'run_config.yml').write_bytes((src/'run_config.yml').read_bytes())
        pre={k:v for k,v in r.items() if k not in ['actual36_geometry_audit','actual36_geometry_audit_sha256','boundary_used','boundary_resolved_event_count']};pp=put('expert_report_pre_actual_audit.json',pre)
        o['trace_sha256']=g['trace_sha256']=sha(tp);o['report_sha256']=g['report_sha256']=sha(pp)
        op=put('actual36_geometry_audit_v13_original.json',o);g['retained_original_v13_audit']={'path':str(op),'sha256':sha(op)}
        g['retained_pre_actual_audit_report']={'path':str(pp),'sha256':sha(pp),'original_bytes_unmodified':True,'matches_original_report_sha256':True}
        g['frozen_trial_identity'].update(run_config_sha256=sha(fixture/'run_config.yml'),scene_sample_sha256=sha(sp))
        gp=put('actual36_geometry_audit.json',g);r['actual36_geometry_audit']={k:v for k,v in g.items() if k!='samples'};r['actual36_geometry_audit_sha256']=sha(gp);put('expert_report.json',r)
        return boundary.audit_geometry_chain(fixture)
    run('synthetic_v9_protocol_full_CPU_baseline_only',packet)
    run('explicit_boundary_flag_mismatch_rejected',lambda:packet(lambda o,g,r,t,s:r.update(boundary_used=False)),reject=True)
    run('explicit_boundary_count_mismatch_rejected',lambda:packet(lambda o,g,r,t,s:g.update(boundary_resolved_event_count=2)),reject=True)
    run('final_other_sample_change_rejected',lambda:packet(lambda o,g,r,t,s:g['samples'][0]['phase'].update(name='tampered')),reject=True)
    run('finger_content_fingerprint_tamper_rejected_CPU',lambda:packet(lambda o,g,r,t,s:g['resolved_closure_boundary_events'][0]['depth_certificate']['event_specific_vertex_certificate'].update(finger_collision_content_sha256='0'*64)),reject=True)
    run('whole_trial_second_frame_event_rejected',lambda:packet(lambda o,g,r,t,s:(o['samples'][pix+1]['bottle_non_touch_collisions'].append(link),o['collision_events'][0].update(count=2,last_step=t[pix+1]['step']))),reject=True)
    run('whole_trial_second_pair_event_rejected',lambda:packet(lambda o,g,r,t,s:(o['samples'][pix]['bottle_non_touch_collisions'].append('l_inspire_left_index_4'),o['collision_events'].append({'category':'bottle_robot_non_touch','pair':['bottle','l_inspire_left_index_4'],'count':1,'first_step':2,'last_step':2,'first_trace_index':1}))),reject=True)
    run('source_pin_tamper_rejected',lambda:packet(lambda o,g,r,t,s:r['source_hashes'].__setitem__(str(a.root/'expert_generation.py'),'0'*64)),reject=True)
    from dataset50_audit_v2 import audit_trial
    run('synthetic_packet_cannot_enter_real_dataset',lambda:(_ for _ in ()).throw(ValueError('Expected rejection: '+audit_trial(fixture)['error'])) if not audit_trial(fixture)['passed'] else None,reject=True)
    require=all(x['passed'] for x in tests) and all(sha(src/n)==v for n,v in originals.items())
    report={'passed':require,'scope':'Synthetic predicate/tamper/protocol fixtures only; copied failed-record geometry is not a training episode',
        'test_count':len(tests),'tests':tests,'source_failed_slot8_unmodified':True,'new_physics_runs':0,'gpu_runs':0,'hardware_io':False,
        'fixture_not_dataset':True,'actual_successful_episodes_generated':0,'boundary_validator_sha256':sha(a.root/'dataset50_boundary_v2.py'),
        'CPU_verifier_sha256':sha(a.root/'dataset50_boundary_cpu_v2.py')}
    (a.out/'synthetic_tests.json').write_text(json.dumps(report,indent=2)+'\n');print('SUMMARY',require,len(tests),flush=True)
    return 0 if require else 2

if __name__=='__main__':raise SystemExit(main())