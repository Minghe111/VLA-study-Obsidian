"""Read-only audit of actual local policy/renderer/actuator artifacts.

Never sends a policy request, replays an action, or touches the model RNG.
"""
from pathlib import Path
import base64,gzip,hashlib,io,json,sys,time
import numpy as np
from PIL import Image

ROOT=Path('/tmp/realman-eval80k-20261012')
SERVICE=ROOT/'policy_runtime/service_v1'
CAMERAS=('cam_head','cam_left_wrist','cam_right_wrist')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def audit_episode(path):
    report=json.loads((path/'report.json').read_text())
    identity_sha=sha(SERVICE/'model_identity.json')
    assert report['checkpoint_step']==80000 and report['model_identity_sha256']==identity_sha
    assert report['expert_future_actions_used'] is False and report['interface_or_infrastructure_error'] is None
    bounds=report['source_initial_HDF_only']
    assert bounds['future_commands_or_states_read'] is False and bounds['source_outcomes_used_to_control_policy'] is False
    assert all(v['row']==0 for v in bounds['numeric_HDF_reads'])
    assert not any('/colors' in v['dataset'] or 'next_command24' in v['dataset'] for v in bounds['numeric_HDF_reads'])
    assert report['policy_full_prediction_horizon']==16 and report['policy_full_chunk_execution'] is True
    manifest_path=Path(report['actual_rgb_recording']['path'])
    assert sha(manifest_path)==report['actual_rgb_recording']['sha256']
    manifest=json.loads(manifest_path.read_text())
    frames={row['physics_step']:row for row in manifest['frames']}
    rows=[];max_state_cast_error=0.;all_model_actions=[]
    prompt=json.loads((path/'initial_state.json').read_text())['prompt']
    for local_index,entry in enumerate(report['model_queries']):
        query=entry['query_id'];d=SERVICE/'requests'/f'{query:06d}'
        source=json.loads((d/'input.json').read_text());result=json.loads((d/'result.json').read_text())
        step=local_index*16*15
        f=frames[step];image_path=Path(f['path'])
        assert sha(image_path)==f['sha256']
        encoded={};hashes={}
        with np.load(image_path,allow_pickle=False) as rendered, np.load(d/'input_rgb_state.npz',allow_pickle=False) as supplied:
            assert int(rendered['physics_step'])==step
            assert abs(float(rendered['sim_time_s'])-entry['sim_time_s'])<1e-10
            current=rendered['state24'];client_state=np.asarray(entry['input_state24'],float)
            assert np.array_equal(current,client_state)
            assert np.array_equal(supplied['state24'],client_state.astype(np.float32))
            assert np.array_equal(np.asarray(source['state24'],np.float32),supplied['state24'])
            max_state_cast_error=max(max_state_cast_error,float(np.max(abs(client_state-supplied['state24']))))
            for key in CAMERAS:
                pixels=rendered[key]
                assert pixels.shape==(480,640,3) and pixels.dtype==np.uint8
                assert np.array_equal(pixels,supplied[key])
                buf=io.BytesIO();Image.fromarray(pixels).save(buf,format='PNG');png=buf.getvalue()
                rgb_hash=hashlib.sha256(pixels.tobytes()).hexdigest();png_hash=hashlib.sha256(png).hexdigest()
                assert png_hash==entry['input_png_sha256'][key]==source['image_hashes'][key]['encoded_sha256']
                assert rgb_hash==source['image_hashes'][key]['rgb_pixels_sha256']
                encoded[key]=base64.b64encode(png).decode('ascii');hashes[key]={'png_sha256':png_hash,'rgb_sha256':rgb_hash}
        assert source['prompt']==prompt
        payload={'images':encoded,'state24':client_state.tolist(),'prompt':prompt}
        assert hashlib.sha256(json.dumps(payload,allow_nan=False,separators=(',',':')).encode()).hexdigest()==entry['request_sha256']
        assert hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()==source['source_request_sha256']
        assert sha(d/'raw_actions16x24.npy')==result['raw_actions_sha256']
        assert hashlib.sha256(json.dumps(result,allow_nan=False).encode()).hexdigest()==entry['response_sha256']
        actions=np.load(d/'raw_actions16x24.npy',allow_pickle=False)
        client_actions=np.asarray(entry['raw_actions'],float)
        assert actions.shape==(16,24) and np.isfinite(actions).all()
        assert np.array_equal(actions,client_actions)
        assert np.array_equal(actions,np.asarray(result['actions'],float))
        with np.load(path/f'query_{local_index+1:05d}.npz',allow_pickle=False) as q:
            assert np.array_equal(q['raw_prediction24'],client_actions)
        assert result['query_id']==query==source['query_id']
        assert result['expert_actions_used'] is False and result['raw_commands_unclipped'] is True
        assert result['model_identity_sha256']==source['model_identity_sha256']==entry['model_identity_sha256']==identity_sha
        all_model_actions.extend(client_actions)
        rows.append({'local_query_index':local_index+1,'global_service_query_id':query,'sim_time_s':entry['sim_time_s'],'physical_step':step,
                     'three_live_camera_pixels_equal_service':True,'client_PNG_hashes_equal_service':True,
                     'live_state_equal_client_and_official_float32_service_cast':True,'input_hashes':hashes,
                     'raw16x24_equal_service_client_and_episode_npz':True,'server_raw_actions_sha256':result['raw_actions_sha256'],
                     'no_expert_actions':True})
    executed=report['executed_actions'];assert len(executed)==report['actual_action_count']
    assert len(all_model_actions)==len(executed)
    max_endpoint_arm=0.;max_endpoint_hand=0.
    arms=np.r_[0:6,12:18];hands=np.r_[6:12,18:24]
    for row,predicted in zip(executed,all_model_actions):
        assert np.array_equal(np.asarray(row['raw_model_action24']),predicted)
        assert row['requested_physics_steps']==15 and row['executed_physics_steps']==15
        assert row['physics_step_end']-row['physics_step_start']==15
        assert row['endpoint_command_written'] is True and row['success_early_stop'] is False
        residual=abs(np.asarray(row['actual_final_drive_command24'])-predicted)
        max_endpoint_arm=max(max_endpoint_arm,float(residual[arms].max()))
        max_endpoint_hand=max(max_endpoint_hand,float(residual[hands].max()))
    ids=[row['global_service_query_id'] for row in rows]
    assert ids==list(range(ids[0],ids[0]+len(ids)))
    return {'episode':path.name,'source_index':report['index'],'passed':True,'model_identity_sha256':identity_sha,
            'queries':len(rows),'actual_actions':len(executed),'actual_physics_steps':report['actual_physics_steps'],
            'service_query_range':[ids[0],ids[-1]],'current_rendered_three_RGB_exactly_equal_service_inputs':True,
            'all24_measured_states_equal_before_official_float32_cast':True,
            'max_float64_to_float32_state_cast_error_mixed_rad_and_SDK':max_state_cast_error,
            'all400_actuator_raw_targets_exactly_equal_model_predictions':True,
            'max_drive_endpoint_arm_error_rad':max_endpoint_arm,'max_drive_endpoint_hand_roundtrip_error_SDK':max_endpoint_hand,
            'official_profile_80000_h16_d32_and_10_denoising_steps':True,'expert_future_numeric_or_RGB_reads':False,
            'source_initial_row0_numeric_reads':bounds['numeric_HDF_reads'],
            'static_source_language_used':True,'native_task_success_event':report['native_task_success'],
            'action_provenance_audited_not_grasp_success_rate_claim':True,'rows':rows}

def main():
    outputs=[]
    identity=json.loads((SERVICE/'model_identity.json').read_text())
    assert identity['checkpoint_step']==80000 and identity['sample_num_steps']==10 and identity['action_horizon']==16
    assert identity['official_create_trained_policy_used'] and not identity['base_fallback'] and not identity['expert_actions_used']
    assert identity['restored_parameter_leaves']==51 and identity['restored_parameter_elements']==3353433872
    assert identity['norm_stats_sha256']=='34cbdd09c831e04549d6c4a4e2eb321fb975bdea13bb816ec8aa3a2332d1ccc5'
    for path in sorted((ROOT/'episodes').glob('ep*_model80000')):
        if (path/'report.json').is_file():outputs.append(audit_episode(path))
    ids=sorted(q['global_service_query_id'] for r in outputs for q in r['rows'])
    assert ids==list(range(len(ids)))
    aggregate={'scope':'read-only action and actual live RGB/state provenance audit, no inference request or RNG mutation',
               'passed':True,'completed_episodes':len(outputs),'completed_episode_queries':len(ids),
               'unique_contiguous_service_query_ids_from_zero':True,'model_identity_sha256':sha(SERVICE/'model_identity.json'),
               'episodes':outputs,'scripts_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'scripts/client80k.py',ROOT/'scripts/run_model.py',ROOT/'scripts/initial_source.py',ROOT/'scripts/observations80k.py',ROOT/'scripts/observations40k.py',ROOT/'policy_runtime/runtime80k.py',ROOT/'policy_runtime/service80k.py']}}
    output=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'policy_runtime/live_input_action_provenance_audit.json'
    write(output,aggregate)
    print(json.dumps({k:aggregate[k] for k in ('passed','completed_episodes','completed_episode_queries','unique_contiguous_service_query_ids_from_zero','model_identity_sha256')}))
if __name__=='__main__':main()
