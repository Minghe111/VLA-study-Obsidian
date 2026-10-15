from pathlib import Path
import json,gzip,hashlib,math,shutil,subprocess
import numpy as np,h5py,cv2
ROOT=Path(__file__).resolve().parent
DEST=Path('/home/admin123/liminghe/VLA-WAM-RL/30-实验记录/证据/2026-10-11-RealMan专家回放')
DEST.mkdir(parents=True,exist_ok=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows=[]
for i in [0,38,30,31,17,7]:
    folder=ROOT/f'episodes/ep{i:02d}_current60';report=json.load(open(folder/'report.json'));initial=json.load(open(folder/'initial_state.json'))
    trace=[json.loads(x) for x in gzip.open(folder/'actual_physical_trace.jsonl.gz','rt')]
    assert not report['interface_or_infrastructure_error'] and len(trace)==report['actual_physics_steps']
    first=np.load(folder/'actual_rgb/observation_000001_step_000000.npz');q=np.array([x['actual_qpos36_rad'] for x in trace]);names=[d['joint'] for d in initial['actual_physical_contract']['drives']]
    armidx=[j for j,n in enumerate(names) if n.startswith(('l_joint','r_joint'))];handidx=[j for j in range(36) if j not in armidx]
    errors={};pixel={}
    with h5py.File(report['source_hdf']) as f:
        source=f['realman_dexterous_physics/full36_qpos_after_rad'][:]
        for key,n in [('cam_head','head_camera'),('cam_left_wrist','left_camera'),('cam_right_wrist','right_camera')]:
            # HDF "extrinsics_matrix" stores camera world model matrix, not CV view matrix.
            pose=first[n+'_world_GL_pose'];model=np.stack([-pose[:,1],pose[:,2],-pose[:,0],pose[:,3]],axis=1)
            errors[key]=float(abs(model-f['vision/'+key+'/extrinsics_matrix'][0]).max())
            rgb=cv2.imdecode(np.frombuffer(bytes(f['vision/'+key+'/colors'][0]),np.uint8),cv2.IMREAD_COLOR)[...,::-1]
            pixel[key]=float(abs(rgb.astype(float)-first[key]).mean())
        command_match=all(np.array_equal(np.asarray(a['raw_model_action24']),t) for a,t in zip(report['executed_actions'],f['realman_dexterous/next_command24'][:]))
    axis=np.array(trace[-1]['bottle_world_pose'])[:3,1];upright=math.degrees(math.acos(np.clip(axis[2]/np.linalg.norm(axis),-1,1)))
    contact=report['observation']['first_positive_hand_contact_step'];before=trace[:contact-1] if contact else trace
    row={'index':i,'source_episode':report['source_episode'],'native_success':report['native_task_success'],'air_grasp_1cm_1s':report['passed'],
        'continuous_air_hold_s':report['observation']['longest_observed_air_grasp_s'],'maximum_bottom_gap_m':report['observation']['maximum_native_bottom_gap_m'],
        'terminal_bottom_gap_m':trace[-1]['native_bottom_gap_m'],'terminal_upright_error_deg':upright,'physical_steps':len(trace),'duration_s':report['physical_duration_s'],
        'source_initial_qpos_error_rad':initial['qpos_max_abs_error_rad'],'source_initial_state24_error':initial['state24_max_abs_error'],
        'source_initial_bottle_pose_matrix_error':initial['source_initial_pose_matrix_max_abs_error'],'source_initial_camera_world_model_matrix_error':errors,
        'source_vs_new_first_RGB_mean_abs_pixel_difference':pixel,'recorded_target_values_exactly_used':command_match,
        'max_pre_hand_contact_bottle_xy_displacement_m':max(x['initial_xy_displacement_m'] for x in before),'max_pre_hand_contact_bottle_rotation_deg':math.degrees(max(x['initial_rotation_rad'] for x in before)),
        'max_arm_actual_qpos_difference_from_source_rad':float(abs(q[:,armidx]-source[:,armidx]).max()),
        'max_hand_actual_qpos_difference_from_source_rad':float(abs(q[:,handidx]-source[:,handidx]).max()),
        'source_hand_contact_offset_m':initial['source_hand_contact_offset_m'],'replay_robot_contact_offsets_m':sorted(set(s['contact_offset_m'] for s in initial['actual_physical_contract']['shapes'])),
        'no_contact_setting_changes':not initial['contact_offset_changes'],'no_hardware_io':not report['hardware_io'],'model_queries':report['model_query_count'],
        'actual_trace_sha256':sha(folder/'actual_physical_trace.jsonl.gz'),'report_sha256':sha(folder/'report.json'),'initial_state_sha256':sha(folder/'initial_state.json'),'source_hdf_sha256':report['source_hdf_sha256']}
    assert command_match and row['source_initial_qpos_error_rad']==0 and row['source_initial_state24_error']==0 and max(errors.values())<1e-6 and row['no_contact_setting_changes']
    target=DEST/f'episode_{i:07d}';target.mkdir()
    for n in ['report.json','initial_state.json','actual_physical_trace.jsonl.gz']:shutil.copy2(folder/n,target/n)
    proof=ROOT/f'source/episodes/episode_{i:07d}'
    for n in ['scene_sample.json','sample_spec.json','run_config.yml','recording_report.json','dataset_validation.json']:shutil.copy2(proof/n,target/('source_'+n))
    rows.append(row)
for n in ['source_inventory.json','selected_transfer.json','proof_transfer.json']:shutil.copy2(ROOT/n,DEST/n)
shutil.copytree(ROOT/'scripts',DEST/'replay_scripts');shutil.copytree(ROOT/'collector_sources',DEST/'read_only_collector_sources')
repo=Path('/home/admin123/liminghe/vla-benchmark/RoboTwin')
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
result={'verified_date':'2026-10-11','scope':'Six selected train50 original expert command24 labels, restored source initial scenes, current inference actuator and physics, new physical RGB; not learned-policy or full50 SR','robotwin_commit':head,
    'current_action_clock_sha256':sha(repo/'integrations/realman_rm65b_gen4/action_clock.py'),'base_task_snapshot_sha256':sha(Path('/tmp/realman-eval60k-20261011/local_eval/scripts/remote_base_task_snapshot.py')),
    'source_manifest_sha256':sha(ROOT/'source_manifest.json'),'hardware_guard_sha256':sha(Path('/tmp/realman-eval60k-20261011/local_eval/hardware_guard.json')),
    'selected_indices':[r['index'] for r in rows],'native_success_count':sum(r['native_success'] for r in rows),'air_hold_success_count':sum(r['air_grasp_1cm_1s'] for r in rows),'total_physics_steps':sum(r['physical_steps'] for r in rows),
    'initialization_limits':['Source initial qvel and original PhysX contact cache absent; robot qvel=0','Initial bottle pose restored once, zero velocity and native sleep before episode; source episode7 explicitly records sleep/zero velocity, others inferred from settled data','No pose/state teleport or forced sleep during replay','Same initial geometry/camera/joints is verified; RGB differs by JPEG/noise/denoising, not bit-identical resumption'],
    'setup_errors_not_episode_failures':['ep0 setup helper used wrong Actor API before physics; fixed','ep38 reconstructed temporary bootstrap pose was unstable; changed creation to saved exact original pose; all final episode starts match recorded state'],
    'source_expert_hand_offset_difference':'Sprite source 1mm vs current default10mm; source code confirms change. Both Sprite current replay succeeded; not sufficient explanation for low model success by itself.',
    'source_feedback_recomputed':False,'model_queries':0,'hardware_io':False,'server_master_closed':not (ROOT/'ssh.sock').exists(),
    'interpretation':'Demonstration actions can execute in current integration for these six scenes. Prior model failures should prioritize predicted action/hand timing/input transforms/generalization; cannot infer all50 data is correct or isolate the learned-model failure cause without model replay on the same restored scenes.',
    'episodes':rows}
(DEST/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps({'summary_path':str(DEST/'summary.json'),'summary_sha256':sha(DEST/'summary.json'),'native':result['native_success_count'],'held':result['air_hold_success_count'],'episodes':[{k:r[k] for k in ['index','continuous_air_hold_s','terminal_upright_error_deg','max_pre_hand_contact_bottle_xy_displacement_m','max_arm_actual_qpos_difference_from_source_rad','max_hand_actual_qpos_difference_from_source_rad']} for r in rows]},indent=2))
