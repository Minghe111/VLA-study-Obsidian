"""Read-only CPU audit of baseline/C2200 native K1 paired rollout receipts."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import math
import numpy as np

ROOT=Path('/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/recap_memory_20261017/experiments/protocol_repair_20261019_v2/reports/postfit_validation_20261019_v1')
issues=[]
def require(ok,message):
    if not ok:issues.append(message)
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def scalar_stats(values):return {'mean':float(np.mean(values)),'min':float(np.min(values)),'max':float(np.max(values))}
conditions={name:read(ROOT/name/'report.json') for name in ['old_task_baseline','old_task_c2200']}
out={'status':'running','scope':'Independent CPU-only report/transition audit; no model weights or GPU/simulator load','issues':issues,'conditions':{},'pairs':{},'files':{}}
all_rows={}
for name,report in conditions.items():
    require(report['status']=='completed',name+' not complete')
    require(report['K_exec']==1 and report['H']==16 and report['physical_action_dim']==24 and report['camera_count']==3,name+' contract differs')
    require(report['success_mode']=='native' and report['MEM'] is False and report['high_level_generation'] is False,name+' oldtask conditions differ')
    require(report['human_or_expert_intervention'] is False and report['training'] is False and report['optimizer_updates']==0,name+' intervention/training differs')
    require(report['parameters_unchanged'] is True and report['parameter_versions_unchanged'] is True and report['no_parameter_gradients'] is True,name+' parameter protection failed')
    require(report['parameter_hash_before']==report['parameter_hash_after'],name+' parameter hashes differ')
    require(len(report['episodes'])==2 and {e['seed'] for e in report['episodes']}=={2010,2013},name+' seedset differs')
    require(report['language_adapters_enabled_during_rollout']==(0 if name=='old_task_baseline' else 72),name+' adapter flags differ')
    receipt=report['load_receipt']
    require(receipt['constructed_language_attention_adapters']==72 and receipt['trainable_manifest_tensors']==144 and receipt['trainable_manifest_numel']==3686400,name+' manifest differs')
    if name=='old_task_baseline':
        require(receipt['all_language_B_zero'] is True and receipt['source_original_sft'].endswith('/models/pi05_80000_openpi'),name+' wrongSFTsource')
    else:
        require(receipt['strict_recap_full_load'] is True and receipt['all144_loaded_adapters_exact'] is True and receipt['source_global_step']==2200 and receipt['nonzero_language_B_matrices']==72 and receipt['full_state_key_count']==1131,name+' missingrealC2200load')
    detailed=[]
    for episode in report['episodes']:
        seed=episode['seed'];p=ROOT/name/f'seed_{seed}'/'transitions.jsonl';rows=[json.loads(l) for l in p.read_text().splitlines() if l.strip()];all_rows[(name,seed)]=rows
        require(sha(p)==episode['transitions_sha256'],name+f' seed{seed}: traceSHA differs')
        require(len(rows)==400 and [r['query_index'] for r in rows]==list(range(400)),name+f' seed{seed}: wrongquerycount')
        require(episode['actual_control_steps']==400 and episode['actual_physics_ticks']==6000 and episode['actual_sim_time_s']==24.0,name+f' seed{seed}: summaryclock differs')
        require(episode['status']=='completed' and episode['workers_closed'] is True,name+f' seed{seed}: incomplete/workernotclosed')
        require(episode['success'] is False and episode['termination_reason']=='time_limit',name+f' seed{seed}: expectedreportedtimeout')
        metrics=[];timestamps=[];intervention_keys=[]
        for index,row in enumerate(rows):
            info=row['info'];clock=info['action_trace'];m=info['episode_metrics'];metrics.append(m)
            require(np.asarray(row['proposal_sdk24_H16']).shape==(16,24),name+f' seed{seed} q{index}: proposalshape')
            require(np.asarray(row['executed_sdk24']).shape==(24,),name+f' seed{seed} q{index}: targetshape')
            require(np.asarray(row['input']['measured_state24']).shape==(24,) and np.asarray(row['next_measured_state24']).shape==(24,),name+f' seed{seed} q{index}: stateshape')
            require(np.isfinite(row['proposal_sdk24_H16']).all() and np.isfinite(row['executed_sdk24']).all(),name+f' seed{seed} q{index}: nonfiniteactions')
            require(np.array_equal(row['executed_sdk24'],row['proposal_sdk24_H16'][0]),name+f' seed{seed} q{index}: notfirstpredictedaction')
            require(row['noise_seed']==7+index,name+f' seed{seed} q{index}: noiseseed')
            require(row['input']['instruction']==report['instruction'],name+f' seed{seed} q{index}: changedinstruction')
            require(info['executed_action_count']==1 and info['valid_action_mask']==[True],name+f' seed{seed} q{index}: actioncount')
            require(info['step_count']==index+1 and info['physics_step_count']==15*(index+1),name+f' seed{seed} q{index}: clockstep')
            require(math.isclose(info['sim_time_s'],.06*(index+1),abs_tol=1e-8),name+f' seed{seed} q{index}: simtime')
            require(info['episode_contract']['seed']==seed,name+f' seed{seed} q{index}: auto-reset')
            require(len(clock)==1 and clock[0]['executed_physics_steps']==15 and clock[0]['requested_physics_steps']==15 and clock[0]['physics_step_start']==15*index and clock[0]['physics_step_end']==15*(index+1),name+f' seed{seed} q{index}: actualphysicalclock')
            require(np.array_equal(clock[0]['raw_model_action24'],row['executed_sdk24']),name+f' seed{seed} q{index}: targetchangedbeforeclock')
            require(not info['episode_terminated'] and info['episode_truncated']==(index==399),name+f' seed{seed} q{index}: terminalsequence')
            require(row['terminated']==[[False]] and row['truncated']==[[index==399]],name+f' seed{seed} q{index}: terminalarrays')
            require(row['reward']==[[0.0]],name+f' seed{seed} q{index}: reward')
            if index+1<len(rows):require(np.array_equal(row['next_measured_state24'],rows[index+1]['input']['measured_state24']),name+f' seed{seed}: statecontinuity')
        require(rows[0]['input']==episode['initial_observation'],name+f' seed{seed}: initialobsdiffers')
        require(np.array_equal(rows[-1]['next_measured_state24'],episode['final_observation']['measured_state24']),name+f' seed{seed}: finalobsdiffers')
        arm_dims=list(range(6))+list(range(12,18));hand_dims=list(range(6,12))+list(range(18,24))
        states=np.asarray([r['input']['measured_state24'] for r in rows]+[rows[-1]['next_measured_state24']])
        result={'seed':seed,'queries':len(rows),'controls':400,'physics_ticks':6000,'sim_seconds':24.,'native_success_count':sum(m['native_success'] for m in metrics),'opposed_contact_steps':sum(m['opposed_contact'] for m in metrics),'both_finger_groups_above_threshold_steps':sum(m['thumb_contact_N']>.05 and m['finger_contact_N']>.05 for m in metrics),'max_thumb_contact_N':max(m['thumb_contact_N'] for m in metrics),'max_finger_contact_N':max(m['finger_contact_N'] for m in metrics),'maximum_sampled_bottom_gap_m':max(m['bottom_gap_m'] for m in metrics),'minimum_sampled_upright_angle_deg':min(m['upright_angle_deg'] for m in metrics),'maximum_sampled_lift_hold_s':max(m['hold_seconds'] for m in metrics),'final_metrics':metrics[-1],'measured_arm_rad_max_delta_from_reset':float(np.abs(states[:,arm_dims]-states[0,arm_dims]).max()),'measured_hand_SDK_max_delta_from_reset':float(np.abs(states[:,hand_dims]-states[0,hand_dims]).max()),'query_wall_s':scalar_stats([r['query_wall_s'] for r in rows]),'environment_step_wall_s':scalar_stats([r['environment_step_wall_s'] for r in rows]),'frame_count':len(episode['frames']),'all_saved_frames_exist':all((ROOT/name/f'seed_{seed}'/f).is_file() for f in episode['frames']),'workers_closed_receipt':episode['workers_closed']}
        require(result['all_saved_frames_exist'],name+f' seed{seed}: missingframefile')
        detailed.append(result);out['files'][str(p)]=sha(p)
    out['conditions'][name]={'SR':'0/2','readiness':'No successful baseline episodes, so task-level retention is NOT established','model_path':report['model_path'],'weights_source':report['source_weights'],'norm_stats':report['norm_stats'],'load_receipt':receipt,'parameter_hash_before':report['parameter_hash_before'],'parameter_hash_after':report['parameter_hash_after'],'parameters_unchanged':report['parameters_unchanged'],'no_parameter_gradients':report['no_parameter_gradients'],'enabled_language_attention_adapters':report['language_adapters_enabled_during_rollout'],'peak_allocated_mib':report['peak_allocated_mib'],'peak_reserved_mib':report['peak_reserved_mib'],'sampled_whole_card_peak_mib':report['sampled_whole_card_peak_mib'],'episodes':detailed}
    out['files'][str(ROOT/name/'report.json')]=sha(ROOT/name/'report.json')
base=conditions['old_task_baseline'];candidate=conditions['old_task_c2200']
require(base['norm_stats']==candidate['norm_stats'],'paired normalization differs')
require(base['environment_config']==candidate['environment_config'] or {k:v for k,v in base['environment_config'].items() if k!='runtime_dir'}=={k:v for k,v in candidate['environment_config'].items() if k!='runtime_dir'},'pairedenvironmentdiffersbeyondruntime')
require(base['parameter_hash_before']['False']==candidate['parameter_hash_before']['False'],'frozenbaseparametersdiffer')
for seed in [2010,2013]:
    left=next(e for e in base['episodes'] if e['seed']==seed);right=next(e for e in candidate['episodes'] if e['seed']==seed)
    a=all_rows[('old_task_baseline',seed)];b=all_rows[('old_task_c2200',seed)]
    hashes={k:left['initial_observation']['rgb_sha256'][k]==right['initial_observation']['rgb_sha256'][k] for k in left['initial_observation']['rgb_sha256']}
    state_exact=left['initial_observation']['measured_state24']==right['initial_observation']['measured_state24']
    noise_equal=[x['noise_sha256']==y['noise_sha256'] for x,y in zip(a,b)]
    require(state_exact and all(hashes.values()),f'seed{seed}: pairedinitialinputsnotexact')
    require(all(noise_equal),f'seed{seed}: actualpairednoisediffers')
    require(base['planned_scenes'][str(seed)]==candidate['planned_scenes'][str(seed)],f'seed{seed}: plannedscenediffers')
    first=np.asarray(a[0]['executed_sdk24'])-np.asarray(b[0]['executed_sdk24']);arms=list(range(6))+list(range(12,18));hands=list(range(6,12))+list(range(18,24))
    out['pairs'][str(seed)]={'initial_measured_state24_exact':state_exact,'initial_RGB_SHA256_exact':hashes,'same_scene_spec':True,'actual_noise_SHA256_equal':sum(noise_equal),'queries_compared':400,'first_action_same_input_arm_rad_rms':float(np.sqrt((first[arms]**2).mean())),'first_action_same_input_hand_SDK_rms':float(np.sqrt((first[hands]**2).mean())),'later_trajectories_expected_to_diverge':True}
out['status']='passed_integrity_but_task_retention_unverified' if not issues else 'failed_integrity'
out['interpretation']=['Four native K1 episodes actually completed400 controls/6000 physics ticks/24 simulated seconds without takeover.','Baseline0/2 and C2200 0/2 do not establish skill retention or noninferiority.','Every episode has zero opposed-contact samples and zero lift/contact hold; neither policy demonstrated a sustained grasp or lift on these seeds.','Right seed2010 bottle tipped (final angle83.93 versus90.32degrees); left seed2013 remained nearly upright but ungrasped. Initial upright orientation is not task success.','C2200 loaded all144 adapter tensor values exactly according to executable strict-load receipt;72 learned B matrices nonzero. Baseline disabled all adapters. Frozen base hash same between conditions.','Checks use actual saved60ms boundary observations/metrics; no250Hz dense object-state trace is available. Worker-close confirmation is the stored live is_alive check, not a new independent PID enumeration.','A separate same-bridge K16 matched diagnostic may isolate execution-policy sensitivity but must not replace K1 outcomes or be equated to historical JAX TOPP.']
path=Path('/tmp/old_task_pair_independent_audit.json');path.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':out['status'],'issues':issues,'pairs':out['pairs'],'physical':[{'condition':v,'episodes':[{k:e[k] for k in ['seed','opposed_contact_steps','max_thumb_contact_N','max_finger_contact_N','maximum_sampled_bottom_gap_m','maximum_sampled_lift_hold_s']} for e in x['episodes']]} for v,x in out['conditions'].items()]},ensure_ascii=False))
if issues:raise SystemExit(1)
