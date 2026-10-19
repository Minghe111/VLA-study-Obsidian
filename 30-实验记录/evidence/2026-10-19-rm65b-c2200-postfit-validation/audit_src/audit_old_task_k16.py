"""Independent CPU-only audit of matched old-task K16 rollout evidence."""
import argparse
from pathlib import Path
import hashlib
import json
import math
import numpy as np

RUN=Path('/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/recap_memory_20261017/experiments/protocol_repair_20261019_v2/reports/postfit_validation_20261019_v1')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline-only',action='store_true');args=p.parse_args()
    issues=[]
    def require(ok,message):
        if not ok:issues.append(message)
    names=['old_task_baseline_k16'] if args.baseline_only else ['old_task_baseline_k16','old_task_c2200_k16']
    reports={n:read(RUN/n/'report.json') for n in names}
    out={'status':'running','scope':'CPU-only inspection of execution receipts; no policy/simulator/GPU load, no new training','K_exec':16,'issues':issues,'conditions':{},'pairs':{},'files':{}}
    all_rows={}
    for name,x in reports.items():
        is_base=name=='old_task_baseline_k16';require(x['status']=='completed',name+': status not completed')
        require(x['K_exec']==16 and x['H']==16 and x['physical_action_dim']==24 and x['camera_count']==3,name+': contractdiffers')
        require(x['success_mode']=='native' and x['MEM'] is False and x['high_level_generation'] is False,name+': mode differs')
        require(x['human_or_expert_intervention'] is False and x['training'] is False and x['optimizer_updates']==0,name+': intervention/training differs')
        require(x['parameters_unchanged'] is True and x['parameter_versions_unchanged'] is True and x['no_parameter_gradients'] is True,name+': mutation or gradients')
        require(x['parameter_hash_before']==x['parameter_hash_after'],name+': parameterhash differs')
        require(x['language_adapters_enabled_during_rollout']==(0 if is_base else 72),name+': adapterflags differ')
        receipt=x['load_receipt']
        require(receipt['trainable_manifest_tensors']==144 and receipt['trainable_manifest_numel']==3686400 and receipt['constructed_language_attention_adapters']==72,name+': manifest differs')
        if is_base:require(receipt['all_language_B_zero'] is True,name+': baseline nonzeroB')
        else:require(receipt['strict_recap_full_load'] and receipt['all144_loaded_adapters_exact'] and receipt['source_global_step']==2200 and receipt['nonzero_language_B_matrices']==72 and receipt['full_state_key_count']==1131,name+': incomplete C2200load')
        episodes=[]
        for e in x['episodes']:
            seed=e['seed'];path=RUN/name/f'seed_{seed}'/'transitions.jsonl';rows=[json.loads(l) for l in path.read_text().splitlines() if l.strip()];all_rows[(name,seed)]=rows
            require(sha(path)==e['transitions_sha256'],name+f' seed{seed}: SHA differs')
            require(len(rows)==25 and [r['query_index'] for r in rows]==list(range(25)),name+f' seed{seed}: actualqueries')
            require(e['actual_control_steps']==400 and e['actual_physics_ticks']==6000 and e['actual_sim_time_s']==24.0,name+f' seed{seed}: actualepisodeclock')
            require(e['workers_closed'] is True and e['status']=='completed',name+f' seed{seed}: workercompletion')
            metrics=[];all_clocks=[];controls=0;ticks=0
            for index,row in enumerate(rows):
                info=row['info'];proposal=np.asarray(row['proposal_sdk24_H16']);requested=np.asarray(row['requested_sdk24_chunk']);actual=np.asarray(row['actually_executed_sdk24']);valid=np.asarray(info['valid_action_mask'],dtype=bool);clock=info['action_trace'];metrics.append(info['episode_metrics']);all_clocks.extend(clock)
                require(proposal.shape==(16,24) and requested.shape==(16,24) and actual.shape==(16,24),name+f' seed{seed} q{index}: SDK24 shapes')
                require(row['requested_chunk_shape']==[1,16,24] and valid.tolist()==[True]*16 and info['executed_action_count']==16,name+f' seed{seed} q{index}: submitted/validshape')
                require(np.isfinite(proposal).all() and np.array_equal(proposal,requested) and np.array_equal(actual,requested),name+f' seed{seed} q{index}: changed actionprefix')
                require(row['noise_seed']==7+index,name+f' seed{seed} q{index}: noiseindex')
                require(row['input']['instruction']==x['instruction'],name+f' seed{seed} q{index}: changedprompt')
                require(row['control_step_start']==controls and row['control_step_end']==controls+16 and info['step_count']==controls+16,name+f' seed{seed} q{index}: controlclock')
                require(len(clock)==16,name+f' seed{seed} q{index}: targetclockcount')
                for offset,c in enumerate(clock):
                    require(c['requested_physics_steps']==15 and c['executed_physics_steps']==15 and c['physics_step_start']==ticks+15*offset and c['physics_step_end']==ticks+15*(offset+1),name+f' seed{seed} q{index} a{offset}: physicalclock')
                    require(np.array_equal(c['raw_model_action24'],requested[offset]),name+f' seed{seed} q{index} a{offset}: rawSDK differs')
                    require(c['endpoint_command_written'] is True and c['success_early_stop'] is False,name+f' seed{seed} q{index} a{offset}: fakeendpoint')
                controls+=16;ticks+=240
                require(info['physics_step_count']==ticks and math.isclose(info['sim_time_s'],ticks*.004,abs_tol=1e-8) and info['episode_contract']['seed']==seed,name+f' seed{seed} q{index}: simclock/reset')
                require(row['postchunk_episode_metrics']==info['episode_metrics'],name+f' seed{seed} q{index}: metricscopy differs')
                require(row['chunk_execution_receipt']=={'executed_target_count':16,'executed_physics_ticks':240,'per_target_physics_ticks':[15]*16,'partial_final_target':False,'padding_target_count':0},name+f' seed{seed} q{index}: chunkexecutionreceipt')
                expect_trunc=[False]*16
                if index==24:expect_trunc[-1]=True
                require(row['reward']==[[0.0]*16] and row['terminated']==[[False]*16] and row['truncated']==[expect_trunc],name+f' seed{seed} q{index}: publicoutcomearrays')
                require(info['chunk_rewards']==[0.0]*16 and info['chunk_terminated']==[False]*16 and info['chunk_truncated']==expect_trunc,name+f' seed{seed} q{index}: receiptoutcomearrays')
                require(info['episode_terminated'] is False and info['episode_truncated']==(index==24),name+f' seed{seed} q{index}: episodestatus')
                if index+1<len(rows):require(row['next_measured_state24']==rows[index+1]['input']['measured_state24'],name+f' seed{seed} q{index}: statecontinuity')
            require(rows[0]['input']==e['initial_observation'],name+f' seed{seed}: initialobs mismatch')
            require(rows[-1]['next_measured_state24']==e['final_observation']['measured_state24'],name+f' seed{seed}: finalstate mismatch')
            require(controls==400 and ticks==6000,name+f' seed{seed}: exactbudget exceeded/missing')
            require(e['success'] is False and e['termination_reason']=='time_limit',name+f' seed{seed}: expected actualtimeout outcome')
            frames_exist=all((RUN/name/f'seed_{seed}'/f).is_file() for f in e['frames']);require(frames_exist,name+f' seed{seed}: framefilemissing')
            stats={'seed':seed,'queries':25,'actual_control_targets':controls,'actual_physics_ticks':ticks,'actual_sim_seconds':24.,'native_success':e['success'],'termination_reason':e['termination_reason'],'postchunk_observation_samples':25,'monitored_opposed_contact_postchunk_samples':sum(m['opposed_contact'] for m in metrics),'maximum_observed_thumb_contact_N':max(m['thumb_contact_N'] for m in metrics),'maximum_observed_finger_contact_N':max(m['finger_contact_N'] for m in metrics),'maximum_observed_postchunk_bottom_gap_m':max(m['bottom_gap_m'] for m in metrics),'minimum_observed_postchunk_upright_angle_deg':min(m['upright_angle_deg'] for m in metrics),'maximum_observed_postchunk_lift_hold_seconds':max(m['hold_seconds'] for m in metrics),'final_metrics':metrics[-1],'all400_clock_endpoints_written':all(c['endpoint_command_written'] for c in all_clocks),'workers_closed_receipt':e['workers_closed'],'frames_saved':len(e['frames']),'all_frames_exist':frames_exist}
            # Compare with the previous K1 run at the shared initial state only;
            # later queries happen at different control times and are not paired.
            k1_name='old_task_baseline' if is_base else 'old_task_c2200';k1_path=RUN/k1_name/f'seed_{seed}'/'transitions.jsonl'
            if k1_path.is_file():
                first=json.loads(k1_path.read_text().splitlines()[0]);stats['same_initial_as_same_condition_K1']={'observation_exact':first['input']==rows[0]['input'],'noise_sha_exact':first['noise_sha256']==rows[0]['noise_sha256'],'first_proposal_H16_exact':np.array_equal(first['proposal_sdk24_H16'],rows[0]['proposal_sdk24_H16'])}
            episodes.append(stats);out['files'][str(path)]=sha(path)
        out['conditions'][name]={'SR':'0/2','episodes':episodes,'parameter_hash_before':x['parameter_hash_before'],'parameter_hash_after':x['parameter_hash_after'],'no_parameter_gradients':x['no_parameter_gradients'],'load_receipt':receipt,'source_weights':x['source_weights'],'norm_stats':x['norm_stats'],'peak_allocated_mib':x['peak_allocated_mib'],'peak_reserved_mib':x['peak_reserved_mib'],'sampled_whole_card_peak_mib':x['sampled_whole_card_peak_mib']}
        out['files'][str(RUN/name/'report.json')]=sha(RUN/name/'report.json')
    if not args.baseline_only:
        base=reports[names[0]];student=reports[names[1]]
        require(base['norm_stats']==student['norm_stats'],'pairednormalization differs')
        require(base['parameter_hash_before']['False']==student['parameter_hash_before']['False'],'pairedfrozenbase differs')
        for seed in [2010,2013]:
            a=all_rows[(names[0],seed)];b=all_rows[(names[1],seed)]
            initial_exact=a[0]['input']==b[0]['input'];noise=[x['noise_sha256']==y['noise_sha256'] for x,y in zip(a,b)]
            require(initial_exact and all(noise),f'seed{seed}: pairedinputs/noise notsame')
            require(base['planned_scenes'][str(seed)]==student['planned_scenes'][str(seed)],f'seed{seed}: pairedscene differs')
            out['pairs'][str(seed)]={'initial_observation_state_and_3RGB_SHA_exact':initial_exact,'noise_SHA_matches':sum(noise),'queries_compared':25,'planned_scene_exact':True}
    out['interpretation']=['BothK16 conditions, if present, are matched withinK16 and kept separate fromK1.','Baseline has no demonstrated successful task episode; equal0/2 cannot establish old skill retention or noninferiority.','Raw action arrays and actual per-target clocks are fully checked, but object/contact metrics are only sampled at chunk boundaries. No claim that unobserved withinchunk contacts never occurred.','Current evidence does not attribute failed behavior to LoRA, task competence, cameras, conversion or control policy.','This fixed60ms K16 bridge remains different from historical JAX TOPP variable physical timing.']
    out['status']='passed_integrity_task_retention_unverified' if not issues else 'failed_integrity'
    output=Path('/tmp/old_task_k16_baseline_audit.json' if args.baseline_only else '/tmp/old_task_k16_pair_independent_audit.json');output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':out['status'],'issues':issues,'output':str(output),'conditions':{n:{'SR':r['SR'],'episodes':r['episodes']} for n,r in out['conditions'].items()},'pairs':out['pairs']},ensure_ascii=False))
    if issues:raise SystemExit(1)
if __name__=='__main__':main()
