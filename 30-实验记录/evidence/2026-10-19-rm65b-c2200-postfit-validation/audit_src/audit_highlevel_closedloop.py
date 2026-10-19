"""CPU-only independent audit of the two autonomous high-level v3 traces."""
from pathlib import Path
import sys
import hashlib
import json
import math
import numpy as np
import sentencepiece

ROOT=Path('/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/recap_memory_20261017/experiments/protocol_repair_20261019_v2')
RUN=ROOT/'reports/postfit_validation_20261019_v1/highlevel_closedloop'
sys.path.insert(0,str(ROOT/'code/RLinf'))
from rlinf.data.storage.episode_memory import EpisodeMemory

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    d=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):d.update(b)
    return d.hexdigest()
def np_sha(v):return hashlib.sha256(np.asarray(v).tobytes()).hexdigest()
issues=[]
def require(ok,text):
    if not ok:issues.append(text)
report=read(RUN/'report.json')
out={'status':'review_running','scope':'CPU-only source/JSON/NPZ/tokenizer audit; no model load, GPU, simulator, oracle or training','issues':issues,'outer_runner_status':report['status'],'case_results':[],'files':{},'version_label_findings':[]}
require(report['status']=='completed','outer runner notcompleted')
require(report['mode']=='autonomous_high_level' and report['optimizer_constructed'] is False and report['optimizer_updates']==0 and report['formal_3000_training'] is False and report['task_SR_claim'] is False,'scope flags differ')
require(report['contract']=={'SDK_dim':24,'cameras':3,'period_s':.06,'H':16,'K_exec':1},'contract differs')
require(report['all_parameters_unchanged'] is True and report['parameters_before']==report['parameters_after'],'parameter hashes changed')
require(report['all_parameter_versions_unchanged'] is True and report['parameter_version_counters_before']==report['parameter_version_counters_after'],'parameter mutation counter changed')
require(report['gradients_before']==report['gradients_after']==0,'gradients present')
require(report['all_source_files_unchanged'] is True and report['source_files_before']==report['source_files_after'],'source receiptchanged')
load=report['load_receipt'];require(load['strict_recap_full_load'] and load['all144_loaded_adapters_exact'] and load['nonzero_language_B_matrices']==72 and load['enabled_language_attention_adapters']==72 and load['source_global_step']==2200 and load['full_state_key_count']==1131,'not actualfullC2200')
require(report['checkpoint_audit']=={'global_step':2200,'trainable_numel':3686400,'trainable_tensors':144,'runtime_mem_enabled':False},'checkpoint scope differs')
require(report['motor_sampling_implementation']=={'sample_actions':'Pi0.sample_actions','predict_eval':'EnvIO._predict_eval','generate_text_tokens':'Pi0RecapHier.generate_text_tokens'},'motor/generation implementation differs')
# Current small source files are rehashed; selected checkpoint hash is supported
# by matching before/after execution receipts, not a second full-weights read.
for label,value in report['source_files_before'].items():
    path=Path(value['path']);require(path.is_file() and path.stat().st_size==value['bytes'],label+': sourcefile missing/sizechanged')
    if label!='selected_weights':require(sha(path)==value['sha256'],label+': current sourceSHA differs')
sp=sentencepiece.SentencePieceProcessor(model_file=report['source_files_before']['tokenizer']['path']);require(sp.eos_id()==1,'tokenizerEOS differs')
allowed_policy={'full_image','instruction','left_wrist_image','right_wrist_image','state'}
first_obs={};policy_versions=set();all_noise_seeds=[]
for entry in report['results']:
    case=entry['case'];cid=case['case_id'];base=RUN/cid
    require(entry['status']=='failed',cid+': case erroneously successful')
    require(entry['workers_closed'] is True and all(x['alive'] is False and x['exitcode']==0 for x in entry['worker_exit_receipts']),cid+': worker closurefailed')
    episode_path=base/'episode/failed_episode.json';episode=read(episode_path)
    require(not (base/'episode/episode.json').exists(),cid+': failed partial evidence exposed as successful episode.json')
    events=[json.loads(l) for l in (base/'policy_trace.jsonl').read_text().splitlines() if l.strip()]
    envs=[json.loads(l) for l in (base/'environment_info.jsonl').read_text().splitlines() if l.strip()]
    expected_order=[('generate','READ'),('generate','WRITE'),('generate','SUBTASK')]+[('actions',None)]*32+[('generate','READ')]
    require([(e['kind'],e.get('mode')) for e in events]==expected_order,cid+': policy sequence differs or output was repaired/continued')
    gen=[e for e in events if e['kind']=='generate'];actions=[e for e in events if e['kind']=='actions'];steps=episode['steps'];decisions=episode['decisions']
    require(len(steps)==len(envs)==len(actions)==32 and len(decisions)==1,cid+': actualcounts')
    require(episode['metadata']['split']=='heldout_eval' and episode['metadata']['strict_memory_protocol'] is True and episode['metadata']['mem_num_frames']==1,cid+': protocol/evalscope')
    require('Sub right right' in episode['metadata']['collection_error'] and 'READ must be one JSON object' in entry['error'],cid+': missing actualfailure')
    for event in events:
        require(set(event['policy_observation_keys'])==allowed_policy,cid+': privileged observation field leaked')
        require(event['inference_guard']=={'guard_enabled':True,'grad_enabled':False,'model_training':False},cid+': inference gradient guard')
        require('oracle_subtask_override' not in event,cid+': unexpectedoracle')
    for event in gen:
        ids=event['token_ids'];require(ids and ids[-1]==1 and ids.count(1)==1 and event['ended_with_eos'] is True,cid+': actualEOS missing')
        require(sp.decode(ids).strip()==event['text'].strip(),cid+': raw decoded text modified')
    require(gen[-1]['text']=='Sub right right' and gen[-1]['mode']=='READ',cid+': unexpected lastinvalidREAD')
    try:json.loads(gen[-1]['text']);last_invalid=False
    except json.JSONDecodeError:last_invalid=True
    require(last_invalid,cid+': lastREAD was validJSON')
    decision=decisions[0];policy_versions.add(decision['policy_version'])
    require(decision['start_step']==0 and decision['end_step']==32,cid+': decisionwindow')
    require(decision['generated_read_text']==gen[0]['text'] and decision['generated_write_text']==gen[1]['text'] and decision['generated_subtask']==gen[2]['text'],cid+': episode text differsrawgeneration')
    require(gen[2]['text']==f'Pick up and hold the sample initially on the {case["first_side"]}.',cid+': firstsubtaskwrongside')
    require(gen[2]['first_role_matches'] is True and gen[2]['generated_side']==case['first_side'],cid+': binding summary differs')
    require(gen[0]['context']['history']=='' and gen[-1]['context']['history']==gen[2]['text'],cid+': actualhistorytransition')
    bank=EpisodeMemory(decision['tool_inputs']['catalogue'],env_id='0',episode_id=cid,strict_protocol=True)
    require(bank.critic_view()==decision['memory_before'],cid+': initial bank not empty')
    read_ids=json.loads(gen[0]['text'])['read_ids'];selected=bank.read(read_ids)
    require(read_ids==['sample_identity'] and selected==decision['tool_outputs']['selected_memory']==gen[1]['context']['visible_memory'],cid+': actualREAD result differs')
    require(selected==[{'slot_id':'sample_identity','status':'empty','value':None,'confirmed':False,'revision':0}],cid+': READ not initiallyempty')
    operations=json.loads(gen[1]['text'])['operations'];require(len(operations)==1 and operations[0]['op']=='WRITE' and operations[0]['slot_id']=='sample_identity' and operations[0]['confirmed'] is True and operations[0]['value']==f'The first requested sample is initially on the {case["first_side"]}.',cid+': WRITE notexpectedrolefact')
    written=[bank.apply(op,source={'kind':'actor_generated','step':0,'policy_version':decision['policy_version']}) for op in operations]
    actor_memory=bank.actor_view(read_ids,list(dict.fromkeys(op['slot_id'] for op in operations)))
    require(bank.critic_view()==decision['memory_after'] and bank.revision==1 and written==decision['current_writes'],cid+': strict toolapply bankdiffers')
    require(actor_memory==decision['actor_memory']==gen[2]['context']['visible_memory'],cid+': SUBTASK did not receive actualupdatedmemory')
    # Raw observation arrays are re-read and matched to every motor input hash.
    observations={}
    for index in range(33):
        path=base/f'episode/observations/{index:06}.npz'
        with np.load(path,allow_pickle=False) as z:
            require(set(z.files)=={'state','full_image','left_wrist_image','right_wrist_image'},cid+f': obs{index} unexpectedfields')
            require(z['state'].shape==(24,) and z['state'].dtype==np.float32 and np.isfinite(z['state']).all(),cid+f': obs{index} invalidstate')
            pixels={}
            for key in ('full_image','left_wrist_image','right_wrist_image'):
                require(z[key].shape==(480,640,3) and z[key].dtype==np.uint8 and np.any(z[key]),cid+f': obs{index} invalidRGB')
                pixels[key]=np_sha(z[key])
            observations[index]={'state24':z['state'].tolist(),'state24_sha256':np_sha(z['state']),'rgb_sha256':pixels}
    first_obs[cid]=observations[0]
    for index,(event,env,step) in enumerate(zip(actions,envs,steps,strict=True)):
        info=env['info'];proposal=np.asarray(event['proposed_H16_SDK24'],dtype=np.float32);issued=np.asarray(env['issued_K1_SDK24']);clock=info['action_trace']
        require(event['call']==index and event['motor_query_index']==index and event['motor_noise_seed']==7+index,cid+f': action{index} noiseseed/query')
        require(event['shape']==[16,24] and proposal.shape==(16,24) and np.isfinite(proposal).all() and event['executed_prefix']==1,cid+f': action{index} proposalcontract')
        require(event['sha256']==np_sha(proposal),cid+f': action{index} proposalhash')
        require(event['actual_motor_input']==observations[index],cid+f': action{index} liveinputnotstoredrawarrays')
        require(event['context']=={'goal':case['goal'],'subtask':gen[2]['text'],'visible_memory':actor_memory},cid+f': action{index} actualconditiondiffers')
        require(np.array_equal(proposal,np.asarray(step['proposal_actions'],dtype=np.float32)) and np.array_equal(proposal[0],step['executed_action']) and issued.shape==(1,1,24) and np.array_equal(issued[0,0],proposal[0]),cid+f': action{index} changedSDKcommand')
        require(env['control']==index+1 and info['step_count']==index+1 and info['executed_action_count']==1 and info['valid_action_mask']==[True],cid+f': action{index} actualcontrolcount')
        require(info['physics_step_count']==15*(index+1) and math.isclose(info['sim_time_s'],.06*(index+1),abs_tol=1e-8),cid+f': action{index} actualphysicalclock')
        require(len(clock)==1 and clock[0]['requested_physics_steps']==clock[0]['executed_physics_steps']==15 and clock[0]['physics_step_start']==15*index and clock[0]['physics_step_end']==15*(index+1) and clock[0]['endpoint_command_written'] is True,cid+f': action{index} clockreceipt')
        require(np.array_equal(clock[0]['raw_model_action24'],proposal[0]) and step['metadata']['action_trace']==clock,cid+f': action{index} rawdriveSDKdiffers')
        require(step['step_index']==index and step['decision_id']==decision['decision_id'] and step['condition_id']==decision['decision_id'],cid+f': action{index} condition/versionlink')
        require(step['physics_steps']==15 and math.isclose(step['physics_dt_seconds'],.06) and step['env_reward']==0 and step['research_reward'] is None and step['terminated'] is False and step['truncated'] is False,cid+f': action{index} fabricatedreward/done')
        require(env['rewards']==[[0.0]] and env['terminated']==[[False]] and env['truncated']==[[False]],cid+f': action{index} envarrays')
        require(info['episode_contract']['seed']==case['seed'] and info['episode_contract']['task_name']=='sample_verify_return',cid+f': action{index} reset/taskchanged')
        require(info['episode_metrics']['verified_count']==info['episode_metrics']['returned_count']==0 and info['episode_metrics']['task_events']==[],cid+f': action{index} missedtaskprogress')
        require(Path(step['observation']['npz_path']).resolve()==(base/f'episode/observations/{index:06}.npz').resolve() and Path(step['next_observation']['npz_path']).resolve()==(base/f'episode/observations/{index+1:06}.npz').resolve(),cid+f': action{index} actualobservationrefs')
    require(entry['motor_noise_seeds']==list(range(7,39)),cid+': seedreceipt')
    require(entry['executed_controls']==32 and entry['diagnosis']['sim_time_s']==1.92 and entry['diagnosis']['physics_step_count']==480,cid+': actualduration summarydiffers')
    require(not entry['diagnosis']['task_SR'] and entry['diagnosis']['verified_count']==entry['diagnosis']['returned_count']==0,cid+': taskresult incorrectlypositive')
    case_result={'case_id':cid,'status':'failed_protocol_at_second_READ','first_side':case['first_side'],'first_cycle':{'actual_READ':gen[0]['text'],'READ_response_empty_slot':selected,'actual_WRITE':gen[1]['text'],'strict_tool_replay_matches_real_record':True,'memory_revision':[0,1],'autonomous_SUBTASK':gen[2]['text'],'first_role_matches':True,'all_three_actual_EOS':True},'executed_motor_queries':32,'executed_K1_control_targets':32,'actual_physics_ticks':480,'actual_sim_seconds':1.92,'stored_actual_observations':33,'everySDKtarget_matches_generated_H16_first_action':True,'every_normal_control_15ticks':True,'next_cycle_READ':{'text':gen[-1]['text'],'token_ids':gen[-1]['token_ids'],'actual_EOS':True,'valid_JSON':False,'history_now_contains_prior_subtask':True,'no_repair_or_later_WRITE_or_actions':True},'env_terminated':False,'env_truncated':False,'stop_cause':'strict READ JSON parser exception; saved as failed_episode','task_verified_count':0,'task_returned_count':0,'workers_closed':True,'worker_exit_receipts':entry['worker_exit_receipts'],'version_string':decision['policy_version'],'motor_noise_seeds':entry['motor_noise_seeds'],'noise_byte_hash_persisted':False}
    out['case_results'].append(case_result)
    for name in ['policy_trace.jsonl','environment_info.jsonl','episode/failed_episode.json','case_report.json']:
        out['files'][str(base/name)]=sha(base/name)
require(first_obs['case00_left_first']==first_obs['case01_right_first'],'first actualvisualstate inputs unexpectedlydifferent')
if policy_versions=={'fit_language_warmup_96_initial'}:out['version_label_findings'].append('policy_version is a generic non-content-addressed alias fit_language_warmup_96_initial; selected checkpoint/version SHA and exact-loaded144 values establish actualC2200 source. The alias alone must not identify model state.')
out.update(status='passed_evidence_integrity_but_autonomous_second_cycle_failed' if not issues else 'failed_evidence_integrity',actual_first_motor_visual_state_exact_between_cases=True,goals_intentionally_differ_between_cases=True,strict_first_role_subtasks_correct='2/2',complete_autonomous_cases='0/2',actual_total_controls=64,actual_total_physics_ticks=960,load_receipt=load,parameters_unchanged=True,no_gradients=True,source_files_unchanged=True,noise_scope={'actual_seed_lists_equal':report['results'][0]['motor_noise_seeds']==report['results'][1]['motor_noise_seeds'],'source':'NativePi05Actor.actions -> _predict_eval(noise=None,rng=None) -> Pi0.sample_actions globaltorch.randn; CPU/CUDAmanual_seed7+query before each call','direct_actual_noise_tensor_or_SHA_logged':False,'claim_limit':'seed protocol verified; no direct byte-level noise-equality audit, no identical later observations or prompts claim'},limits=['Outer completed means diagnostic runner finished and preserved failures; both autonomous cases failed.','Only first static requested-side binding and one memory-tool cycle succeeded; no phase-dependent verify/return or long-horizon semantic success was demonstrated.','Memory confirmed=true is an actor-authored role fact, not measured physical verification. Environment verified/returned/events stayed0/0/empty.','No oracle was executed; no manual/fixed-library answer repaired the failed READ.','Actualinitial RGB/state was identical while goals differed; later action histories and observations need not match.','This audit loaded only tokenizer/NPZ and pure CPU memory tooling; model/source parameter protection is supported by live execution receipts.'])
output=Path('/tmp/highlevel_closedloop_independent_audit.json');output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':out['status'],'issues':issues,'first_subtask_binding':out['strict_first_role_subtasks_correct'],'autonomous_cases':out['complete_autonomous_cases'],'controls':out['actual_total_controls'],'ticks':out['actual_total_physics_ticks'],'version_label_findings':out['version_label_findings']},ensure_ascii=False))
if issues:raise SystemExit(1)
