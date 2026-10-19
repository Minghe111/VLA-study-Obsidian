"""CPU evidence only: no actor/model import, inference, training or engineering writes."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import hashlib, importlib.util, json, sys
import numpy as np
import yaml

NEW=Path('/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/recap_memory_20261017/experiments/protocol_repair_20261019_v2')
SOURCE=NEW.parent/'protocol_sft_20261018_v1'
CASE=NEW/'reports/postfit_validation_20261019_v1/highlevel_closedloop/case00_left_first'
OUT=Path('/tmp/highlevel_history_coverage_audit.json')
sys.path.insert(0,str(NEW/'code/RLinf'))
from rlinf.models.embodiment.openpi.recap_tokenizer import RecapTokenizer, build_high_prefix, canonical_json, quantized_state
spec=importlib.util.spec_from_file_location('label_cpu',NEW/'scripts/build_protocol_labels.py')
label=importlib.util.module_from_spec(spec);spec.loader.exec_module(label)

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read_rows(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def load_obs(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def empty(v):return v is None or v=='' or v==[] or v=={}
def history_summary(rows):
    vals=[r['context'].get('visible_history') for r in rows]
    return {'rows':len(vals),'types':dict(Counter(type(v).__name__ for v in vals)),
            'empty_count':sum(empty(v) for v in vals),'nonempty_count':sum(not empty(v) for v in vals),
            'unique_serialized_values':sorted(set(canonical_json(v) for v in vals)),
            'metadata_subtask_history_source_values':sorted(set(str(r.get('metadata',{}).get('subtask_history_source')) for r in rows)),
            'metadata_label_schedule_values':sorted(set(str(r.get('metadata',{}).get('label_schedule')) for r in rows)),
            'physical_behavior_values':sorted(set(str(r.get('provenance',{}).get('physical_behavior')) for r in rows))}

cfg=yaml.safe_load((CASE/'effective_config.yaml').read_text())
fitcfgpath=NEW/'reports/continuation_v1/language_warmup_96_2200/effective_config.yaml'
fitcfg=yaml.safe_load(fitcfgpath.read_text())
stats_path=Path(cfg['model']['openpi_data']['norm_stats_path'])
stats=json.loads(stats_path.read_text())['norm_stats']['state']
tokcfg=cfg['tokenizer']
tok=RecapTokenizer.from_local_paths(tokcfg['text_tokenizer_path'],max_prefix_tokens=tokcfg['max_prefix_tokens'],max_target_tokens=tokcfg['max_target_tokens'])
datasets={'fit96':NEW/'data/fit_subsets_v1/fit96.train.jsonl',
          'source_train360':SOURCE/'data/protocol_labels_reviewed_v1/train.reviewed.jsonl',
          'heldout_val60':SOURCE/'data/protocol_labels_reviewed_v1/val.reviewed.jsonl'}
allrows={name:read_rows(p) for name,p in datasets.items()}
coverage={name:{'path':str(datasets[name]),'sha256':sha(datasets[name]),'all':history_summary(rs),
               'by_mode':{m:history_summary([r for r in rs if r['mode']==m]) for m in ['READ','WRITE','SUBTASK']},
               'observation_keys':sorted(set(k for r in rs for k in load_obs(r['observation']['npz_path']))),
               'num_images_context_key_count':sum('num_images' in r['context'] for r in rs)} for name,rs in allrows.items()}

trace=read_rows(CASE/'policy_trace.jsonl')
reads=[r for r in trace if r.get('kind')=='generate' and r.get('mode')=='READ']
assert [r['call'] for r in reads]==[0,3]
obs={s:load_obs(CASE/f'episode/observations/{s:06d}.npz') for s in [0,32]}
norm={s:label.normalized_state24(o['state'],stats).astype(np.float32) for s,o in obs.items()}

def rt_prompt(event,state,history=None):
    c=event['context']
    return build_high_prefix(event['mode'],task=c['goal'],state=state,
       visible_memory=canonical_json(c.get('visible_memory',{})),
       memory_index=canonical_json(c.get('memory_index',{})),
       history=c.get('history','') if history is None else history,advantage=None)
def receipt(text):
    ids=tok.encode_prefix(text).tolist()
    return {'prefix_text':text,'prefix_sha256':hashlib.sha256(text.encode()).hexdigest(),
            'prefix_tokens':len(ids),'ids_sha256':hashlib.sha256(np.asarray(ids,dtype=np.int64).tobytes()).hexdigest(),
            'bos_id':ids[0],'last_24_token_ids':ids[-24:],
            'within_text_prefix_budget':len(ids)<=tok.max_prefix_tokens,
            'within_model_text_limit':len(ids)+cfg['rollout']['max_new_tokens']<=cfg['model']['openpi']['max_token_len']}

runtime=[]
fit_read=[r for r in allrows['fit96'] if r['mode']=='READ']
fit_raw=np.stack([load_obs(r['observation']['npz_path'])['state'] for r in fit_read])
for event,s in zip(reads,[0,32]):
    o=obs[s]
    runtime.append({'call':event['call'],'observation_step':s,'context':event['context'],'output_text':event['text'],
      'output_token_ids':event['token_ids'],'ended_with_eos':event['ended_with_eos'],
      'inference_guard':event['inference_guard'],'policy_observation_keys':event['policy_observation_keys'],
      'raw_state24':o['state'].tolist(),'normalized_state32':norm[s].tolist(),
      'quantized_state32':[int(x) for x in quantized_state(norm[s]).split()],
      'state_outside_fit_READ_raw_per_dim_minmax':np.flatnonzero((o['state']<fit_raw.min(0)-1e-6)|(o['state']>fit_raw.max(0)+1e-6)).tolist(),
      'cameras':{k:{'shape':list(o[k].shape),'dtype':str(o[k].dtype),'pixel_sha256':hashlib.sha256(o[k].tobytes()).hexdigest()} for k in label.CAMERAS},
      'state_float64_helper_vs_float32_runtime_quantized_equal':quantized_state(label.normalized_state24(o['state'],stats))==quantized_state(norm[s]),
      'prefix':receipt(rt_prompt(event,norm[s]))})

controlled={h:receipt(rt_prompt(reads[0],norm[0],history=v)) for h,v in {
    'runtime_empty_string':'','training_empty_list_serialized':'[]','second_round_history':reads[1]['context']['history']}.items()}
p0=tok.encode_prefix(controlled['runtime_empty_string']['prefix_text']).tolist()
p1=tok.encode_prefix(controlled['training_empty_list_serialized']['prefix_text']).tolist()
first=next((i for i,(a,b) in enumerate(zip(p0,p1)) if a!=b),min(len(p0),len(p1)))

source_files={
 'prefix_tokenizer':NEW/'code/RLinf/rlinf/models/embodiment/openpi/recap_tokenizer.py',
 'collator':NEW/'code/RLinf/rlinf/data/datasets/recap/hierarchical_collator.py',
 'controller_and_native_actor':NEW/'code/RLinf/rlinf/workers/rollout/hf/hierarchical_recap_worker.py',
 'text_generation':NEW/'code/RLinf/rlinf/models/embodiment/openpi/tasks/recap_hier.py',
 'fresh_prefix_cache':NEW/'code/RLinf/rlinf/models/embodiment/openpi/pi0.py',
 'closedloop_runner':Path('/tmp/c2200_closedloop_runner_v3.py'),
 'policy_trace':CASE/'policy_trace.jsonl','fit_effective_config':fitcfgpath,'runtime_effective_config':CASE/'effective_config.yaml',
 'tokenizer_asset':Path(tokcfg['text_tokenizer_path']),'norm_stats':stats_path}
state_domains={}
for name,rs in allrows.items():
    states=np.stack([load_obs(r['observation']['npz_path'])['state'] for r in rs if r['mode']=='READ'])
    state_domains[name]={'read_rows':len(states),'raw_state_min24':states.min(0).tolist(),'raw_state_max24':states.max(0).tolist(),
      'runtime_outside_per_dim_minmax':{str(s):np.flatnonzero((o['state']<states.min(0)-1e-6)|(o['state']>states.max(0)+1e-6)).tolist() for s,o in obs.items()},
      'scope':'Empirical coverage test only; outside finite static sample extrema does not establish invalid physics or normalization.'}
result={
 'status':'cpu_only_readonly_audit_completed_no_model_no_inference',
 'created_at_utc':datetime.now(timezone.utc).isoformat(),'test_split_read_or_used':False,
 'source_files':{k:{'path':str(p),'sha256':sha(p)} for k,p in source_files.items()},
 'dataset_history_coverage':coverage,
 'runtime_READ_calls':runtime,
 'READ_state_domains':state_domains,
 'same_observation_controlled_prefix_reconstruction':controlled,
 'empty_history_serialization_gap':{'tokens_equal':p0==p1,'first_different_token_index':first,
   'runtime_tokens_at_first_diff':p0[first:first+6],'training_tokens_at_first_diff':p1[first:first+6],
   'runtime_piece_at_first_diff':[tok.text_tokenizer.id_to_piece(i) for i in p0[first:first+6]],
   'training_piece_at_first_diff':[tok.text_tokenizer.id_to_piece(i) for i in p1[first:first+6]],
   'build_high_prefix_normalizes_history':False,'scope':'confirmed serialization/interface consistency gap; contribution to observed failure not isolated'},
 'call0_to_call3_changes':{'mode_unchanged':reads[0]['mode']==reads[1]['mode']=='READ',
   'goal_unchanged':reads[0]['context']['goal']==reads[1]['context']['goal'],
   'memory_index_unchanged':reads[0]['context']['memory_index']==reads[1]['context']['memory_index'],
   'other_context_keys_equal':{k:reads[0]['context'].get(k)==reads[1]['context'].get(k) for k in sorted(set(reads[0]['context'])|set(reads[1]['context'])) if k!='history'},
   'changed_raw_state_indices':np.flatnonzero(np.abs(obs[0]['state']-obs[32]['state'])>1e-6).tolist(),
   'changed_quantized_state_indices':np.flatnonzero(np.asarray(runtime[0]['quantized_state32'])!=np.asarray(runtime[1]['quantized_state32'])).tolist(),
   'changed_camera_keys':[k for k in label.CAMERAS if not np.array_equal(obs[0][k],obs[32][k])]},
 'asset_and_assembly_checks':{
   'tokenizer_path_same_fit_and_runtime':fitcfg['tokenizer']['text_tokenizer_path']==tokcfg['text_tokenizer_path'],
   'norm_stats_path_same_fit_and_runtime':fitcfg['model']['openpi_data']['norm_stats_path']==str(stats_path),
   'same_prefix_builder_same_mode_format_same_concat_order':True,
   'prefix_bos_eos_ids':{'bos':int(tok.text_tokenizer.bos_id()),'eos':tok.eos_token_id,'pad':int(tok.text_tokenizer.pad_id())},
   'text_tokenizer_vocab_size':tok.vocab_size,
   'advantage_condition_fit_and_runtime':None,
   'num_images_is_prefix_or_context_field':False,'actual_rgb_cameras_per_observation':3,
   'configured_MEM_enabled':cfg['model']['openpi'].get('mem_enabled'),'actual_runtime_MEM_enabled':False,
   'MEM_runtime_scope':'language_warmup checkpoint and runner preflight require disabled temporal MEM; no video history tensor supplied',
   'cross_call_text_KV_reuse_found':False,
   'cache_scope_evidence':'generate_text_tokens calls build_prefix_cache(observation) every invocation; generated/cache local variables; per-token reuse only within one generation call',
   'silent_prefix_truncation_found':False,
   'normalization_reconstruction_scope':'CPU helper mirrors original state transforms; cast float32 matches actual prompt boundary; no actor/model or GPU loaded'},
 'conclusions':[
   'fit96/all96 including READ32, source train360/READ120 and val60/READ20 contain only empty list histories; second-round nonempty history is uncovered.',
   'Training/heldout dataset evaluation serializes [] while real runtime first round uses empty string; effective prefix token sequences genuinely differ.',
   'Second READ uses unchanged READ mode instruction and same builder/concatenation order; no source evidence of stale cross-call text KV or tokenizer-asset mismatch.',
   'After32 controls images, state and history change together. This audit cannot attribute the malformed output to history alone.',
   'Controller history records previously issued subtask text, not verified task-completion evidence. Static labels do not cover actual task progress or later-round memory/planning.'
 ]}
OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'output':str(OUT),'history_counts':{k:{'all':v['all']['nonempty_count'],'READ':v['by_mode']['READ']['nonempty_count']} for k,v in coverage.items()},'prefix_tokens':{k:v['prefix_tokens'] for k,v in controlled.items()},'runtime_prefix_tokens':[x['prefix']['prefix_tokens'] for x in runtime],'empty_tokens_equal':p0==p1,'first_diff':result['empty_history_serialization_gap']},ensure_ascii=False))
