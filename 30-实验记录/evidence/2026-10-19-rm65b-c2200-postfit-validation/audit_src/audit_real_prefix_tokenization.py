from pathlib import Path
import sys,json,dataclasses,collections,hashlib,types
import numpy as np
import torch
ROOT=Path('/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/recap_memory_20261017/experiments/protocol_repair_20261019_v2')
sys.path.insert(0,str(ROOT/'code/RLinf'));sys.path.insert(0,str(ROOT/'scripts'))
from rlinf.models.embodiment.openpi.recap_tokenizer import RecapTokenizer,canonical_json
from rlinf.data.datasets.recap.hierarchical_collator import NativeJointCollator
from rlinf.workers.rollout.hf.hierarchical_recap_worker import NativePi05Actor
from build_protocol_labels import normalized_state24
import yaml
cfg=yaml.safe_load((ROOT/'reports/continuation_v1/language_warmup_96_1200/effective_config.yaml').read_text())
labels=Path(cfg['data']['verified_high_sft_paths'][0]); records=[json.loads(x) for x in labels.read_text().splitlines()]
tokcfg=cfg['tokenizer']; tokenizer=RecapTokenizer.from_local_paths(tokcfg['text_tokenizer_path'],max_prefix_tokens=tokcfg['max_prefix_tokens'],max_target_tokens=tokcfg['max_target_tokens'])
stats_all=json.loads(Path(cfg['model']['openpi_data']['norm_stats_path']).read_text());stats=stats_all.get('norm_stats',stats_all)['state']
@dataclasses.dataclass
class TextOnlyObservation:
 state:torch.Tensor
 tokenized_prompt:object=None
 tokenized_prompt_mask:object=None
 token_ar_mask:object=None
 token_loss_mask:object=None
class TextOnlyCapture:
 device=torch.device('cpu')
 max_token_len=cfg['model']['openpi']['max_token_len']
 mem_enabled=False
 def eval(self):return self
 def generate_text_tokens(self,obs,**kwargs):
  self.captured=obs
  return torch.tensor([[tokenizer.eos_token_id]],dtype=torch.long)
model=TextOnlyCapture(); collator=NativeJointCollator(model,tokenizer,loss_weights={'high_ce':1.0},training_mode='verified_sft')
state_cache={}
def state_for(reference):
 p=reference['npz_path']
 if p not in state_cache:
  with np.load(p,allow_pickle=False) as archive: raw=archive['state'].copy()
  state_cache[p]=normalized_state24(raw,stats).astype(np.float32)
 return torch.from_numpy(state_cache[p]).unsqueeze(0)
collator._processed=lambda ref,actions=None: (None,TextOnlyObservation(state_for(ref)),None)
actor=NativePi05Actor(model,tokenizer,version='readonly_token_capture',max_new_tokens=cfg['rollout']['max_new_tokens'],eos_token_id=tokenizer.eos_token_id,advantage_condition=False)
actor._base_observation=lambda observation:TextOnlyObservation(state_for(observation))
rows=[];groups=collections.defaultdict(list);pair_examples=[]
for rec in records:
 batch=collator([rec],[]); train=batch['high_observation'];target=batch['high_target_ids'][batch['high_loss_mask']].tolist(); ids=train.tokenized_prompt[train.tokenized_prompt_mask].tolist()
 context=rec['context'];history=context['visible_history'] if isinstance(context['visible_history'],str) else canonical_json(context['visible_history'])
 actor.generate(rec['observation'],rec['mode'],{'goal':context['goal'],'memory_index':context['catalogue'],'history':history,'visible_memory':context.get('actor_memory',context.get('selected_memory',{}))})
 infer=model.captured; inference=infer.tokenized_prompt[infer.tokenized_prompt_mask].tolist()
 assert ids==inference
 assert torch.equal(train.state,infer.state)
 assert target[-1]==tokenizer.eos_token_id and target.count(tokenizer.eos_token_id)==1
 assert train.tokenized_prompt.shape[-1]==len(ids) # microbatch1: no padding
 row={'sample_id':rec['sample_id'],'mode':rec['mode'],'target':rec['target_text'],'effective_token_ids':ids,'prefix_len':len(ids),'target_len':len(target),'prefix_sha256':hashlib.sha256(np.asarray(ids,dtype=np.int64).tobytes()).hexdigest(),'target_token_ids':target,'bos_count':ids.count(tokenizer.text_tokenizer.bos_id()),'eos_in_prefix':ids.count(tokenizer.eos_token_id),'unk_in_prefix':ids.count(tokenizer.text_tokenizer.unk_id()),'unk_in_target':target.count(tokenizer.text_tokenizer.unk_id()),'prefix_plus_target':len(ids)+len(target),'prefix_plus_generation96':len(ids)+96,'recorded_builder_prefix_len':rec['token_lengths']['prefix'],'target_side':rec['evaluation']['target_side']}
 rows.append(row);groups[tuple(ids)].append(row)
conflicts=[{'sample_ids':[x['sample_id'] for x in xs],'targets':list({x['target'] for x in xs})} for xs in groups.values() if len({x['target'] for x in xs})>1]
# Token distinctions for matched records are reported with sentencepiece pieces.
for mode in ['SUBTASK','WRITE']:
 for left in rows:
  if left['mode']!=mode or left['target_side']!='left':continue
  for right in rows:
   if right['mode']!=mode or right['target_side']!='right':continue
   if mode=='WRITE':
    lop=json.loads(left['target'])['operations'];rop=json.loads(right['target'])['operations']
    if not lop or not rop or lop[0]['op']!=rop[0]['op']:continue
   a,b=left['effective_token_ids'],right['effective_token_ids'];changes=[{'position':i,'left_id':x,'left_piece':tokenizer.text_tokenizer.id_to_piece(x),'right_id':y,'right_piece':tokenizer.text_tokenizer.id_to_piece(y)} for i,(x,y) in enumerate(zip(a,b)) if x!=y]
   pair_examples.append({'kind':mode+'_left_vs_right','left_sample':left['sample_id'],'right_sample':right['sample_id'],'different_tokens':len(changes),'first_differences':changes[:20]});break
  if pair_examples and pair_examples[-1]['kind']==mode+'_left_vs_right':break
# Same episode confidence/corruption WRITE contexts must differ at revision/value side.
for op1,op2 in [('KEEP','CORRECT'),('EMPTY','CORRECT')]:
 a=next(x for x in rows if x['mode']=='WRITE' and (json.loads(x['target'])['operations'][0]['op'] if json.loads(x['target'])['operations'] else 'EMPTY')==op1)
 b=next(x for x in rows if x['mode']=='WRITE' and (json.loads(x['target'])['operations'][0]['op'] if json.loads(x['target'])['operations'] else 'EMPTY')==op2 and x['target_side']==a['target_side'])
 pair_examples.append({'kind':f'WRITE_{op1}_vs_{op2}','sample_a':a['sample_id'],'sample_b':b['sample_id'],'same_prefix_ids':a['effective_token_ids']==b['effective_token_ids'],'prefix_lengths':[a['prefix_len'],b['prefix_len']]})
summary={'read_only':True,'GPU_or_real_model_loaded':False,'scope':'Real SentencePiece tokenizer plus actual NativeJointCollator and NativePi05Actor.generate text wrappers; CPU TextOnlyCapture substitutes network/image computation. State uses actual raw24 NPZ, original normalization and32 padding mirrored by existing verified builder.','tokenizer_path':tokcfg['text_tokenizer_path'],'tokenizer_sha256':hashlib.sha256(Path(tokcfg['text_tokenizer_path']).read_bytes()).hexdigest(),'vocab':tokenizer.vocab_size,'BOS':tokenizer.text_tokenizer.bos_id(),'EOS':tokenizer.eos_token_id,'UNK':tokenizer.text_tokenizer.unk_id(),'PAD':tokenizer.text_tokenizer.pad_id(),'max_prefix_tokens':tokenizer.max_prefix_tokens,'max_model_token_len':model.max_token_len,'max_target_tokens':tokenizer.max_target_tokens,'generation_max_new_tokens':actor.max_new_tokens,'prefix_len_range':[min(x['prefix_len'] for x in rows),max(x['prefix_len'] for x in rows)],'max_prefix_plus_generation96':max(x['prefix_plus_generation96'] for x in rows),'max_prefix_plus_target':max(x['prefix_plus_target'] for x in rows),'unique_effective_prefixes':len(groups),'label_conflicts':conflicts,'train_inference_token_id_matches':len(rows),'train_inference_state_matches':len(rows),'microbatch1_no_padding':True,'prefix_BOS_count_set':sorted({x['bos_count'] for x in rows}),'prefix_EOS_count_set':sorted({x['eos_in_prefix'] for x in rows}),'total_unknown_tokens_prefix':sum(x['unk_in_prefix'] for x in rows),'total_unknown_tokens_target':sum(x['unk_in_target'] for x in rows),'builder_prefix_length_mismatches':[x['sample_id'] for x in rows if x['prefix_len']!=x['recorded_builder_prefix_len']],'literal_token_ids':{word:tokenizer.encode_text(word).tolist() for word in ['left','right','CORRECT','KEEP']},'pair_examples':pair_examples,'conclusions':['No truncation or prefix collisions remove role/memory distinctions on these96.','encode_prefix raises if >1400; no slicing. Collator checks prefix+target<=2048; generate checks prefix+96<=2048.','One BOS in both paths; no prefix EOS; one target EOS; unknown-token counts are reported.','Training batch_size1 means prefix padding absent. Both wrappers use identical effective token IDs and state.','This is tokenizer/wrapper evidence, not network activation or visual preprocessing parity.'], 'rows':rows}
p=Path('/tmp/real_prefix_tokenization_audit.json');p.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if k not in ['rows','pair_examples','literal_token_ids','conclusions']},ensure_ascii=False));print('report',str(p))
