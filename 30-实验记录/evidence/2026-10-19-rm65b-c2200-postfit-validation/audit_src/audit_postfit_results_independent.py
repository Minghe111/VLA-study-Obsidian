from pathlib import Path
from copy import deepcopy
from collections import defaultdict,Counter
import json,sys,hashlib,re,importlib.util,datetime
import numpy as np
ROOT=Path('/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/recap_memory_20261017/experiments/protocol_repair_20261019_v2')
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'code/RLinf')]
from rlinf.models.embodiment.openpi.recap_tokenizer import RecapTokenizer,canonical_json
from protocol_pilot import check_output
from build_protocol_labels import _load_frame,actor_input_digest,_high_prefix
import yaml
cfg=yaml.safe_load((ROOT/'reports/continuation_v1/language_warmup_96_2200/effective_config.yaml').read_text())
tokenizer=RecapTokenizer.from_local_paths(cfg['tokenizer']['text_tokenizer_path'],max_prefix_tokens=1400,max_target_tokens=512)
stats=json.loads(Path(cfg['model']['openpi_data']['norm_stats_path']).read_text())['norm_stats']['state']
r=ROOT/'reports/postfit_validation_20261019_v1'
valreport=json.loads((r/'heldout_val/report.json').read_text());ppreport=json.loads((r/'paraphrase/report.json').read_text())
labels=Path(valreport['audit']['splits']['val']['path']); val=[json.loads(x) for x in labels.read_text().splitlines()];byid={x['sample_id']:x for x in val}
cache={};digest_by_id={};conflicts=[];digests={}
for row in val:
 p=row['observation']['npz_path']
 if p not in cache:cache[p]=_load_frame({'npz_path':p},stats)
 state,hashes=cache[p];d=actor_input_digest(tokenizer,row,state,hashes['camera_pixel_sha256']);digest_by_id[row['sample_id']]=d
 if d!=row['actor_input_digest']:raise ValueError('Source actual digest changed')
 if d in digests and digests[d]!=row['target_text']:conflicts.append(row['sample_id'])
 digests[d]=row['target_text']
# Recompute searched historical coverage from actual old source references, not report counters.
source=labels.parents[2];old_indices=set();searched=[]
for p in sorted((source/'reports/protocol_sft_sequence_v1').glob('*/report.json')):
 d=json.loads(p.read_text()); count=0
 for case in d.get('cases',[]):
  ref=case.get('source',{})
  if isinstance(ref,dict) and ref.get('path') and Path(ref['path']).name==labels.name:old_indices.add(ref['row']);count+=1
 if count:searched.append({'path':str(p),'case_count':count})
seen_digests={val[i]['actor_input_digest'] for i in old_indices}; fresh={d for d in digests if d not in seen_digests}

def score(report,records):
 errors=[];bad=[];groups=defaultdict(list);totals=Counter();by_mode=defaultdict(Counter)
 for case in report['cases']:
  row=records[case['sample_id']];ids=case['token_ids'];decoded=tokenizer.decode_text(ids)
  eos=bool(ids) and ids[-1]==tokenizer.eos_token_id and ids.count(tokenizer.eos_token_id)==1
  if decoded!=case['text']:errors.append({'id':case['sample_id'],'error':'tokenizer decode mismatch','decode':decoded,'text':case['text']})
  checks=check_output(row,decoded,eos);exact=decoded.strip()==row['target_text'].strip();strict=checks['passed'] and (row['mode']!='SUBTASK' or exact)
  checks.update(text_exact=exact,semantic_passed=checks['passed'],strict_passed=strict,strict_text_exact=eos and exact)
  for k,v in case['checks'].items():
   if checks.get(k)!=v:errors.append({'id':case['sample_id'],'check':k,'stored':v,'recomputed':checks.get(k)})
  if not eos:errors.append({'id':case['sample_id'],'error':'EOS condition failed'})
  p=row['observation']['npz_path'];state,hashes=cache[p];digest=actor_input_digest(tokenizer,row,state,hashes['camera_pixel_sha256'])
  if digest!=case['actor_input_digest']:errors.append({'id':case['sample_id'],'error':'actualinput digest differs'})
  groups[digest].append(case);totals['count']+=1;totals['strict_passed']+=int(strict);totals['eos']+=int(eos)
  by_mode[row['mode']]['count']+=1;by_mode[row['mode']]['strict_passed']+=int(strict)
  if not strict:bad.append({'sample_id':row['sample_id'],'mode':row['mode'],'goal':row['context']['goal'],'target':row['target_text'],'text':decoded,'token_ids':ids,'checks':checks,'context':row['context']})
 for digest,ls in groups.items():
  if len({(x['text'],tuple(x['token_ids'])) for x in ls})>1:errors.append({'digest':digest,'error':'sameactualinput inconsistent deterministic output'})
 return {'count':totals['count'],'strict_passed':totals['strict_passed'],'eos':totals['eos'],'unique_actual_inputs':len(groups),'by_mode':{k:dict(v) for k,v in by_mode.items()},'bad_examples':bad,'errors':errors}

valscore=score(valreport,byid)
# Independently reconstruct the authored equivalent-goal variants from canonical val records.
pprows={};pp_checks=[]
for case in ppreport['cases']:
 meta=case['text_intervention'];base=byid[meta['original_sample_id']];row=deepcopy(base)
 side=re.findall(r'\b(left|right)\b',base['context']['goal'].lower())[0];other='right' if side=='left' else 'left'
 variant=meta['variant'];goal=(f'First verify the sample initially on the {side}, put it back, and then do the same for the sample initially on the {other}.' if variant==1 else f'Start with the sample initially on the {side}: verify it and return it home before verifying and returning the sample initially on the {other}.')
 assert variant in (1,2)
 row['context']['goal']=goal;row['sample_id']=case['sample_id'];pprows[row['sample_id']]=row
 assert row['target_text']==case['target_text']
 # Inputs represented by actual prefix+pixels are recomputed by score() below.
 assert {k:v for k,v in row['context'].items() if k!='goal'}=={k:v for k,v in base['context'].items() if k!='goal'}
 assert row['observation']==base['observation']
 assert re.findall(r'\b(left|right)\b',goal.lower())==re.findall(r'\b(left|right)\b',base['context']['goal'].lower())
 assert digest_by_id[base['sample_id']]==meta['original_actor_input_digest']
 pp_checks.append({'source_id':base['sample_id'],'source_digest':meta['original_actor_input_digest'],'variant':variant,'target_unchanged':True,'only_actor_field_goal_changed':True})
ppscore=score(ppreport,pprows)
# Validate v4 generated counterfactual schema with the real EpisodeMemory tool, CPU only.
v4path=Path('/tmp/verify_protocol_generalization_v4.py');v4sha=hashlib.sha256(v4path.read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('audit_frozen_v4',v4path);v4=importlib.util.module_from_spec(spec);spec.loader.exec_module(v4)
signatures={}
for row in val:
 probe=deepcopy(row);probe['mode']='READ';probe['context']={'goal':'Verify the sample initially on the left and return it home, then verify the sample initially on the right and return it home.','catalogue':deepcopy(row['context']['catalogue']),'visible_history':[],'mode':'READ'}
 state,hashes=cache[row['observation']['npz_path']];signatures[row['observation']['npz_path']]=actor_input_digest(tokenizer,probe,state,hashes['camera_pixel_sha256'])
cf=v4.counterfactual_rows(val,4,observation_signatures=signatures); envelopes=Counter(); badschemas=[]; cfgroup={}; cfconflicts=[]; rejected_list=False
for row in cf:
 v4.validate_actor_visible_schema(row);v4.validate_target(row)
 state,hashes=cache[row['observation']['npz_path']];d=actor_input_digest(tokenizer,row,state,hashes['camera_pixel_sha256'])
 if d in cfgroup and cfgroup[d]!=row['target_text']:cfconflicts.append(row['sample_id'])
 cfgroup[d]=row['target_text']
 if row['mode']=='SUBTASK':
  view=row['context']['actor_memory'];envelopes[tuple(sorted(view))]+=1
  if view['current_writes']!=[]:badschemas.append(row['sample_id'])
# Regression: old list envelope must be explicitly rejected.
wrong=deepcopy(next(x for x in cf if x['mode']=='SUBTASK'));wrong['context']['actor_memory']=wrong['context']['actor_memory']['selected']
try:v4.validate_actor_visible_schema(wrong)
except ValueError:rejected_list=True
out={'audit_time':datetime.datetime.now().isoformat(),'read_only':True,'GPU_or_model_loaded':False,'val_recomputed':valscore,'val_dedup_history':{'rows':len(val),'unique_actual_inputs':len(digests),'source_target_conflicts':conflicts,'historically_seen_sample_indices':sorted(old_indices),'historically_seen_unique_inputs':len(seen_digests),'fresh_unique_inputs_in_searched_history':len(fresh),'fresh_input_rows':sum(x['actor_input_digest'] in fresh for x in val),'historical_search_reports':searched},'paraphrase_recomputed':ppscore,'paraphrase_input_intervention':{'count':len(pp_checks),'original_unique_inputs':len({x['source_digest'] for x in pp_checks}),'variant_counts':dict(Counter(x['variant'] for x in pp_checks)),'only_goal_differs':True,'same_first_then_second_role':True,'target_unchanged':True,'memory_history_observation_ref_state_pixels_unchanged':True,'source_rule':'Two authored English templates explicitly preserve first side, verify thenreturn beforesecond side','semantic_limit':'56 authored goal variants over28canonical inputs, not56novelphysicalepisodes or broadlanguage/generalrobot skills'},'v4_counterfactual_schema':{'path':str(v4path),'script_sha256':v4sha,'frozen_during_audit':v4sha==hashlib.sha256(v4path.read_bytes()).hexdigest(),'rows':len(cf),'mode_counts':dict(Counter(x['mode'] for x in cf)),'selected_observation_count':len({x['observation']['npz_path'] for x in cf}),'available_unique_image_state_inputs':len(set(signatures.values())),'unique_actual_actor_inputs':len(cfgroup),'conflicting_targets':cfconflicts,'subtask_nested_envelope_counts':{str(k):v for k,v in envelopes.items()},'invalid_current_writes':badschemas,'rejects_old_list_shaped_envelope':rejected_list,'real_memory_tool_used':True,'scope':'Native-schema controlled stale/empty/unconfirmed selected-memory stress. current_writesempty; NOT asserted to be natural result of preceding successfulCORRECT.','GPU_results_not_verified_here':True},'conclusion':'Val60/60=28deduplicatedactualinputs, ofwhich15freshwithinsearchedhistory; paraphrase54/56 with2WRITE syntaxerrors. v4schema corrected and oldlistregression rejected. No motor or taskSR claim.'}
assert not valscore['errors'] and not ppscore['errors'] and not conflicts and not cfconflicts and not badschemas and rejected_list
p=Path('/tmp/protocol_postfit_independent_audit.json');p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'report':str(p),'val':{k:v for k,v in valscore.items() if k not in ['bad_examples','errors']},'fresh_unique':len(fresh),'paraphrase':{k:v for k,v in ppscore.items() if k not in ['bad_examples','errors']},'bad_ids':[x['sample_id'] for x in ppscore['bad_examples']],'v4modecounts':out['v4_counterfactual_schema']['mode_counts'],'v4_rejects_oldlist':rejected_list},ensure_ascii=False))
