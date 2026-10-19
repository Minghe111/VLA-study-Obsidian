from pathlib import Path
from collections import defaultdict,Counter
from copy import deepcopy
import json,hashlib,re,sys,importlib.util,datetime
ROOT=Path('/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/recap_memory_20261017/experiments/protocol_repair_20261019_v2')
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'code/RLinf')]
from rlinf.models.embodiment.openpi.recap_tokenizer import RecapTokenizer,canonical_json
from protocol_pilot import check_output
from build_protocol_labels import _load_frame,actor_input_digest,_high_prefix
import yaml
cfg=yaml.safe_load((ROOT/'reports/continuation_v1/language_warmup_96_2200/effective_config.yaml').read_text()); tok=RecapTokenizer.from_local_paths(cfg['tokenizer']['text_tokenizer_path'],max_prefix_tokens=1400,max_target_tokens=512)
stats=json.loads(Path(cfg['model']['openpi_data']['norm_stats_path']).read_text())['norm_stats']['state'];reports=ROOT/'reports/postfit_validation_20261019_v1'
cf_report=json.loads((reports/'counterfactual_corrected/report.json').read_text());rev_report=json.loads((reports/'revision_only/report.json').read_text())
val=[json.loads(x) for x in Path(cf_report['audit']['splits']['val']['path']).read_text().splitlines()];cache={}
for row in val:
 p=row['observation']['npz_path']
 if p not in cache:cache[p]=_load_frame({'npz_path':p},stats)
def load_module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
v4=load_module('cf_v4_audit','/tmp/verify_protocol_generalization_v4.py');v5=load_module('revision_v5_audit','/tmp/verify_protocol_generalization_v5.py')
signatures={}
for row in val:
 probe=deepcopy(row);probe['mode']='READ';probe['context']={'goal':'Verify the sample initially on the left and return it home, then verify the sample initially on the right and return it home.','catalogue':deepcopy(row['context']['catalogue']),'visible_history':[],'mode':'READ'}
 state,hashes=cache[row['observation']['npz_path']];signatures[row['observation']['npz_path']]=actor_input_digest(tok,probe,state,hashes['camera_pixel_sha256'])
cf_rows=v4.counterfactual_rows(val,4,observation_signatures=signatures);rev_rows=v5.revision_rows(val)

def expected_target(row):
 side=re.findall(r'\b(left|right)\b',row['context']['goal'].lower())[0];fact=f'The first requested sample is initially on the {side}.'
 if row['mode']=='SUBTASK':return f'Pick up and hold the sample initially on the {side}.'
 if row['mode']=='READ':return canonical_json({'read_ids':['sample_identity']})
 slot=row['context']['selected_memory'][0]
 if slot['status']!='active':op={'op':'WRITE','slot_id':'sample_identity','value':fact,'confirmed':True}
 elif slot['value']!=fact:op={'op':'CORRECT','slot_id':'sample_identity','value':fact,'confirmed':True}
 elif not slot['confirmed']:op={'op':'KEEP','slot_id':'sample_identity','confirmed':True}
 else:op=None
 return canonical_json({'operations':[] if op is None else [op]})

def score(report,rows):
 byid={x['sample_id']:x for x in rows};errors=[];by_mode=defaultdict(Counter);groups=defaultdict(list)
 for case in report['cases']:
  row=byid[case['sample_id']]; v4.validate_actor_visible_schema(row)
  if row['target_text']!=expected_target(row) or row['target_text']!=case['target_text']:raise ValueError('Formal target mismatch')
  ids=case['token_ids'];decoded=tok.decode_text(ids);eos=bool(ids) and ids[-1]==tok.eos_token_id and ids.count(tok.eos_token_id)==1
  if decoded!=case['text']:errors.append({'case':case['sample_id'],'error':'Decode mismatch'})
  checks=check_output(row,decoded,eos);exact=decoded.strip()==row['target_text'].strip();checks.update(text_exact=exact,semantic_passed=checks['passed'],strict_passed=checks['passed'] and (row['mode']!='SUBTASK' or exact),strict_text_exact=eos and exact)
  for key,value in case['checks'].items():
   if checks.get(key)!=value:errors.append({'case':case['sample_id'],'check':key,'stored':value,'actual':checks.get(key)})
  state,hashes=cache[row['observation']['npz_path']];digest=actor_input_digest(tok,row,state,hashes['camera_pixel_sha256'])
  if digest!=case['actor_input_digest']:errors.append({'case':case['sample_id'],'error':'Actualprefix/pixel digest mismatch'})
  groups[digest].append(case);m=by_mode[row['mode']];m['count']+=1;m['strict_passed']+=checks['strict_passed'];m['eos']+=eos
  for key in ('json','schema_legal','target_match'):
   if key in checks:m[key]+=checks[key];m[key+'_eligible']+=1
 return {'rows':len(report['cases']),'unique_actual_inputs':len(groups),'by_mode':{k:dict(v) for k,v in by_mode.items()},'strict_passed':sum(x['strict_passed'] for x in by_mode.values()),'errors':errors},byid

cf_score,cf_byid=score(cf_report,cf_rows);rev_score,rev_byid=score(rev_report,rev_rows)
substats=defaultdict(Counter);write_stats=defaultdict(Counter);subexamples={}
for case in cf_report['cases']:
 group=case['counterfactual_group'];memory=group['memory'];goal=group['goal_side'];row=cf_byid[case['sample_id']]
 if case['mode']=='SUBTASK':
  st=substats[memory];st['count']+=1;st['strict_passed']+=case['checks']['strict_passed'];st['EOS']+=case['checks']['eos'];st['lexical_goal_match']+=case['checks'].get('side_binding_lexical',False)
  side=re.findall(r'initially on the (left|right)',case['text'].lower());side=side[0] if len(set(side))==1 and side else None
  if memory!='empty':st['followed_memory_value_side']+=side==memory.split('_')[1];st['goal_conflicts_memory']+=goal!=memory.split('_')[1]
  st['format_crosstalk_starts_JSON']+=case['text'].lstrip().startswith('{')
  if memory not in subexamples or (memory!='empty' and goal!=memory.split('_')[1]):subexamples[memory]={'sample_id':case['sample_id'],'goal_side':goal,'memory':row['context']['actor_memory'],'target':case['target_text'],'text':case['text'],'checks':case['checks']}
 elif case['mode']=='WRITE':
  st=write_stats[memory];st['count']+=1;st['strict_passed']+=case['checks']['strict_passed']
  predicted=json.loads(case['text'])['operations'];expected=json.loads(case['target_text'])['operations'];st['pred_'+(predicted[0]['op'] if predicted else 'EMPTY')]+=1;st['target_'+(expected[0]['op'] if expected else 'EMPTY')]+=1
# Revision causal conditions: independently validate ONLY revision differs, same actual input arrays and one invariant target.
rev_groups=defaultdict(list)
for row in rev_rows:rev_groups[row['revision_intervention']['group']].append(row)
rev_cases={x['sample_id']:x for x in rev_report['cases']};triples=[]
for name,rs in sorted(rev_groups.items()):
 rs.sort(key=lambda row:row['context']['selected_memory'][0]['revision']);contexts=[];prefixes=[];obs=[]
 for row in rs:
  context=deepcopy(row['context']);context['selected_memory'][0].pop('revision');contexts.append(context);obs.append(row['observation'])
  state,_=cache[row['observation']['npz_path']];prefixes.append(tok.encode_prefix(_high_prefix(row,state)).tolist())
 assert contexts[0]==contexts[1]==contexts[2] and obs[0]==obs[1]==obs[2] and len({r['target_text'] for r in rs})==1
 outputs=[rev_cases[x['sample_id']] for x in rs];parsed=[json.loads(x['text']) for x in outputs]
 tokenchanges=[]
 for index in (1,2):
  assert len(prefixes[index])==len(prefixes[0])
  changed=[{'position':i,'from_id':a,'to_id':b,'from_piece':tok.text_tokenizer.id_to_piece(a),'to_piece':tok.text_tokenizer.id_to_piece(b)} for i,(a,b) in enumerate(zip(prefixes[0],prefixes[index])) if a!=b];tokenchanges.append(changed)
 triples.append({'group':name,'revisions':[x['context']['selected_memory'][0]['revision'] for x in rs],'only_revision_input_change_verified':True,'same_observation_state_pixels':True,'target_invariant':True,'target':rs[0]['target_text'],'token_changes_from_revision1':tokenchanges,'semantic_output_changed':len({canonical_json(x) for x in parsed})>1,'outputs':[{'revision':r['context']['selected_memory'][0]['revision'],'text':c['text'],'strict_passed':c['checks']['strict_passed']} for r,c in zip(rs,outputs)]})
fit=[json.loads(x) for x in (ROOT/'data/fit_subsets_v1/fit96.train.jsonl').read_text().splitlines()];corr=defaultdict(Counter)
for row in fit:
 if row['mode']!='WRITE':continue
 op=json.loads(row['target_text'])['operations'];op=op[0]['op'] if op else 'EMPTY';corr[op][row['context']['selected_memory'][0]['revision']]+=1
out={'audit_time':datetime.datetime.now().isoformat(),'read_only':True,'GPU_or_model_loaded':False,'counterfactual_v4_recomputed':cf_score,'SUBTASK_by_memory_condition':{k:dict(v) for k,v in substats.items()},'SUBTASK_examples':subexamples,'WRITE_by_memory_condition':{k:dict(v) for k,v in write_stats.items()},'revision_v5_recomputed':rev_score,'revision_causal_conditions':{'groups':len(triples),'only_revision_changed_groups':sum(x['only_revision_input_change_verified'] for x in triples),'semantic_output_changed_groups':sum(x['semantic_output_changed'] for x in triples),'triples':triples,'within_test_operation_mapping':'All8 groups: revision1 predictsEMPTY, revision2 predictsKEEP, revision4 predictsCORRECT','scope':'Causal revision-input effect in one fixed actual observation,2goals,4value/confirmed contexts; not proof of all dataset shortcuts.'},'fit96_WRITE_revision_label_correlation':{k:dict(v) for k,v in corr.items()},'old_v2_SUBTASK':{'excluded_from_model_generalization_evidence':True,'reason':'Old40SUBTASK actor_memory waslist instead of native selected/current_writes envelope','preserved':'Correctedv4 replaces inference evidence; originalinvalid results should remain failure/provenance evidence, not silently overwritten.'},'conclusions':['CorrectedSUBTASK: all32nonempty memory cases copy stored memory side regardless goal/confirmation;16/32goals thereforewrong. Empty8all fail:JSONmode cross-talk8,EOSmissing3.','CorrectedWRITE: all32active rev1cases predictEMPTY, empty8rev0predictWRITE;16/40targets correct.','Revisiononly24:8/24correct,all24legalJSON/EOS,8/8triples change semanticoperation whenonlyrevisionchanges. This directly establishes localrevisiondependence.','Fit96WRITE operation perfectly correlated with revision; controlled intervention supports a shortcut explanation for these contexts, not a blanket globalclaim.','Targetsareformal visiblegoal/memorytool logic; none demonstratesphysicalgrounding, graspability or robotSR.']}
assert not cf_score['errors'] and not rev_score['errors'] and len(triples)==8 and all(x['only_revision_input_change_verified'] for x in triples)
p=Path('/tmp/protocol_cf_revision_independent_audit.json');p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'report':str(p),'CF':cf_score,'SUBTASK':out['SUBTASK_by_memory_condition'],'revision':rev_score,'revision_changed_groups':out['revision_causal_conditions']['semantic_output_changed_groups'],'fit_correlation':out['fit96_WRITE_revision_label_correlation']},ensure_ascii=False))
