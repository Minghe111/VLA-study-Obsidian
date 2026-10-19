"""Independent receipt acceptance, CPU only; no model weights are loaded."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import sentencepiece
import yaml

ROOT=Path('/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/recap_memory_20261017/experiments/protocol_repair_20261019_v2')
RUN=ROOT/'reports/continuation_v1'
issues=[]
def require(value,message):
    if not value:issues.append(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def stat(a):
    a=sorted(a)
    def q(p):
        k=(len(a)-1)*p;i=int(k)
        return a[i] if i+1==len(a) else a[i]+(a[i+1]-a[i])*(k-i)
    return {'count':len(a),'mean':sum(a)/len(a),'min':min(a),'median':q(.5),'p90':q(.9),'p95':q(.95),'max':max(a)}
summary=read(RUN/'summary.json')
result={'status':'review_running','review_scope':'CPU-only independent audit of JSON/log receipts; no model weights, GPU loads, or optimizer steps','stop_reason':summary['stop_reason'],'issues':issues,'segments':[],'final_checkpoints':{},'provenance':{}}
require(summary['stop_reason']=='converged_all96_and_local_retention_twice','unexpected stop reason')
require(summary['formal_3000_training'] is False and summary['task_SR_measured'] is False,'scope incorrect')
all_updates=[];reports=[];previous=300;freeze_hashes=set();data_fingerprints=set()
for target in range(600,2201,200):
    p=RUN/f'language_warmup_96_{target:04}';x=read(p/'report.json');u=read(p/'update_metrics.json');reports.append(x)
    expected=list(range(previous+1,target+1));actual=[r['global_step'] for r in u]
    require(actual==expected,f'C{target} update range differs')
    require(x['status']=='completed',f'C{target} not completed')
    require(x['resume']['exact_resume'] is True and x['resume']['global_step']==previous,f'C{target} resume differs')
    require(x['training']['global_step']==target and x['training']['updates']==target-previous,f'C{target} training steps differ')
    require(x['sampler_before']['microbatch_cursor']==4*previous and x['sampler_after']['microbatch_cursor']==4*target,f'C{target} sampler cursor differs')
    freeze_hashes.update([x['frozen_sha_before'],x['frozen_sha_after']]);data_fingerprints.update([x['sampler_before']['data_fingerprint'],x['sampler_after']['data_fingerprint']])
    require(x['frozen_parameters_exact'] is True and x['frozen_sha_before']==x['frozen_sha_after'],f'C{target} frozen changed')
    require(x['gradient_audit']=={'trainable_with_grad':144,'trainable_finite_grad':144,'frozen_with_grad':0},f'C{target} gradient audit differs')
    require(x['changed_trainable_tensors']==144 and x['max_trainable_delta']>0,f'C{target} missing tensor updates')
    require(x['model_audit']['trainable_numel']==3686400 and len(x['model_audit']['trainable_parameters'])==144,f'C{target} training scope differs')
    require(x['contract']=={'action_dim':24,'cameras':3,'period_s':.06,'H':16,'execute':1},f'C{target} robot contract differs')
    require(all(math.isfinite(v) for row in u for v in row.values() if isinstance(v,(int,float))),f'C{target} nonfinite update scalar')
    result['provenance'][str(p/'report.json')]=sha(p/'report.json');result['provenance'][str(p/'update_metrics.json')]=sha(p/'update_metrics.json')
    result['segments'].append({'target_step':target,'resume_step':previous,'updates':len(u),'first_last_update':[actual[0],actual[-1]],'exact_resume':x['resume']['exact_resume'],'sampler_microbatch_cursor':[x['sampler_before']['microbatch_cursor'],x['sampler_after']['microbatch_cursor']],'frozen_parameters_exact':x['frozen_parameters_exact'],'finite_and_changed_trainable_tensors':144,'evaluation_ce':x['evaluation']['teacher_forced_ce'],'fit_passed':x['evaluation']['fit_passed'],'memory':x['memory'],'sampled_card_peak_mib':x['sampled_card_peak_mib']})
    all_updates.extend(u);previous=target
require([x['global_step'] for x in all_updates]==list(range(301,2201)),'all 1900 update IDs not exactly301..2200')
require(len(freeze_hashes)==1,'frozen hashes differ across segments')
require(len(data_fingerprints)==1,'sampler data fingerprints differ across segments')
require(summary['total_additional_optimizer_updates']==1900,'summary additional update count differs')
result['continuation']={'segments':len(reports),'actual_additional_updates':len(all_updates),'global_step_range':[301,2200],'no_overlap_or_gaps':[x['global_step'] for x in all_updates]==list(range(301,2201)),'same_frozen_sha256_all_segments':list(freeze_hashes),'same_sampler_data_fingerprint':list(data_fingerprints),'all_update_grad_norm':stat([u['grad_norm_before_clip'] for u in all_updates]),'clipped_fraction':sum(u['exceeds_clip_threshold'] for u in all_updates)/len(all_updates)}
# CPU tokenizer only; no VLM or checkpoint weights.
config=yaml.safe_load((RUN/'language_warmup_96_2200/effective_config.yaml').read_text())
sp=sentencepiece.SentencePieceProcessor(model_file=config['tokenizer']['text_tokenizer_path'])
eos=sp.eos_id();require(eos==1,'unexpected EOS id')
label_path=Path(config['data']['verified_high_sft_paths'][0]);records=[json.loads(l) for l in label_path.read_text().splitlines() if l.strip()];by_id={r['sample_id']:r for r in records}
require(len(records)==96 and len(by_id)==96,'original labels not96 unique')
result['tokenizer']={'path':config['tokenizer']['text_tokenizer_path'],'sha256':sha(config['tokenizer']['text_tokenizer_path']),'eos_id':eos,'model_weights_loaded':False}
result['labels']={'path':str(label_path),'sha256':sha(label_path),'rows':len(records)}
for target in [2000,2200]:
    p=RUN/f'language_warmup_96_{target:04}';x=read(p/'report.json');evaluation=read(p/'evaluation.json');cases=evaluation['cases']
    require(len(cases)==96 and len(set(c['sample_id'] for c in cases))==96,f'C{target} not96 unique eval cases')
    require(set(c['sample_id'] for c in cases)==set(by_id),f'C{target} evaluated IDs differ from labels')
    require(evaluation['summary']==x['evaluation'],f'C{target} embedded/eval summaries differ')
    count=Counter();fails=[]
    for case in cases:
        sid=case['sample_id'];ref=by_id[sid];ids=case['token_ids'];checks=case['checks']
        checks_ok=checks.get('fit_passed') is True and checks.get('passed') is True and checks.get('eos') is True
        actual_eos=bool(ids) and ids[-1]==eos and ids.count(eos)==1
        decoded=sp.decode(ids)
        exact=case['text'].strip()==ref['target_text'].strip() and case['target_text']==ref['target_text'] and decoded.strip()==case['text'].strip()
        schema=True;content=True
        if case['mode'] in ('READ','WRITE'):
            value=json.loads(case['text']);expected=json.loads(ref['target_text'])
            require(type(value) is dict,f'C{target} {sid}: JSON object required')
            if case['mode']=='READ':
                read_ids=value.get('read_ids');available={r['slot_id'] for r in ref['context']['catalogue']}
                schema=set(value)=={'read_ids'} and type(read_ids) is list and all(type(i) is str for i in read_ids) and len(set(read_ids))==len(read_ids) and set(read_ids)<=available
                content=set(read_ids)==set(expected['read_ids'])
            else:
                operations=value.get('operations');schema=set(value)=={'operations'} and type(operations) is list
                states={r['slot_id']:r for r in ref['context']['selected_memory']}
                for op in operations:
                    allowed={'op','slot_id','value','confirmed','expected_revision'}
                    schema &= type(op) is dict and set(op)<=allowed and 'expected_revision' not in op and op.get('op') in {'WRITE','KEEP','CORRECT','REVOKE'} and op.get('slot_id') in ref['context']['read_ids'] and op.get('slot_id') in states and type(op.get('confirmed')) is bool
                    active=states[op['slot_id']].get('status')=='active'
                    schema &= (not active if op['op']=='WRITE' else active)
                    schema &= (type(op.get('value')) is str and bool(op['value'].strip())) if op['op'] in {'WRITE','CORRECT'} else op.get('value') in (None,'')
                content=operations==expected['operations']
            count['json']+=1;count['schema']+=bool(schema);count['content']+=bool(content)
            require(checks.get('json') is True and checks.get('schema_legal') is True and checks.get('target_match') is True,f'C{target} {sid}: stored JSON checks differ')
        else:
            count['subtask_exact']+=exact
        count['actual_eos']+=actual_eos;count['text_exact_from_original_labels']+=exact;count['fit_passed']+=checks_ok;count[case['mode']]+=1
        if not all([checks_ok,actual_eos,exact,schema,content]):fails.append(sid)
    require(not fails,f'C{target} per-case language failures:{fails}')
    require(count['fit_passed']==96 and count['actual_eos']==96 and count['text_exact_from_original_labels']==96 and count['schema']==64 and count['content']==64 and count['subtask_exact']==32,f'C{target} derived counts differ')
    retention={}
    for pool in ['retention','retention_holdout']:
        stored=x[pool];rp=p/pool/'report.json';actual=read(rp)
        require(actual==stored,f'C{target} {pool}: embedded/file report mismatch')
        require(len(actual['cases'])==8,f'C{target} {pool}: not8 cases')
        threshold=actual['thresholds'];require(threshold['normalized_physical24_rms_max']==.02 and threshold['masked_all_layer_kv_relative_rms_max']==.05,f'C{target} {pool}: thresholdschanged')
        vals=[]
        for c in actual['cases']:
            action=c['action_metrics']['normalized_physical24']['rms'];kv=c['kv_metrics']['relative_rms'];passed=c['context_exact'] and action<=.02 and kv<=.05
            require(passed and c['passed'],f'C{target} {pool} {c["case_id"]}: gatefailed')
            require(c['sdk_action_shape']==[1,16,24] and c['normalized_internal_shape']==[1,16,32],f'C{target} {pool}: shapecontractwrong')
            require(len(c['kv_metrics']['layers'])==18,f'C{target} {pool}: missingKVlayers')
            vals.append((action,kv))
        require(actual['passed'] is True and actual['summary']['passed']==8,f'C{target} {pool}: summaryfail')
        require(max(a for a,k in vals)==actual['summary']['max_case_normalized_physical24_rms'] and max(k for a,k in vals)==actual['summary']['max_case_masked_all_layer_kv_relative_rms'],f'C{target} {pool}: maxvaluesdiffersummary')
        retention[pool]={'cases':8,'derived_passed':8,'action_rms_max':max(a for a,k in vals),'kv_relative_rms_max':max(k for a,k in vals),'sdk_arm_rad_rms_max':max(c['action_metrics']['sdk_arm_rad']['rms'] for c in actual['cases']),'sdk_arm_rad_abs_max':max(c['action_metrics']['sdk_arm_rad']['max_abs'] for c in actual['cases']),'sdk_hand_units_rms_max':max(c['action_metrics']['sdk_hand_units']['rms'] for c in actual['cases']),'sdk_hand_units_abs_max':max(c['action_metrics']['sdk_hand_units']['max_abs'] for c in actual['cases']),'all_actions_exact':actual['summary']['all_sdk_actions_exact'],'context_exact_all':all(c['context_exact'] for c in actual['cases'])}
        result['provenance'][str(rp)]=sha(rp)
    result['final_checkpoints'][str(target)]={'derived_language_counts':dict(count),'failed_sample_ids':fails,'teacher_forced_ce':evaluation['summary']['teacher_forced_ce'],'gold_token_correct':evaluation['summary']['gold_correct'],'gold_token_total':evaluation['summary']['gold_tokens'],'min_gold_eos_probability':min(c['gold_eos_probability'] for c in cases),'retention':retention}
    result['provenance'][str(p/'evaluation.json')]=sha(p/'evaluation.json')
result['resource_maxima_mib']={'peak_allocated_mib':max(x['memory']['peak_allocated_mib'] for x in reports),'peak_reserved_mib':max(x['memory']['peak_reserved_mib'] for x in reports),'host_peak_rss_mib':max(x['memory']['host_peak_rss_mib'] for x in reports),'sampled_card_peak_mib':max(x['sampled_card_peak_mib'] for x in reports)}
result['resource_maxima_gib']={k.replace('_mib','_gib'):v/1024 for k,v in result['resource_maxima_mib'].items()}
result['sum_segment_elapsed_seconds']=sum(x['elapsed_s'] for x in reports)
result['limits']=['Same96 training diagnostic rows at two checkpoints, not an independent semantic generalization assessment.','Retention uses two8-state pools from one expert episode; heldout times were not KV training anchors but are not new-task success evidence.','Parameters and gradients were audited from stored execution receipts; this reviewer did not re-run forward/backward or load full checkpoint tensors.','Old actions meet local drift thresholds, not bitwise equality or closed-loop SR.','Whole-card peak is sampled rather than a guaranteed instantaneous maximum.']
result['status']='passed' if not issues else 'failed'
q=Path('/tmp/final_protocol_repair_acceptance.json');q.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':result['status'],'issues':issues,'additional_updates':len(all_updates),'final_checkpoints':result['final_checkpoints'],'resource_maxima_gib':result['resource_maxima_gib']},ensure_ascii=False))
if issues:raise SystemExit(1)
