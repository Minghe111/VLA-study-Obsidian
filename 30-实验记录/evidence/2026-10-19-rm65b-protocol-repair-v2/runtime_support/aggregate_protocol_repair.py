"""Read-only CPU aggregation of RM65-B small-sample continuation receipts."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

DEFAULT_ROOT = Path('/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/recap_memory_20261017/experiments/protocol_repair_20261019_v2')


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def label_records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def summarize_cases(cases):
    result = {'count': len(cases)}
    if not cases:
        return result
    result['teacher_forced_ce_macro'] = statistics.mean(c['ce'] for c in cases)
    result['gold_tokens'] = sum(c['gold_tokens'] for c in cases)
    result['gold_correct'] = sum(c['gold_correct'] for c in cases)
    result['teacher_forced_ce_token_weighted'] = sum(c['ce']*c['gold_tokens'] for c in cases)/result['gold_tokens']
    for field in ('eos', 'first_token_match', 'json', 'schema_legal', 'target_match', 'required_terms', 'side_binding_lexical', 'text_exact', 'fit_passed'):
        eligible = [c['checks'][field] for c in cases if field in c['checks']]
        result[field] = {'passed': sum(eligible), 'eligible': len(eligible)}
    result['all_fit_passed'] = all(c['checks'].get('fit_passed') is True for c in cases)
    return result


def evaluation_summary(directory, subset_ids):
    data = read_json(directory/'evaluation.json')
    if not data or not data.get('summary'):
        return None
    cases = data['cases']
    return {'reported_summary': data['summary'], 'all': summarize_cases(cases),
        'by_mode': {m:summarize_cases([c for c in cases if c['mode']==m]) for m in ('READ','WRITE','SUBTASK')},
        'common_subsets': {name:{'expected':len(ids), 'coverage_complete':sum(c['sample_id'] in ids for c in cases)==len(ids),
            **summarize_cases([c for c in cases if c['sample_id'] in ids])} for name,ids in subset_ids.items()},
        'failed_cases': [{'sample_id':c['sample_id'], 'mode':c['mode'], 'target_text':c['target_text'],
            'text':c['text'], 'checks':c['checks']} for c in cases if not c['checks'].get('fit_passed')]}


def retention_summary(receipt):
    if not receipt:
        return None
    cases = receipt.get('cases', [])
    if not cases:
        return {'available':False}
    thresholds = receipt['thresholds']
    action_gate=thresholds['normalized_physical24_rms_max']
    kv_gate=thresholds['masked_all_layer_kv_relative_rms_max']
    return {'available':True,'count':len(cases),'reported_passed':receipt['passed'],
        'reported_summary':receipt['summary'],'thresholds':thresholds,
        'physical_action_gate':{'passed':sum(c['action_metrics']['normalized_physical24']['rms']<=action_gate for c in cases), 'eligible':len(cases)},
        'kv_gate':{'passed':sum(c['kv_metrics']['relative_rms']<=kv_gate for c in cases),'eligible':len(cases)},
        'context_exact':{'passed':sum(c['context_exact'] for c in cases),'eligible':len(cases)},
        'max_normalized_physical24_rms':max(c['action_metrics']['normalized_physical24']['rms'] for c in cases),
        'max_normalized_physical24_abs':max(c['action_metrics']['normalized_physical24']['max_abs'] for c in cases),
        'max_pooled_all_layer_kv_relative_rms':max(c['kv_metrics']['relative_rms'] for c in cases),
        'max_sdk_arm_error_rad':max(c['action_metrics']['sdk_arm_rad']['max_abs'] for c in cases),
        'max_sdk_arm_case_rms_rad':max(c['action_metrics']['sdk_arm_rad']['rms'] for c in cases),
        'max_sdk_hand_error_units':max(c['action_metrics']['sdk_hand_units']['max_abs'] for c in cases),
        'max_sdk_hand_case_rms_units':max(c['action_metrics']['sdk_hand_units']['rms'] for c in cases),
        'local_mapping_only_not_rollout_SR':True}


def metric_window(metrics):
    if not metrics:
        return {'count':0}
    fields=('loss','high_ce','kv_anchor','grad_norm_before_clip','clip_scale_estimate')
    return {'count':len(metrics),'first_global_step':metrics[0]['global_step'],
        'last_global_step':metrics[-1]['global_step'],
        'clip_fraction':statistics.mean(bool(x['exceeds_clip_threshold']) for x in metrics),
        **{field:{'mean':statistics.mean(x[field] for x in metrics),'min':min(x[field] for x in metrics),
            'max':max(x[field] for x in metrics)} for field in fields if all(field in x for x in metrics)}}


def update_summary(directory, report):
    path = directory/'update_metrics.json'
    metrics=read_json(path)
    if metrics is None:
        path=directory/'step_metrics.json'
        metrics=read_json(path)
    if not metrics:
        return {'available':False,'reported_training':report.get('training')}
    start=report['branch_initialization']['branch_global_step_start']
    end=report['training']['global_step']
    return {'available':True,'path':str(path),'ordered_global_steps_exact':
        [x['global_step'] for x in metrics]==list(range(start+1,end+1)),
        'all':metric_window(metrics),'last50':metric_window(metrics[-50:])}


def get_source_paths(root, state):
    source=Path(state['source_checkpoint']).resolve()
    phase1root=source.parents[3]
    baseline=source.parent
    history=[]
    for p in (phase1root/'reports/comparison_v1').glob('language_warmup_96_*'):
        report=read_json(p/'report.json',{})
        if report.get('status')=='completed' and report.get('training',{}).get('global_step',10**9)<=300:
            history.append((report['training']['global_step'],p,report))
    return baseline,sorted(history)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    p.add_argument('--output',type=Path,default=Path('/tmp/protocol_repair_aggregate.json'))
    a=p.parse_args(argv)
    root=a.root.resolve()
    state=read_json(root/'reports/continuation_v1/summary.json')
    if state is None:
        raise ValueError('Orchestrator summary is not yet readable')
    baseline,source_history=get_source_paths(root,state)
    subset_ids={name:{r['sample_id'] for r in label_records(root/f'data/fit_subsets_v1/{file}.train.jsonl')}
        for name,file in (('common9','fit9'),('quick12','fit96.quick12'))}
    runs=[]
    in_progress=[]
    for directory in (root/'reports/continuation_v1').glob('language_warmup_96_*'):
        report=read_json(directory/'report.json',{})
        if report.get('status')=='completed':
            runs.append((report['training']['global_step'],directory,report))
        else:
            in_progress.append({'directory':str(directory),'status':report.get('status','report_unreadable'),
                'progress':read_json(directory/'progress.json')})
    runs.sort()
    import yaml
    config=yaml.safe_load((baseline/'effective_config.yaml').read_text())
    sys.path[:0]=[str(root/'scripts'),str(root/'code/RLinf')]
    from rlinf.models.embodiment.openpi.recap_tokenizer import RecapTokenizer
    from rlinf.runners.hierarchical_recap_runner import ActorMicrobatchSampler
    from protocol_pilot import read_records
    tokenizer=RecapTokenizer.from_local_paths(config['tokenizer']['text_tokenizer_path'],
        max_prefix_tokens=config['tokenizer']['max_prefix_tokens'],max_target_tokens=config['tokenizer']['max_target_tokens'])
    labels=Path(config['data']['verified_high_sft_paths'][0])
    rows=read_records([labels])
    tokens={row['sample_id']:len(tokenizer.encode_target(row['target_text'])) for row in rows}
    modes={row['sample_id']:row['mode'] for row in rows}
    if len(tokens)!=96:
        raise ValueError('Expected exactly96 distinct audited training labels')
    source_sft=Path('/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/recap_memory_20261017/experiments/protocol_sft_20261018_v1')
    manifest_path=source_sft/'data/retention_anchor/manifest.json'
    refs=[{'observation':{'npz_path':str(manifest_path.parent/x['observation']['npz_path'])},'instruction':x['instruction']}
        for x in read_json(manifest_path)['cases']]
    for row in rows:
        row['_fit_retention_contract']=refs
    sampler=ActorMicrobatchSampler(rows,[],[],[],batch_size=1,gradient_accumulation=4,anchor_interval=4,
        training_mode='verified_sft',loss_weights=config['actor_training']['loss_weights'],shuffle=True,seed=20261019)
    sequence=source_history+runs
    cumulative_tokens=cumulative_presentations=0
    exposure=Counter()
    boundaries=[]
    trace_reports=[]
    previous=None
    for step,directory,report in sequence:
        trace=read_json(directory/'sample_trace.json',[])
        ordered=True
        trace_tokens=0
        counts=Counter()
        for item in trace:
            high,_,_=sampler.next_batch()
            actual_id=item['sample_id']
            ordered &= (item['microbatch']==sampler.microbatch_cursor and actual_id==high[0]['sample_id'])
            if actual_id not in tokens:
                raise ValueError('Actual sampled label is absent from96-row source')
            trace_tokens+=tokens[actual_id]
            exposure[actual_id]+=1
            counts[modes[actual_id]]+=1
        cumulative_presentations+=len(trace)
        cumulative_tokens+=trace_tokens
        before=report['sampler_before'];after=report['sampler_after']
        boundary={'step':step,'path':str(directory),'actual_id_order_matches_replayed_sampler':bool(ordered),
            'before_cursor':before['microbatch_cursor'],'after_cursor':after['microbatch_cursor'],
            'cursor_delta_matches_trace':after['microbatch_cursor']-before['microbatch_cursor']==len(trace),
            'saved_after_matches_replayed_state':after==json.loads(json.dumps(sampler.state_dict())),
            'fingerprint_matches_original_data':before['data_fingerprint']==after['data_fingerprint']==sampler.fingerprint,
            'resume_exact':report.get('resume',{}).get('exact_resume'),
            'previous_after_equals_current_before':None if previous is None else previous['sampler_after']==before}
        if previous is not None:
            boundary['previous_frozen_hash_equals_current']=previous['frozen_sha_after']==report['frozen_sha_before']
        boundaries.append(boundary)
        trace_reports.append({'step':step,'path':str(directory),'presentations':len(trace),'supervised_tokens_including_EOS':trace_tokens,
            'mode_presentations':dict(counts),'cumulative_presentations':cumulative_presentations,
            'cumulative_supervised_tokens_including_EOS':cumulative_tokens,
            'trace_sha256':sha256(directory/'sample_trace.json'),
            'ordered_sample_ids_sha256':hashlib.sha256(json.dumps([x['sample_id'] for x in trace]).encode()).hexdigest()})
        previous=report
    def run_receipt(step,directory,report):
        return {'step':step,'directory':str(directory),'status':report['status'],
            'evaluation':evaluation_summary(directory,subset_ids),
            'retention_anchor8':retention_summary(report.get('retention')),
            'retention_holdout8':retention_summary(report.get('retention_holdout')),
            'updates':update_summary(directory,report),'memory':report.get('memory'),
            'sampled_card_peak_mib':report.get('sampled_card_peak_mib'),
            'resume':report.get('resume'),'frozen_parameters_exact':report.get('frozen_parameters_exact'),
            'gradient_audit':report.get('gradient_audit')}
    baseline_report=read_json(baseline/'report.json')
    all_retained=[]
    for _,_,report in runs:
        all_retained.append(report['evaluation'].get('all_fit_passed') is True and report['evaluation'].get('count')==96
            and report.get('retention',{}).get('passed') is True and report.get('retention_holdout',{}).get('passed') is True)
    output={'format':'rm65b_protocol_repair_aggregate_v1','generated_utc':datetime.now(timezone.utc).isoformat(),
        'root':str(root),'script_sha256':sha256(Path(__file__)),
        'scope':'Read-only CPU receipts, training-set fitting and local action mapping; no heldout task SR',
        'formal_3000_training':False,'orchestrator_status':state['status'],'stop_reason':state.get('stop_reason'),
        'confirmed_converged':state.get('status')=='completed' and state.get('stop_reason')=='converged_all96_and_local_retention_twice'
            and len(all_retained)>=2 and all(all_retained[-2:]),
        'latest_completed_global_step':runs[-1][0] if runs else 300,'in_progress':in_progress,
        'source_C300':run_receipt(300,baseline,baseline_report),
        'completed_continuation_runs':[run_receipt(*item) for item in runs],
        'continuous_sampler_boundaries':boundaries,'continuous_sample_traces':trace_reports,
        'cumulative_presentations':cumulative_presentations,'cumulative_supervised_tokens_including_EOS':cumulative_tokens,
        'per_sample_exposure':[{'sample_id':sample_id,'mode':modes[sample_id],'presentations':exposure[sample_id],
            'supervised_tokens_per_presentation_including_EOS':tokens[sample_id],
            'total_supervised_tokens_including_EOS':tokens[sample_id]*exposure[sample_id]} for sample_id in sorted(tokens)],
        'training_labels':{'path':str(labels),'sha256':sha256(labels),'rows':96,'one_pass_supervised_tokens_including_EOS':sum(tokens.values()),
            'tokenizer':config['tokenizer']['text_tokenizer_path'],'tokenizer_sha256':sha256(config['tokenizer']['text_tokenizer_path'])},
        'all_completed_sampler_checks_passed':all(b['actual_id_order_matches_replayed_sampler'] and b['cursor_delta_matches_trace']
            and b['saved_after_matches_replayed_state'] and b['fingerprint_matches_original_data']
            and b['previous_after_equals_current_before'] is not False for b in boundaries)}
    a.output.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:output[k] for k in ('orchestrator_status','latest_completed_global_step','confirmed_converged',
        'cumulative_presentations','cumulative_supervised_tokens_including_EOS','all_completed_sampler_checks_passed')}))
    return output


if __name__=='__main__':
    main()
