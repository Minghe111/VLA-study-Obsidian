"""Confirm the first passing C2000 checkpoint with exactly200 more same-contract updates."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback

ROOT=Path('/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/recap_memory_20261017/experiments/protocol_repair_20261019_v2')
CONTINUATION=ROOT/'reports/continuation_v1'


def load(path):
    return json.loads(Path(path).read_text())


def validate_completed(directory,target,previous):
    report=load(directory/'report.json')
    if (report.get('status')!='completed' or report['training']['global_step']!=target
        or report['training']['updates']!=target-previous['training']['global_step']
        or report.get('evaluation',{}).get('count')!=96 or report.get('resume',{}).get('exact_resume') is not True):
        raise ValueError('Completed segment/update/full96/exact-resume verification failed')
    if (report.get('frozen_parameters_exact') is not True or report['gradient_audit'].get('frozen_with_grad')!=0
        or report['frozen_sha_before']!=report['frozen_sha_after']):
        raise ValueError('Frozen parameter protection failed')
    if report['sampler_before']!=previous['sampler_after'] or report['sampler_after']['microbatch_cursor']!=target*4:
        raise ValueError('Complete sampler continuity or cursor verification failed')
    checkpoint=directory/'checkpoint'
    version=load(checkpoint/'version.json')
    if version.get('global_step')!=target or version['training_manifest']['training_stage']!='language_warmup':
        raise ValueError('Saved checkpoint version differs')
    if not all((checkpoint/name).is_file() for name in ('full_weights.pt','trainer_state.pt')):
        raise ValueError('Full resumable checkpoint missing')
    for pool in ('retention','retention_holdout'):
        receipt=report.get(pool,{})
        if len(receipt.get('cases',[]))!=8 or receipt.get('summary',{}).get('count')!=8:
            raise ValueError('Both local retention receipts must contain8 actual cases')
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--enable-training',action='store_true')
    a=p.parse_args(argv)
    if a.execute and not a.enable_training:
        p.error('Optimizer updates require --enable-training')
    sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'code/RLinf')]
    import run_language_repair as helper
    from fit_comparison import resolve_training_labels
    from protocol_pilot import write_json
    import yaml
    summary_path=CONTINUATION/'summary.json'
    original_bytes=summary_path.read_bytes()
    state=json.loads(original_bytes)
    if (state.get('status')!='completed' or state.get('max_total_updates')!=2000
        or state.get('source_checkpoint_global_step')!=300):
        raise ValueError('Confirmation expects a completed original C300-to-C2000 run')
    c300=Path(state['source_checkpoint'])
    baseline=c300.parent.parent/'00_baseline/retention/snapshot.pt'
    if not baseline.is_file():
        raise ValueError('Original all-adapters-off retention baseline missing')
    previous=load(c300.parent/'report.json')
    reports=[]
    for target in (600,800,1000,1200,1400,1600,1800,2000):
        directory=CONTINUATION/f'language_warmup_96_{target:04d}'
        report=validate_completed(directory,target,previous)
        reports.append(report)
        previous=report
    if not helper.both_gates_pass(reports[-1]) or any(helper.both_gates_pass(x) for x in reports[:-1]):
        raise ValueError('Expected C2000 to be the first full96+anchor8+holdout8 passing checkpoint')
    source=CONTINUATION/'language_warmup_96_2000/checkpoint'
    source_config=yaml.safe_load((source.parent/'effective_config.yaml').read_text())
    training=source_config['actor_training']
    if training['learning_rate']!=5e-5 or training['loss_weights']['kv_anchor']!=1.0:
        raise ValueError('Confirmation must keep original LR5e-5/KV1 objectives')
    labels,label_provenance=resolve_training_labels({},checkpoint=source)
    out=CONTINUATION/'language_warmup_96_2200'
    if out.exists() or any(x.get('name')==out.name for x in state['steps']):
        raise ValueError('C2200 confirmation already exists; no retraining or overwrite permitted')
    args=[sys.executable,str(ROOT/'scripts/fit_comparison.py'),'fit',
        '--stage','language_warmup','--size','96','--output',str(out),
        '--resume',str(source),'--training-labels',str(labels),'--baseline',str(baseline),
        '--total-updates','2200','--max-fit-updates','2200',
        '--learning-rate','5e-05','--kv-weight','1.0','--eval-all','--execute','--enable-training']
    extension={'at_utc':datetime.now(timezone.utc).isoformat(),'previous_cap':2000,
        'previous_stop_reason':state.get('stop_reason'),'previous_summary_snapshot':'summary_cap2000.json',
        'new_cap':2200,'first_all_pass_checkpoint':2000,'additional_optimizer_updates':200,
        'source_checkpoint_global_step':300,'total_additional_optimizer_updates_after_confirmation':1900,
        'reason':'second checkpoint confirmation after first C2000allpass',
        'same_learning_rate':5e-5,'same_kv_anchor_weight':1.0,
        'training_label_provenance':label_provenance,'no_lr_kv_branches_started':True}
    if not a.execute:
        result={'status':'planned','optimizer_updates_executed':0,'extension':extension,'argv':args}
        print(json.dumps(result,indent=2))
        return result
    with (CONTINUATION/'summary_cap2000.json').open('xb') as stream:
        stream.write(original_bytes)
    state['confirmation_extension']=extension
    state['status']='running'
    state['max_total_updates']=2200
    state['additional_optimizer_update_cap']=1900
    if state.get('eval_targets') and 2200 not in state['eval_targets']:
        state['eval_targets'].append(2200)
    state.pop('stop_reason',None)
    state.pop('finished_unix',None)
    log=CONTINUATION/'logs/language_warmup_96_2200.log'
    entry={'name':out.name,'argv':args,'status':'running','log':str(log),
        'start_global_step':2000,'target_global_step':2200,'additional_optimizer_updates':200,
        'started_unix':time.time(),'confirmation_only':True}
    state['steps'].append(entry)
    def save():
        write_json(summary_path,state)
    save()
    try:
        helper.run_subprocess(args,ROOT,log)
        report=validate_completed(out,2200,reports[-1])
        reports.append(report)
        entry.update(status='completed',exit_code=0,evaluation=report['evaluation'],memory=report['memory'],
            sampled_card_peak_mib=report.get('sampled_card_peak_mib'),
            retention_summary=report['retention']['summary'],retention_holdout_summary=report['retention_holdout']['summary'],
            both_gates_pass=helper.both_gates_pass(report),elapsed_s=report.get('elapsed_s'))
        raw=helper.decide_stop(reports,2200,2200)
        state['confirmation_raw_stop_decision']=raw
        state['stop_reason']=raw if raw=='converged_all96_and_local_retention_twice' else 'not_confirmed_at_cap'
        state['final_checkpoint']=str(out/'checkpoint')
        state['total_additional_optimizer_updates']=1900
        state['status']='completed'
    except BaseException:
        state['status']='failed'
        state['error']=traceback.format_exc()
        if entry['status']=='running':
            entry['status']='failed'
        raise
    finally:
        state['finished_unix']=time.time()
        save()
    return state


if __name__=='__main__':
    main()
