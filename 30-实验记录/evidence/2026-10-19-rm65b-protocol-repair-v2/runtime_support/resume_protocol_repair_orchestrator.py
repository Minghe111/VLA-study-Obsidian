"""Recover interrupted serial orchestration; never rerun a completed fit segment."""
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
CAP=2000


def load(path):
    return json.loads(Path(path).read_text())


def validate_run(directory, target, previous_report):
    report=load(directory/'report.json')
    if (report.get('status')!='completed' or report.get('stage')!='language_warmup' or report.get('size')!=96
        or report.get('training',{}).get('global_step')!=target or report.get('evaluation',{}).get('count')!=96
        or report.get('resume',{}).get('exact_resume') is not True):
        raise ValueError(f'Completed run/resume/full96 receipt failed: {directory}')
    if report.get('frozen_parameters_exact') is not True or report.get('gradient_audit',{}).get('frozen_with_grad')!=0:
        raise ValueError(f'Frozen parameter protection failed: {directory}')
    if report.get('frozen_sha_before')!=report.get('frozen_sha_after'):
        raise ValueError(f'Frozen hashes disagree: {directory}')
    start=previous_report['training']['global_step']
    if report['training']['updates']!=target-start:
        raise ValueError('Reported additional optimizer updates differ from exact continuation')
    if report['sampler_before']!=previous_report['sampler_after']:
        raise ValueError('Full sampler state does not match previous completed checkpoint')
    if (report['sampler_before']['microbatch_cursor']!=start*4
        or report['sampler_after']['microbatch_cursor']!=target*4):
        raise ValueError('Sampler boundary does not match update/accumulation contract')
    if report['sampler_before']['data_fingerprint']!=report['sampler_after']['data_fingerprint']:
        raise ValueError('Strict sampler data fingerprint changed')
    checkpoint=directory/'checkpoint'
    version=load(checkpoint/'version.json')
    if (version.get('global_step')!=target
        or version.get('training_manifest',{}).get('training_stage')!='language_warmup'
        or version.get('metadata',{}).get('fit_dataset_size')!=96):
        raise ValueError('Checkpoint version differs from completed training receipt')
    if any(not (checkpoint/name).is_file() for name in ('full_weights.pt','trainer_state.pt')):
        raise ValueError('Checkpoint weights or trainer state missing')
    if not report.get('retention') or not report.get('retention_holdout'):
        raise ValueError('Completed run lacks one of the local retention receipts')
    return report


def complete_entry(entry, report, helper):
    entry.update(status='completed', evaluation=report['evaluation'], memory=report['memory'],
        sampled_card_peak_mib=report.get('sampled_card_peak_mib'),
        retention_summary=report['retention'].get('summary'),
        retention_holdout_summary=report['retention_holdout'].get('summary'),
        both_gates_pass=helper.both_gates_pass(report),
        elapsed_s=report.get('elapsed_s'))
    return entry


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--enable-training',action='store_true')
    a=p.parse_args(argv)
    if a.execute and not a.enable_training:
        p.error('Recovery mutation and additional optimizer updates require --enable-training')
    sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'code/RLinf')]
    import run_language_repair as helper
    from fit_comparison import resolve_training_labels
    from protocol_pilot import write_json
    summary_path=CONTINUATION/'summary.json'
    original_bytes=summary_path.read_bytes()
    state=json.loads(original_bytes)
    if state.get('status')!='running' or state.get('max_total_updates')!=CAP or state.get('source_checkpoint_global_step')!=300:
        raise ValueError('Recovery is fixed to the interrupted original C300-to-C2000 continuation')
    source=Path(state['source_checkpoint'])
    # C300 lives under comparison_v1/C300/checkpoint; its sibling baseline is comparison_v1/00_baseline.
    baseline=source.parent.parent/'00_baseline/retention/snapshot.pt'
    if not baseline.is_file():
        raise ValueError('Original all-adapters-off retention baseline is missing')
    previous=load(source.parent/'report.json')
    if previous['training']['global_step']!=300:
        raise ValueError('Source baseline must be actual C300')
    labels,_=resolve_training_labels({},checkpoint=source)
    reports=[]
    recovered=[]
    for target in (600,800,1000,1200,1400):
        name=f'language_warmup_96_{target:04d}'
        directory=CONTINUATION/name
        report=validate_run(directory,target,previous)
        entry=next((x for x in state['steps'] if x.get('name')==name),None)
        if entry is None:
            raise ValueError('Completed run absent from original orchestrator step list')
        old_status=entry.get('status')
        complete_entry(entry,report,helper)
        if old_status!='completed':
            entry['recovered_from_completed_disk_receipts']=True
            entry['original_orchestrator_status']=old_status
            entry['recovery_did_not_observe_original_child_exit_code']=True
            recovered.append(name)
        reports.append(report)
        previous=report
        source=directory/'checkpoint'
    reason=helper.decide_stop(reports,1400,CAP)
    remaining=[] if reason is not None else [1600,1800,2000]
    for target in remaining:
        if (CONTINUATION/f'language_warmup_96_{target:04d}').exists():
            raise ValueError('Recovery refuses to replace or rerun an existing remaining segment')
    recovery={'at_utc':datetime.now(timezone.utc).isoformat(),
        'root_tool_observed_parent_session_exit_code':143,
        'observation_source':'Root agent reported tool session 94875 exited143; this recovery script did not observe that exit itself',
        'cause':'unknown; not established as a training numerical failure',
        'actual_last_completed_global_step':1400,'source_checkpoint_global_step':300,
        'completed_disk_receipts_verified':[600,800,1000,1200,1400],
        'entries_reconciled':recovered,'remaining_targets':remaining,
        'decide_stop_at_recovery':reason,'additional_optimizer_update_cap':600,
        'max_total_updates':CAP,'do_not_retrain_completed_segments':True}
    if not a.execute:
        result={'status':'planned','optimizer_updates_executed':0,'recovery':recovery}
        print(json.dumps(result,indent=2))
        return result
    backup=CONTINUATION/'summary_interrupted.json'
    with backup.open('xb') as stream:
        stream.write(original_bytes)
    state['recovery']=recovery
    state['status']='running'
    state['final_checkpoint']=str(source)
    state['total_additional_optimizer_updates']=1100
    state.pop('error',None)
    state.pop('finished_unix',None)
    def save():
        write_json(summary_path,state)
    save()
    try:
        if reason is not None:
            state['stop_reason']=reason
        else:
            for target in remaining:
                name=f'language_warmup_96_{target:04d}'
                out=CONTINUATION/name
                args=[sys.executable,str(ROOT/'scripts/fit_comparison.py'),'fit',
                    '--stage','language_warmup','--size','96','--output',str(out),
                    '--resume',str(source),'--training-labels',str(labels),'--baseline',str(baseline),
                    '--total-updates',str(target),'--max-fit-updates','2000',
                    '--learning-rate','5e-05','--kv-weight','1.0','--eval-all','--execute','--enable-training']
                log=CONTINUATION/'logs'/f'{name}.log'
                entry={'name':name,'argv':args,'status':'running','log':str(log),
                    'start_global_step':previous['training']['global_step'],'target_global_step':target,
                    'additional_optimizer_updates':target-previous['training']['global_step'],
                    'started_unix':time.time(),'created_by_recovery':True}
                state['steps'].append(entry)
                save()
                helper.run_subprocess(args,ROOT,log)
                report=validate_run(out,target,previous)
                entry['exit_code']=0  # helper returned only after observing successful child exit
                complete_entry(entry,report,helper)
                reports.append(report)
                previous=report
                source=out/'checkpoint'
                state['final_checkpoint']=str(source)
                state['total_additional_optimizer_updates']=target-300
                reason=helper.decide_stop(reports,target,CAP)
                if reason is not None:
                    state['stop_reason']=reason
                    break
                save()
        state['status']='completed'
    except BaseException:
        state['status']='failed'
        state['error']=traceback.format_exc()
        if state['steps'][-1]['status']=='running':
            state['steps'][-1]['status']='failed'
        raise
    finally:
        state['finished_unix']=time.time()
        save()
    return state


if __name__=='__main__':
    main()
