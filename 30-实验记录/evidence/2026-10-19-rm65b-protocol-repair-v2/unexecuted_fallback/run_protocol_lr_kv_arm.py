"""One controlled C2000 weights-only LR x KV arm. Plans by default; no formal data."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT=Path('/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/recap_memory_20261017/experiments/protocol_repair_20261019_v2')
GRID=ROOT/'reports/lr_kv_from_C2000_v1'
SOURCE=ROOT/'reports/continuation_v1/language_warmup_96_2000/checkpoint'
ARMS={'lr5e5_kv1':(5e-5,1.0),'lr1e4_kv1':(1e-4,1.0),
      'lr5e5_kv0':(5e-5,0.0),'lr1e4_kv0':(1e-4,0.0)}
SEED=20261019
RESERVE=20*2**30


def load(path,default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(16*2**20),b''):
            h.update(chunk)
    return h.hexdigest()


def tree_bytes(path):
    if not Path(path).exists():
        return 0
    return sum(x.stat().st_size for x in Path(path).rglob('*') if x.is_file() and not x.is_symlink())


def disk_ledger(root,grid,source):
    reference=root/'reports/continuation_v1/language_warmup_96_1800'
    estimate=tree_bytes(reference)
    if estimate<=0:
        raise ValueError('Need the completed C1800 segment for conservative size budgeting')
    grid_written=tree_bytes(grid)
    pending_primary=max(0,estimate-tree_bytes(source.parent))
    remaining_grid=max(0,5*estimate-grid_written)  # Four arms plus one optional best-arm confirmation
    free=shutil.disk_usage(root).free
    required=pending_primary+remaining_grid+RESERVE
    return {'free_bytes':free,'reference_segment_bytes':estimate,'grid_written_bytes':grid_written,
        'pending_primary_estimate_bytes':pending_primary,'remaining_four_arms_plus_optional_confirmation_estimate_bytes':remaining_grid,
        'planned_saved_segments':5,
        'reserve_bytes':RESERVE,'required_bytes':required,'projected_margin_above_reserve_bytes':free-required,
        'passes':free>=required,'estimate_includes_full_weights_optimizer_and_two_retention_snapshots':True}


def arm_plan(arm,*,root=ROOT,grid=GRID,source=SOURCE):
    lr,kv=ARMS[arm]
    primary=load(root/'reports/continuation_v1/summary.json',{})
    report=load(source.parent/'report.json',{})
    available=primary.get('status')=='completed' and report.get('status')=='completed'
    model_audit=report.get('model_audit') or load(root/'reports/continuation_v1/language_warmup_96_1800/report.json',{})['model_audit']
    if model_audit.get('trainable_numel')!=3686400 or model_audit.get('training_stage')!='language_warmup':
        raise ValueError('Expected the existing full18-layer q/k/v/o rank16 attention scope')
    return {'status':'planned','arm':arm,'formal_3000_training':False,'primary_completed_and_source_ready':available,
        'conditional_trigger':'Execute only after primary C2000 is fully completed and strict full96 fitting still fails',
        'primary_status':primary.get('status'),'source_report_status':report.get('status'),
        'source_checkpoint':str(source),'source_checkpoint_global_step':2000,
        'branch_global_step_start':0,'branch_eval_steps':[400],
        'optional_confirmation':'Best passing arm may later be exactly resumed400->600 in a separate explicit run, within current authorized diagnostic scope',
        'branch_optimizer_update_budget':400,'diagnostic_window_not_convergence_limit':True,
        'seed':SEED,'optimizer':'AdamW','optimizer_reset':True,'sampler_reset':True,'rng_reset':True,
        'learning_rate':lr,'kv_anchor_weight':kv,'max_grad_norm':1.0,'weight_decay':0.0,
        'batch_size':1,'gradient_accumulation':4,'sequential_objectives':True,
        'loss_weights':{'high_ce':1.0,'fast_ce':0.0,'fm':0.0,'kv_anchor':kv},
        'parameter_ledger':{'stage':'language_warmup','trainable_numel':3686400,
            'trainable_tensors':len(model_audit['trainable_parameters']),
            'trainable_parameters':model_audit['trainable_parameters'],
            'vision_frozen':True,'MEM_bypassed':True,'action_expert_frozen':True,
            'MLP_norm_embedding_frozen':True},
        'contract':{'physical_action_dim':24,'camera_count':3,'period_s':.06,'H':16,'execute':1},
        'evaluations':'All96 unrestricted greedy+realEOS and anchor8/holdout8 local action/KV at branch400 only; per-update metrics and full sample traces retained',
        'all_arms':{name:{'learning_rate':v[0],'kv_anchor_weight':v[1]} for name,v in ARMS.items()},
        'disk_ledger':disk_ledger(root,grid,source),'output':str(grid/arm),'optimizer_updates_executed':0}


def validate_segment(out,target,*,source,previous=None):
    report=load(out/'report.json')
    if (report.get('status')!='completed' or report['training']['global_step']!=target
        or report.get('evaluation',{}).get('count')!=96 or report.get('frozen_parameters_exact') is not True
        or report.get('gradient_audit',{}).get('frozen_with_grad')!=0):
        raise ValueError('Branch segment acceptance failed')
    version=load(out/'checkpoint/version.json')
    if version.get('global_step')!=target:
        raise ValueError('Saved branch checkpoint step differs')
    if previous is None:
        initialization=report['branch_initialization']
        if (initialization.get('kind')!='weights_only_fork' or initialization.get('source_checkpoint_global_step')!=2000
            or initialization.get('branch_global_step_start')!=0 or report['sampler_before']['microbatch_cursor']!=0
            or report['resume'].get('restored') is not False):
            raise ValueError('First branch must load C2000 weights and reset Adam/sampler/RNG')
    elif (report['resume'].get('exact_resume') is not True or report['sampler_before']!=previous['sampler_after']):
        raise ValueError('Second branch segment requires exact same-contract resume')
    expected_updates=target if previous is None else target-previous['training']['global_step']
    if report['training'].get('updates')!=expected_updates:
        raise ValueError('Actual optimizer update count differs from the controlled arm budget')
    if report['sampler_before'].get('contract',{}).get('seed')!=SEED:
        raise ValueError('Actual fit sampler seed differs from common seed')
    if previous is None and report['branch_initialization'].get('seed')!=SEED:
        raise ValueError('Actual branch RNG initialization seed differs from common seed')
    if report['sampler_after']['microbatch_cursor']!=target*4:
        raise ValueError('Branch sampler/update boundary differs')
    for pool in ('retention','retention_holdout'):
        receipt=report.get(pool,{})
        if len(receipt.get('cases',[]))!=8 or receipt.get('summary',{}).get('count')!=8:
            raise ValueError('Both local retention evaluations require8 actual case receipts')
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=list(ARMS),required=True)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--enable-training',action='store_true')
    a=parser.parse_args(argv)
    if a.execute and not a.enable_training:
        parser.error('Optimizer updates require --enable-training')
    plan=arm_plan(a.arm)
    if not a.execute:
        print(json.dumps(plan,indent=2))
        return plan
    if not plan['primary_completed_and_source_ready']:
        raise ValueError('Primary C2000 is not fully completed; conditional experiment must wait')
    source_report=load(SOURCE.parent/'report.json')
    if source_report['training']['global_step']!=2000:
        raise ValueError('Controlled branches require exactly the same C2000 source checkpoint')
    if source_report['evaluation'].get('all_fit_passed') is True:
        raise ValueError('C2000 already fits all96; the proposed conditional LR/KV diagnostic is not triggered')
    if not plan['disk_ledger']['passes']:
        raise ValueError('Insufficient space for the complete four-arm grid plus20GiB reserve; preserve existing files')
    sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'code/RLinf')]
    from fit_comparison import resolve_training_labels, build_config, reset_branch_rng
    import inspect
    fit_config=build_config('language_warmup',96,ARMS[a.arm][0],ARMS[a.arm][1],max_fit_updates=400)
    if (fit_config['actor_training']['sampler']['seed']!=SEED
        or inspect.signature(reset_branch_rng).parameters['seed'].default!=SEED):
        raise ValueError('Actual fit configuration/RNG defaults do not match common seed')
    import run_language_repair as helper
    from protocol_pilot import write_json
    labels,label_provenance=resolve_training_labels({},checkpoint=SOURCE)
    version=load(SOURCE/'version.json')
    if version.get('global_step')!=2000 or version['training_manifest']['training_stage']!='language_warmup':
        raise ValueError('Source checkpoint manifest differs from C2000 full-language stage')
    weights=SOURCE/'full_weights.pt'
    if not weights.is_file() or not (SOURCE/'trainer_state.pt').is_file():
        raise ValueError('Source full checkpoint is incomplete')
    source_weights_sha=file_sha(weights)
    common={'source_checkpoint':str(SOURCE),'source_global_step':2000,'weights_sha256':source_weights_sha,
        'version_sha256':file_sha(SOURCE/'version.json'),'training_label_provenance':label_provenance,
        'seed':SEED,'branch_eval_steps':[400],'branch_start_step':0,
        'fit_driver_sha256':file_sha(ROOT/'scripts/fit_comparison.py'),
        'recap_objective_sha256':file_sha(ROOT/'code/RLinf/rlinf/models/embodiment/openpi/tasks/recap_hier.py'),
        'trainable_parameter_manifest_sha256':hashlib.sha256(json.dumps(plan['parameter_ledger']['trainable_parameters'],sort_keys=True).encode()).hexdigest()}
    GRID.mkdir(parents=True,exist_ok=True)
    common_path=GRID/'common_manifest.json'
    if common_path.exists():
        if load(common_path)!=common:
            raise ValueError('Common source/seed/data/parameter manifest differs from previous arms')
    else:
        with common_path.open('x') as stream:
            json.dump(common,stream,indent=2)
    arm_dir=GRID/a.arm
    arm_dir.mkdir(exist_ok=False)
    (arm_dir/'logs').mkdir()
    state={**plan,'status':'running','started_unix':time.time(),'common_manifest':common,'steps':[],
        'source_weights_unchanged_sha256':source_weights_sha}
    write_json(arm_dir/'plan.json',plan)
    def save():
        write_json(arm_dir/'summary.json',state)
    save()
    previous=None
    source=SOURCE
    lr,kv=ARMS[a.arm]
    primary=load(ROOT/'reports/continuation_v1/summary.json')
    c300=Path(primary['source_checkpoint'])
    baseline=c300.parent.parent/'00_baseline/retention/snapshot.pt'
    if not baseline.is_file():
        raise ValueError('Original retention baseline is unavailable')
    try:
        for target in (400,):
            out=arm_dir/f'branch_{target:04d}'
            args=[sys.executable,str(ROOT/'scripts/fit_comparison.py'),'fit','--stage','language_warmup',
                '--size','96','--output',str(out),'--total-updates',str(target),'--max-fit-updates','400',
                '--training-labels',str(labels),'--learning-rate',str(lr),'--kv-weight',str(kv),
                '--baseline',str(baseline),'--eval-all','--execute','--enable-training',
                '--fork-from' if previous is None else '--resume',str(source)]
            log=arm_dir/'logs'/f'branch_{target:04d}.log'
            entry={'target_branch_step':target,'source_checkpoint_global_step':2000,
                'argv':args,'status':'running','log':str(log),'started_unix':time.time()}
            state['steps'].append(entry)
            save()
            helper.run_subprocess(args,ROOT,log)
            report=validate_segment(out,target,source=source,previous=previous)
            entry.update(status='completed',exit_code=0,evaluation=report['evaluation'],
                retention=report['retention']['summary'],retention_holdout=report['retention_holdout']['summary'],
                memory=report['memory'],elapsed_s=report.get('elapsed_s'))
            state['optimizer_updates_executed']=target
            source=out/'checkpoint'
            state['final_checkpoint']=str(source)
            state['latest_branch_step']=target
            previous=report
            save()
        state['status']='completed'
        state['full96_fitted_at_end']=previous['evaluation']['all_fit_passed']
        state['local_retention_passed_at_end']=previous['retention']['passed'] and previous['retention_holdout']['passed']
        state['budget_interpretation']='400-update conditional diagnostic window, not proof of convergence or model capacity'
    except BaseException:
        state['status']='failed'
        state['error']=traceback.format_exc()
        if state['steps'] and state['steps'][-1]['status']=='running':
            state['steps'][-1]['status']='failed'
        raise
    finally:
        state['finished_unix']=time.time()
        save()
    return state


if __name__=='__main__':
    main()
