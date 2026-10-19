"""Whitelist-only evidence archiving. Default is plan-only; never copy model assets."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

DEFAULT_ROOT = Path('/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/recap_memory_20261017/experiments/protocol_repair_20261019_v2')
DEFAULT_DESTINATION = Path('/home/admin123/liminghe/VLA-WAM-RL/30-实验记录/evidence/2026-10-19-rm65b-protocol-repair-v2/runtime')
RUN_FILES = ('report.json','evaluation.json','update_metrics.json','progress.json',
    'sample_trace.json','loss_trace.json','effective_config.yaml',
    'retention/report.json','retention_holdout/report.json')
REQUIRED_FINAL_RUN_FILES = set(RUN_FILES)-{'progress.json'}
RUN_PATTERN = re.compile(r'language_warmup_96_[0-9]{4}')


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def validated_bytes(source, *, max_bytes=32*2**20):
    source=Path(source)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f'Evidence must be a regular file, without symlink: {source}')
    if source.stat().st_size>max_bytes:
        raise ValueError(f'Evidence exceeds the explicit small-file limit: {source}')
    data=source.read_bytes()
    if len(data)>max_bytes:
        raise ValueError(f'Evidence grew beyond small-file limit: {source}')
    if source.suffix=='.json':
        def reject_nonfinite(value):
            raise ValueError(f'Nonfinite JSON value in {source}: {value}')
        parsed=json.loads(data, parse_constant=reject_nonfinite)
        kind='json'
    elif source.name=='effective_config.yaml':
        import yaml
        parsed=yaml.safe_load(data)
        if not isinstance(parsed,dict):
            raise ValueError('Effective configuration must be a YAML mapping')
        kind='yaml'
    else:
        raise ValueError(f'Not a whitelisted evidence format: {source}')
    return data,parsed,kind


def gather(root, aggregate, diagnostics, allow_running=False):
    root=Path(root).resolve()
    continuation=root/'reports/continuation_v1'
    summary_path=continuation/'summary.json'
    summary_bytes,summary,_=validated_bytes(summary_path)
    final=summary.get('status')=='completed'
    if not final and not allow_running:
        raise ValueError(f"Final archive refused: orchestrator status={summary.get('status')}; use --allow-running for an explicit snapshot")
    entries=[(summary_path,'continuation_v1/summary.json','reports/continuation_v1/summary.json')]
    for optional in ('summary_interrupted.json','summary_cap2000.json'):
        source=continuation/optional
        if source.is_file():
            entries.append((source,f'continuation_v1/{optional}',source.relative_to(root).as_posix()))
    runs=[]
    for directory in sorted(continuation.iterdir()):
        if not directory.is_dir() or not RUN_PATTERN.fullmatch(directory.name):
            continue
        if directory.is_symlink():
            raise ValueError('Run evidence directory must not be a symlink')
        report_path=directory/'report.json'
        _,report,_=validated_bytes(report_path)
        if final and report.get('status')!='completed':
            raise ValueError(f'Final archive contains an incomplete run: {directory.name}')
        available={name for name in RUN_FILES if (directory/name).is_file()}
        if final and not REQUIRED_FINAL_RUN_FILES<=available:
            raise ValueError(f'Final run lacks required receipts: {directory.name}: {sorted(REQUIRED_FINAL_RUN_FILES-available)}')
        for name in RUN_FILES:
            source=directory/name
            if source.is_file():
                entries.append((source,f'continuation_v1/{directory.name}/{name}',source.relative_to(root).as_posix()))
        runs.append({'name':directory.name,'status':report.get('status'),
            'global_step':report.get('training',{}).get('global_step')})
    if final and not runs:
        raise ValueError('Final archive requires at least one completed training segment')
    _,aggregate_data,_=validated_bytes(aggregate)
    aggregate_matches=(Path(aggregate_data.get('root','')).resolve()==root and aggregate_data.get('orchestrator_status')==summary.get('status')
        and aggregate_data.get('stop_reason')==summary.get('stop_reason'))
    latest=max((run['global_step'] for run in runs if isinstance(run['global_step'],int)), default=None)
    aggregate_latest_matches=aggregate_data.get('latest_completed_global_step')==latest
    if final and (not aggregate_matches or not aggregate_latest_matches):
        raise ValueError('Final aggregate is stale or belongs to another run; regenerate it before archiving')
    entries.extend([(Path(aggregate),'protocol_repair_aggregate.json',None),
        (Path(diagnostics),'diagnostics/continuation_600_1000_independent_numerics_review.json',None)])
    # Read and validate every file before creating destination directories.
    receipts=[]
    for source,destination_relative,source_relative in entries:
        if source == summary_path:
            data,kind=summary_bytes,'json'
        else:
            data,_,kind=validated_bytes(source)
        relative=Path(destination_relative)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe archive relative path')
        receipts.append({'source':source,'destination_relative':destination_relative,'source_relative_path':source_relative,
            'data':data,'sha256':sha256(data),'bytes':len(data),'validated_format':kind})
    summary_end_sha=sha256(summary_path.read_bytes())
    if final and summary_end_sha!=sha256(summary_bytes):
        raise ValueError('Final summary changed during validation; retry with stable receipts')
    return receipts,{'format':'rm65b_protocol_repair_runtime_archive_v1','created_utc':datetime.now(timezone.utc).isoformat(),
        'archive_kind':'final_completed_run' if final else 'explicit_running_or_failed_snapshot',
        'source_root':str(root),'orchestrator_status':summary.get('status'),'stop_reason':summary.get('stop_reason'),
        'summary_sha256':sha256(summary_bytes),'summary_at_validation_end_sha256':summary_end_sha,
        'summary_changed_during_validation':summary_end_sha!=sha256(summary_bytes),'runs':runs,'latest_completed_global_step':latest,
        'aggregate_matches_summary':aggregate_matches,'aggregate_latest_matches_runs':aggregate_latest_matches,
        'confirmed_converged':final and aggregate_data.get('confirmed_converged') is True,
        'formal_3000_training':False,'task_SR_measured':False,
        'excluded':'All model checkpoints, .pt, .npz, raw images, logs and non-whitelisted files',
        'source_snapshot_is_atomic':False,'consistency_note':'Completed run requires a stable summary; running snapshots may span progress changes',
        'file_count':len(receipts),'total_bytes':sum(x['bytes'] for x in receipts)}


def archive(*, root, destination, aggregate, diagnostics, allow_running=False, execute=False):
    destination=Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f'Archive destination already exists; existing evidence is preserved: {destination}')
    receipts,manifest=gather(root,aggregate,diagnostics,allow_running)
    manifest['destination']=str(destination)
    manifest['files']=[{'source_path':str(x['source']),'source_relative_path':x['source_relative_path'],
        'relative_path':x['destination_relative'],'sha256':x['sha256'],'bytes':x['bytes'],
        'validated_format':x['validated_format']} for x in receipts]
    if not execute:
        return {'status':'planned','optimizer_updates_executed':0,**manifest}
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=Path(tempfile.mkdtemp(prefix='.'+destination.name+'-pending-',dir=destination.parent))
    try:
        for receipt in receipts:
            target=temporary/receipt['destination_relative']
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(receipt['data'])
            copied,_,_=validated_bytes(target)
            if sha256(copied)!=receipt['sha256']:
                raise ValueError('Copied receipt SHA differs from source bytes')
        (temporary/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
        validated_bytes(temporary/'manifest.json')
        if destination.exists():
            raise FileExistsError('Destination appeared during archive; refusing to replace it')
        # Rename the new directory only after every whitelisted receipt passes.
        os.rename(temporary,destination)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return {'status':'archived',**manifest}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--destination',type=Path,default=DEFAULT_DESTINATION)
    parser.add_argument('--aggregate',type=Path,default=Path('/tmp/protocol_repair_aggregate.json'))
    parser.add_argument('--diagnostics',type=Path,default=Path('/tmp/continuation_600_1000_independent_numerics_review.json'))
    parser.add_argument('--allow-running',action='store_true')
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args(argv)
    result=archive(root=args.root,destination=args.destination,aggregate=args.aggregate,diagnostics=args.diagnostics,
        allow_running=args.allow_running,execute=args.execute)
    print(json.dumps({k:v for k,v in result.items() if k!='files'},ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':
    main()
