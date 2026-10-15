"""Strict final dataset audit: frozen RGB/24/36 readback plus v14 evidence chain.

Canonical accepted records use readback only. Boundary-used records additionally
rebuild one recorded event in a private zero-step CPU scene; never a new episode.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import traceback
import dataset50_audit as base
from dataset50_reader_v3 import audit_geometry_chain, original_readback

ROOT=Path(__file__).resolve().parent
BASE_SHA='f631ef652497b9c6797cb47ea6dd53a2a9ac22491ed05fca04e35ba34f7fed5b'
require=base.require;sha=base.sha;read_json=base.read_json;inventory=base.inventory;accepted_paths=base.accepted_paths


def audit_trial(directory, *, codec_path=None, allow_synthetic=False):
    require(sha(base.__file__)==BASE_SHA,'Frozen original RGB/24/36 validator changed')
    try:
        report=read_json(Path(directory)/'expert_report.json')
        require(report.get('accepted') is True and report.get('result')=='physics_success' and report.get('hold_passed') is True,'Expert not accepted physics success (v2 first gate)')
    except Exception as error:
        return {'passed':False,'trial_dir':str(Path(directory).resolve()),'readback_only':True,'new_physics_runs':0,'hardware_io':False,
            'validator_version':3,'boundary_used':None,'boundary_used_verified':False,'error':str(error),'error_type':type(error).__name__}
    result=original_readback.audit_trial(directory,codec_path=codec_path,allow_synthetic=allow_synthetic)
    result.update(validator_version=3,boundary_used=None,boundary_used_verified=False)
    if not result['passed']:return result
    try:
        evidence=audit_geometry_chain(directory)
        require(inventory(directory)==result['source_files'],'Source changed during strengthened geometry readback')
        result.update(geometry_evidence=evidence,boundary_used=evidence['boundary_used'],boundary_used_verified=True,
            zero_step_cpu_event_reconstructions=int(evidence['independent_cpu_event_reconstruction']),
            legacy_validator_sha256=BASE_SHA,validator_sha256=sha(Path(__file__)),
            private_reader_sha256=sha(ROOT/'dataset50_reader_v3.py'),private_original_readback_ast_sha256=original_readback.private_ast_sha256,
            pose_report_semantics='Exact float32 producer heading, independent float64 physical class and nearest-arm; unchanged physical thresholds')
        result['limitations']+=['Boundary numerical reconciliation, if used, is explicitly recorded; original full v13 audit remains retained.']
    except Exception as error:
        result.update(passed=False,error=str(error),error_type=type(error).__name__,traceback=traceback.format_exc())
    return result


def audit_dataset(paths, *, codec_path=None, allow_synthetic=False):
    episodes=[audit_trial(path,codec_path=codec_path,allow_synthetic=allow_synthetic) for path in paths]
    passed=[r for r in episodes if r['passed']]
    counts={k:dict(Counter(str(r['identity'][k]) for r in passed)) for k in ['arm','model_id','pose_class']}
    targets={'arm':{'left':25,'right':25},'model_id':{'13':20,'16':20,'4':10},'pose_class':{'upright':20,'horizontal':15,'diagonal':15}}
    seeds=[r['identity']['seed'] for r in passed];indices=[r['identity']['spec_index'] for r in passed]
    unique=len(set(seeds))==len(seeds) and len(set(indices))==len(indices)
    boundary=[r['identity']['spec_index'] for r in passed if r['boundary_used'] is True]
    complete=len(episodes)==len(passed)==50 and counts==targets and unique and set(indices)==set(range(50))
    return {'schema_version':3,'complete':complete,'status':'complete_independently_validated50' if complete else 'partial_or_rejected_not50complete',
        'new_physics_runs':0,'readback_only':True,'hardware_io':False,'input_count':len(episodes),'accepted_count':len(passed),
        'failed_count':len(episodes)-len(passed),'actual_counts':counts,'target_counts':targets,'unique_seeds_and_specs':unique,
        'boundary_used_count':len(boundary),'boundary_used_spec_indices':sorted(boundary),
        'zero_step_cpu_event_reconstructions':sum(r.get('zero_step_cpu_event_reconstructions',0) for r in episodes),
        'episodes':episodes,'synthetic_fixture_only':any(r.get('synthetic_fixture_only',False) for r in episodes),
        'validator_sha256':sha(Path(__file__)),'legacy_validator_sha256':BASE_SHA,
        'boundary_validator_sha256':sha(ROOT/'dataset50_boundary_v2.py'),'boundary_cpu_verifier_sha256':sha(ROOT/'dataset50_boundary_cpu_v2.py'),'private_reader_sha256':sha(ROOT/'dataset50_reader_v3.py')}


def main():
    ap=argparse.ArgumentParser();group=ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--trial',type=Path);group.add_argument('--accepted-list',type=Path);group.add_argument('--trials-root',type=Path)
    ap.add_argument('--codec',type=Path);ap.add_argument('--output',type=Path);ap.add_argument('--allow-synthetic-fixtures',action='store_true')
    args=ap.parse_args()
    if args.trial:result=audit_trial(args.trial,codec_path=args.codec,allow_synthetic=args.allow_synthetic_fixtures);success=result['passed']
    else:result=audit_dataset(accepted_paths(accepted_list=args.accepted_list,trials_root=args.trials_root),codec_path=args.codec,allow_synthetic=args.allow_synthetic_fixtures);success=result['complete']
    if args.output:
        rows=result.get('episodes',[result]);require(all(not args.output.resolve().is_relative_to(Path(r['trial_dir'])) for r in rows),'Audit output overlaps source trial')
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['source_files','episodes','traceback']},ensure_ascii=False,indent=2))
    return 0 if success else 2

if __name__=='__main__':raise SystemExit(main())