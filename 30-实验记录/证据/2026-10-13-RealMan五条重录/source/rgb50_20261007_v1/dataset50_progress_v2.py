"""Read-only candidate observer using frozen v2 CPU dataset validation.

Only own progress/receipt JSON files are written. Source candidates, original20,
GPU workers and robot state are never modified. Stop when all50 slots qualify.
"""
from __future__ import annotations
import argparse
from collections import Counter
import datetime
import hashlib
import json
import os
from pathlib import Path
import time
import dataset50_audit_v2 as audit

ROOT=Path(__file__).resolve().parent


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic(path,value):
    path=Path(path);tmp=path.with_name('.'+path.name+'.tmp-'+str(os.getpid()))
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)


def verify_sources(freeze):
    for name,digest in freeze['source_sha256'].items():audit.require(audit.sha(ROOT/name)==digest,'Frozen validator source changed: '+name)


def stamp(path):
    s=path.stat();return [s.st_size,s.st_mtime_ns]


def ready_snapshot(directory):
    receipt_path=directory/'worker_receipt.json';receipt=audit.read_json(receipt_path)
    audit.require(receipt.get('accepted') is True and receipt.get('rgb_recorded') is True,'Worker not a completed RGB success')
    report=audit.read_json(directory/'expert_report.json');spec=report['sample_spec']
    audit.require(type(spec['index']) is int and 0<=spec['index']<50,'Not formal slot0..49')
    audit.require(report.get('accepted') is True and report.get('synthetic_fixture_only') is not True,'Expert failed or synthetic')
    audit.require(receipt['physics_steps']==report['physical_steps'],'Worker/expert endpoint differs')
    recording=audit.read_json(directory/'recording_report.json')
    audit.require(recording.get('accepted') is True,'Recording incomplete')
    required=['scene_sample.json','expert_report.json','physics_trace.json','physics_telemetry.npz','actual36_geometry_audit.json',
        'recording_samples.json','recording_report.json','data/episode_0000000.hdf5','worker_receipt.json']
    for name in required:audit.require((directory/name).is_file() and (directory/name).stat().st_size>0,'Incomplete '+name)
    key={'small_proof_hashes':{name:audit.sha(directory/name) for name in ['worker_receipt.json','expert_report.json','recording_report.json','scene_sample.json']},
        'completed_file_stats':{name:stamp(directory/name) for name in required},'recorded_hdf5_sha256':recording['hdf5_sha256']}
    return spec,key


def same_formal_spec(actual,expected):
    return actual.get('index',actual.get('spec_index'))==expected['index'] and all(actual[k]==expected[k] for k in ['seed','model_id','arm','pose_class'])


def observe(watch,interval):
    freeze_path=ROOT/'dataset50_validator_v2_source_freeze.json';freeze=audit.read_json(freeze_path);freeze_digest=audit.sha(freeze_path)
    verify_sources(freeze)
    plan=audit.read_json(ROOT/'sample_plan.json');plan_digest=audit.sha(ROOT/'sample_plan.json');expected={s['index']:s for s in plan['specs']}
    audit.require(set(expected)==set(range(50)),'Formal plan must contain all50 distinct slots')
    targets={'arm':{'left':25,'right':25},'model_id':{'13':20,'16':20,'4':10},'pose_class':{'upright':20,'horizontal':15,'diagonal':15}}
    progress_path=ROOT/'formal_accepted_progress_v2.json';list_path=ROOT/'accepted_list_v2.json';state_path=ROOT/'formal_accepted_progress_v2_state.json'
    proof_dir=ROOT/'formal_independent_v2_receipts';proof_dir.mkdir(exist_ok=True)
    state=audit.read_json(state_path) if state_path.exists() else {'entries':{},'first_seen':{},'cycle':0,'started_utc':now()}
    # Reuse only the two existing full RGB/24/36 v2 receipts after every source hash matches.
    reuse=ROOT/'independent_v2_real_record_readback_20261007.json'
    if reuse.is_file():
        for row in audit.read_json(reuse)['episodes']:
            if not row.get('passed') or row.get('boundary_used') is not False:continue
            directory=Path(row['trial_dir']);keyname=str(directory)
            if keyname in state['entries']:continue
            try:
                spec,key=ready_snapshot(directory)
                audit.require(same_formal_spec(spec,expected[spec['index']]),'Existing record formal spec differs')
                audit.require(audit.inventory(directory)==row['source_files'],'Existing accepted source full hashes changed')
                evidence=audit.audit_geometry_chain(directory) if hasattr(audit,'audit_geometry_chain') else None
                audit.require(evidence is not None and evidence['boundary_used'] is False,'Canonical reuse geometry differs')
                row=dict(row);row.update(geometry_evidence=evidence,reused_full_RGB_readback=True,reuse_receipt_sha256=audit.sha(reuse),
                    frozen_validator_source_sha256=freeze['source_sha256'])
                proof=proof_dir/(f"slot{spec['index']:07d}_"+hashlib.sha256(keyname.encode()).hexdigest()[:12]+'.json');atomic(proof,row)
                state['entries'][keyname]={'key':key,'passed':True,'identity':row['identity'],'boundary_used':False,
                    'audit_receipt_path':str(proof),'audit_receipt_sha256':audit.sha(proof),'qualified_utc':now()}
            except Exception as exc:print('REUSE_DEFERRED',str(directory),str(exc),flush=True)
    while True:
        verify_sources(freeze);audit.require(audit.sha(freeze_path)==freeze_digest,'Validator freeze registry changed')
        audit.require(audit.sha(ROOT/'sample_plan.json')==plan_digest,'Formal slot plan changed')
        state['cycle']+=1;cycle_start=time.monotonic();pending=[];changed=False;current=set()
        paths=sorted((ROOT/'batch_candidates').rglob('worker_receipt.json'),key=lambda p:(p.stat().st_mtime_ns,str(p)))
        for receipt in paths:
            directory=receipt.parent;name=str(directory)
            try:spec,key=ready_snapshot(directory)
            except Exception:continue
            if not same_formal_spec(spec,expected[spec['index']]):
                pending.append({'trial_dir':name,'index':spec['index'],'reason':'Formal slot seed/model/arm/pose differs'});continue
            current.add(name)
            previous=state['entries'].get(name)
            if previous and previous['key']==key:continue
            if previous:state['entries'].pop(name);changed=True
            seen=state['first_seen'].get(name)
            if not seen or seen['key']!=key:
                state['first_seen'][name]={'key':key,'time':time.time()}
                pending.append({'trial_dir':name,'index':spec['index'],'reason':'Waiting for unchanged completion proofs'});continue
            if time.time()-seen['time']<min(interval,30):
                pending.append({'trial_dir':name,'index':spec['index'],'reason':'Stability interval not complete'});continue
            print('INDEPENDENT_AUDIT_START',spec['index'],name,flush=True)
            row=audit.audit_trial(directory,codec_path=ROOT.parents[2]/'RoboTwin/data/decode_image_bit.py')
            try:
                _,after=ready_snapshot(directory);audit.require(after==key,'Completion proof changed during v2 audit')
                audit.require(row.get('passed') is not True or same_formal_spec(row['identity'],expected[spec['index']]),'Accepted actual identity differs')
                verify_sources(freeze)
            except Exception as exc:row.update(passed=False,error=str(exc))
            # The audit receipt is immutable for this completion-proof generation.
            generation=hashlib.sha256(json.dumps(key,sort_keys=True).encode()).hexdigest()[:12]
            proof=proof_dir/(f"slot{spec['index']:07d}_"+hashlib.sha256(name.encode()).hexdigest()[:12]+'_'+generation+'.json')
            audit.require(not proof.exists(),'Refusing overwrite independent audit receipt')
            row.update(frozen_validator_source_sha256=freeze['source_sha256'],worker_receipt_sha256=key['small_proof_hashes']['worker_receipt.json'])
            atomic(proof,row)
            state['entries'][name]={'key':key,'passed':row['passed'],'identity':row.get('identity'),'boundary_used':row.get('boundary_used'),
                'audit_receipt_path':str(proof),'audit_receipt_sha256':audit.sha(proof),'qualified_utc':now(),'error':row.get('error')}
            changed=True;print('INDEPENDENT_AUDIT_RESULT',spec['index'],row['passed'],row.get('boundary_used'),row.get('error'),flush=True)
        qualified=sorted([(name,e) for name,e in state['entries'].items() if e['passed'] and name in current],key=lambda x:(x[1]['qualified_utc'],x[0]))
        first={}
        for name,e in qualified:first.setdefault(e['identity']['spec_index'],(name,e))
        accepted=[{'trial_dir':name,'identity':e['identity'],'boundary_used':e['boundary_used'],'audit_receipt_path':e['audit_receipt_path'],
            'audit_receipt_sha256':e['audit_receipt_sha256'],'worker_receipt_sha256':e['key']['small_proof_hashes']['worker_receipt.json'],
            'expert_report_sha256':e['key']['small_proof_hashes']['expert_report.json'],'qualified_utc':e['qualified_utc']}
            for _,(name,e) in sorted(first.items())]
        counts={k:dict(Counter(str(e['identity'][k]) for e in accepted)) for k in ['arm','model_id','pose_class']}
        complete=len(accepted)==50 and counts==targets and set(first)==set(range(50))
        boundary=[r['identity']['spec_index'] for r in accepted if r['boundary_used'] is True]
        rejected=[{'trial_dir':name,'identity':e.get('identity'),'error':e.get('error'),'audit_receipt_path':e['audit_receipt_path']}
            for name,e in state['entries'].items() if not e['passed']]
        common={'schema_version':2,'complete':complete,'status':'all50_independently_qualified_not_published' if complete else 'partial_independent_qualification_not50complete',
            'updated_utc':now(),'accepted_count':len(accepted),'actual_counts':counts,'target_counts':targets,'boundary_used_count':len(boundary),
            'boundary_used_spec_indices':boundary,'new_physics_runs':0,'gpu_runs':0,'hardware_io':False,'dataset_published':False,
            'validator_freeze_sha256':freeze_digest,'formal_plan_sha256':audit.sha(ROOT/'sample_plan.json'),
            'qualification_rule':'First independently passing stable completed worker per formal slot; final publication reruns every v2 data audit.'}
        atomic(list_path,{**common,'trials':accepted})
        atomic(progress_path,{**common,'cycle':state['cycle'],'accepted':accepted,'pending':pending,'rejected_completed_workers':rejected,
            'duplicate_qualifying_candidates_excluded':len(qualified)-len(accepted),'last_cycle_elapsed_s':time.monotonic()-cycle_start})
        atomic(state_path,state)
        if changed or state['cycle']==1:print('FORMAL_PROGRESS',json.dumps({k:common[k] for k in ['accepted_count','actual_counts','boundary_used_count','complete']}),flush=True)
        if complete or not watch:return 0
        time.sleep(interval)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--watch',action='store_true');ap.add_argument('--interval',type=float,default=30);args=ap.parse_args()
    audit.require(1<=args.interval<=60,'Interval must be1..60 seconds')
    return observe(args.watch,args.interval)

if __name__=='__main__':raise SystemExit(main())