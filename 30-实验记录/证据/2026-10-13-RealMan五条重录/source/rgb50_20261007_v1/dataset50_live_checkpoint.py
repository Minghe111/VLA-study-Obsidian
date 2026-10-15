"""Live view of immutable independent v3 receipts; never a final dataset publisher.
Streams existing expert traces for stage metrics, with no scene/GPU/hardware IO.
"""
from __future__ import annotations
import argparse,codecs,datetime,hashlib,json,math,os,time
from collections import Counter
from pathlib import Path
import numpy as np
import dataset50_audit_v3 as audit
ROOT=Path(__file__).resolve().parent
TARGETS={'arm':{'left':25,'right':25},'model_id':{'13':20,'16':20,'4':10},'pose_class':{'upright':20,'horizontal':15,'diagonal':15}}

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def atomic(path,value):
    temp=path.with_name('.'+path.name+'.tmp-'+str(os.getpid()))
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(temp,path)
def stream_rows(path,evidence):
    decoder=json.JSONDecoder();utf=codecs.getincrementaldecoder('utf-8')();digest=hashlib.sha256()
    with Path(path).open('rb') as src:
        buf='';pos=0;eof=False
        def more():
            nonlocal buf,pos,eof
            raw=src.read(1024*1024);digest.update(raw);buf=buf[pos:]+utf.decode(raw,final=not raw);pos=0;eof=not raw
        def token():
            nonlocal pos
            while True:
                while pos<len(buf) and buf[pos].isspace():pos+=1
                if pos<len(buf):return buf[pos]
                if eof:return None
                more()
        audit.require(token()=='[','Trace is not a root array');pos+=1;first=True
        while True:
            current=token()
            if current==']':pos+=1;break
            if not first:
                audit.require(current==',','Trace separator missing');pos+=1
                audit.require(token()!=']','Trailing trace comma')
            while True:
                try:row,end=decoder.raw_decode(buf,pos);pos=end;break
                except json.JSONDecodeError:
                    if eof:raise
                    more()
            first=False;yield row
        audit.require(token() is None,'Trailing trace content')
    evidence['sha256']=digest.hexdigest()

def polar(rotation):
    u,_,v=np.linalg.svd(rotation);s=np.eye(3);s[2,2]=np.linalg.det(u@v);return u@s@v

def stage_summary(directory,report,trace_sha):
    initial=np.asarray(report['before']['pose'],float);initial_fp=np.asarray(report['before']['functional_point0'][:3],float)
    r0=polar(initial[:3,:3]);stages={};endpoint=0;evidence={};hold=[]
    for row in stream_rows(directory/'physics_trace.json',evidence):
        endpoint+=1;audit.require(row['step']==endpoint,'Trace physical sequence changed')
        phase=row['phase'];terminal=phase.get('terminal_hold') is True
        key=str(phase.get('dense_call'))+'|'+phase['name']+'|'+str(terminal)
        m=np.asarray(row['bottle']['pose'],float);fp=np.asarray(row['bottle']['functional_point0'][:3],float)
        audit.require(m.shape==(4,4) and np.isfinite(m).all() and np.isfinite(fp).all(),'Nonfinite stage pose')
        origin=float(np.linalg.norm(m[:3,3]-initial[:3,3]));functional=float(np.linalg.norm(fp-initial_fp))
        rotation=float(np.degrees(np.arccos(np.clip((np.trace(r0.T@polar(m[:3,:3]))-1)/2,-1,1))))
        audit.require(math.isfinite(row['rotation_delta_deg']) and math.isfinite(row['fp_delta_m']),'Nonfinite recorded stage motion')
        if key not in stages:
            stages[key]={'name':phase['name'],'dense_call':phase.get('dense_call'),'terminal_hold':terminal,'samples':0,
                'first_step':endpoint,'first_sim_time_s':row['sim_time_s'],'first_actor_origin_world_m':m[:3,3].tolist(),
                'maximum_actor_origin_displacement_from_episode_initial_m':0.,'maximum_functional_point_displacement_from_episode_initial_m':0.,
                'maximum_recorded_functional_point_delta_m':0.,'maximum_recorded_rotation_delta_deg':0.,
                'maximum_independent_polar_rotation_from_episode_initial_deg':0.,'official_success_samples':0}
        s=stages[key];s['samples']+=1;s['last_step']=endpoint;s['last_sim_time_s']=row['sim_time_s'];s['last_actor_origin_world_m']=m[:3,3].tolist()
        for k,value in [('maximum_actor_origin_displacement_from_episode_initial_m',origin),('maximum_functional_point_displacement_from_episode_initial_m',functional),
            ('maximum_recorded_functional_point_delta_m',float(row['fp_delta_m'])),('maximum_recorded_rotation_delta_deg',float(row['rotation_delta_deg'])),
            ('maximum_independent_polar_rotation_from_episode_initial_deg',rotation)]:s[k]=max(s[k],value)
        s['official_success_samples']+=int(row['bottle']['official_success'] is True)
        if terminal:
            grip=row['grip'];hold.append({'success':row['bottle']['official_success'] is True,'gap':float(row['bottle_bottom_gap_m']),
                'table':float(grip['table_normal_N']),'thumb':float(grip['thumb_normal_N']),'fingers':float(grip['fingers_normal_N']),
                'dot':float(grip['opposing_normals_dot']),'translation':float(row['relative_translation_drift_m']),'rotation':float(row['relative_rotation_drift_deg'])})
    audit.require(evidence['sha256']==trace_sha,'Trace SHA changed since complete independent v3 readback')
    audit.require(endpoint==report['physical_steps'] and bool(hold),'Endpoint/hold changed')
    dt=report['action_clock']['physics_dt_s'];audit.require(len(hold)*dt>=1-1e-8,'Terminal hold under1s')
    for row in hold:
        audit.require(row['success'] and row['gap']>=.01 and row['table']<1e-6 and row['thumb']>.05 and row['fingers']>.05 and row['dot']<-.3
            and row['translation']<.002 and row['rotation']<2,'Previously qualified terminal hold no longer agrees')
    return {'trace_sha256':trace_sha,'trace_steps_read':endpoint,'physics_dt_s':dt,'phase_metrics':list(stages.values()),
        'terminal_hold':{'samples':len(hold),'duration_s':len(hold)*dt,'official_success_fraction_of_hold_physics_frames':sum(x['success'] for x in hold)/len(hold),
            'minimum_air_gap_m':min(x['gap'] for x in hold),'maximum_table_force_N':max(x['table'] for x in hold),'minimum_thumb_force_N':min(x['thumb'] for x in hold),
            'minimum_opposing_fingers_force_N':min(x['fingers'] for x in hold),'maximum_opposing_normals_dot':max(x['dot'] for x in hold),
            'maximum_palm_relative_translation_drift_m':max(x['translation'] for x in hold),'maximum_palm_relative_rotation_drift_deg':max(x['rotation'] for x in hold)},
        'initial_official_success':report['before']['official_success'],'immediate_official_success':report['immediate']['official_success'],
        'after_hold_official_success':report['after_hold']['official_success'],
        'rotation_semantics':'Recorded rotation delta remains separate from independently polar-normalized native float32 matrix geodesic; no acceptance threshold changed.'}

def snapshot(previous):
    freeze=audit.read_json(ROOT/'dataset50_validator_v3_source_freeze.json')
    for n,h in freeze['source_sha256'].items():audit.require(audit.sha(ROOT/n)==h,'Frozen v3 source changed '+n)
    plan=audit.read_json(ROOT/'sample_plan.json');expected={x['index']:x for x in plan['specs']}
    cache={x['audit_receipt_sha256']:x.get('stage_evidence') for x in previous.get('accepted',[]) if x.get('stage_evidence')}
    accepted=[];excluded=[];first={}
    for proof in sorted((ROOT/'formal_independent_v3_receipts').glob('slot*.json'),key=lambda p:(p.stat().st_mtime_ns,str(p))):
        try:
            receipt_sha=audit.sha(proof);row=audit.read_json(proof)
            if row.get('passed') is not True:continue
            audit.require(row['validator_version']==3 and row['boundary_used_verified'] is True and row.get('synthetic_fixture_only') is not True,'Not a strengthened real v3 receipt')
            audit.require(row['frozen_validator_source_sha256']==freeze['source_sha256'],'Receipt source pins differ')
            identity=row['identity'];i=identity['spec_index'];audit.require(type(i) is int and 0<=i<50,'Not a formal0..49 slot')
            audit.require(all(identity[k]==expected[i][k] for k in ['model_id','arm','pose_class']) and type(identity['seed']) is int and identity['seed']>=0,'Formal slot identity changed')
            directory=Path(row['trial_dir']).resolve();sources=row['source_files']
            audit.require(directory.is_relative_to(ROOT/'batch_candidates') or any(directory.is_relative_to(p) for p in ROOT.glob('smoke_worker*_v9') if p.is_dir()),'Outside explicit formal collection roots')
            current={str(p.relative_to(directory)):p.stat().st_size for p in directory.rglob('*') if p.is_file() and not p.is_symlink()}
            audit.require(set(current)==set(sources) and all(current[n]==v['bytes'] for n,v in sources.items()),'Qualified source file set or sizes changed')
            for name in ['scene_sample.json','expert_report.json','worker_receipt.json','recording_report.json','recording_samples.json','physics_telemetry.npz']:
                audit.require(audit.sha(directory/name)==sources[name]['sha256'],'Qualified small/telemetry proof changed '+name)
            report=audit.read_json(directory/'expert_report.json');worker=audit.read_json(directory/'worker_receipt.json')
            audit.require(report['accepted'] is True and report['result']=='physics_success' and report['hold_passed'] is True and worker['accepted'] is True and worker['rgb_recorded'] is True,'Current expert/worker no longer accepted')
            audit.require(report['sample_spec']['seed']==identity['seed'] and report['sample_spec']['index']==i,'Current actual seed/index differs')
            audit.require(worker['physics_steps']==report['physical_steps']==row['physical_steps'],'Current step endpoint differs')
            audit.require(report.get('boundary_used',False)==row['boundary_used'],'Current boundary flag differs')
            audit.require(audit.sha(proof)==receipt_sha,'Independent receipt changed while read')
            if i in first:excluded.append({'trial_dir':str(directory),'index':i,'reason':'Later independently passing duplicate formal slot'});continue
            stage=cache.get(receipt_sha) or stage_summary(directory,report,sources['physics_trace.json']['sha256'])
            first[i]=True;accepted.append({'identity':identity,'trial_dir':str(directory),'boundary_used':row['boundary_used'],
                'physical_steps':row['physical_steps'],'state_action_pairs':row['state_action_pairs'],'raw_observations':row['raw_observations'],
                'audit_receipt_path':str(proof),'audit_receipt_sha256':receipt_sha,'source_hdf5_sha256':row['source_hdf5_sha256'],
                'expert_report_sha256':sources['expert_report.json']['sha256'],'worker_receipt_sha256':sources['worker_receipt.json']['sha256'],
                'stage_evidence':stage})
        except Exception as error:excluded.append({'audit_receipt_path':str(proof),'reason':str(error)})
    accepted.sort(key=lambda x:x['identity']['spec_index']);counts={k:dict(Counter(str(x['identity'][k]) for x in accepted)) for k in TARGETS}
    unique=len({x['identity']['seed'] for x in accepted})==len(accepted)
    return {'schema_version':1,'reader_version':3,'updated_utc':now(),'scope':'Live checkpoint of immutable independently qualified receipts; final publisher must rerun every full dataset audit',
        'independently_qualified_unique_slots':len(accepted),'accepted_count':len(accepted),'actual_counts':counts,'target_counts':TARGETS,
        'all50_receipt_slots_seen':len(accepted)==50 and set(first)==set(range(50)) and counts==TARGETS and unique,
        'complete':False,'dataset_published':False,'unique_actual_seeds':unique,'boundary_used_count':sum(x['boundary_used'] is True for x in accepted),
        'resampled_seed_spec_indices':[x['identity']['spec_index'] for x in accepted if x['identity']['seed']!=expected[x['identity']['spec_index']]['seed']],
        'policy_inference_SR_measured':False,'SR_semantics':'Terminal hold fraction is expert trace evidence; no policy inference SR is inferred from this accepted dataset.',
        'current_verification_scope':'Frozen validator and receipt pins; unchanged source file sets/sizes and small proof/telemetry hashes. Full raw RGB/HDF/geometry hashes are in the immutable original v3 receipts, and are rerun before final publication.',
        'new_physics_runs':0,'gpu_runs':0,'hardware_io':False,'source_sha256':audit.sha(Path(__file__)),'accepted':accepted,'excluded':excluded}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--watch',action='store_true');ap.add_argument('--interval',type=float,default=30);args=ap.parse_args()
    audit.require(1<=args.interval<=60,'Interval must be1..60');output=ROOT/'formal_accepted_live_checkpoint_v3.json';source=audit.sha(Path(__file__))
    while True:
        audit.require(audit.sha(Path(__file__))==source,'Live checkpoint source changed')
        result=snapshot(audit.read_json(output) if output.is_file() else {});atomic(output,result)
        print('LIVE_CHECKPOINT',json.dumps({k:result[k] for k in ['accepted_count','actual_counts','boundary_used_count','complete']}),flush=True)
        if not args.watch or result['all50_receipt_slots_seen']:return 0
        time.sleep(args.interval)
if __name__=='__main__':raise SystemExit(main())