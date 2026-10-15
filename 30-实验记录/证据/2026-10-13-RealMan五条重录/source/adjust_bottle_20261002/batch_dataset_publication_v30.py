"""Stdlib-only accepted replay publication. Never modifies HDF5 or simulation.

One coordinator calls publish_accepted_episode sequentially. Per-attempt replay
remains episode0; only verified byte copies receive the official batch index.
The manifest is authoritative for resume; HDF5 existence alone is insufficient.
"""
from __future__ import annotations
from pathlib import Path
import copy
import datetime
import hashlib
import json
import math
import os
import shutil
import uuid

BATCH_NAME='realman_adjust_bottle_batch20_20261003'
_REQUIRED_SPEC={'task_name':'adjust_bottle','episode_num':20,'profile_version':'v29','qpose_tag':0,'model_id':13}
_VIDEO_DEFAULT={'head_realtime':'video/head_realtime_h264.mp4','external_realtime':'video/external_realtime_h264.mp4'}


def _require(ok,message):
    if not ok:raise ValueError(message)


def _sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def _json(path):
    with Path(path).open('r',encoding='utf-8') as stream:return json.load(stream)


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _integer(value,label,minimum=0,maximum=None):
    _require(type(value) is int and value>=minimum and (maximum is None or value<=maximum),label+' must be an integer in range')
    return value


def _hash_string(value):return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def _inside(path,root):return Path(path).resolve().is_relative_to(Path(root).resolve())


def _file(path,root=None):
    p=Path(path)
    _require(p.is_absolute(),'Evidence path must be absolute: '+str(p));p=p.resolve()
    _require(p.is_file(),'Missing evidence file: '+str(p))
    if root is not None:_require(_inside(p,root),'Evidence escaped replay: '+str(p))
    return p


def _check_hashes(values,expected=None,artifact_roots=()):
    _require(isinstance(values,dict) and bool(values),'Missing evidence hash map')
    result={}
    for file,wanted in values.items():
        p=_file(file);_require(_hash_string(wanted),'Invalid SHA256 '+str(p));_require(_sha(p)==wanted,'Current SHA mismatch '+str(p))
        if expected is not None and not any(_inside(p,root) for root in artifact_roots):
            _require(str(p) in expected and expected[str(p)]==wanted,'Unfrozen shared source '+str(p))
        result[str(p)]=wanted
    return result


def _validate_spec(spec):
    _require(isinstance(spec,dict),'batch_spec must be a dict')
    for key,value in _REQUIRED_SPEC.items():
        _require(key in spec and type(spec[key]) is type(value) and spec[key]==value,'Batch specification mismatch '+key)
    videos=spec.get('delivered_video_paths',_VIDEO_DEFAULT)
    _require(videos==_VIDEO_DEFAULT,'Delivered video paths differ from approved profile')
    return copy.deepcopy(dict(spec,delivered_video_paths=videos))


def _check_flags(r):
    _require(r.get('result')=='one_strictly_validated_episode_collected' and r.get('accepted') is True,'Incomplete/unaccepted replay')
    for key in ('canonical_sources_unchanged','runtime_sources_unchanged','source_trajectory_unchanged','saved_paths_consumed','candidate_wrench_sources_unchanged','candidate_wrench_actions_complete','support_departure_guard_complete','native14D_preserved','actual24D_preserved'):
        _require(r.get(key) is True,'Replay guard failed/missing '+key)
    _require(r.get('hardware_io') is False and r.get('planner_calls_allowed') is False,'Replay IO/planning scope differs')
    _require(r.get('accepted_replay_version')=='v29','Replay profile must be v29')
    setting=r.get('grasp_setting',{})
    approved={'lift_axis':'world','pitch_deg':10,'hand_x':0,'hand_y':.085,'hand_z':.165,'close_sdk':[650,670,610,610,887,0],'ready_near_sdk':[650,670,610,610,887,0],'lp_min_main_force_N':1.15,'feedback_joint_rate_cap_rad_s':.16,'hand_joint_stiffness':20.,'hand_joint_damping':.2,'open_sdk':[1000,1000,1000,1000,1000,0]}
    for key,value in approved.items():_require(setting.get(key)==value,'Frozen V29 grasp configuration differs '+key)
    for phase in ('before_motion','after_motion'):
        drive=r.get('hardware_hand_drive',{}).get(phase,{})
        _require(drive.get('passed') is True and drive.get('joint_count')==24 and drive.get('primary_count')==12 and drive.get('follower_count')==12,'Missing actual V29 hand drive readback '+phase)
        records=drive.get('records',[]);_require(len(records)==24 and len({row['joint'] for row in records})==24,'Incomplete actual24 hand drive readback')
        for row in records:
            role=row.get('role');_require(role in ('primary','follower') and row.get('side') in ('left','right'),'Invalid hand drive role/side')
            values=(20.,.2,10.) if role=='primary' else (0.,0.,0.)
            actual=row.get('actual',{})
            for key,value in zip(('stiffness','damping','force_limit'),values):_require(type(actual.get(key)) in (int,float) and math.isfinite(actual[key]) and abs(actual[key]-value)<=1e-5,'Actual drive property differs '+row['joint']+'/'+key)
            _require(actual.get('drive_mode')=='force','Actual drive mode differs')


def _helper_hash(proof,key,expected):
    value=proof.get(key);_require(_hash_string(value) and value in expected.values(),'QA helper source is not frozen: '+key)


def _validate_episode(replay,index,seed,expected,spec):
    replay=Path(replay).resolve();rp=_file(replay/'replay_report.json',replay);r=_json(rp);_check_flags(r)
    _require(r.get('task')==spec['task_name'] and type(r.get('seed')) is int and r['seed']==seed,'Task/official seed mismatch')
    source_out=Path(r['source_candidate_out']).resolve();roots=(replay,source_out)
    observed={str(rp):_sha(rp)}
    for key in ('source_hashes','replay_helper_source_hashes'):
        observed.update(_check_hashes(r.get(key),expected,roots))
    runtime=r.get('runtime_source_hashes');_require(runtime==r.get('runtime_source_hashes_final')==r.get('candidate_runtime_source_hashes')==r.get('candidate_runtime_source_hashes_final'),'Runtime begin/end candidate/replay mismatch')
    observed.update(_check_hashes(runtime,expected,roots))
    candidate_file=_file(replay/'accepted_candidate_report.json',replay);candidate=_json(candidate_file)
    _require(candidate.get('accepted') is True and candidate.get('result')=='physics_success' and candidate.get('seed')==seed,'Candidate is not current accepted seed')
    attempts=r.get('bottle_attempts');_require(isinstance(attempts,list) and bool(attempts),'Missing official scene identity')
    identity=attempts[-1]['before'];_require(type(identity.get('qpose_tag')) is int and identity['qpose_tag']==0 and type(identity.get('model_id')) is int and identity['model_id']==13,'Batch accepts only official left/model13 scenes')
    ds=r['dataset'];pairs=_integer(ds.get('pairs'),'pairs',1);frames=_integer(ds.get('recorded_frames'),'recorded_frames',2);steps=_integer(r.get('physical_steps'),'physical_steps',1)
    dt=ds.get('physical_step_s');_require(type(dt) in (int,float) and math.isfinite(dt) and dt>0 and frames==pairs+1,'Invalid dynamic counts/physics clock')
    _require(ds.get('save_interval_steps')==15,'Native recording interval must be15 physics steps')
    hdf=_file(ds['hdf5'],replay);native=_file(ds['video'],replay);external=_file(ds['external_realtime_video'],replay)
    _require(hdf.parent==replay/'data' and hdf.name=='episode_0000000.hdf5' and _sha(hdf)==ds.get('sha256'),'Native HDF source/path/SHA mismatch')
    _require(_sha(external)==ds.get('external_realtime_video_sha256'),'External real-time source hash mismatch')
    trace=_file(replay/'physics_trace.json',replay);raw=_file(replay/'physics_command24_trace.npz',replay)
    actual=_json(replay/'actual36_geometry_audit.json');_require(actual.get('passed') is True and actual.get('physical_scene_unchanged') is True,'Missing full actual36 geometry acceptance')
    _require(actual.get('recorded_samples')==steps and actual.get('full_temporal_coverage') is True and actual.get('complete_to_reported_endpoint') is True and actual.get('trace_sha256')==_sha(trace),'Actual36 trace/endpoint coverage mismatch')
    observed.update(_check_hashes(actual.get('source_hashes'),expected,roots))
    qa=_json(replay/'independent_short_episode_physics_qa.json');_require(qa.get('full_success_verified') is True and qa.get('report_accepted') is True and qa.get('report_result')==r['result'],'Independent physical/timeline QA rejected')
    _require(qa.get('physics_clock',{}).get('samples')==steps and qa['physics_clock'].get('last_step')==steps and qa['physics_clock'].get('dt_s')==dt and all(v is True for v in qa.get('independent_recorded_checks',{}).values()) and bool(qa.get('independent_recorded_checks')),'Incomplete independent physical clock/checks')
    observed.update(_check_hashes(qa.get('source_sha256'),expected,roots));_helper_hash(qa,'helper_sha256',expected)
    _require(qa['source_sha256'].get(str(rp))==_sha(rp) and qa['source_sha256'].get(str(trace))==_sha(trace),'Timeline QA is stale for report/trace')
    validation=_json(replay/'dataset_validation.json');_require(validation.get('passed') is True,'Independent dataset readback rejected');_helper_hash(validation,'validator_source_sha256',expected)
    counts=validation.get('counts',{});_require(counts.get('native_pairs')==pairs and counts.get('cache_frames')==frames and counts.get('physical_steps')==steps and counts.get('physics_dt_s')==dt,'Dataset validator dynamic counts mismatch')
    checks=validation.get('checks',{})
    for key in ('publication_and_provenance','per_physics_step_24d_36d','native14d_24d_cache_rgb_depth_calibration','video_decode_and_physical_timing','evidence_unchanged'):_require(checks.get(key,{}).get('passed') is True,'Dataset section not passed '+key)
    evidence=checks['evidence_unchanged'].get('evidence_sha256');observed.update(_check_hashes(evidence,expected,roots))
    for p in (rp,hdf,raw,trace):_require(evidence.get(str(p))==_sha(p),'Dataset readback missing/current hash '+str(p))
    support=_json(replay/'actual_support_readback.json');_helper_hash(support,'helper_sha256',expected)
    for key in ('passed','geometry_passed','report_accepted','completed_episode_endpoint_covered','source_files_unchanged','full_actual36_finite_coverage'):_require(support.get(key) is True,'Actual support check not passed '+key)
    _require(support.get('report_result')==r['result'] and support.get('physics_steps')==steps and support.get('physics_dt_s')==dt,'Support report/clock mismatch')
    telemetry=support.get('contact_telemetry',{});selfs=support.get('recorded_actual_self_contacts',{})
    _require(telemetry.get('full_coverage') is True and telemetry.get('missing_steps')==0 and telemetry.get('active_table_impulse_after_release')==[] and selfs.get('full_coverage') is True and selfs.get('active_normal_contacts')==[],'Actual support/self contact coverage failed')
    observed.update(_check_hashes(support.get('source_sha256'),expected,roots))
    _require(support['source_sha256'].get(str(rp))==_sha(rp) and support['source_sha256'].get(str(trace))==_sha(trace),'Support QA stale for report/trace')
    videoqa=_json(replay/'delivery_video_validation.json');_require(videoqa.get('passed') is True and videoqa.get('source_report_sha256')==_sha(rp),'Delivered-video validation missing/stale')
    _require(abs(float(videoqa.get('simulation_duration_s',-1))-steps*dt)<1e-8,'Delivered-video physics duration differs')
    headclock=_json(replay/'head_realtime_video.json')
    _require(headclock.get('result')=='accepted_raw_head_realtime_exported' and headclock.get('replay_report_path')==str(rp) and headclock.get('replay_report_sha256')==_sha(rp),'Head real-time clock proof missing/stale')
    _helper_hash(headclock,'exporter_source_sha256',expected)
    _require(headclock.get('physical_steps')==steps and headclock.get('physical_step_s')==dt and headclock.get('simulation_duration_s')==steps*dt and headclock.get('source_observation_count')==frames,'Head actual-time physical clock/count mismatch')
    _require(headclock.get('simulation_time_origin_s')==0 and headclock.get('first_source_observation_time_s')==0 and headclock.get('last_source_observation_time_s')==steps*dt and headclock.get('final_endpoint_source_hold_s')==0,'Head source coverage does not include exact physical endpoints')
    fps=headclock.get('encoded_fps');_require(type(fps) in (int,float) and math.isfinite(fps) and fps>0,'Invalid head actual-time FPS')
    _require(headclock.get('decoded_frames')==math.ceil(steps*dt*fps),'Head encoded frame count differs from actual clock')
    _require(0<=float(headclock.get('max_source_age_s',-1))<=float(headclock.get('max_allowed_source_age_s',-1))+1e-8 and float(headclock.get('max_allowed_source_age_s',-1))<=15*dt+1e-8,'Head source sample age exceeds native recording cadence')
    rawhead=_file(headclock['video'],replay);_require(_sha(rawhead)==headclock.get('video_sha256'),'Head actual-time source video changed')
    _require(videoqa.get('videos',{}).get('head_realtime',{}).get('source_sha256')==_sha(rawhead) and videoqa.get('videos',{}).get('external_realtime',{}).get('source_sha256')==_sha(external),'Video validation does not bind current actual-time sources')
    observed[str(rawhead)]=_sha(rawhead)
    stem=f'episode_{index:07d}';copies=[(hdf,'data/'+stem+'.hdf5'),(native,'video/'+stem+'.mp4'),(external,'video/external_native/'+stem+'.mp4')]
    for label,relative in spec['delivered_video_paths'].items():
        delivered=_file(replay/relative,replay);entry=videoqa.get('videos',{}).get(label,{})
        _require(entry.get('delivery_path')==str(delivered) and entry.get('delivery_sha256')==_sha(delivered) and entry.get('source_unchanged') is True and entry.get('speed_changed') is False,'Delivered video not current/no speed change '+label)
        copies.append((delivered,'video/'+label+'/'+stem+'.mp4'))
    scene=_json(replay/'scene_info.json');_require(set(scene)=={'episode_0'},'Local official scene_info must contain episode_0 only');info=scene['episode_0'];_require(info.get('info',{}).get('{A}')=='001_bottle/base13' and info['info'].get('{a}')=='left','Official language scene identity differs')
    instruction_path=_file(replay/'instruction/episode_0000000.json',replay);instructions=_json(instruction_path)
    for split in ('seen','unseen'):_require(isinstance(instructions.get(split),list) and bool(instructions[split]) and all(isinstance(t,str) and t.strip() for t in instructions[split]),'Missing official language '+split)
    _require(instructions['seen']==r.get('instructions',{}).get('seen')==checks['native14d_24d_cache_rgb_depth_calibration'].get('instructions'),'HDF/report/official instruction texts differ')
    _require(instructions['unseen']==r['instructions'].get('unseen'),'Unseen instruction/report differs')
    observed[str(replay/'scene_info.json')]=_sha(replay/'scene_info.json');observed[str(instruction_path)]=_sha(instruction_path)
    copies.append((instruction_path,'instruction/'+stem+'.json'))
    trajectory=_file(r['source_trajectory']);_require(_inside(trajectory,source_out) and _sha(trajectory)==r['source_trajectory_sha256'],'Candidate trajectory source hash/path mismatch');copies.append((trajectory,f'_traj_data/episode{index}.pkl'))
    for name in ('replay_report.json','accepted_candidate_report.json','actual36_geometry_audit.json','dataset_validation.json','actual_support_readback.json','independent_short_episode_physics_qa.json','delivery_video_validation.json','head_realtime_video.json'):
        copies.append((_file(replay/name,replay),'episodes/'+stem+'/'+name))
    observed.update({str(source):_sha(source) for source,_ in copies});observed[str(trace)]=_sha(trace);observed[str(raw)]=_sha(raw)
    return {'index':index,'seed':seed,'job_id':f'seed_{seed:07d}','replay_dir':str(replay),'hdf5_sha256':_sha(hdf),'source_replay_dir':str(replay),'source_report_sha256':_sha(rp),'source_files':observed,'pairs':pairs,'recorded_frames':frames,'physical_steps':steps,'physics_dt_s':dt,'physical_duration_s':steps*dt,'scene_info':info,'instructions':instructions,'copies':[(str(p),target) for p,target in copies]}


def _atomic_json(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    with tmp.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(payload,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(tmp,path)


def _atomic_text(path,value):
    path=Path(path);tmp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    with tmp.open('x',encoding='utf-8',newline='\n') as stream:stream.write(value);stream.flush();os.fsync(stream.fileno())
    os.replace(tmp,path)


def _copy_verified(source,destination,wanted):
    source=_file(source);destination=Path(destination);_require(_sha(source)==wanted,'Source changed before byte-copy')
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():_require(destination.is_file() and _sha(destination)==wanted,'Conflicting prior staging/output '+str(destination));return
    with source.open('rb') as src,destination.open('xb') as dst:shutil.copyfileobj(src,dst,1024*1024);dst.flush();os.fsync(dst.fileno())
    _require(_sha(source)==wanted and _sha(destination)==wanted,'Source/copy SHA mismatch; staging preserved')


def _validate_records(records):
    _require(isinstance(records,list),'Manifest records must be a list');indices=[];seeds=[]
    for record in records:indices.append(_integer(record['index'],'manifest index',0,19));seeds.append(_integer(record['seed'],'manifest seed',0,2**32-1))
    _require(len(indices)==len(set(indices)) and len(seeds)==len(set(seeds)),'Duplicate official episode index/seed')
    _require(sorted(indices)==list(range(len(records))),'Committed indices must be contiguous')


def _reserve_identity(path,identity):
    path=Path(path)
    if path.exists():_require(_json(path)==identity,'Pending publication belongs to different accepted replay; preserved')
    else:_atomic_json(path,identity)


def _verify_committed(root,records):
    _validate_records(records)
    for record in records:
        _check_hashes(record['source_files'])
        for relative,wanted in record['published_files'].items():
            p=(root/relative).resolve();_require(_inside(p,root),'Published path escaped batch root');_require(p.is_file() and _sha(p)==wanted,'Committed artifact missing/changed '+relative)


def _materialize_metadata(root,manifest):
    records=manifest['episodes'];_validate_records(records)
    _atomic_json(root/'scene_info.json',{f"episode_{r['index']}":r['scene_info'] for r in records})
    _atomic_text(root/'seed.txt',''.join(str(r['seed'])+'\n' for r in records))
    _atomic_json(root/'metadata.json',{'source_format':'RoboTwin','task_name':'adjust_bottle','episode_num':len(records),'target_episode_num':20,'profile_version':'v29','state_action_schema':'native14D with unchanged actual24D/full36 supplemental groups','clock':'per-episode actual physics times; no concatenation or video-fps substitution','hardware_calibrated':False,'manifest_sha256':_sha(root/'manifest.json')})


def _complete(root,manifest):
    _verify_committed(root,manifest['episodes']);_materialize_metadata(root,manifest)
    if len(manifest['episodes'])!=20:return None
    hashes={str(p.relative_to(root)):_sha(p) for p in (root/'manifest.json',root/'scene_info.json',root/'seed.txt',root/'metadata.json')}
    for record in manifest['episodes']:
        hashes.update(record['published_files']);p=root/'instruction'/f"episode_{record['index']:07d}.json";hashes[str(p.relative_to(root))]=_sha(p)
    completion={'passed':True,'completed_episode_count':20,'unique_official_seeds':20,'official_seeds':[r['seed'] for r in manifest['episodes']],'official_indices':list(range(20)),'batch_spec_sha256':manifest['batch_spec_sha256'],'expected_sources':manifest['expected_sources'],'files':hashes}
    if (root/'complete.json').exists():_require(_json(root/'complete.json')==completion,'Existing completion marker conflicts')
    else:_atomic_json(root/'complete.json',completion)
    return completion


def publish_accepted_episode(replay_dir,dataset_root,index,seed,expected_sources,batch_spec):
    """Publish exactly one complete accepted replay. Return committed record/state.

    Resume by calling again with the same index/seed/replay. A pending intent
    reserves that identity; conflicting files/seeds never get overwritten. A
    failed staging directory and pending intent remain for diagnosis/retry.
    """
    index=_integer(index,'episode index',0,19);seed=_integer(seed,'official seed',0,2**32-1);spec=_validate_spec(batch_spec)
    root=Path(dataset_root).resolve();_require(root.name==BATCH_NAME or (root.name=='dataset' and root.parent.name==BATCH_NAME),'Use the requested independent batch directory basename')
    expected={str(Path(p).resolve()):h for p,h in expected_sources.items()};_check_hashes(expected)
    _require(str(Path(__file__).resolve()) in expected and expected[str(Path(__file__).resolve())]==_sha(__file__),'Publisher itself must be frozen in expected_sources')
    payload=_validate_episode(replay_dir,index,seed,expected,spec)
    root.mkdir(parents=True,exist_ok=True);manifest_path=root/'manifest.json';spec_sha=_digest(spec)
    if manifest_path.exists():
        manifest=_json(manifest_path);_require(manifest['batch_spec_sha256']==spec_sha and manifest['expected_sources']==expected,'Batch resume specification/shared sources differ')
    else:
        _require(not any(root.iterdir()),'New batch root contains unmanaged files')
        manifest={'schema_version':1,'batch_name':BATCH_NAME,'task_name':'adjust_bottle','batch_spec':spec,'batch_spec_sha256':spec_sha,'expected_sources':expected,'episodes':[]};_atomic_json(manifest_path,manifest)
    _verify_committed(root,manifest['episodes'])
    for record in manifest['episodes']:
        if record['index']==index:
            _require(record['seed']==seed and record['source_replay_dir']==str(Path(replay_dir).resolve()) and record['source_report_sha256']==payload['source_report_sha256'],'Existing committed index belongs to another seed/replay')
            _complete(root,manifest);return copy.deepcopy(record)
        _require(record['seed']!=seed,'Official seed already committed at another index')
    _require(index==len(manifest['episodes']),'Only the next contiguous accepted index may be published')
    intent_path=root/'_publication_state'/f'episode_{index:07d}.json';identity={'index':index,'seed':seed,'source_replay_dir':payload['source_replay_dir'],'source_report_sha256':payload['source_report_sha256'],'source_files':payload['source_files'],'copies':payload['copies']}
    _reserve_identity(intent_path,identity)
    stage=root/'_staging'/f'episode_{index:07d}_seed{seed}_{uuid.uuid4().hex}';stage.mkdir(parents=True,exist_ok=False);published={}
    for source,relative in payload['copies']:
        target=(root/relative).resolve();_require(_inside(target,root),'Destination escaped batch root');wanted=payload['source_files'][source]
        temp=(stage/relative).resolve();_require(_inside(temp,stage),'Staging destination escaped');_copy_verified(source,temp,wanted)
        if target.exists():_require(target.is_file() and _sha(target)==wanted,'Uncommitted destination conflict; staging preserved')
        else:target.parent.mkdir(parents=True,exist_ok=True);os.replace(temp,target)
        published[relative]=wanted
    _check_hashes(payload['source_files']);_check_hashes(expected)
    record={k:v for k,v in payload.items() if k!='copies'};record['published_files']=published
    record['committed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();record['publication_staging_dir']=str(stage)
    manifest['episodes'].append(record);_validate_records(manifest['episodes']);_atomic_json(manifest_path,manifest)
    complete=_complete(root,manifest)
    return copy.deepcopy(record)


def finalize_dataset(dataset_root,expected_sources,batch_spec):
    """Revalidate all20 committed artifacts and return the final completion record.

    No complete marker is published for19 or fewer episodes. The coordinator
    may copy the returned record into its separate outer batch/complete.json.
    """
    root=Path(dataset_root).resolve();spec=_validate_spec(batch_spec)
    _require(root.name==BATCH_NAME or (root.name=='dataset' and root.parent.name==BATCH_NAME),'Wrong independent batch dataset directory')
    expected={str(Path(p).resolve()):h for p,h in expected_sources.items()};_check_hashes(expected)
    _require(str(Path(__file__).resolve()) in expected and expected[str(Path(__file__).resolve())]==_sha(__file__),'Publisher itself must be frozen in expected_sources')
    manifest=_json(root/'manifest.json')
    _require(manifest.get('batch_spec_sha256')==_digest(spec) and manifest.get('expected_sources')==expected,'Final source/specification freeze mismatch')
    _validate_records(manifest['episodes']);_require(len(manifest['episodes'])==20,'Cannot complete a partial dataset')
    return _complete(root,manifest)
