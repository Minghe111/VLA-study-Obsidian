"""Independent CPU-only readback of a published accepted RealMan episode.
No SAPIEN imports, scene calls, planner, GPU or physical IO. Source/data are read
only; this script creates one new validation JSON and refuses to overwrite it.
Native 14D action is the NEXT OBSERVED official joint vector. Actual motor drive
commands are separately preserved in 24D physics telemetry and raw hand radians.
"""
from pathlib import Path
import argparse,datetime,hashlib,importlib.util,json,pickle,traceback
import cv2,h5py,numpy as np

FIELDS=('physics_step','sim_time_before_s','sim_time_after_s','measured_state24_before','applied_command24','measured_state24_after','hand24_qpos_before_rad','hand24_qpos_after_rad','hand24_drive_target_rad','hand24_actual_fd_rad_s','hand24_engine_qvel_rad_s','full36_qpos_before_rad','full36_qpos_after_rad')
NPZ_FIELDS=tuple(f for f in FIELDS if f not in ('hand24_drive_target_rad','hand24_engine_qvel_rad_s'))
CAMERAS={'head_camera':'cam_head','left_camera':'cam_left_wrist','right_camera':'cam_right_wrist','front_camera':'cam_third_view'}
JOINTS={'left_arm':'left_arm_joint_states','left_gripper':'left_ee_joint_states','right_arm':'right_arm_joint_states','right_gripper':'right_ee_joint_states'}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def require(value,message):
    if not bool(value):raise AssertionError(message)

def exact(a,b,message):
    a,b=np.asarray(a),np.asarray(b)
    require(a.shape==b.shape and np.array_equal(a,b),message+f' shapes {a.shape}/{b.shape}')

def finite(a,message):
    require(np.isfinite(np.asarray(a)).all(),message)

def text(value):
    return value.decode('utf-8') if isinstance(value,bytes) else str(value)

def video_readback(path,frames,fps,shape):
    cap=cv2.VideoCapture(str(path));require(cap.isOpened(),'Cannot decode video '+str(path))
    meta={'path':str(path),'sha256':sha(path),'declared_frames':int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
          'fps':float(cap.get(cv2.CAP_PROP_FPS)),'width':int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
          'height':int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    count=0
    try:
        while True:
            ok,img=cap.read()
            if not ok:break
            require(img.shape==tuple(shape) and img.dtype==np.uint8,'Invalid decoded video frame')
            count+=1
    finally:cap.release()
    require(count==frames and meta['declared_frames']==frames,'Decoded video count mismatch')
    require(abs(meta['fps']-fps)<1e-5,'Video FPS mismatch')
    meta.update(decoded_frames=count,encoded_duration_s=count/meta['fps'])
    return meta

def validate(replay,repo):
    replay=Path(replay).resolve();repo=Path(repo).resolve()
    out={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'scope':'CPU data/source readback only; no new physics or geometry execution',
         'replay_dir':str(replay),'validator_source_sha256':sha(__file__),
         'versions':{'numpy':np.__version__,'h5py':h5py.__version__,'cv2':cv2.__version__},
         'checks':{},'limitations':['SDK angle mapping and friction are simulation settings, not real-robot calibration.',
                                  'Official 14D action is next observed joint state; actual drive command24 is separate.',
                                  'Native head MP4 is an observation sequence at 30 FPS; use recording timestamps for real-time playback.']}
    def section(name,fn):
        try:out['checks'][name]={'passed':True,**fn()}
        except Exception:out['checks'][name]={'passed':False,'traceback':traceback.format_exc()}
    report=json.loads((replay/'replay_report.json').read_text())
    require(report.get('result')=='one_strictly_validated_episode_collected' and report.get('accepted') is True,'Episode not completely published/accepted')
    require(report.get('actual36_geometry_audit',{}).get('passed') is True,'Missing mandatory full actual36 acceptance')
    require(report.get('canonical_sources_unchanged') is True and report.get('source_trajectory_unchanged') is True,'Source acceptance failed')
    ds=report['dataset'];n=int(ds['pairs']);frames=int(ds['recorded_frames']);steps=int(report['physical_steps']);dt=float(ds['physical_step_s'])
    require(n>0 and frames==n+1 and steps>0 and dt>0,'Invalid counts or physical timestep')
    hdf=Path(ds['hdf5']);npzpath=replay/'physics_command24_trace.npz';tracepath=replay/'physics_trace.json'
    require(hdf.resolve().parent==replay/'data','Dataset points outside its replay publication')
    files=sorted((replay/'.cache/episode0').glob('*.pkl'),key=lambda p:int(p.stem))
    require([int(p.stem) for p in files]==list(range(frames)),'Cache missing, duplicated, or out-of-order frames')
    traces=json.loads(tracepath.read_text());require(len(traces)==steps,'Trace step count mismatch')
    codecfile=repo/'data/decode_image_bit.py'
    spec=importlib.util.spec_from_file_location('_accepted_readback_codec',codecfile)
    codec=importlib.util.module_from_spec(spec);spec.loader.exec_module(codec)
    out['image_codec_source_sha256']=sha(codecfile)
    initial_hashes={str(p):sha(p) for p in (hdf,npzpath,tracepath,replay/'replay_report.json')}
    initial_sizes={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in files}
    out['counts']={'native_pairs':n,'cache_frames':frames,'physical_steps':steps,'physics_dt_s':dt,'physics_duration_s':steps*dt}
    def provenance():
        require(sha(hdf)==ds['sha256'],'Published HDF5 SHA mismatch')
        groups={key:report[key] for key in ('source_hashes','replay_helper_source_hashes')}
        count=0
        for values in groups.values():
            for file,expected in values.items():
                require(sha(file)==expected,'Source hash drift: '+file);count+=1
        require(sha(report['source_trajectory'])==report['source_trajectory_sha256'],'Original trajectory drift')
        for entry in report['candidate_wrench_manifest']:
            for file,expected in entry['source_hashes'].items():
                require(sha(file)==expected,'Accepted path/wrench artifact drift: '+file);count+=1
        require(report.get('candidate_wrench_sources_unchanged') is True and report.get('candidate_wrench_actions_complete') is True,'Wrench provenance incomplete')
        require(report.get('hardware_io') is False and report.get('planner_calls_allowed',report.get('planner_calls_allowed')) is False,'Replay IO/planner policy mismatch')
        audit=json.loads((replay/'actual36_geometry_audit.json').read_text())
        require(audit.get('passed') is True and audit.get('physical_scene_unchanged') is True,'Actual audit report invalid')
        return {'verified_source_and_carry_artifact_files':count,'hdf5_sha256':ds['sha256'],'source_trajectory_sha256':report['source_trajectory_sha256']}
    section('publication_and_provenance',provenance)
    with h5py.File(hdf,'r') as f,np.load(npzpath,allow_pickle=False) as raw:
        g=f['realman_dexterous_physics'];data={key:np.asarray(g[key]) for key in FIELDS}
        fullnames=json.loads(text(g.attrs['full36_joint_names']));handnames=json.loads(text(g.attrs['hand24_joint_names']))
        ids=[fullnames.index(name) for name in handnames]
        def telemetry():
            require(len(fullnames)==36 and len(set(fullnames))==36 and len(handnames)==24 and len(set(handnames))==24,'Invalid raw joint ordering')
            exact(fullnames,report['full_qpos_joint_names'],'Report/HDF full36 joint name order differs')
            require(set(g)==set(FIELDS) and set(raw.files)==set(NPZ_FIELDS),'Unexpected raw telemetry schema')
            require(bool(g.attrs['sdk_mapping_calibrated']) is False,'SDK incorrectly marked calibrated')
            require(text(g.attrs['ordering'])=='Larm6 rad, Lprimary SDK6, Rarm6 rad, Rprimary SDK6','24D ordering attr mismatch')
            require('actual drive targets' in text(g.attrs['command_source']),'Command source attr mismatch')
            for key,value in data.items():
                cols=36 if key.startswith('full36') else 24
                expected=(steps,) if key in FIELDS[:3] else (steps,cols)
                require(value.shape==expected,'Raw field shape mismatch '+key);finite(value,'Nonfinite field '+key)
                if key in raw.files:exact(value,raw[key],'HDF/NPZ mismatch '+key)
            exact(data['physics_step'],np.arange(1,steps+1),'Physics steps not complete')
            exact(data['sim_time_before_s'],np.arange(steps)*dt,'Raw before times mismatch')
            exact(data['sim_time_after_s'],np.arange(1,steps+1)*dt,'Raw after times mismatch')
            exact(data['full36_qpos_before_rad'][1:],data['full36_qpos_after_rad'][:-1],'Raw physical state continuity mismatch')
            exact(data['measured_state24_before'][1:],data['measured_state24_after'][:-1],'Measured 24D continuity mismatch')
            for part in ('before','after'):
                exact(data[f'hand24_qpos_{part}_rad'],data[f'full36_qpos_{part}_rad'][:,ids],'Raw hand24/full36 name mapping mismatch')
            fd=(data['hand24_qpos_after_rad']-data['hand24_qpos_before_rad'])/dt
            exact(data['hand24_actual_fd_rad_s'],fd,'Physical finite-difference velocity mismatch')
            sdk=np.concatenate([data['measured_state24_before'][:,6:12],data['measured_state24_before'][:,18:24],data['measured_state24_after'][:,6:12],data['measured_state24_after'][:,18:24],data['applied_command24'][:,6:12],data['applied_command24'][:,18:24]],axis=1)
            require(float(sdk.min())>=-1e-7 and float(sdk.max())<=1000+1e-7,'SDK24 range outside model units')
            for idx,row in enumerate(traces):
                require(row['step']==idx+1 and row['sim_time_s']==data['sim_time_after_s'][idx],'JSON/raw step clock mismatch')
                exact(row['full_qpos_actual_rad'],data['full36_qpos_after_rad'][idx],'JSON/HDF actual full36 mismatch')
                exact(row['command24_actual'],data['applied_command24'][idx],'JSON/HDF applied command24 mismatch')
                observed=data['measured_state24_after'][idx]
                exact(row['arm_qpos']['left'],observed[:6],'JSON measured left arm mismatch')
                exact(row['arm_qpos']['right'],observed[12:18],'JSON measured right arm mismatch')
                require(not any(c.get('active') for c in row['actual_self_contacts']),'Active forbidden self contact in accepted trace')
            return {'all_fields_and_shared_npz_exact':True,'full36_and_raw_hand24_name_mapping_exact':True,
                    'physics_state_continuity_exact':True,'true_hand_fd_max_rad_s':float(np.max(np.abs(fd))),
                    'engine_qvel_max_rad_s':float(np.max(np.abs(data['hand24_engine_qvel_rad_s']))),
                    'engine_qvel_is_not_physical_finite_difference':True,'raw_full_joint_names':fullnames,'raw_hand_joint_names':handnames}
        section('per_physics_step_24d_36d',telemetry)
        clocks=[];camstats={};headshape=[None];rgberrors=[]
        def observations():
            require(text(f.attrs['source_format'])=='RoboTwin' and text(f.attrs['source_path'])=='native_collection','Native format attrs mismatch')
            require(text(f['data_format_version'][()])=='v1.0','Native data format version mismatch')
            instructions=json.loads(text(f['instructions'][()]));require(isinstance(instructions,list) and all(isinstance(v,str) for v in instructions),'Invalid instructions')
            require(int(f['additional_info/frequency'][()])==15,'Native frequency metadata no longer save_freq15')
            dex=f['realman_dexterous'];require(bool(dex.attrs['sdk_mapping_calibrated']) is False,'Native SDK mapping incorrectly calibrated')
            require('actual drive targets' in text(dex.attrs['command_source']),'Native command source attr mismatch')
            require(set(f['vision'])==set(ds['cameras']) and 'cam_head' in f['vision'],'Camera list mismatch')
            for group in ('state','action'):
                for src,target in JOINTS.items():require(f[group+'/'+target].shape==(n,6 if 'arm' in src else 1),'Official14D shape mismatch')
            for idx,path in enumerate(files):
                sample=pickle.loads(path.read_bytes());clock=sample['recording_clock'];step=int(clock['physics_step']);t=float(clock['sim_time_s'])
                require(0<=step<=steps and t==step*dt,'PKL clock outside actual physical timeline');clocks.append((step,t))
                st=np.asarray(sample['dexterous_state']['state24']);cmd=np.asarray(sample['dexterous_state']['command24'])
                require(st.shape==(24,) and cmd.shape==(24,),'PKL24D shape mismatch');finite(st,'PKL state24 not finite');finite(cmd,'PKL command24 not finite')
                state0=data['measured_state24_before'][0] if step==0 else data['measured_state24_after'][step-1]
                exact(st,state0,'PKL measured state24 vs actual physics boundary mismatch')
                for src,target in JOINTS.items():
                    v=np.asarray(sample['joint_action'][src],dtype=np.float32).reshape(-1)
                    if idx<n:exact(f['state/'+target][idx],v,'Official native state vs PKL mismatch '+target)
                    if idx>0:exact(f['action/'+target][idx-1],v,'Official native NEXT state action vs PKL mismatch '+target)
                for src,target in (('left_endpose','left_ee_poses'),('right_endpose','right_ee_poses')):
                    v=np.asarray(sample['endpose'][src],dtype=np.float32).reshape(-1);finite(v,'Endpose nonfinite')
                    require(v.shape==(7,) and abs(np.linalg.norm(v[3:])-1)<2e-4,'Endpose quaternion invalid')
                    if idx<n:exact(f['state/'+target][idx],v,'State endpose mismatch')
                    if idx>0:exact(f['action/'+target][idx-1],v,'Action endpose mismatch')
                if idx<n:
                    exact(dex['measured_state24'][idx],st,'Native24D state vs PKL mismatch');require(dex['sample_time_s'][idx]==t,'Native state timestamp mismatch')
                if idx>0:
                    exact(dex['next_command24'][idx-1],cmd,'Native24D next drive command vs PKL mismatch');require(dex['next_sample_time_s'][idx-1]==t,'Native command timestamp mismatch')
                for name,cam in sample['observation'].items():
                    require(name in CAMERAS,'Unsupported cache camera name');hn=CAMERAS[name];cg=f['vision/'+hn]
                    rgb=np.asarray(cam['rgb']);depth=np.asarray(cam['depth']);K=np.asarray(cam['intrinsic_cv']);E=np.asarray(cam['extrinsic_cv']);C=np.asarray(cam['cam2world_gl'])
                    require(rgb.dtype==np.uint8 and rgb.ndim==3 and rgb.shape[2]==3,'RGB dtype/shape invalid')
                    require(depth.shape==rgb.shape[:2],'Depth shape mismatch');finite(depth,'Depth nonfinite');require(float(depth.min())>=0,'Negative depth')
                    finite(K,'K not finite');finite(E,'E not finite');finite(C,'C not finite')
                    require(K.shape==(3,3) and E.shape==(3,4) and C.shape==(4,4),'Camera matrix schema invalid')
                    h,w=rgb.shape[:2];require(K[0,0]>0 and K[1,1]>0 and 0<=K[0,2]<=w and 0<=K[1,2]<=h,'Camera focal/principal point invalid')
                    E4=np.eye(4);E4[:3]=E
                    require(np.allclose(E4@C,np.diag([1,-1,-1,1]),atol=2e-5),'CV world-to-camera and GL camera-to-world mismatch')
                    require(np.allclose(C[:3,:3].T@C[:3,:3],np.eye(3),atol=2e-5) and abs(np.linalg.det(C[:3,:3])-1)<2e-5,'Camera rotation invalid')
                    if name=='head_camera':headshape[0]=tuple(rgb.shape)
                    stats=camstats.setdefault(hn,{'frames_decoded':0,'shape':list(rgb.shape),'depth_unit':'millimetres','depth_min_mm':float('inf'),'depth_max_mm':0.,'intrinsic_first':K.tolist(),'extrinsics_convention':'GL camera-to-world (HDF); CV world-to-camera also preserved in PKL','jpeg_exact_reencoded':True})
                    stats['depth_min_mm']=min(stats['depth_min_mm'],float(depth.min()));stats['depth_max_mm']=max(stats['depth_max_mm'],float(depth.max()))
                    if idx<n:
                        encoded=bytes(cg['colors'][idx]).rstrip(b'\0');decoded=codec.decode_image_bit(encoded)
                        require(decoded.shape==rgb.shape and decoded.dtype==np.uint8,'HDF JPEG decode shape invalid')
                        require(codec._has_rgb_marker(encoded),'HDF RGB marker missing')
                        require(encoded==codec.encode_image_bit(rgb),'JPEG bytes differ from official RGB encoder on PKL')
                        rgberrors.append(float(np.mean(np.abs(decoded.astype(np.int16)-rgb.astype(np.int16)))))
                        exact(cg['depths'][idx],depth,'HDF depth differs from PKL millimetres')
                        exact(cg['intrinsic_matrix'][idx],K,'HDF intrinsic differs from PKL')
                        exact(cg['extrinsics_matrix'][idx],C.astype(np.float32),'HDF extrinsics differ from cam2worldGL')
                        exact(cg['shape'][:],rgb.shape,'HDF camera shape differs')
                        stats['frames_decoded']+=1
            cs=np.asarray(clocks);require(cs[0,0]==0 and np.all(np.diff(cs[:,0])>=0),'Cache clock not initial-zero/nondecreasing')
            require(np.all(np.diff(cs[:,0])<=15),'Cache has gap larger than configured save_freq15')
            for hn,s in camstats.items():require(s['frames_decoded']==n,'Camera decoded frame count mismatch')
            require(steps-int(cs[-1,0])<=15,'Cache misses more than last save interval')
            return {'native14D_exact_vs_pkls':True,'native24D_exact_vs_pkls':True,'state24_exact_vs_physics_boundaries':True,
                    'cache_first_step':int(cs[0,0]),'cache_last_step':int(cs[-1,0]),'unrecorded_tail_steps':steps-int(cs[-1,0]),
                    'duplicate_action_boundary_frames':int(np.count_nonzero(np.diff(cs[:,0])==0)),
                    'max_cache_step_gap':int(np.max(np.diff(cs[:,0]))),'camera_stats':camstats,
                    'jpeg_lossy_mean_absolute_error_mean':float(np.mean(rgberrors)),'instructions':instructions}
        section('native14d_24d_cache_rgb_depth_calibration',observations)
    def videos():
        require(headshape[0] is not None,'No decoded head camera shape')
        native=video_readback(ds['video'],frames,30.,headshape[0])
        ext=report['external_realtime_video'];ev=Path(ds['external_realtime_video'])
        require(sha(ev)==ds['external_realtime_video_sha256'],'External video source hash mismatch')
        expected=np.arange(0,steps+1,10);exact(ext['frame_physics_steps'],expected,'External video not sampled at every 10 physical steps including t0')
        exact(ext['frame_times_s'],expected*dt,'External video times not actual physics timeline')
        require(ext['physics_stride']==10 and abs(ext['physics_dt_s']-dt)<1e-12 and ext['captured_initial_t0'] is True,'External physical timing schema mismatch')
        require(abs(10*dt-1/25)<1e-8,'External encoded 25fps cannot represent physical dt')
        observer=video_readback(ev,len(expected),25.,(480,640,3))
        require(ext['frames']==len(expected) and abs(ext['physics_duration_s']-steps*dt)<1e-10,'External reported duration/count mismatch')
        require(ext['trailing_unsampled_physics_steps']==steps%10,'External final unsampled remainder mismatch')
        return {'native_head':native,'native_head_realtime':False,'native_video_contains_last_unpaired_cache_frame':True,
                'external_observer':observer,'external_observer_realtime':True,'external_initial_t0':True,
                'external_tail_unsampled_steps':steps%10,'video_visual_quality_review':'Separate read-only visual QA; this validator checks decode and physical timing.'}
    section('video_decode_and_physical_timing',videos)
    def unchanged():
        for file,expected in initial_hashes.items():require(sha(file)==expected,'Evidence mutated during readback: '+file)
        for file,(size,mtime) in initial_sizes.items():
            st=Path(file).stat();require((st.st_size,st.st_mtime_ns)==(size,mtime),'PKL cache changed during readback: '+file)
        return {'read_only_dataset_evidence':True,'evidence_sha256':initial_hashes,'cache_size_mtime_unchanged':True}
    section('evidence_unchanged',unchanged)
    out['passed']=all(c['passed'] for c in out['checks'].values())
    return out

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--replay-dir',required=True);parser.add_argument('--repo',required=True);parser.add_argument('--output')
    args=parser.parse_args();replay=Path(args.replay_dir);output=Path(args.output) if args.output else replay/'dataset_validation.json'
    require(not output.exists(),'Refusing to overwrite validation evidence')
    try:result=validate(replay,args.repo)
    except Exception:result={'passed':False,'scope':'CPU readback only','traceback':traceback.format_exc()}
    with output.open('x',encoding='utf-8') as file:json.dump(result,file,indent=2,ensure_ascii=False)
    print(json.dumps({'passed':result['passed'],'output':str(output),'counts':result.get('counts'),'checks':{k:v['passed'] for k,v in result.get('checks',{}).items()}},ensure_ascii=False),flush=True)
    raise SystemExit(0 if result['passed'] else 2)

if __name__=='__main__':main()
