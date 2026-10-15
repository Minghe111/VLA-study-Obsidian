"""Synthetic CPU-only unit regression; no user simulation, SAPIEN or GPU."""
import importlib.util,json,pickle,tempfile,pathlib,copy,unittest
import numpy as np,cv2,h5py
BASE=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('dataset_readback',BASE/'validate_collected_dataset.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
REPO=pathlib.Path('/bigdata2/liminghe/VLA-benchmark/RoboTwin')
spec=importlib.util.spec_from_file_location('native_codec',REPO/'data/decode_image_bit.py');codec=importlib.util.module_from_spec(spec);spec.loader.exec_module(codec)

def make_fixture(p):
    (p/'data').mkdir();(p/'video').mkdir();(p/'.cache/episode0').mkdir(parents=True)
    nsteps=30;dt=.004;names=[f'q{i}' for i in range(36)];hand=names[12:]
    q=np.arange(nsteps+1)[:,None]*np.arange(36)[None,:]*1e-6
    st=np.concatenate([q[:,:6],np.full((31,6),900.),q[:,6:12],np.full((31,6),900.)],axis=1)
    data={'physics_step':np.arange(1,31),'sim_time_before_s':np.arange(30)*dt,'sim_time_after_s':np.arange(1,31)*dt,
          'measured_state24_before':st[:-1],'measured_state24_after':st[1:],'applied_command24':st[1:].copy(),
          'full36_qpos_before_rad':q[:-1],'full36_qpos_after_rad':q[1:],
          'hand24_qpos_before_rad':q[:-1,12:],'hand24_qpos_after_rad':q[1:,12:],
          'hand24_drive_target_rad':q[1:,12:],'hand24_actual_fd_rad_s':(q[1:,12:]-q[:-1,12:])/dt,
          'hand24_engine_qvel_rad_s':np.zeros((30,24))}
    ts=[0,15,15,30];rgb=np.zeros((24,32,3),np.uint8);rgb[:,:,0]=150;rgb[:,:,1]=50
    K=np.array([[30.,0,16],[0,30,12],[0,0,1]],np.float32);C=np.eye(4,dtype=np.float32);E=np.diag([1.,-1,-1,1]).astype(np.float32)[:3]
    obs=[]
    for i,step in enumerate(ts):
        s={'recording_clock':{'physics_step':step,'sim_time_s':step*dt},'dexterous_state':{'state24':st[step].tolist(),'command24':st[step].tolist()},
           'joint_action':{'left_arm':st[step,:6].tolist(),'left_gripper':1.,'right_arm':st[step,12:18].tolist(),'right_gripper':1.},
           'endpose':{'left_endpose':[0,0,0,1,0,0,0],'right_endpose':[0,0,0,1,0,0,0]},
           'observation':{'head_camera':{'rgb':rgb.copy(),'depth':np.ones((24,32))*1000,'intrinsic_cv':K,'extrinsic_cv':E,'cam2world_gl':C}}}
        (p/f'.cache/episode0/{i}.pkl').write_bytes(pickle.dumps(s));obs.append(s)
    hdf=p/'data/episode_0000000.hdf5'
    with h5py.File(hdf,'w') as f:
        f.attrs.update(source_format='RoboTwin',source_path='native_collection')
        f.create_dataset('data_format_version',data='v1.0');f.create_dataset('instructions',data=json.dumps(['adjust bottle']))
        f.create_group('additional_info').create_dataset('frequency',data=15)
        for group in ('state','action'):
            g=f.create_group(group);selected=obs[:-1] if group=='state' else obs[1:]
            for source,target in v.JOINTS.items():g.create_dataset(target,data=np.array([np.asarray(s['joint_action'][source]).reshape(-1) for s in selected],np.float32))
            for source,target in (('left_endpose','left_ee_poses'),('right_endpose','right_ee_poses')):g.create_dataset(target,data=np.array([s['endpose'][source] for s in selected],np.float32))
        cam=f.create_group('vision').create_group('cam_head');enc=[codec.encode_image_bit(s['observation']['head_camera']['rgb']) for s in obs[:-1]]
        cam.create_dataset('colors',data=enc,dtype=f'S{max(map(len,enc))}');cam.create_dataset('shape',data=[24,32,3]);cam.create_dataset('depths',data=np.ones((3,24,32))*1000)
        cam.create_dataset('intrinsic_matrix',data=np.repeat(K[None],3,axis=0));cam.create_dataset('extrinsics_matrix',data=np.repeat(C[None],3,axis=0))
        dex=f.create_group('realman_dexterous');dex.attrs.update(sdk_mapping_calibrated=False,command_source='actual drive targets at native frame timestamps')
        dex.create_dataset('measured_state24',data=st[ts[:-1]]);dex.create_dataset('next_command24',data=st[ts[1:]])
        dex.create_dataset('sample_time_s',data=np.array(ts[:-1])*dt);dex.create_dataset('next_sample_time_s',data=np.array(ts[1:])*dt)
        g=f.create_group('realman_dexterous_physics');g.attrs.update(sdk_mapping_calibrated=False,ordering='Larm6 rad, Lprimary SDK6, Rarm6 rad, Rprimary SDK6',command_source='read actual drive targets after feedback',full36_joint_names=json.dumps(names),hand24_joint_names=json.dumps(hand))
        for k,a in data.items():g.create_dataset(k,data=a)
    np.savez_compressed(p/'physics_command24_trace.npz',**{k:data[k] for k in v.NPZ_FIELDS})
    trace=[{'step':i+1,'sim_time_s':(i+1)*dt,'full_qpos_actual_rad':q[i+1].tolist(),'command24_actual':data['applied_command24'][i].tolist(),
            'arm_qpos':{'left':st[i+1,:6].tolist(),'right':st[i+1,12:18].tolist()},'actual_self_contacts':[]} for i in range(30)]
    (p/'physics_trace.json').write_text(json.dumps(trace));(p/'actual36_geometry_audit.json').write_text(json.dumps({'passed':True,'physical_scene_unchanged':True}))
    for file,shape,num,fps in ((p/'video/episode_0000000.mp4',(32,24),4,30.),(p/'video/external_realtime.mp4',(640,480),4,25.)):
        w=cv2.VideoWriter(str(file),cv2.VideoWriter_fourcc(*'mp4v'),fps,shape);assert w.isOpened()
        for _ in range(num):w.write(np.zeros((shape[1],shape[0],3),np.uint8))
        w.release()
    source=p/'source.py';source.write_text('protected dummy source\n')
    trajectory=p/'episode0.pkl';trajectory.write_bytes(b'synthetic trajectory')
    r={'result':'one_strictly_validated_episode_collected','accepted':True,'canonical_sources_unchanged':True,'source_trajectory_unchanged':True,
       'actual36_geometry_audit':{'passed':True},'physical_steps':30,'source_hashes':{str(source):v.sha(source)},'replay_helper_source_hashes':{},
       'source_trajectory':str(trajectory),'source_trajectory_sha256':v.sha(trajectory),'full_qpos_joint_names':names,
       'candidate_wrench_manifest':[],'candidate_wrench_sources_unchanged':True,'candidate_wrench_actions_complete':True,'hardware_io':False,'planner_calls_allowed':False,
       'dataset':{'pairs':3,'recorded_frames':4,'physical_step_s':dt,'hdf5':str(hdf),'sha256':v.sha(hdf),'cameras':['cam_head'],
                  'video':str(p/'video/episode_0000000.mp4'),'external_realtime_video':str(p/'video/external_realtime.mp4'),
                  'external_realtime_video_sha256':v.sha(p/'video/external_realtime.mp4')},
       'external_realtime_video':{'frame_physics_steps':[0,10,20,30],'frame_times_s':(np.arange(0,31,10)*dt).tolist(),
          'physics_stride':10,'physics_dt_s':dt,'captured_initial_t0':True,'frames':4,'physics_duration_s':.12,'trailing_unsampled_physics_steps':0}}
    (p/'replay_report.json').write_text(json.dumps(r));return r

class Readback(unittest.TestCase):
    def setUp(self):self.temp=tempfile.TemporaryDirectory(prefix='readback_cpu_',dir=BASE);self.p=pathlib.Path(self.temp.name);self.r=make_fixture(self.p)
    def tearDown(self):self.temp.cleanup()
    def save_report(self):self.r['dataset']['sha256']=v.sha(self.r['dataset']['hdf5']);(self.p/'replay_report.json').write_text(json.dumps(self.r))
    def check(self):return v.validate(self.p,REPO)
    def test_valid_all_fields_and_boundary_duplicates(self):
        r=self.check();self.assertTrue(r['passed'],r['checks']);self.assertEqual(r['checks']['native14d_24d_cache_rgb_depth_calibration']['duplicate_action_boundary_frames'],1)
    def test_reject_not_published(self):
        self.r['result']='physics_success';self.save_report()
        with self.assertRaises(AssertionError):self.check()
    def test_reject_failed_actual36(self):
        self.r['actual36_geometry_audit']['passed']=False;self.save_report()
        with self.assertRaises(AssertionError):self.check()
    def test_reject_official14_next_state_mismatch(self):
        with h5py.File(self.r['dataset']['hdf5'],'a') as f:f['action/left_arm_joint_states'][0,0]=.5
        self.save_report();self.assertFalse(self.check()['checks']['native14d_24d_cache_rgb_depth_calibration']['passed'])
    def test_reject_shared_npz_command_mismatch(self):
        file=self.p/'physics_command24_trace.npz'
        with np.load(file) as a:d={k:a[k].copy() for k in a.files}
        d['applied_command24'][0,0]=1.;np.savez_compressed(file,**d);self.assertFalse(self.check()['checks']['per_physics_step_24d_36d']['passed'])
    def test_reject_raw_joint_order(self):
        with h5py.File(self.r['dataset']['hdf5'],'a') as f:f['realman_dexterous_physics'].attrs['full36_joint_names']=json.dumps(list(reversed(self.r['full_qpos_joint_names'])))
        self.save_report();self.assertFalse(self.check()['checks']['per_physics_step_24d_36d']['passed'])
    def test_reject_rgb_channel_swap(self):
        file=self.p/'.cache/episode0/1.pkl';s=pickle.loads(file.read_bytes());s['observation']['head_camera']['rgb']=s['observation']['head_camera']['rgb'][...,::-1];file.write_bytes(pickle.dumps(s))
        self.assertFalse(self.check()['checks']['native14d_24d_cache_rgb_depth_calibration']['passed'])
    def test_reject_cv_gl_extrinsics_confusion(self):
        file=self.p/'.cache/episode0/0.pkl';s=pickle.loads(file.read_bytes());s['observation']['head_camera']['extrinsic_cv']=np.eye(4,dtype=np.float32)[:3];file.write_bytes(pickle.dumps(s))
        self.assertFalse(self.check()['checks']['native14d_24d_cache_rgb_depth_calibration']['passed'])
    def test_reject_physics_clock_mismatch(self):
        file=self.p/'.cache/episode0/1.pkl';s=pickle.loads(file.read_bytes());s['recording_clock']['sim_time_s']+=.1;file.write_bytes(pickle.dumps(s))
        self.assertFalse(self.check()['checks']['native14d_24d_cache_rgb_depth_calibration']['passed'])
    def test_reject_external_video_time_mismatch(self):
        self.r['external_realtime_video']['frame_times_s'][2]+=.001;self.save_report();self.assertFalse(self.check()['checks']['video_decode_and_physical_timing']['passed'])
    def test_reject_source_changed(self):
        (self.p/'source.py').write_text('changed\n');self.assertFalse(self.check()['checks']['publication_and_provenance']['passed'])
    def test_no_simulator_or_step_in_source(self):
        source=(BASE/'validate_collected_dataset.py').read_text();self.assertNotIn('import sapien',source);self.assertNotIn('scene.step(',source);self.assertNotIn('cuda',source.lower())

if __name__=='__main__':unittest.main(verbosity=2)
