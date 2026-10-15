"""RoboTwin RGB50 collector v2 with creation-time axial roll: additive sampled scene, genuine dynamics and native RGB schema."""
from pathlib import Path
import argparse, copy, hashlib, json, os, pickle, sys, traceback
import numpy as np

WORK = Path(__file__).resolve().parent
ROOT = Path('/bigdata2/liminghe/VLA-benchmark/RoboTwin')
LEGACY = ROOT.parent / 'deploy/realman_gen4/adjust_bottle_20261002'
OLD20 = ROOT.parent / 'deploy/realman_gen4/rgb20_final_20261004/dataset'
CONFIG = ROOT.parent / 'deploy/realman_gen4/batch_runs_v32/realman_adjust_bottle_batch20_20261003/attempts/seed_0000007/bootstrap/run_config.yml'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def json_write(path,data):
    Path(path).write_text(json.dumps(data,indent=2,ensure_ascii=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x)),encoding='utf-8')

def prepare_task(spec, out, *, root=ROOT, need_planner=True, reuse_robot=None, backend="curobo"):
    import yaml, sapien, trimesh
    from transforms3d.quaternions import mat2quat
    from transforms3d.axangles import axangle2mat
    from envs.adjust_bottle import adjust_bottle
    from envs.utils.create_actor import create_actor
    from integrations.realman_rm65b_gen4.three_camera_rgb import attach_rgb_recording
    from geometry_overlay import install_native_shape_count_overlay
    overlay_receipt=install_native_shape_count_overlay(LEGACY)
    import importlib
    install_initializer=importlib.import_module(os.environ.get("RGB50_EXPERT_MODULE","expert_generation")).install_initializer
    import full_mesh_path_audit as fma
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    args=yaml.safe_load(CONFIG.read_text())
    profile=yaml.safe_load((OLD20/'episodes/episode_0000000/profile_used.yml').read_text())
    ready=json.loads((LEGACY/'ready_home_elevated.json').read_text())
    for side in ('left','right'):
        args[side+'_embodiment_config'].update(hand_joint_stiffness=20.,hand_joint_damping=.2,
            hand_synergy={'open':[1000,1000,1000,1000,1000,0],'close':[650,670,610,610,887,0]})
    model=int(spec['model_id']); arm=spec['arm'];tag=0 if arm=='left' else 1
    data=json.loads((root/f'assets/objects/001_bottle/model_data{model}.json').read_text())
    scene=trimesh.load(root/f'assets/objects/001_bottle/collision/base{model}.glb',force='scene')
    mesh=scene.dump(concatenate=True); vertices=np.asarray(mesh.vertices)*np.asarray(data['scale'])
    local_center=(vertices.min(axis=0)+vertices.max(axis=0))/2
    # World bottle-axis heading is measured from +X. All assets use local +Y.
    yaw=np.deg2rad(float(spec.get('axis_yaw_deg',spec.get('yaw_deg',0.))))
    kind=spec['pose_class']
    if kind=='upright':
        R=axangle2mat([0,0,1],yaw)@axangle2mat([1,0,0],np.pi/2)
    elif kind in ('horizontal','diagonal'):
        R=axangle2mat([0,0,1],yaw-np.pi/2)
    else: raise ValueError('Unsupported physical pose class '+kind)
    axial_roll_deg=float(spec.get('axial_roll_deg',0.))
    if axial_roll_deg: R=R@axangle2mat([0,1,0],np.deg2rad(axial_roll_deg))
    center_xy=np.asarray(spec.get('center_xy',spec.get('xy')),float)
    if center_xy.shape!=(2,): raise ValueError('Explicit bottle center_xy required')
    origin=np.zeros(3);origin[:2]=center_xy-(R@local_center)[:2]
    origin[2]=.74-float((vertices@R.T)[:,2].min())+.002
    class SampleScene(adjust_bottle):
        def load_robot(self, **kwargs):
            if need_planner and backend=='curobo':
                return super().load_robot(**kwargs)
            if need_planner and backend=='full_native':
                from integrations.realman_rm65b_gen4.native_robot import RealManRobot
                class FullNativeRobot(RealManRobot):
                    def set_planner(self,scene=None):
                        self.communication_flag=False;self.need_topp=True
                        self._rebind_mplib(scene)
                        self.left_planner=self.left_mplib_planner;self.right_planner=self.right_mplib_planner
                    def reset(self,scene,need_topp=False,**kw):
                        self._init_robot_(scene,need_topp,**kw);self.set_planner(scene);self.init_joints()
                if hasattr(self,'robot'):self.robot.reset(self.scene,True,**kwargs)
                else:
                    self.robot=FullNativeRobot(self.scene,need_topp=True,**kwargs)
                    self.robot.set_planner(self.scene);self.robot.init_joints()
                return
            from integrations.realman_rm65b_gen4.native_robot import RealManRobot
            class ViewRobot(RealManRobot):
                def set_planner(self, scene=None): pass
            self.robot=ViewRobot(self.scene,need_topp=False,**kwargs)
            self.robot.communication_flag=False
            self.robot.init_joints()
        def together_open_gripper(self, **kwargs):
            if need_planner: return super().together_open_gripper(**kwargs)
            for side in ('left','right'): self.robot.set_gripper(1,side,1.0)
        def delay(self, duration):
            if need_planner: return super().delay(duration)
            for _ in range(int(round(duration/.004))): self.scene.step()
        def load_actors(self):
            self.model_id=model;self.qpose_tag=tag
            self.bottle=create_actor(self,sapien.Pose(origin,mat2quat(R)),modelname='001_bottle',convex=True,model_id=model)
            # Ordinary gravity settles the sampled pose. This is initialization,
            # before collection begins; no object pose/velocity writes after it.
            self.delay(4)
            self.add_prohibit_area(self.bottle,padding=.15)
            self.left_target_pose=[-.25,-.12,.95,0,1,0,0]
            self.right_target_pose=[.25,-.12,.95,0,1,0,0]
    args.update(seed=int(spec['seed']),save_data=False,save_freq=None,need_plan=True,
                now_ep_num=0,save_path=str(out),render_freq=0)
    task=SampleScene()
    if reuse_robot is not None: task.robot=reuse_robot
    original_initializer=install_initializer(ready,LEGACY,task=task)
    args=attach_rgb_recording(task,args,profile)
    task.setup_demo(**args)
    comp=next(c for c in task.bottle.actor.get_components() if hasattr(c,'collision_shapes'))
    actual_vertices,_=fma._link_vertices(comp)
    M=task.bottle.get_pose().to_transformation_matrix()
    world_vertices=actual_vertices@M[:3,:3].T+M[:3,3]
    center=(world_vertices.min(0)+world_vertices.max(0))/2
    shoulders={s:np.asarray(task.robot.left_entity.find_link_by_name(('l_' if s=='left' else 'r_')+'link1').get_pose().p) for s in ('left','right')}
    distances={s:float(np.linalg.norm(center-p)) for s,p in shoulders.items()}
    nearest=min(distances,key=distances.get)
    axis=M[:3,1];upright=float(np.degrees(np.arccos(np.clip(abs(axis[2]),0,1))))
    actual_heading=float(np.degrees(np.arctan2(axis[1],axis[0])))
    receipt={'shape_count_overlay':overlay_receipt,'planning_backend':backend,'creation_axial_roll_deg':axial_roll_deg,'sample_spec':spec,'actual_bottle_center_world':center.tolist(),'actual_collision_vertices':len(world_vertices),'actual_collision_shape_count':len(comp.collision_shapes),
             'nearest_arm':nearest,'selected_arm':arm,'nearest_rule':'Euclidean distance from actual collision bbox center to fixed shoulder joint1 origins',
             'shoulder_origins_world':{s:p.tolist() for s,p in shoulders.items()},'arm_distances_m':distances,
             'nearest_margin_m':abs(distances['left']-distances['right']),
             'initial_pose':M.tolist(),'actual_axis_world':axis.tolist(),'upright_angle_deg':upright,'actual_axis_yaw_deg':actual_heading,
             'sampled_pose_settled_by_gravity':True,'source_config_sha256':sha(CONFIG),'camera_profile_sha256':sha(OLD20/'episodes/episode_0000000/profile_used.yml'),
             'old20_manifest_sha256':sha(OLD20/'manifest.json'),'hardware_io':False}
    json_write(out/'scene_sample.json',receipt)
    (out/'run_config.yml').write_text(yaml.safe_dump(args,sort_keys=False),encoding='utf-8')
    if nearest!=arm or receipt['nearest_margin_m']<.020: raise RuntimeError('Sampled arm is not unambiguously nearest')
    if (kind=='upright' and upright>5) or (kind!='upright' and upright<85): raise RuntimeError('Settled orientation class differs')
    if kind!='upright':
        heading_distance=min(actual_heading%180,180-actual_heading%180)
        if kind=='horizontal' and heading_distance>12: raise RuntimeError('Settled horizontal heading exceeds12deg')
        if kind=='diagonal' and not 25<=heading_distance<=75: raise RuntimeError('Settled diagonal heading outside25..75deg')
    return task,receipt,original_initializer

def export_recording(task,out,report,frames):
    import h5py
    from envs.utils.pkl2hdf5 import process_folder_to_hdf5_video
    # Exporter does not run or alter the expert module.
    out=Path(out); n=len(frames)-1
    if n<=0: raise RuntimeError('Incomplete N+1 recording')
    for d in ('data','video','instruction','_traj_data'): (out/d).mkdir(exist_ok=True)
    instruction='Pick up the soda bottle with the nearer arm, turn it upright, and hold it steadily in the air.'
    hdf=out/'data/episode_0000000.hdf5'
    pairs=process_folder_to_hdf5_video(str(out/'.cache/episode0'),str(hdf),str(out/'video/episode_0000000.mp4'),instructions=[instruction],frequency=15)
    if pairs!=n: raise RuntimeError('Native exporter pair count differs')
    samples=json.loads((out/'recording_samples.json').read_text())
    times=np.asarray([r['sim_time_s'] for r in samples]);states=np.asarray([r['measured_state24'] for r in samples]);commands=np.asarray([r['command24'] for r in samples])
    if len(samples)!=n+1 or times[0]!=0 or not np.allclose(np.diff(times),.060,atol=1e-8): raise RuntimeError('Fixed60ms capture timeline differs')
    telemetry=np.load(out/'physics_telemetry.npz')
    names=[j.get_name() for j in task.robot.left_entity.get_active_joints()]
    hand_names=[name for side in ('left','right') for name in task.robot._manifest['hands'][side]['limits']]
    with h5py.File(hdf,'a') as f:
        g=f.create_group('realman_dexterous');g.attrs['sdk_mapping_calibrated']=False
        g.attrs['ordering']='Larm6 rad,Lprimary SDK6,Rarm6 rad,Rprimary SDK6'
        g.attrs['command_source']='actual drive targets at next observation timestamps'
        for name,values in {'measured_state24':states[:-1],'next_command24':commands[1:],'sample_time_s':times[:-1],'next_sample_time_s':times[1:]}.items():g.create_dataset(name,data=values,compression='gzip')
        p=f.create_group('realman_dexterous_physics');p.attrs['sdk_mapping_calibrated']=False
        p.attrs['hand24_joint_names']=json.dumps(hand_names);p.attrs['full36_joint_names']=json.dumps(names)
        p.attrs['ordering']=g.attrs['ordering'];p.attrs['command_source']='actual drive targets before each physics step'
        for name in telemetry.files:p.create_dataset(name,data=telemetry[name],compression='gzip')
        f.attrs['physical_success_report_sha256']=sha(out/'expert_report.json')
        f.attrs['physical_calibration_available']=False;f.attrs['new_physics_episode']=True
        f.attrs['recording_clock']='60ms observations,4ms physical steps'
    json_write(out/'instruction/episode_0000000.json',{'instructions':[instruction],'seen':[instruction],'unseen':[]})
    task.save_traj_data(0)
    json_write(out/'scene_info.json',{'episode_0':task.info})
    # Independent reader decodes all three RGB streams and checks every native
    # and dexterous field. Actual success remains separate from RGB validation.
    import importlib.util
    path=OLD20.parent/'validate_rgb_three_camera.py'
    spec=importlib.util.spec_from_file_location('rgb50_rgb_validator',path);validator=importlib.util.module_from_spec(spec);spec.loader.exec_module(validator)
    validation=validator.validate(hdf,ROOT,None);json_write(out/'dataset_validation.json',validation)
    import subprocess
    video_info={}
    for source_name,target_name in (("head_camera","head"),("left_camera","left_wrist"),("right_camera","right_wrist")):
        dest=out/f"video/{target_name}_realtime.mp4"
        command=["ffmpeg","-nostdin","-loglevel","error","-f","rawvideo","-pixel_format","rgb24",
            "-video_size","640x480","-framerate","30","-i","pipe:0","-c:v","libx264",
            "-pix_fmt","yuv420p","-crf","18","-movflags","+faststart",str(dest)]
        process=subprocess.Popen(command,stdin=subprocess.PIPE)
        sample=-1;rgb=None;count=int(np.ceil(times[-1]*30));max_age=0.
        try:
            for tick in range(count):
                t=min(tick/30,times[-1]);idx=max(int(np.searchsorted(times,t,side="right")-1),0)
                if idx!=sample:
                    with (out/f".cache/episode0/{idx}.pkl").open('rb') as f: rgb=pickle.load(f)["observation"][source_name]["rgb"]
                    sample=idx
                max_age=max(max_age,float(t-times[idx]));process.stdin.write(rgb.tobytes())
        finally:process.stdin.close()
        if process.wait()!=0:raise RuntimeError("Video encoding failed")
        video_info[target_name]={"path":str(dest),"frames":count,"playback_fps":30,
            "duration_s":count/30,"max_observation_age_s":max_age,"method":"zero-order hold actual sample clock"}
    json_write(out/'recording_report.json',{'accepted':True,'native_state_action_pairs':n,'raw_observations':n+1,'hdf5':str(hdf),'hdf5_sha256':sha(hdf),'rgb_only':True,'training_cameras':['cam_head','cam_left_wrist','cam_right_wrist'],'new_physics_episode':True,'hardware_io':False,'actual_duration_s':float(times[-1]),'realtime_videos':video_info})
    return str(hdf)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--spec',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--no-rgb',action='store_true');ap.add_argument('--setup-only',action='store_true')
    opt=ap.parse_args();spec=json.loads(opt.spec.read_text());out=opt.output.resolve()
    if out.exists() and any(out.iterdir()):raise RuntimeError('Refusing to overwrite '+str(out))
    out.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('ROBOTWIN_RT_DENOISER','none');sys.path[:0]=[str(ROOT),str(LEGACY),str(WORK)];os.chdir(ROOT)
    task=None;original=None
    try:
        task,receipt,original=prepare_task(spec,out,need_planner=not opt.setup_only)
        if opt.setup_only:
            from PIL import Image
            obs=task.get_obs()
            for name in ('head_camera','left_camera','right_camera'):Image.fromarray(obs['observation'][name]['rgb']).save(out/(name+'_initial.png'))
            return
        from expert_generation import run_episode
        frames=[];cache=out/'.cache/episode0'
        if not opt.no_rgb:cache.mkdir(parents=True)
        def capture(obs,step):
            index=len(frames)
            with (cache/f'{index}.pkl').open('wb') as stream:pickle.dump(obs,stream)
            frames.append(step)
        report=run_episode(task,spec,out,legacy_dir=LEGACY,capture_callback=None if opt.no_rgb else capture)
        if not opt.no_rgb:export_recording(task,out,report,frames)
    except BaseException as e:
        json_write(out/'runner_error.json',{'error':str(e),'traceback':traceback.format_exc(),'accepted':False});raise
    finally:
        if task is not None:
            try:task.close_env()
            except Exception:pass
        if original is not None:
            from integrations.realman_rm65b_gen4.native_robot import RealManRobot
            RealManRobot._initialize_hands=original

if __name__=='__main__':main()
