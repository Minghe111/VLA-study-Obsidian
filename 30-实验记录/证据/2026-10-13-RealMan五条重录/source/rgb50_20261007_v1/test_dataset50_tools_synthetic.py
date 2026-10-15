"""Explicit synthetic fixture tests. Zero physics/model/GPU runs.

These objects only exercise reader/publisher schema checks and must never be
used as robot demonstrations. Production audit rejects them by default.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import pickle
import shutil
import uuid

import h5py
import numpy as np

import dataset50_audit as audit
import dataset50_publish as pub
from sample_plan import quat_matrix, SHOULDERS_WORLD_M


def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+"\n",encoding="utf-8")


def build_base(path, codec_path):
    path.mkdir();n=20;p=300;dt=.004
    times=np.arange(n+1)*.06;steps=np.arange(n+1)*15
    samples=[{"physics_step":int(steps[i]),"sim_time_s":float(times[i]),"measured_state24":[0.]*24,"command24":[0.]*24} for i in range(n+1)]
    write_json(path/"recording_samples.json",samples)
    telemetry={name:np.zeros((p,)+shape,dtype=np.float64) for name,shape in audit.PHYSICS.items()}
    telemetry["physics_step"]=np.arange(1,p+1,dtype=np.int64)
    telemetry["sim_time_before_s"]=np.arange(p)*dt;telemetry["sim_time_after_s"]=np.arange(1,p+1)*dt
    np.savez_compressed(path/"physics_telemetry.npz",**telemetry)
    codec,_=audit.find_codec(codec_path);native={key:0. if key.endswith("gripper") else [0.]*6 for key in ("left_arm","left_gripper","right_arm","right_gripper")}
    K=np.array([[400.,0,320],[0,400,240],[0,0,1]]);C=np.eye(4)
    end={"left_endpose":[0,0,0,1,0,0,0],"right_endpose":[0,0,0,1,0,0,0]}
    encoded={name:[] for name in audit.CAMERAS}
    cache=path/".cache/episode0";cache.mkdir(parents=True)
    for i in range(n+1):
        observations={}
        for j,(name,raw_name) in enumerate(audit.CAMERAS.items()):
            image=np.full((480,640,3),[i*3,20+j*30,100],dtype=np.uint8)
            observations[raw_name]={"rgb":image,"intrinsic_cv":K,"cam2world_gl":C}
            if i<n:encoded[name].append(codec.encode_image_bit(image))
        raw={"observation":observations,"joint_action":native,"endpose":end,
             "recording_clock":{"physics_step":int(steps[i]),"sim_time_s":float(times[i])},
             "dexterous_state":{"state24":[0.]*24,"command24":[0.]*24}}
        (cache/f"{i}.pkl").write_bytes(pickle.dumps(raw,protocol=4))
    full=[f"l_joint{i}" for i in range(1,7)]+[f"r_joint{i}" for i in range(1,7)]
    hands=[prefix+f"synthetic_hand_joint{i}" for prefix in ("l_","r_") for i in range(12)]
    full+=hands
    hdf=path/"data/episode_0000000.hdf5";hdf.parent.mkdir()
    with h5py.File(hdf,"w") as f:
        f.attrs.update(source_format="RoboTwin",new_physics_episode=True,physical_calibration_available=False)
        f.create_dataset("data_format_version",data="v1.0");f.create_dataset("instructions",data='["SYNTHETIC FIXTURE ONLY"]')
        f.create_dataset("additional_info/frequency",data=15)
        for group in ("state","action"):
            for name,(_,width) in audit.NATIVE.items():f.create_dataset(group+"/"+name,data=np.zeros((n,width or 1),np.float32))
            for name in ("left_ee_poses","right_ee_poses"):f.create_dataset(group+"/"+name,data=np.tile([0,0,0,1,0,0,0],(n,1)).astype(np.float32))
        for name in audit.CAMERAS:
            cg=f.create_group("vision/"+name);width=max(map(len,encoded[name]));cg.create_dataset("colors",data=encoded[name],dtype=f"S{width}")
            cg.create_dataset("shape",data=[480,640,3]);cg.create_dataset("intrinsic_matrix",data=np.tile(K,(n,1,1)));cg.create_dataset("extrinsics_matrix",data=np.tile(C,(n,1,1)))
        g=f.create_group("realman_dexterous");g.attrs["sdk_mapping_calibrated"]=False
        for name,data in {"measured_state24":np.zeros((n,24)),"next_command24":np.zeros((n,24)),"sample_time_s":times[:-1],"next_sample_time_s":times[1:]}.items():g.create_dataset(name,data=data)
        g=f.create_group("realman_dexterous_physics");g.attrs["full36_joint_names"]=json.dumps(full);g.attrs["hand24_joint_names"]=json.dumps(hands)
        for name,data in telemetry.items():g.create_dataset(name,data=data)
    (path/"_traj_data").mkdir();(path/"_traj_data/episode0.pkl").write_bytes(pickle.dumps({"left_joint_path":[],"right_joint_path":[]},protocol=4))
    write_json(path/"instruction/episode_0000000.json",{"instructions":["SYNTHETIC FIXTURE ONLY"],"seen":["SYNTHETIC FIXTURE ONLY"]})
    (path/"video").mkdir();(path/"video/episode_0000000.mp4").write_bytes(b"SYNTHETIC FIXTURE NOT A REAL VIDEO")
    for name in ("head","left_wrist","right_wrist"):(path/f"video/{name}_realtime.mp4").write_bytes(b"SYNTHETIC PREVIEW ONLY")
    return full


def clone(base, target):
    def copier(src,dst):
        if "/.cache/" in str(src).replace("\\","/"): __import__("os").link(src,dst)
        else:shutil.copy2(src,dst)
    shutil.copytree(base,target,copy_function=copier)


def stamp_identity(path, spec, full):
    p=300;M=np.eye(4);M[:3,:3]=quat_matrix(spec["actor_q_wxyz"]);M[:3,3]=spec["actor_origin_world_m"]
    center=np.asarray(spec["initial_geometry_center_world_m"]);shoulders={k:np.asarray(v) for k,v in SHOULDERS_WORLD_M.items()}
    distances={s:float(np.linalg.norm(center-v)) for s,v in shoulders.items()};axis=M[:3,1]
    scene={"sample_spec":spec,"actual_bottle_center_world":center.tolist(),"actual_collision_vertices":100,
        "nearest_arm":spec["arm"],"selected_arm":spec["arm"],"shoulder_origins_world":SHOULDERS_WORLD_M,
        "arm_distances_m":distances,"nearest_margin_m":abs(distances["left"]-distances["right"]),
        "initial_pose":M.tolist(),"actual_axis_world":axis.tolist(),"upright_angle_deg":float(np.degrees(np.arccos(np.clip(abs(axis[2]),0,1)))),
        "actual_axis_yaw_deg":float(np.degrees(np.arctan2(axis[1],axis[0]))),"sampled_pose_settled_by_gravity":True,
        "old20_manifest_sha256":"97b33109a740ae2d9b99155dbf228e5d2a19761b323e60fc622c8998aec0e577","hardware_io":False}
    write_json(path/"scene_sample.json",scene)
    final=np.eye(4);final[:3,:3]=quat_matrix([2**-.5,2**-.5,0,0]);final[:3,3]=[(-.25 if spec["arm"]=="left" else .25),-.12,.85]
    bottle={"model_id":spec["model_id"],"qpose_tag":0 if spec["arm"]=="left" else 1,"pose":final.tolist(),"official_success":True,"upright_angle_deg":0.}
    trace=[]
    for i in range(p):
        trace.append({"step":i+1,"sim_time_s":(i+1)*.004,"phase":{"name":"after_grasp","terminal_hold":i>=50},
            "full_qpos_actual_rad":[0.]*36,"command24_actual":[0.]*24,"bottle":bottle,"fp_delta_m":0.,"rotation_delta_deg":0.,
            "actual_self_contacts":[],"grip":{"table_normal_N":0.,"thumb_normal_N":2.3,"fingers_normal_N":2.3,"opposing_normals_dot":-1.,"contacts":[]},
            "bottle_bottom_gap_m":.1,"relative_translation_drift_m":0.,"relative_rotation_drift_deg":0.})
    write_json(path/"physics_trace.json",trace)
    write_json(path/"actual36_geometry_audit.json",{"passed":True,"full_temporal_coverage":True,"complete_to_reported_endpoint":True,
        "physical_scene_unchanged":True,"trace_sha256":audit.sha(path/"physics_trace.json"),"synthetic_fixture_only":True})
    report={"accepted":True,"result":"physics_success","hold_passed":True,"canonical_sources_unchanged":True,"official_task_methods_unmodified":True,
        "hardware_io":False,"active_arm":spec["arm"],"sample_spec":spec,"source_hashes":{},"physical_steps":p,"recorded_frames":21,
        "full_qpos_joint_names":full,"after_hold":bottle,"action_clock":{"physics_dt_s":.004,"record_every_physics_steps":15},"synthetic_fixture_only":True}
    write_json(path/"expert_report.json",report)
    hdf=path/"data/episode_0000000.hdf5"
    with h5py.File(hdf,"a") as f:f.attrs["physical_success_report_sha256"]=audit.sha(path/"expert_report.json")
    write_json(path/"recording_report.json",{"accepted":True,"rgb_only":True,"new_physics_episode":True,"native_state_action_pairs":20,"raw_observations":21,"hdf5_sha256":audit.sha(hdf),"synthetic_fixture_only":True})


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,required=True);ap.add_argument("--codec",type=Path,required=True);args=ap.parse_args()
    root=args.root.resolve();root.mkdir(parents=True,exist_ok=False)
    specs=audit.read_json(Path(__file__).parent/"sample_plan.json")["specs"]
    base=root/"synthetic_base";full=build_base(base,args.codec)
    trials=[]
    for spec in specs:
        path=root/f"synthetic_trial_{spec['index']:03d}";clone(base,path);stamp_identity(path,spec,full);trials.append(path)
    valid=audit.audit_trial(trials[0],codec_path=args.codec,allow_synthetic=True)
    assert valid["passed"],valid.get("error")
    tests=[{"case":"valid_schema_only_synthetic", "passed":True}]
    assert not audit.audit_trial(trials[0],codec_path=args.codec)["passed"]
    tests.append({"case":"production_rejects_synthetic_by_default","passed":True})
    float_path=root/"synthetic_actual_sapien_float_dt";clone(trials[0],float_path);float_dt=.004000000189989805
    rewrite_dt(float_path,float_dt)
    float_result=audit.audit_trial(float_path,codec_path=args.codec,allow_synthetic=True)
    assert float_result["passed"],float_result.get("error")
    tests.append({"case":"accept_actual_sapien_float_dt_and15step_clock","passed":True,"dt_s":float_dt})
    mutations={
        "nearest_label":lambda path:write_json(path/"scene_sample.json",dict(audit.read_json(path/"scene_sample.json"),selected_arm="right" if specs[0]["arm"]=="left" else "left")),
        "clock_offset":lambda path:modify_hdf(path,"realman_dexterous/sample_time_s",0,.001),
        "full36_mismatch":lambda path:modify_hdf(path,"realman_dexterous_physics/full36_qpos_after_rad",(1,0),.1),
        "depth_present":lambda path:add_depth(path),
        "raw_cache_missing":lambda path:(path/".cache/episode0/20.pkl").unlink(),
        "pose_class_mismatch":lambda path:alter_class(path),
        "jpeg_wrong_capture":lambda path:shift_rgb(path),
    }
    for name,mutate in mutations.items():
        path=root/("synthetic_bad_"+name);clone(trials[0],path);mutate(path)
        result=audit.audit_trial(path,codec_path=args.codec,allow_synthetic=True)
        assert not result["passed"],name
        tests.append({"case":name,"passed":True,"rejection":result["error"]})
    incomplete=pub.publish(trials[:1],root/"must_not_publish_insufficient",codec_path=args.codec,allow_synthetic=True)
    assert not incomplete["complete"] and not (root/"must_not_publish_insufficient").exists()
    tests.append({"case":"insufficient50_never_publishes","passed":True})
    published=pub.publish(trials,root/"SYNTHETIC_TEST_PUBLICATION_NOT_ROBOT_DATA",codec_path=args.codec,allow_synthetic=True,hardlink=True)
    assert published["complete"] and published["status"]=="synthetic_fixture50_test_only",published
    manifest=audit.read_json(root/"SYNTHETIC_TEST_PUBLICATION_NOT_ROBOT_DATA/manifest.json")
    assert manifest["synthetic_fixture_only"] and len(manifest["episodes"])==50
    assert len(list((root/"SYNTHETIC_TEST_PUBLICATION_NOT_ROBOT_DATA/video").glob("*_realtime.mp4")))==150
    assert all(len(list((root/"SYNTHETIC_TEST_PUBLICATION_NOT_ROBOT_DATA"/ep["raw_cache_dir"]).glob("*.pkl")))==21 for ep in manifest["episodes"])
    tests.append({"case":"synthetic50_atomic_publish_copies_all_raw_cache_and3_realtime_videos","passed":True})
    receipt={"synthetic_fixture_only":True,"actual_robot_episodes_generated":0,"new_physics_runs":0,"gpu_runs":0,
             "model_queries":0,"real_dataset_complete":False,"tests":tests,"all_passed":True,"test_root":str(root)}
    write_json(root/"synthetic_test_results.json",receipt)
    print(json.dumps(receipt,indent=2))


def rewrite_dt(path,dt):
    with np.load(path/"physics_telemetry.npz",allow_pickle=False) as f:data={k:f[k].copy() for k in f.files}
    p=len(data["physics_step"]);data["sim_time_before_s"]=np.arange(p)*dt;data["sim_time_after_s"]=np.arange(1,p+1)*dt
    np.savez_compressed(path/"physics_telemetry.npz",**data)
    samples=audit.read_json(path/"recording_samples.json")
    for row in samples:row["sim_time_s"]=row["physics_step"]*dt
    write_json(path/"recording_samples.json",samples)
    trace=audit.read_json(path/"physics_trace.json")
    for row in trace:row["sim_time_s"]=row["step"]*dt
    write_json(path/"physics_trace.json",trace)
    geo=audit.read_json(path/"actual36_geometry_audit.json");geo["trace_sha256"]=audit.sha(path/"physics_trace.json");write_json(path/"actual36_geometry_audit.json",geo)
    report=audit.read_json(path/"expert_report.json");report["action_clock"]["physics_dt_s"]=dt;write_json(path/"expert_report.json",report)
    with h5py.File(path/"data/episode_0000000.hdf5","a") as f:
        f["realman_dexterous/sample_time_s"][:]=[row["sim_time_s"] for row in samples[:-1]]
        f["realman_dexterous/next_sample_time_s"][:]=[row["sim_time_s"] for row in samples[1:]]
        for name,values in data.items():f["realman_dexterous_physics/"+name][:]=values
        f.attrs["physical_success_report_sha256"]=audit.sha(path/"expert_report.json")
    for file,row in zip(sorted((path/".cache/episode0").glob("*.pkl"),key=lambda p:int(p.stem)),samples):
        raw=audit.numeric_pickle(file);raw["recording_clock"]["sim_time_s"]=row["sim_time_s"];file.unlink();file.write_bytes(pickle.dumps(raw,protocol=4))
    repin(path)

def repin(path):
    hdf=path/"data/episode_0000000.hdf5";r=audit.read_json(path/"recording_report.json");r["hdf5_sha256"]=audit.sha(hdf);write_json(path/"recording_report.json",r)


def modify_hdf(path,name,index,value):
    with h5py.File(path/"data/episode_0000000.hdf5","a") as f:f[name][index]=value
    repin(path)


def add_depth(path):
    with h5py.File(path/"data/episode_0000000.hdf5","a") as f:f.create_dataset("vision/cam_head/depths",data=np.zeros((20,1)))
    repin(path)


def alter_class(path):
    s=audit.read_json(path/"scene_sample.json");s["sample_spec"]["pose_class"]="horizontal" if s["sample_spec"]["pose_class"]=="upright" else "upright";write_json(path/"scene_sample.json",s)


def shift_rgb(path):
    with h5py.File(path/"data/episode_0000000.hdf5","a") as f:
        d=f["vision/cam_head/colors"];d[0]=d[1]
    repin(path)

if __name__=="__main__":main()
