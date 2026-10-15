"""Independent CPU readback for fresh50. Never imports expert or simulator.

RGB uses the native marked JPEG codec. Original raw N+1 caches are read with
an allowlisted numeric-only unpickler; every source file is hash-guarded.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import pickle
import traceback

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent
BRANDS = {13: "Coca-Cola", 16: "Sprite", 4: "Fanta"}
CAMERAS = {"cam_head": "head_camera", "cam_left_wrist": "left_camera", "cam_right_wrist": "right_camera"}
NATIVE = {"left_arm_joint_states": ("left_arm", 6), "left_ee_joint_states": ("left_gripper", None),
          "right_arm_joint_states": ("right_arm", 6), "right_ee_joint_states": ("right_gripper", None)}
PHYSICS = {"physics_step": (), "sim_time_before_s": (), "sim_time_after_s": (),
    "measured_state24_before": (24,), "applied_command24": (24,), "measured_state24_after": (24,),
    "hand24_qpos_before_rad": (24,), "hand24_qpos_after_rad": (24,), "hand24_drive_target_rad": (24,),
    "hand24_actual_fd_rad_s": (24,), "hand24_engine_qvel_rad_s": (24,),
    "full36_qpos_before_rad": (36,), "full36_qpos_after_rad": (36,)}
FORBIDDEN = {"depth", "depths", "depth_image", "depth_images", "pointcloud", "point_cloud", "pointclouds",
    "segmentation", "mesh_segmentation", "actor_segmentation", "segmentation_mesh", "segmentation_actor"}


def require(value, message):
    if not bool(value): raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""): h.update(block)
    return h.hexdigest()


def text(value):
    return value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def close(a, b, label, *, atol=1e-8):
    a, b = np.asarray(a), np.asarray(b)
    require(a.shape == b.shape and a.dtype.kind in "biuf" and b.dtype.kind in "biuf", "Shape/numeric mismatch: "+label)
    require(np.isfinite(a).all() and np.isfinite(b).all(), "Nonfinite: "+label)
    require(np.allclose(a, b, atol=atol, rtol=0), "Values differ: "+label)


def numeric_pickle(path):
    """No simulator classes or arbitrary GLOBAL/REDUCE constructors allowed."""
    path = Path(path)
    require(path.stat().st_size <= 32*1024*1024, "Oversized numeric pickle")
    data = path.read_bytes()
    def checked_dtype(*args):
        d = np.dtype(*args)
        require(not d.hasobject and d.fields is None and d.kind in "biufc", "Object/structured pickle dtype")
        return d
    def checked_frombuffer(data, dtype, shape, order):
        dtype = checked_dtype(dtype)
        require(isinstance(data, (bytes, bytearray)) and len(data) <= 32*1024*1024, "Array buffer invalid")
        require(isinstance(shape, (tuple, list)) and len(shape) <= 8 and all(type(v) is int and v >= 0 for v in shape), "Array shape invalid")
        require(math.prod(shape)*dtype.itemsize == len(data), "Array buffer/shape differ")
        return np.frombuffer(data, dtype=dtype).reshape(shape, order=order)
    allowed = {("numpy", "ndarray"): np.ndarray, ("numpy", "dtype"): checked_dtype}
    for module in ("numpy.core.multiarray", "numpy._core.multiarray"):
        allowed[module, "_reconstruct"] = np.core.multiarray._reconstruct
        allowed[module, "scalar"] = np.core.multiarray.scalar
    for module in ("numpy.core.numeric", "numpy._core.numeric"):
        allowed[module, "_frombuffer"] = checked_frombuffer
    class NumericUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            require((module, name) in allowed, "Disallowed pickle global: "+module+"."+name)
            return allowed[module, name]
        def persistent_load(self, pid): raise ValueError("Persistent pickle reference forbidden")
    stream = io.BytesIO(data)
    result = NumericUnpickler(stream).load()
    require(stream.tell() == len(data), "Trailing pickle bytes")
    seen = set()
    def visit(v, depth=0):
        require(depth <= 64, "Pickle nesting limit")
        if v is None or type(v) in (bool, int, float, str, bytes, bytearray): return
        if isinstance(v, np.ndarray):
            require(not v.dtype.hasobject and v.dtype.fields is None and v.dtype.kind in "biufc" and v.nbytes <= 32*1024*1024, "Unsafe array")
            return
        if isinstance(v, np.generic): checked_dtype(v.dtype); return
        require(type(v) in (dict, list, tuple), "Non-numeric/basic pickle object")
        require(id(v) not in seen, "Cyclic pickle")
        seen.add(id(v))
        for x in (list(v.keys())+list(v.values()) if isinstance(v, dict) else v): visit(x, depth+1)
        seen.remove(id(v))
    visit(result)
    return result


def find_codec(explicit=None):
    candidates = [Path(explicit)] if explicit else [p/"RoboTwin/data/decode_image_bit.py" for p in ROOT.parents]
    candidates += [ROOT/"decode_image_bit.py"]
    for p in candidates:
        if p.is_file():
            spec = importlib.util.spec_from_file_location("rgb50_native_codec", p)
            module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
            return module, p.resolve()
    raise FileNotFoundError("Pass --codec pointing to native RoboTwin/data/decode_image_bit.py")


def inventory(directory):
    directory = Path(directory).resolve()
    files = {}
    for p in sorted(directory.rglob("*")):
        require(not p.is_symlink(), "Trial symlink forbidden: "+str(p))
        if p.is_file(): files[str(p.relative_to(directory))] = {"sha256": sha(p), "bytes": p.stat().st_size}
    return files


def pose_receipt(scene):
    spec = scene["sample_spec"]; arm = spec["arm"]; kind = spec["pose_class"]
    require(arm in ("left", "right") and kind in ("upright", "horizontal", "diagonal"), "Unknown arm or pose class")
    require(int(spec["model_id"]) in BRANDS, "Unverified soda brand")
    center = np.asarray(scene["actual_bottle_center_world"], float)
    shoulders = {s: np.asarray(scene["shoulder_origins_world"][s], float) for s in ("left", "right")}
    require(center.shape == (3,) and all(v.shape == (3,) for v in shoulders.values()), "Near-arm geometry dimensions")
    require(np.isfinite(center).all() and all(np.isfinite(v).all() for v in shoulders.values()), "Nonfinite near geometry")
    distances = {s: float(np.linalg.norm(center-v)) for s,v in shoulders.items()}
    actual = min(distances, key=distances.get); margin = abs(distances["left"]-distances["right"])
    require(actual == arm == scene["selected_arm"] == scene["nearest_arm"], "Selected arm is not actual nearest")
    require(margin >= .020, "Near-arm margin below20mm")
    for s in shoulders: close(distances[s], scene["arm_distances_m"][s], "near distance "+s)
    close(margin, scene["nearest_margin_m"], "near margin")
    matrix = np.asarray(scene["initial_pose"], float)
    require(matrix.shape == (4,4), "Initial pose shape")
    close(matrix[3], [0,0,0,1], "Initial pose bottom row")
    close(matrix[:3,:3].T @ matrix[:3,:3], np.eye(3), "Initial rotation orthonormal", atol=2e-5)
    require(abs(np.linalg.det(matrix[:3,:3])-1) < 2e-5, "Initial improper rotation")
    axis = matrix[:3,1]
    close(axis, scene["actual_axis_world"], "Actual bottle local+Y axis", atol=2e-6)
    vertical = float(np.degrees(np.arccos(np.clip(abs(axis[2]),0,1))))
    heading = float(np.degrees(np.arctan2(axis[1],axis[0])))
    offset = abs((heading+90)%180-90)
    close(vertical, scene["upright_angle_deg"], "Initial vertical angle", atol=1e-5)
    close(heading, scene["actual_axis_yaw_deg"], "Initial heading", atol=1e-5)
    if kind == "upright": require(vertical <= 5, "Upright sample settled outside5deg")
    else:
        require(vertical >= 85, "Lying sample settled outside5deg from table")
        require(offset <= 20 if kind == "horizontal" else 35 <= offset <= 65, "Horizontal/diagonal heading class differs")
    require(scene.get("sampled_pose_settled_by_gravity") is True, "No gravity-settle evidence")
    return {"model_id": int(spec["model_id"]), "brand": BRANDS[int(spec["model_id"])], "arm": arm,
            "pose_class": kind, "seed": int(spec["seed"]), "spec_index": int(spec["index"]),
            "nearest_margin_m": margin, "upright_angle_deg": vertical, "heading_offset_deg": offset}


def audit_trial(directory, *, codec_path=None, allow_synthetic=False):
    directory = Path(directory).resolve()
    result = {"passed": False, "trial_dir": str(directory), "readback_only": True, "new_physics_runs": 0, "hardware_io": False}
    try:
        before = inventory(directory)
        required = ["scene_sample.json", "expert_report.json", "physics_trace.json", "physics_telemetry.npz",
                    "recording_samples.json", "recording_report.json", "actual36_geometry_audit.json",
                    "data/episode_0000000.hdf5", "instruction/episode_0000000.json", "_traj_data/episode0.pkl",
                    "video/episode_0000000.mp4"]
        for name in required: require(name in before and before[name]["bytes"] > 0, "Required proof missing: "+name)
        scene = read_json(directory/"scene_sample.json"); report = read_json(directory/"expert_report.json")
        identity = pose_receipt(scene); result["identity"] = identity
        synthetic = report.get("synthetic_fixture_only") is True
        require(not synthetic or allow_synthetic, "Synthetic fixture rejected for real dataset")
        result["synthetic_fixture_only"] = synthetic
        require(scene.get("old20_manifest_sha256") == "97b33109a740ae2d9b99155dbf228e5d2a19761b323e60fc622c8998aec0e577", "Historical old20 manifest pin differs")
        require(report.get("accepted") is True and report.get("result") == "physics_success" and report.get("hold_passed") is True, "Expert not accepted physics success")
        require(report.get("canonical_sources_unchanged") is True and report.get("official_task_methods_unmodified") is True, "Canonical source/task acceptance missing")
        require(report.get("hardware_io") is False and scene.get("hardware_io") is False, "Physical IO evidence differs")
        require(report["active_arm"] == identity["arm"], "Expert active arm differs")
        for key in ("seed", "model_id", "arm", "pose_class", "index"):
            require(report["sample_spec"][key] == scene["sample_spec"][key], "Scene/expert sample identity differs: "+key)
        require(synthetic or bool(report.get("source_hashes")), "Missing frozen source hash registry")
        for path,digest in report.get("source_hashes", {}).items():
            require(Path(path).is_file() and sha(path) == digest, "Frozen expert source missing/changed: "+path)
        geometry = read_json(directory/"actual36_geometry_audit.json")
        require(geometry.get("passed") is True and geometry.get("full_temporal_coverage") is True
                and geometry.get("complete_to_reported_endpoint") is True and geometry.get("physical_scene_unchanged") is True, "Actual36 geometry proof incomplete")
        require(geometry.get("trace_sha256") == sha(directory/"physics_trace.json"), "Geometry trace pin differs")
        trace = read_json(directory/"physics_trace.json"); samples = read_json(directory/"recording_samples.json")
        require(isinstance(trace,list) and isinstance(samples,list) and len(samples)>1, "Trace/recording list invalid")
        with np.load(directory/"physics_telemetry.npz", allow_pickle=False) as z:
            require(set(z.files) == set(PHYSICS), "Unexpected/missing telemetry columns")
            telemetry = {key: z[key].copy() for key in z.files}
        p = len(telemetry["physics_step"]); n = len(samples)-1
        require(p > 0 and p == len(trace) == int(report["physical_steps"]), "Full physical endpoint differs")
        require(p == 15*n and int(report["recorded_frames"]) == n+1, "N+1/raw physical15-step coverage differs")
        for key,shape in PHYSICS.items():
            require(telemetry[key].shape == (p,)+shape and np.isfinite(telemetry[key]).all(), "Bad telemetry shape/value: "+key)
        close(telemetry["physics_step"], np.arange(1,p+1), "Physical contiguous steps")
        dt = float(telemetry["sim_time_after_s"][0]-telemetry["sim_time_before_s"][0])
        require(abs(dt-.004) <= 1e-8, "Physical dt is not approximately4ms")
        reported_dt=float(report.get("action_clock",{}).get("physics_dt_s",dt))
        close(dt,reported_dt,"Reported actual physics dt",atol=1e-10)
        close(telemetry["sim_time_before_s"], np.arange(p)*dt, "Physical before clock", atol=1e-10)
        close(telemetry["sim_time_after_s"], np.arange(1,p+1)*dt, "Physical after clock", atol=1e-10)
        close(telemetry["full36_qpos_after_rad"][:-1], telemetry["full36_qpos_before_rad"][1:], "Full36 temporal continuity")
        close(telemetry["measured_state24_after"][:-1], telemetry["measured_state24_before"][1:], "Measured24 temporal continuity")
        close(telemetry["hand24_actual_fd_rad_s"], (telemetry["hand24_qpos_after_rad"]-telemetry["hand24_qpos_before_rad"])/dt, "Hand actual FD")
        times = np.asarray([row["sim_time_s"] for row in samples], float)
        steps = np.asarray([row["physics_step"] for row in samples], int)
        states = np.asarray([row["measured_state24"] for row in samples], float)
        commands = np.asarray([row["command24"] for row in samples], float)
        close(steps, np.arange(n+1)*15, "Observation15-step cadence")
        close(times, np.arange(n+1)*15*dt, "Observation actual15-step clock", atol=1e-10)
        close(states[0], telemetry["measured_state24_before"][0], "Initial measured24")
        close(states[1:], telemetry["measured_state24_after"][steps[1:]-1], "Recorded24 physical after linkage")
        close(commands[1:], telemetry["applied_command24"][steps[1:]-1], "Next commands at capture linkage")
        close([row["step"] for row in trace], np.arange(1,p+1), "Historical trace sequence")
        close([row["sim_time_s"] for row in trace], telemetry["sim_time_after_s"], "Historical trace time")
        close([row["full_qpos_actual_rad"] for row in trace], telemetry["full36_qpos_after_rad"], "Historical trace actual36")
        close([row["command24_actual"] for row in trace], telemetry["applied_command24"], "Historical trace actual24 command")
        hold = [row for row in trace if row.get("phase", {}).get("terminal_hold") is True]
        require(len(hold)*dt >= 1-1e-8, "Less than1s terminal hold")
        for row in trace:
            require(int(row["bottle"]["model_id"]) == identity["model_id"] and int(row["bottle"]["qpose_tag"]) == (0 if identity["arm"] == "left" else 1), "Historical bottle identity differs")
            phase = row["phase"]["name"]
            if phase == "approach": require(row["fp_delta_m"] <= .001+1e-8 and row["rotation_delta_deg"] <= 1+1e-8 and not row["grip"]["contacts"], "Approach pushed/touched bottle")
            if phase in ("close", "close_hold"): require(row["fp_delta_m"] <= .002+1e-8 and row["rotation_delta_deg"] <= 2+1e-8, "Closure pushed bottle")
            require(not any(contact.get("active",False) for contact in row.get("actual_self_contacts",[])), "Historical native self collision")
        for row in hold:
            grip = row["grip"]
            require(row["bottle"]["official_success"] is True and row["bottle_bottom_gap_m"] >= .01, "Terminal hold not airborne success")
            require(grip["table_normal_N"] < 1e-6 and grip["thumb_normal_N"] > .05 and grip["fingers_normal_N"] > .05 and grip["opposing_normals_dot"] < -.3, "Terminal hold not real opposing hand support")
            require(row["relative_translation_drift_m"] < .002 and row["relative_rotation_drift_deg"] < 2, "Terminal hold slipped")
        final = report["after_hold"]
        require(final.get("official_success") is True, "Final official success false")
        final_matrix = np.asarray(final["pose"],float)
        require(final_matrix.shape == (4,4) and final_matrix[2,1] >= math.cos(math.radians(5)), "Final bottle not upright within5deg")
        recording = read_json(directory/"recording_report.json")
        require(recording.get("accepted") is True and recording.get("rgb_only") is True and recording.get("new_physics_episode") is True, "Recording receipt rejected")
        require(int(recording["native_state_action_pairs"]) == n and int(recording["raw_observations"]) == n+1, "Recording receipt counts differ")
        require(recording["hdf5_sha256"] == sha(directory/"data/episode_0000000.hdf5"), "Recording HDF pin differs")
        codec, codec_file = find_codec(codec_path); codec_before = sha(codec_file)
        cache = directory/".cache/episode0"
        cache_files = sorted(cache.glob("*.pkl"), key=lambda p:int(p.stem) if p.stem.isdigit() else -1)
        require([p.name for p in cache_files] == [f"{i}.pkl" for i in range(n+1)], "Raw N+1 contiguous cache missing")
        with h5py.File(directory/"data/episode_0000000.hdf5","r") as f:
            require(text(f["data_format_version"][()]) == "v1.0" and text(f.attrs["source_format"]) == "RoboTwin", "Native HDF format differs")
            require(set(f["vision"]) == set(CAMERAS), "Exactly3 RGB cameras required")
            paths=[]; f.visititems(lambda name,obj: paths.append(name) if isinstance(obj,h5py.Dataset) else None)
            require(not [path for path in paths if any(part.lower() in FORBIDDEN for part in path.split("/"))], "Depth/pointcloud/segmentation in formal HDF")
            require(f.attrs.get("physical_success_report_sha256") == sha(directory/"expert_report.json"), "Physical report/HDF pin differs")
            require(bool(f.attrs.get("new_physics_episode",False)) and not bool(f.attrs.get("physical_calibration_available",True)), "New-episode/calibration claims differ")
            native_shapes={}
            for group in ("state","action"):
                for name,(_,width) in NATIVE.items():
                    ds=f[group+"/"+name]; actual=ds.shape
                    require(len(actual)==2 and actual[0]==n and (actual[1]==width if width is not None else actual[1] in (1,6)), "Native joint dimensions differ: "+ds.name)
                    require(np.isfinite(ds[:]).all(), "Nonfinite native state/action")
                    native_shapes[name]=actual[1]
                for name in ("left_ee_poses","right_ee_poses"):
                    values=f[group+"/"+name][:]
                    require(values.shape==(n,7) and np.isfinite(values).all(), "Endpose shape/value")
                    close(np.linalg.norm(values[:,3:],axis=1),np.ones(n),"Endpose unit quaternion",atol=2e-4)
            dex=f["realman_dexterous"]
            close(dex["measured_state24"][:],states[:-1],"HDF measured24")
            close(dex["next_command24"][:],commands[1:],"HDF next command24")
            close(dex["sample_time_s"][:],times[:-1],"HDF current clock")
            close(dex["next_sample_time_s"][:],times[1:],"HDF next clock")
            require(not bool(dex.attrs.get("sdk_mapping_calibrated",True)), "Unverified SDK calibration claimed")
            g=f["realman_dexterous_physics"]
            require(set(g)==set(PHYSICS), "HDF physics columns differ")
            for name in PHYSICS:
                actual=g[name][:]
                require(actual.dtype==telemetry[name].dtype and np.array_equal(actual,telemetry[name]), "HDF physics exact bytes/values differ: "+name)
            full_names=json.loads(text(g.attrs["full36_joint_names"])); hand_names=json.loads(text(g.attrs["hand24_joint_names"]))
            require(len(full_names)==len(set(full_names))==36 and len(hand_names)==len(set(hand_names))==24, "Physical joint names incomplete")
            require(full_names==report["full_qpos_joint_names"] and set(hand_names).issubset(full_names), "Full joint ordering differs")
            hand_ids=[full_names.index(name) for name in hand_names]
            for when in ("before","after"):
                close(telemetry[f"full36_qpos_{when}_rad"][:,hand_ids],telemetry[f"hand24_qpos_{when}_rad"],"Full36/hand24 mapping "+when)
                for arm,block,prefix in (("left",slice(0,6),"l_"),("right",slice(12,18),"r_")):
                    ids=[full_names.index(prefix+f"joint{i}") for i in range(1,7)]
                    close(telemetry[f"full36_qpos_{when}_rad"][:,ids],telemetry[f"measured_state24_{when}"][:,block],"Full36/primary24 arm "+arm+when)
            for name in CAMERAS:
                cg=f["vision/"+name]
                require(cg["colors"].shape==(n,) and cg["colors"].dtype.kind=="S", "RGB compressed column differs")
                close(cg["shape"][:],[480,640,3],"RGB dimensions")
                K,C=cg["intrinsic_matrix"][:],cg["extrinsics_matrix"][:]
                require(K.shape==(n,3,3) and C.shape==(n,4,4) and np.isfinite(K).all() and np.isfinite(C).all(),"Camera matrix dimensions/values")
                require(np.all(K[:,0,0]>0) and np.all(K[:,1,1]>0),"Camera focal lengths invalid")
                close(K[:,2,:],np.tile([0,0,1],(n,1)),"Camera K bottom row",atol=1e-6)
                close(C[:,3,:],np.tile([0,0,0,1],(n,1)),"Camera C bottom row",atol=2e-5)
                close(np.swapaxes(C[:,:3,:3],1,2)@C[:,:3,:3],np.tile(np.eye(3),(n,1,1)),"Camera rigid rotation",atol=2e-5)
                close(np.linalg.det(C[:,:3,:3]),np.ones(n),"Camera proper rotation",atol=2e-5)
            rgb_streams={name:hashlib.sha256() for name in CAMERAS}
            for index,path in enumerate(cache_files):
                raw=numeric_pickle(path)
                require(isinstance(raw,dict),"Raw observation not dictionary")
                clock=raw["recording_clock"]
                close(clock["physics_step"],steps[index],"Raw physics clock")
                close(clock["sim_time_s"],times[index],"Raw sim time")
                close(raw["dexterous_state"]["state24"],states[index],"Raw measured24")
                close(raw["dexterous_state"]["command24"],commands[index],"Raw command24")
                require(set(CAMERAS.values()).issubset(raw["observation"]),"Raw cameras missing")
                for name,raw_name in CAMERAS.items():
                    camera=raw["observation"][raw_name]
                    require(not any(key.lower() in FORBIDDEN for key in camera),"Raw depth/segmentation retained")
                    image=np.asarray(camera["rgb"])
                    require(image.dtype==np.uint8 and image.shape==(480,640,3),"Raw RGB shape/order invalid")
                    rgb_streams[name].update(image.tobytes())
                    if index<n:
                        stored=bytes(f["vision/"+name+"/colors"][index]).rstrip(b"\0")
                        require(codec._has_rgb_marker(stored),"Stored JPEG not standard RGB marker")
                        require(stored==codec.encode_image_bit(image),"RGB image does not belong to same raw capture: "+name+"/"+str(index))
                        decoded=codec.decode_image_bit(stored)
                        require(decoded.shape==image.shape and decoded.dtype==np.uint8,"RGB decode dimensions")
                        close(f["vision/"+name+"/intrinsic_matrix"][index],camera.get("intrinsic_cv",camera.get("intrinsic_matrix")),"Raw/HDF intrinsics",atol=2e-5)
                        C=camera.get("cam2world_gl",camera.get("extrinsic_cv",camera.get("extrinsics_matrix")))
                        C=np.asarray(C)
                        if C.shape==(3,4): C=np.vstack([C,[0,0,0,1]])
                        close(f["vision/"+name+"/extrinsics_matrix"][index],C,"Raw/HDF extrinsics",atol=2e-5)
                for group,row in (("state",index),("action",index-1)):
                    if not 0<=row<n:continue
                    for name,(raw_name,_) in NATIVE.items():
                        close(f[group+"/"+name][row],np.asarray(raw["joint_action"][raw_name]).reshape(-1),"Raw/native next-pair "+group+"/"+name,atol=3e-5)
                    for name,raw_name in (("left_ee_poses","left_endpose"),("right_ee_poses","right_endpose")):
                        close(f[group+"/"+name][row],raw["endpose"][raw_name],"Raw/native endpose",atol=3e-5)
        require(sha(codec_file)==codec_before,"Native codec changed during readback")
        # Merely parse safe numeric trajectory; its native schema differs from dense telemetry.
        trajectory=numeric_pickle(directory/"_traj_data/episode0.pkl")
        require(isinstance(trajectory,dict),"Native trajectory object invalid")
        instructions=read_json(directory/"instruction/episode_0000000.json")
        require(isinstance(instructions,dict) and bool(instructions.get("instructions",instructions.get("seen",[]))),"Instruction missing")
        require(inventory(directory)==before,"Trial source changed during readback")
        result.update(passed=True,source_files=before,source_hdf5_sha256=before["data/episode_0000000.hdf5"]["sha256"],
            physical_steps=p,state_action_pairs=n,raw_observations=n+1,physical_dt_s=dt,observation_dt_s=15*dt,
            hold_samples=len(hold),minimum_hold_gap_m=min(row["bottle_bottom_gap_m"] for row in hold),
            native_joint_widths=native_shapes,rgb_raw_stream_sha256={k:v.hexdigest() for k,v in rgb_streams.items()},
            codec_sha256=codec_before,codec_path=str(codec_file),
            verified_alignment="Every3-camera stored JPEG reencoded from same raw capture; raw clock/state/command equals physical NPZ; all24/hand24/full36 records checked", 
            limitations=["Static historical readback only; does not rerun physics", "Force/SDK/TCP hardware calibration remains unavailable",
                         "Video is preview only; authoritative timeline is raw clock and HDF/NPZ timestamps",
                         "No swept-volume guarantee beyond stored actual36 geometry audit"])
    except Exception as error:
        result.update(error=str(error),error_type=type(error).__name__,traceback=traceback.format_exc())
    return result


def accepted_paths(*, accepted_list=None, trials_root=None):
    require(bool(accepted_list) != bool(trials_root), "Choose one accepted-list or trials-root")
    if accepted_list:
        value=read_json(accepted_list)
        rows=value if isinstance(value,list) else value.get("trials",value.get("accepted_trials",[]))
        paths=[Path(row if isinstance(row,str) else row["trial_dir"]).resolve() for row in rows]
    else:
        paths=[]
        for p in sorted(Path(trials_root).resolve().rglob("expert_report.json")):
            report=read_json(p)
            if report.get("synthetic_fixture_only") is True: continue
            # Published proof copies must not be counted as new physical trials.
            published=any((parent/"manifest.json").is_file() and read_json(parent/"manifest.json").get("complete") is True for parent in p.parents if parent.is_relative_to(Path(trials_root).resolve()))
            if report.get("accepted") is True and not published: paths.append(p.parent.resolve())
    require(len(paths)==len(set(paths)),"Duplicate accepted trial directory")
    return paths


def audit_dataset(paths, *, codec_path=None, allow_synthetic=False):
    episodes=[audit_trial(path,codec_path=codec_path,allow_synthetic=allow_synthetic) for path in paths]
    passed=[row for row in episodes if row["passed"]]
    counts={key:dict(Counter(str(row["identity"][key]) for row in passed)) for key in ("arm","model_id","pose_class")}
    targets={"arm":{"left":25,"right":25},"model_id":{"13":20,"16":20,"4":10},"pose_class":{"upright":20,"horizontal":15,"diagonal":15}}
    seeds=[row["identity"]["seed"] for row in passed]; indices=[row["identity"]["spec_index"] for row in passed]
    unique=len(set(seeds))==len(seeds) and len(set(indices))==len(indices)
    complete=len(episodes)==len(passed)==50 and counts==targets and unique and set(indices)==set(range(50))
    return {"schema_version":1,"complete":complete,"status":"complete_independently_validated50" if complete else "partial_or_rejected_not50complete",
            "new_physics_runs":0,"readback_only":True,"hardware_io":False,"input_count":len(episodes),"accepted_count":len(passed),
            "failed_count":len(episodes)-len(passed),"actual_counts":counts,"target_counts":targets,"unique_seeds_and_specs":unique,
            "episodes":episodes,"synthetic_fixture_only":any(row.get("synthetic_fixture_only",False) for row in episodes),"validator_sha256":sha(Path(__file__))}


def main():
    ap=argparse.ArgumentParser(); group=ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--trial",type=Path);group.add_argument("--accepted-list",type=Path);group.add_argument("--trials-root",type=Path)
    ap.add_argument("--codec",type=Path);ap.add_argument("--output",type=Path);ap.add_argument("--allow-synthetic-fixtures",action="store_true")
    args=ap.parse_args()
    if args.trial: result=audit_trial(args.trial,codec_path=args.codec,allow_synthetic=args.allow_synthetic_fixtures);success=result["passed"]
    else: result=audit_dataset(accepted_paths(accepted_list=args.accepted_list,trials_root=args.trials_root),codec_path=args.codec,allow_synthetic=args.allow_synthetic_fixtures);success=result["complete"]
    if args.output:
        require(not any(args.output.resolve().is_relative_to(Path(row["trial_dir"])) for row in result.get("episodes",[])),"Audit output must be outside source trials")
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k not in ("source_files","episodes","traceback")},ensure_ascii=False,indent=2))
    return 0 if success else 2

if __name__=="__main__": raise SystemExit(main())
