"""Standalone CPU export of physically timed head video from accepted raw PKL.

No scene, robot, HDF5, rerendering, physics, GPU or collector changes.
Use: python export_timed_preview_v12.py REPLAY_DIR
Outputs video/head_realtime.mp4 and head_realtime_video.json; refuses overwrite.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import pickle
import time

import numpy as np


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()


def _published(path):
    data=json.loads(path.read_text(encoding="utf-8"))
    if data.get("result")!="one_strictly_validated_episode_collected" or data.get("accepted") is not True:
        raise RuntimeError("Replay is not a published strictly accepted episode")
    return data


def _sample(path,index,dt,previous_time,previous_step):
    raw=path.read_bytes();sample=pickle.loads(raw)
    clock=sample["recording_clock"]
    t=float(clock["sim_time_s"]);raw_step=clock["physics_step"];step=int(raw_step)
    if isinstance(raw_step,bool) or float(raw_step)!=step:raise ValueError("Physics step must be an integer")
    if not np.isfinite(t) or step<0 or abs(t-step*dt)>1e-8:
        raise ValueError("Cache recording clock is invalid: "+str(path))
    if t<previous_time-1e-10 or step<previous_step:
        raise ValueError("Cache clocks must be nondecreasing: "+str(path))
    image=np.asarray(sample["observation"]["head_camera"]["rgb"])
    if image.shape!=(240,320,3) or image.dtype!=np.uint8:
        raise ValueError("Expected unmodified raw RGB uint8 head frame 320x240: "+str(path))
    return {"path":path,"index":index,"t":t,"step":step,"rgb":image.copy(),
            "sha256":hashlib.sha256(raw).hexdigest(),"size":len(raw)}


def export_head_realtime(replay_dir,*,output_video=None,sidecar=None,max_source_age_s=.06):
    started=time.monotonic();root=Path(replay_dir).resolve()
    report_path=root/"replay_report.json";report_sha=_sha(report_path);report=_published(report_path)
    dataset=report.get("dataset",{})
    dt=float(dataset.get("physical_step_s",.004))
    steps=int(report["physical_steps"]);duration=steps*dt
    if dt<=0 or steps<=0 or not math.isfinite(duration):raise ValueError("Invalid physics duration")
    if not math.isfinite(max_source_age_s) or max_source_age_s<=0:raise ValueError("Invalid maximum source age")
    paths=sorted((root/".cache/episode0").glob("*.pkl"),key=lambda p:int(p.stem))
    if not paths:raise ValueError("No accepted raw PKL observations")
    if len({int(p.stem) for p in paths})!=len(paths):raise ValueError("Duplicate cache numeric names")
    if len(paths)!=int(dataset["recorded_frames"]) or [int(p.stem) for p in paths]!=list(range(len(paths))):
        raise ValueError("Raw cache count/numeric sequence differs from published native observation count")
    video=Path(output_video).resolve() if output_video else root/"video/head_realtime.mp4"
    metadata=Path(sidecar).resolve() if sidecar else root/"head_realtime_video.json"
    if video.exists() or metadata.exists():raise FileExistsError("Refusing to overwrite realtime output")
    temporary=video.with_name(video.stem+".partial"+video.suffix)
    if temporary.exists():raise FileExistsError("Stale partial video exists: "+str(temporary))
    if not video.parent.is_dir():raise ValueError("Published video directory is missing")
    fps=30.;count=int(math.ceil(duration*fps-1e-10))
    times=np.arange(count,dtype=float)/fps  # absolute replay time, origin 0
    manifest=[];last_time=-math.inf;last_step=-1;next_index=0;duplicates=0
    def load_next():
        nonlocal last_time,last_step,next_index,duplicates
        if next_index>=len(paths):return None
        sample=_sample(paths[next_index],next_index,dt,last_time,last_step)
        if sample["t"]==last_time:duplicates+=1
        last_time,last_step=sample["t"],sample["step"];next_index+=1
        manifest.append({"cache_index":sample["index"],"cache_numeric_name":int(sample["path"].stem),
                         "path":str(sample["path"]),"sha256":sample["sha256"],"size_bytes":sample["size"],
                         "physics_step":sample["step"],"sim_time_s":sample["t"]})
        return sample
    current=load_next()
    if abs(current["t"])>1e-8:raise ValueError("Initial raw observation must cover absolute replay time 0")
    following=load_next()
    import cv2
    writer=cv2.VideoWriter(str(temporary),cv2.VideoWriter_fourcc(*"mp4v"),fps,(320,240))
    if not writer.isOpened():raise RuntimeError("CPU VideoWriter could not open")
    indices=[];source_steps=[];source_times=[];ages=[]
    try:
        for t in times:
            while following is not None and following["t"]<=t+1e-10:
                current=following;following=load_next()
            age=float(t-current["t"])
            if age < -1e-8 or age>max_source_age_s+1e-8:
                raise ValueError(f"No current raw sample within {max_source_age_s}s at output time {t}; age={age}")
            writer.write(cv2.cvtColor(current["rgb"],cv2.COLOR_RGB2BGR))
            indices.append(current["index"]);source_steps.append(current["step"])
            source_times.append(current["t"]);ages.append(max(0.,age))
        # Validate all final/duplicate endpoint observations without retaining images.
        while following is not None:
            current=following;following=load_next()
    finally:writer.release()
    if last_time>duration+1e-8:raise ValueError("Raw observation extends beyond completed physics duration")
    if duration-last_time>max_source_age_s+1e-8:
        raise ValueError("Final raw observation is too stale for completed physics endpoint")
    if _sha(report_path)!=report_sha or _published(report_path).get("physical_steps")!=steps:
        raise RuntimeError("Published replay report changed during export")
    cap=cv2.VideoCapture(str(temporary));decoded=0;actual_fps=float(cap.get(cv2.CAP_PROP_FPS))
    actual_width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));actual_height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    while True:
        ok,frame=cap.read()
        if not ok:break
        if frame.shape!=(240,320,3):raise RuntimeError("Decoded head frame resolution changed")
        decoded+=1
    cap.release()
    if decoded!=count or abs(actual_fps-fps)>1e-6 or (actual_width,actual_height)!=(320,240):
        raise RuntimeError("Encoded video does not match output frame count/fps/resolution")
    if len(manifest)!=len(paths):raise RuntimeError("Not all raw cache clocks validated")
    if any(_sha(Path(x["path"]))!=x["sha256"] for x in manifest):
        raise RuntimeError("An original raw observation changed during export")
    manifest_bytes=json.dumps(manifest,sort_keys=True,separators=(",",":")).encode()
    result={
        "result":"accepted_raw_head_realtime_exported","video":str(video),
        "source":"Recorded raw head RGB, unchanged physics; no rerendering or synthetic frames",
        "method":"Zero-order hold of last available observation, duplicate clocks choose last cache index",
        "encoded_fps":fps,"encoded_resolution":[320,240],"decoded_frames":decoded,
        "physical_steps":steps,"physical_step_s":dt,"simulation_duration_s":duration,
        "encoded_duration_s":count/fps,"simulation_time_origin_s":0.,
        "first_source_observation_time_s":manifest[0]["sim_time_s"],
        "last_source_observation_time_s":last_time,"final_endpoint_source_hold_s":duration-last_time,
        "max_source_age_s":max(ages),"max_allowed_source_age_s":max_source_age_s,
        "duplicate_source_clock_count":duplicates,"source_observation_count":len(manifest),
        "frame_times_s":times.tolist(),"source_cache_index":indices,"source_physics_steps":source_steps,
        "source_observation_times_s":source_times,"source_age_s":ages,
        "source_cache_manifest":manifest,"source_cache_manifest_sha256":hashlib.sha256(manifest_bytes).hexdigest(),
        "replay_report_path":str(report_path),"replay_report_sha256":report_sha,
        "exporter_source_sha256":_sha(Path(__file__)),"elapsed_s":time.monotonic()-started,
        "native_export_video_retained":True,
        "timing_scope":"Encoded frame time is continuous output time; source observation clock may repeat by zero-order hold",
    }
    result["video_sha256"]=_sha(temporary)
    temporary.replace(video)
    metadata.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay_dir");parser.add_argument("--output-video");parser.add_argument("--sidecar")
    parser.add_argument("--max-source-age-s",type=float,default=.06)
    args=parser.parse_args()
    r=export_head_realtime(args.replay_dir,output_video=args.output_video,sidecar=args.sidecar,max_source_age_s=args.max_source_age_s)
    print(json.dumps({k:r[k] for k in ("result","video","decoded_frames","simulation_duration_s","encoded_duration_s","max_source_age_s","video_sha256")},indent=2))

