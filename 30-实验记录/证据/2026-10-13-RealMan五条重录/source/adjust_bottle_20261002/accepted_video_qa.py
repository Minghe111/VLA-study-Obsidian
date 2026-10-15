"""Pure CPU decoded-video QA of a published accepted replay.

python accepted_video_qa.py REPLAY_DIR
Reads completed original videos/clock JSON/physics trace; writes only new QA
PNG/CSV/JSON. No HDF/cache audit, scene, physics, rerendering, GPU or robot IO.
Video appearance cannot certify contact/collision or object manipulation.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _iter_json_array(path,chunk_size=1024*1024):
    """Bounded-memory read of the multi-GB ordered physics JSON array."""
    decoder=json.JSONDecoder()
    with Path(path).open("r",encoding="utf-8") as stream:
        buffer="";eof=False
        def fill():
            nonlocal buffer,eof
            chunk=stream.read(chunk_size)
            if chunk:buffer+=chunk
            else:eof=True
        fill();buffer=buffer.lstrip()
        if not buffer.startswith("["):raise ValueError("Expected top-level trace array")
        buffer=buffer[1:];first=True
        while True:
            buffer=buffer.lstrip()
            while not buffer and not eof:fill();buffer=buffer.lstrip()
            if buffer.startswith("]"):
                buffer=buffer[1:]
                while not eof:fill()
                if buffer.strip():raise ValueError("Trailing data after trace array")
                return
            if not first:
                if not buffer.startswith(","):raise ValueError("Missing trace item separator")
                buffer=buffer[1:].lstrip()
            while True:
                buffer=buffer.lstrip()
                try:item,end=decoder.raw_decode(buffer);break
                except json.JSONDecodeError:
                    if eof:raise ValueError("Incomplete or invalid trace JSON")
                    fill()
            buffer=buffer[end:];first=False
            yield item


def _compact_trace(path):
    trace=[]
    for r in _iter_json_array(path):
        grip=r.get("grip",{});bottle=r["bottle"]
        # Explicitly omit joint/controller histories and large physical snapshots.
        compact_grip={k:v for k,v in grip.items() if k=="contacts" or k=="relative_bottle_palm"
                      or "drift" in k or "slip" in k}
        shells=r.get("actual_self_contacts",[])
        if any("active" not in x for x in shells):raise ValueError("Self-contact records lack actual active flags")
        active=[x for x in shells if x["active"] is True]
        trace.append({"step":r["step"],"sim_time_s":r["sim_time_s"],"phase":r["phase"],
            "bottle":{k:bottle.get(k) for k in ("pose","functional_point0","upright_angle_deg","official_success")},
            "grip":compact_grip,"actual_self_contacts":active,"inactive_self_contact_shells":len(shells)-len(active),
            "fp_delta_m":r.get("fp_delta_m"),"rotation_delta_deg":r.get("rotation_delta_deg")})
    return trace


def _normal_forces(record):
    contacts=record.get("grip",{}).get("contacts",record.get("contact_forces",[]))
    return [float(sum(c.get("normal_force_N",0.)
                     for c in contacts if token in c.get("link","")))
            for token in ("middle","index","thumb")]


def _rotation_angle_deg(first,second):
    r=np.asarray(first)[:3,:3]@np.asarray(second)[:3,:3].T
    u,_,vt=np.linalg.svd(r)
    return float(np.degrees(np.arccos(np.clip((np.trace(u@vt)-1)/2,-1,1))))


def _key_records(trace,report):
    events=[]
    def add(label,records,index=-1):
        if not records:raise ValueError("Missing accepted key event: "+label)
        events.append((label,records[index]))
    add("start",trace,0)
    add("approach_end",[r for r in trace if r["phase"]["name"]=="approach"])
    pairs=[r for r in trace if r["phase"]["name"]=="close" and min(_normal_forces(r))>.05]
    add("first_main3_contact",pairs,0)
    add("closed_hold_end",[r for r in trace if r["phase"]["name"]=="close_hold"])
    after=[r for r in trace if r["phase"]["name"]=="after_grasp"]
    dense=[x["action_dense_call"] for x in report.get("replay_wrench_actions",[])]
    if len(dense)<2:raise ValueError("Published report lacks exact lift/turn action dense-call provenance")
    lift=[r for r in after if r["phase"].get("dense_call")==dense[0]] if dense else []
    turn=[r for r in after if len(dense)>1 and r["phase"].get("dense_call")==dense[1]]
    add("lift_middle",lift,len(lift)//2 if lift else 0)
    add("lift_end",lift)
    add("turn_middle",turn,len(turn)//2 if turn else 0)
    add("turn_end",turn)
    add("final_hold_end",trace)
    return events


def _metadata(root,report,kind,explicit=None):
    if explicit:
        path=Path(explicit);return _load(path),path
    if kind=="head":
        path=root/"head_realtime_video.json"
        if path.exists():return _load(path),path
    else:
        for name in ("external_realtime_video.json","external_realtime_diagnostic.json"):
            path=root/name
            if path.exists():return _load(path),path
        for candidate in (report.get("external_realtime_video"),report.get("dataset",{}).get("external_realtime_video")):
            if isinstance(candidate,dict):return candidate,root/"replay_report.json"
    raise ValueError(f"Missing exact {kind} realtime clock metadata; do not infer frame clocks")


def _video(root,report,meta,kind,explicit=None):
    choices=[explicit,meta.get("video"),meta.get("path")]
    if kind=="external":choices.append(report.get("dataset",{}).get("external_realtime_video"))
    choices.extend([root/"video"/f"{kind}_realtime.mp4",root/"external_realtime_diagnostic.mp4"] if kind=="external"
                   else [root/"video/head_realtime.mp4"])
    for candidate in choices:
        if isinstance(candidate,(str,Path)):
            path=Path(candidate)
            if not path.is_absolute():path=root/path
            if path.is_file():return path.resolve()
    raise ValueError(f"Completed {kind} realtime video is missing")


def _clock(meta,kind,expected_fps):
    raw=meta.get("frame_times_s",meta.get("output_frame_times_s"))
    if raw is None:raise ValueError(f"{kind} sidecar lacks absolute output frame times")
    t=np.asarray(raw,dtype=float)
    source=meta.get("source_observation_times_s",meta.get("source_observationtime_s",raw))
    source=np.asarray(source,dtype=float)
    steps=np.asarray(meta.get("source_physics_steps",meta.get("frame_physics_steps",[])))
    if (t.ndim!=1 or not len(t) or source.shape!=t.shape or steps.shape!=t.shape
            or not np.isfinite(t).all() or not np.isfinite(source).all()
            or not np.issubdtype(steps.dtype,np.integer)):
        raise ValueError(f"{kind} clock arrays have invalid shapes/values")
    if np.any(np.diff(t)<=0) or np.any(np.diff(source)<-1e-10) or np.any(np.diff(steps)<0):
        raise ValueError(f"{kind} output/source clocks must advance monotonically")
    if abs(t[0])>1e-8 or np.any(steps<0):raise ValueError("Realtime mapping must start at absolute t0 with nonnegative physics steps")
    age=t-source
    if np.min(age)<-1e-8:raise ValueError("Source image comes from the future")
    if len(t)>1 and np.max(np.abs(np.diff(t)-1/expected_fps))>1e-7:
        raise ValueError(f"{kind} clock is not {expected_fps} fps realtime cadence")
    if kind=="head" and np.max(age)>.06+1e-8:
        raise ValueError("Head source observation exceeds allowed .06s zero-order hold")
    return t,source,steps.astype(int),np.maximum(age,0)


def _select_keyframes(events,source_steps):
    selected=[]
    for label,event in events:
        eligible=np.flatnonzero(source_steps>=event["step"] if label=="first_main3_contact"
                                else source_steps<=event["step"])
        if not len(eligible):raise ValueError("No causally matching image for key event: "+label)
        selected.append(int(eligible[0] if label=="first_main3_contact" else eligible[-1]))
    return selected


def qa_accepted_videos(replay_dir,*,output_dir=None,external_video=None,head_video=None,
                       external_clock=None,head_clock=None):
    started=time.monotonic();root=Path(replay_dir).resolve()
    report_file=root/"replay_report.json";report=_load(report_file)
    if report.get("result")!="one_strictly_validated_episode_collected" or report.get("accepted") is not True:
        raise RuntimeError("Only a completed published accepted replay can be checked")
    trace_file=root/"physics_trace.json";trace=_compact_trace(trace_file)
    if not trace or trace[0]["step"]!=1 or any(trace[i]["step"]+1!=trace[i+1]["step"] for i in range(len(trace)-1)):
        raise ValueError("Trace must contain every ordered physical step")
    dt=float(report.get("dataset",{}).get("physical_step_s",.004))
    duration=int(report["physical_steps"])*dt
    if trace[-1]["step"]!=report["physical_steps"]:
        raise ValueError("Trace is incomplete relative to the published report")
    trace_times=np.asarray([r.get("sim_time_s",r["step"]*dt) for r in trace])
    events=_key_records(trace,report)
    output=Path(output_dir).resolve() if output_dir else root/"video_qa_20261003_v2"
    if output.exists():raise FileExistsError("Refusing to overwrite existing QA output")
    configs=[]
    for kind,fps,explicit_video,explicit_clock in (("external",25.,external_video,external_clock),
                                                ("head",30.,head_video,head_clock)):
        meta,meta_file=_metadata(root,report,kind,explicit_clock)
        video=_video(root,report,meta,kind,explicit_video)
        clock,source_times,source_steps,ages=_clock(meta,kind,fps)
        if np.any(np.abs(source_times-source_steps*dt)>1e-8):
            raise ValueError(f"{kind} source clock/physics steps disagree")
        if clock[-1]>duration+1e-8 or duration-clock[-1]>1/fps+1e-8:
            raise ValueError(f"{kind} realtime video does not cover the accepted physics duration")
        expected=meta.get("video_sha256",meta.get("sha256"))
        if kind=="external" and expected is None:
            expected=report.get("dataset",{}).get("external_realtime_video_sha256")
        actual=_sha(video)
        if isinstance(expected,str) and len(expected)==64 and expected!=actual:
            raise RuntimeError(f"{kind} original video hash differs from metadata")
        selections=_select_keyframes(events,source_steps)
        configs.append((kind,fps,meta,meta_file,video,actual,clock,source_times,source_steps,ages,selections))
    output.mkdir()
    import cv2
    analyses=[];rows=[];source_hashes={str(report_file):_sha(report_file),str(trace_file):_sha(trace_file)}
    carry_reference_record=dict(events)["closed_hold_end"]
    carry_reference=np.asarray(carry_reference_record["grip"]["relative_bottle_palm"])
    for kind,fps,meta,meta_file,video,video_sha,clock,source_times,source_steps,ages,selections in configs:
        source_hashes[str(meta_file)]=_sha(meta_file);source_hashes[str(video)]=video_sha
        cap=cv2.VideoCapture(str(video))
        if not cap.isOpened():raise RuntimeError("CPU video decoder could not open "+str(video))
        actual_fps=float(cap.get(cv2.CAP_PROP_FPS));header_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if abs(actual_fps-fps)>1e-5:raise ValueError(f"{kind} encoded fps mismatch")
        expected_size=(320,240) if kind=="head" else (int(meta["width"]),int(meta["height"]))
        if (width,height)!=expected_size:raise ValueError(f"{kind} actual video resolution differs from accepted source")
        selected_images={};decoded=0
        while True:
            ok,frame=cap.read()
            if not ok:break
            if frame.shape!=(height,width,3):raise RuntimeError("Decoded video resolution changed")
            if decoded in selections:selected_images[decoded]=frame.copy()
            decoded+=1
        cap.release()
        if decoded!=len(clock) or header_count!=decoded:
            raise RuntimeError(f"{kind} full decode count disagrees with exact frame clocks/header")
        cards=[]
        for (label,event),frame_index in zip(events,selections):
            frame=selected_images[frame_index]
            step=int(source_steps[frame_index]);nearest=int(np.argmin(np.abs(np.array([r["step"] for r in trace])-step)))
            actual=trace[nearest];contacts=actual.get("grip",{}).get("contacts",actual.get("contact_forces",[]))
            if label=="first_main3_contact" and min(_normal_forces(actual))<=.05:
                raise ValueError("Selected image source does not yet satisfy actual three-contact event")
            png=output/f"{kind}_{len(cards)+1:02}_{label}.png"
            if not cv2.imwrite(str(png),frame):raise RuntimeError("Could not write exact decoded keyframe PNG")
            item={"camera":kind,"event":label,"frame_index":frame_index,
                  "output_time_s":float(clock[frame_index]),"source_observation_time_s":float(source_times[frame_index]),
                  "source_physics_step":step,"matched_trace_step":actual["step"],
                  "trace_source_alignment_error_s":abs(actual.get("sim_time_s",actual["step"]*dt)-float(source_times[frame_index])),
                  "requested_event_step":event["step"],"event_time_error_s":abs(float(clock[frame_index])-float(event.get("sim_time_s",event["step"]*dt))),
                  "source_age_s":float(ages[frame_index]),"phase":actual["phase"],
                  "main3_normal_N":_normal_forces(actual),"upright_angle_deg":actual["bottle"].get("upright_angle_deg"),
                  "official_success":actual["bottle"].get("official_success"),
                  "actual_self_contacts":actual.get("actual_self_contacts",[]),"contact_points":contacts,
                  "inactive_self_contact_shells":actual.get("inactive_self_contact_shells",0),
                  "bottle_pose_world":actual["bottle"]["pose"],
                  "bottle_functional_point0":actual["bottle"]["functional_point0"],
                  "grip_measured":actual.get("grip",{}),
                  "object_delta_from_initial_m":actual.get("fp_delta_m"),
                  "object_rotation_delta_from_initial_deg":actual.get("rotation_delta_deg"),
                  "relative_bottle_palm":actual.get("grip",{}).get("relative_bottle_palm"),"decoded_png":str(png)}
            relative=np.asarray(actual["grip"]["relative_bottle_palm"])
            item["relative_drift_reference_trace_step"]=carry_reference_record["step"]
            item["relative_translation_drift_from_pre_carry_m"]=float(np.linalg.norm(relative[:3,3]-carry_reference[:3,3])) if actual["phase"]["name"]=="after_grasp" else None
            item["relative_rotation_drift_from_pre_carry_deg"]=_rotation_angle_deg(relative,carry_reference) if actual["phase"]["name"]=="after_grasp" else None
            rows.append(item)
            # Contact sheet thumbnails are separate; the extracted PNG above is native resolution.
            thumb=cv2.resize(frame,(480,360),interpolation=cv2.INTER_AREA)
            card=np.zeros((405,480,3),dtype=np.uint8);card[:360]=thumb
            cv2.putText(card,f"{len(cards)+1} {label} t={clock[frame_index]:.3f}s",(8,378),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
            cv2.putText(card,f"Fn m/i/t="+"/".join(f"{x:.3f}" for x in item["main3_normal_N"]),(8,398),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
            cards.append(card)
        sheet=np.vstack([np.hstack(cards[j:j+3]) for j in range(0,9,3)])
        sheet_file=output/f"{kind}_key9_contact_sheet.png"
        if not cv2.imwrite(str(sheet_file),sheet):raise RuntimeError("Could not write contact sheet")
        analyses.append({"camera":kind,"original_video":str(video),"original_video_sha256":video_sha,
             "decoded_frames":decoded,"encoded_fps":actual_fps,"actual_resolution":[width,height],
             "encoded_duration_s":decoded/actual_fps,"simulation_duration_s":duration,
             "max_source_age_s":float(ages.max()),"key9_contact_sheet":str(sheet_file),
             "all_frames_decoded":True,"exact_clock_mapping_verified":True})
    if any(_sha(Path(p))!=value for p,value in source_hashes.items()):
        raise RuntimeError("An original video/report/trace/clock changed during QA")
    raw_positions=np.array([r["bottle"]["functional_point0"][:3] for r in trace])
    moves=np.linalg.norm(np.diff(raw_positions,axis=0),axis=1)
    rotations=np.array([_rotation_angle_deg(trace[i+1]["bottle"]["pose"],trace[i]["bottle"]["pose"]) for i in range(len(trace)-1)])
    summary={"result":"completed_decoded_video_timing_and_contact_QA","source_sha256":source_hashes,
             "qa_source_sha256":_sha(Path(__file__)),"published_accepted_replay":True,
             "physical_steps":report["physical_steps"],"physical_step_s":dt,"simulation_duration_s":duration,
             "videos":analyses,"key9_contacts":rows,
             "trace_max_per_physics_step_functional_point_motion_m":float(moves.max()),
             "trace_max_per_physics_step_body_rotation_deg":float(rotations.max()),
             "trace_max_motion_step":trace[int(np.argmax(moves))+1]["step"],
             "trace_max_rotation_step":trace[int(np.argmax(rotations))+1]["step"],
             "trace_actual_active_self_contact_steps":sum(bool(r.get("actual_self_contacts")) for r in trace),
             "self_contact_scope":"Only actual active=True normal-impulse contacts; inactive PhysX contact shells are counted separately and not collisions",
             "accepted_trace_slip_audit":{k:report.get("actual36_geometry_audit",{}).get(k) for k in (
                 "passed","full_temporal_coverage","recorded_samples",
                 "max_post_grasp_relative_translation_drift_m","max_post_grasp_relative_rotation_drift_deg")},
             "scope":"Full video decode/timing and 9 keyframe images with matching real object pose and grip records; collision/slip acceptance comes from actual physics and full36 mesh audit, not visual inference",
             "elapsed_s":time.monotonic()-started}
    (output/"video_qa_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    with (output/"key9_contacts.csv").open("w",newline="",encoding="utf-8-sig") as stream:
        writer=csv.writer(stream);writer.writerow(["camera","event","frame","output_time_s","source_time_s","source_step","trace_step","phase","middle_N","index_N","thumb_N","upright_deg","self_contacts","decoded_png"])
        for r in rows:writer.writerow([r["camera"],r["event"],r["frame_index"],r["output_time_s"],r["source_observation_time_s"],r["source_physics_step"],r["matched_trace_step"],r["phase"]["name"],*r["main3_normal_N"],r["upright_angle_deg"],bool(r["actual_self_contacts"]),r["decoded_png"]])
    return summary


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("replay_dir");p.add_argument("--output-dir")
    p.add_argument("--external-video");p.add_argument("--head-video");p.add_argument("--external-clock");p.add_argument("--head-clock")
    a=p.parse_args();r=qa_accepted_videos(a.replay_dir,output_dir=a.output_dir,external_video=a.external_video,head_video=a.head_video,external_clock=a.external_clock,head_clock=a.head_clock)
    print(json.dumps({k:r[k] for k in ("result","physical_steps","simulation_duration_s","elapsed_s")},indent=2))

