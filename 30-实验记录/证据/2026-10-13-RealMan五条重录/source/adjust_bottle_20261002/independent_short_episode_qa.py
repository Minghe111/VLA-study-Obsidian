"""Pure CPU summary of recorded five-finger episode timing and physical evidence.

Reads immutable candidate/replay reports and a streamed physics_trace JSON array.
Never imports SAPIEN/FCL, steps physics, changes source data, or uses robot IO.
Independent temporal/contact/recorded-state checks do not replace native full36
geometry, inertia-wrench, dataset or hardware calibration acceptance.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np

TOKENS=("little","ring","middle","index","thumb")
EXPECTED_RESULTS=("physics_success","one_strictly_validated_episode_collected")

def sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):digest.update(chunk)
    return digest.hexdigest()

def iter_trace(path,chunk_size=65536):
    """Bounded memory; retain at most one JSON record plus an input chunk."""
    decoder=json.JSONDecoder()
    with Path(path).open("r",encoding="utf-8",newline="") as stream:
        buffer="";eof=False
        def fill():
            nonlocal buffer,eof
            value=stream.read(chunk_size)
            if value:buffer+=value
            else:eof=True
        fill();buffer=buffer.lstrip()
        if not buffer.startswith("["):raise ValueError("Expected a top-level physics trace JSON array")
        buffer=buffer[1:];first=True
        while True:
            buffer=buffer.lstrip()
            while not buffer and not eof:fill();buffer=buffer.lstrip()
            if buffer.startswith("]"):
                buffer=buffer[1:]
                while not eof:fill()
                if buffer.strip():raise ValueError("Trailing data after trace JSON array")
                return
            if not first:
                if not buffer.startswith(","):raise ValueError("Missing trace separator")
                buffer=buffer[1:].lstrip()
            while True:
                buffer=buffer.lstrip()
                try:record,end=decoder.raw_decode(buffer);break
                except json.JSONDecodeError:
                    if eof:raise ValueError("Incomplete or invalid trace JSON")
                    fill()
            if not isinstance(record,dict):raise ValueError("Trace item must be an object")
            buffer=buffer[end:];first=False
            yield record

def rotation_difference_deg(reference,current):
    return float(np.degrees(np.arccos(np.clip((np.trace(reference.T@current)-1.)/2.,-1.,1.))))

def pose(record):
    value=np.asarray(record["bottle"]["pose"],dtype=float)
    if value.shape!=(4,4) or not np.isfinite(value).all():raise ValueError("Bad recorded bottle pose")
    return value

def relative_pose(record):
    value=record.get("grip",{}).get("relative_bottle_palm")
    if value is None:return None
    value=np.asarray(value,dtype=float)
    if value.shape!=(4,4) or not np.isfinite(value).all():raise ValueError("Bad recorded palm-relative pose")
    return value

def actual_forces(record):
    contacts=record.get("grip",{}).get("contacts",record.get("contact_forces",[]))
    forces=np.zeros(5);directions=np.zeros((5,3))
    for contact in contacts:
        name=contact.get("link",contact.get("hand_link",""))
        index=next((i for i,token in enumerate(TOKENS) if token in name and name.startswith("l_")),None)
        if index is None:continue
        force=float(contact.get("normal_force_N",0.))
        if not math.isfinite(force) or force<0:raise ValueError("Invalid true normal contact force")
        forces[index]+=force
        normal=contact.get("normal_bottle_to_hand")
        if normal is not None:
            normal=np.asarray(normal,dtype=float)
            if normal.shape!=(3,) or not np.isfinite(normal).all():raise ValueError("Invalid contact normal")
            directions[index]+=normal*force
    dot=record.get("grip",{}).get("opposing_normals_dot")
    if dot is None:
        fingers=directions[:4].sum(axis=0);thumb=directions[4]
        divisor=np.linalg.norm(fingers)*np.linalg.norm(thumb)
        dot=float(np.dot(fingers,thumb)/divisor) if divisor>0 else None
    return forces,None if dot is None else float(dot)

def self_contact_counts(record):
    active=inactive=unknown=0
    for contact in record.get("actual_self_contacts",[]):
        if "active" in contact:
            value=contact["active"]
            if not isinstance(value,bool):raise ValueError("Self-contact active flag must be boolean")
        elif "normal_impulse_Ns" in contact:
            value=float(contact["normal_impulse_Ns"])>float(contact.get("normal_impulse_threshold_Ns",1e-7))
        else:
            unknown+=1;continue  # Never count an unqualified PhysX shell as active.
        if value:active+=1
        else:inactive+=1
    return active,inactive,unknown

def resolve_dt(report,explicit):
    if explicit is not None:candidates=[("explicit_cli_physics_dt_s",explicit)]
    else:
        candidates=[("report.dataset.physical_step_s",report.get("dataset",{}).get("physical_step_s")),
                    ("report.external_realtime_video.physics_dt_s",report.get("external_realtime_video",{}).get("physics_dt_s"))]
        for segment in report.get("continuous_hand_profile",[]):
            intervals=int(segment.get("end_step",0))-int(segment.get("start_step",0))
            if intervals>0:candidates.append(("report.continuous_hand_profile.duration_s/intervals",float(segment["duration_s"])/intervals))
        for segment in report.get("action_timing",[]):
            intervals=int(segment["end_step"])-int(segment["start_step"])
            if intervals>0:candidates.append(("report.action_timing.duration_s/physics_steps",float(segment["duration_s"])/intervals))
    for source,value in candidates:
        if value is not None and math.isfinite(float(value)) and float(value)>0:return float(value),source
    raise ValueError("No exact physics dt; pass --dt-s (do not use video fps)")

def analyze_episode(trace_path,report_path,manifest_path,*,dt_s=None,stable_contact_steps=50,
                    measured_start_threshold_rad=.001,command_start_threshold_sdk=.001,
                    first_lift_dense_call=None):
    started=time.monotonic()
    paths=[Path(trace_path),Path(report_path),Path(manifest_path)]
    before={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in paths}
    report=json.loads(paths[1].read_text(encoding="utf-8"))
    manifest=json.loads(paths[2].read_text(encoding="utf-8"))
    metadata=manifest["hands"]["left"];channels=tuple(metadata["channels"])
    if len(channels)!=6:raise ValueError("Expected SDK6 primary mapping")
    joint_names=report["full_qpos_joint_names"]
    if len(set(joint_names))!=len(joint_names):raise ValueError("Duplicate full-qpos names")
    joint_index={n:i for i,n in enumerate(joint_names)}
    primary_ids=[joint_index[n] for n in channels]
    hand_limits={n:lim for h in manifest["hands"].values() for n,lim in h["limits"].items()}
    hand_ids=[joint_index[n] for n in hand_limits if n in joint_index]
    if len(hand_ids)!=len(hand_limits):raise ValueError("Recorded full qpos omits vendor hand joints")
    dt,dt_source=resolve_dt(report,dt_s)
    if isinstance(stable_contact_steps,bool) or int(stable_contact_steps)!=stable_contact_steps or stable_contact_steps<=0:
        raise ValueError("Stable contact sample count must be a positive integer")
    widths=np.array([metadata["limits"][n]["upper"]-metadata["limits"][n]["lower"] for n in channels])
    lower=np.array([metadata["limits"][n]["lower"] for n in channels])
    finger=[{"channel":token,"primary_joint":channels[i],"first_measured_motion_step":None,
             "first_command_motion_step":None,"first_true_normal_contact_step":None,
             "first_capture_latch_step":None,"max_abs_joint_delta_rad":0.,"max_true_normal_N":0.}
             for i,token in enumerate(TOKENS)]
    counts={};samples=0;first_step=None;previous=None;previous_q=None;last_time=None
    full_coverage=True;max_clock_error=0.;max_recorded_fd=0.;max_recomputed_fd=0.;max_recomputed_fd_ratio=0.
    max_engine=None;max_mimic=0.;self_active_steps=0;self_active_count=0;self_inactive_count=0;self_unknown_count=0
    close_started=False;close_done=False;close_first=None;close_last=None;close_reference=None;close_ref_step=None
    close_q0=None;close_sdk0=None;close_command0=None;close_pose0=None;close_fp0=None
    max_close_fp=0.;max_close_origin=0.;max_close_rotation=0.;max_close_global_fp=0.;max_close_global_rotation=0.
    first_main3=None;main3_run_start=None;streak=0;max_streak=0
    first_after=None;first_lift=None;first_observed_lift=None;after_samples=0
    carry_reference=None;carry_ref_source=None;carry_count=0;carry_missing=0;max_carry_delta=0.;max_carry_rotation=0.
    for row in iter_trace(paths[0]):
        step=int(row["step"]);time_s=float(row.get("sim_time_s",step*dt))
        if not math.isfinite(time_s):raise ValueError("Nonfinite physics clock")
        if first_step is None:first_step=step;full_coverage=step==1
        if previous is not None:
            full_coverage=full_coverage and step==int(previous["step"])+1
            if time_s<=last_time:raise ValueError("Physics times must be strictly increasing")
        max_clock_error=max(max_clock_error,abs(time_s-step*dt))
        phase=row["phase"];name=phase["name"];dense=phase.get("dense_call")
        counts[name]=counts.get(name,0)+1;samples+=1
        q=np.asarray(row["full_qpos_actual_rad"],dtype=float)
        if q.shape!=(len(joint_names),) or not np.isfinite(q).all():raise ValueError("Recorded qpos shape/finite failure")
        measured_sdk=row.get("sdk_measured",{}).get("left")
        sdk=np.asarray(measured_sdk if measured_sdk is not None else np.clip(1000*(1-(q[primary_ids]-lower)/widths),0,1000),dtype=float)
        command=row.get("sdk_target",{}).get("left")
        if command is None and "command24_actual" in row:command=row["command24_actual"][6:12]
        if command is not None:
            command=np.asarray(command,dtype=float)
            if command.shape!=(6,) or not np.isfinite(command).all():raise ValueError("Bad recorded SDK6 command")
        if previous_q is not None:
            qd=(q-previous_q)/(time_s-last_time)
            max_recomputed_fd=max(max_recomputed_fd,float(np.max(np.abs(qd[hand_ids]))))
            max_recomputed_fd_ratio=max(max_recomputed_fd_ratio,max(abs(qd[joint_index[n]])/float(lim["velocity"]) for n,lim in hand_limits.items()))
        max_recorded_fd=max(max_recorded_fd,float(row.get("hand_actual_fd_max_rad_s",0.)))
        if "hand_engine_qvel_max_rad_s" in row:
            max_engine=max(0. if max_engine is None else max_engine,float(row["hand_engine_qvel_max_rad_s"]))
        max_mimic=max(max_mimic,float(row.get("max_mimic_residual_rad",row.get("mimic_residual_rad",0.))))
        active,inactive,unknown=self_contact_counts(row)
        self_active_steps+=int(active>0);self_active_count+=active;self_inactive_count+=inactive;self_unknown_count+=unknown
        body=pose(row);fp=np.asarray(row["bottle"]["functional_point0"][:3],dtype=float)
        f,dot=actual_forces(row)
        closing=name in ("close","close_hold") and not close_done
        if closing:
            if not close_started:
                close_started=True;close_first={"step":step,"time_s":time_s}
                close_reference=previous if previous is not None else row
                close_ref_step=int(close_reference["step"])
                close_pose0=pose(close_reference)
                close_fp0=np.asarray(close_reference["bottle"]["functional_point0"][:3],dtype=float)
                close_q0=q[primary_ids].copy();close_sdk0=sdk.copy()
                close_command0=command.copy() if command is not None else None
                for i,item in enumerate(finger):item.update({"initial_sdk_measured":float(sdk[i]),"initial_joint_rad":float(close_q0[i])})
            close_last={"step":step,"time_s":time_s,"pose":body,"relative":relative_pose(row),"force_N":f.tolist()}
            max_close_fp=max(max_close_fp,float(np.linalg.norm(fp-close_fp0)))
            max_close_origin=max(max_close_origin,float(np.linalg.norm(body[:3,3]-close_pose0[:3,3])))
            max_close_rotation=max(max_close_rotation,rotation_difference_deg(close_pose0[:3,:3],body[:3,:3]))
            max_close_global_fp=max(max_close_global_fp,float(row.get("fp_delta_m",0.)))
            max_close_global_rotation=max(max_close_global_rotation,float(row.get("rotation_delta_deg",0.)))
            for i,item in enumerate(finger):
                item.update({"final_sdk_measured":float(sdk[i]),"final_joint_rad":float(q[primary_ids[i]]),
                             "final_sdk_command":None if command is None else float(command[i]),"final_true_normal_N":float(f[i])})
                item["max_abs_joint_delta_rad"]=max(item["max_abs_joint_delta_rad"],abs(float(q[primary_ids[i]]-close_q0[i])))
                item["max_true_normal_N"]=max(item["max_true_normal_N"],float(f[i]))
                if item["first_measured_motion_step"] is None and abs(q[primary_ids[i]]-close_q0[i])>=measured_start_threshold_rad:
                    item["first_measured_motion_step"]=step;item["first_measured_motion_time_s"]=time_s
                if item["first_command_motion_step"] is None and command is not None and close_command0 is not None and abs(command[i]-close_command0[i])>=command_start_threshold_sdk:
                    item["first_command_motion_step"]=step;item["first_command_motion_time_s"]=time_s
                if item["first_true_normal_contact_step"] is None and f[i]>1e-5:item["first_true_normal_contact_step"]=step
                if item["first_capture_latch_step"] is None and row.get("capture_latched",[False]*5)[i]:item["first_capture_latch_step"]=step
            paired=bool(np.all(f[2:5]>.05) and dot is not None and dot<-.3)
            if paired:
                if streak==0:main3_run_start={"step":step,"time_s":time_s}
                streak+=1;max_streak=max(max_streak,streak)
                if streak>=stable_contact_steps and first_main3 is None:
                    first_main3={"run_start":main3_run_start.copy(),"confirmed":{"step":step,"time_s":time_s},
                                 "samples":stable_contact_steps,"main3_normal_N_at_confirmation":f[2:5].tolist(),
                                 "opposing_normals_dot_at_confirmation":dot}
            else:streak=0;main3_run_start=None
        if name=="after_grasp":
            close_done=True;after_samples+=1
            if first_after is None:first_after={"first_trace_step":step,"action_boundary_step":step-1,
                                               "first_trace_time_s":time_s,"action_boundary_time_s":time_s-dt,
                                               "dense_call":dense}
            if first_lift is None and (first_lift_dense_call is None or dense==first_lift_dense_call):
                first_lift={"first_trace_step":step,"action_boundary_step":step-1,"first_trace_time_s":time_s,
                            "action_boundary_time_s":time_s-dt,"dense_call":dense,
                            "definition":"Explicit dense call" if first_lift_dense_call is not None else "First post-grasp arm action"}
            if close_last is not None and first_observed_lift is None and body[2,3]-close_last["pose"][2,3]>=.001:
                first_observed_lift={"step":step,"time_s":time_s,"bottle_body_world_z_gain_m":float(body[2,3]-close_last["pose"][2,3]),"threshold_m":.001}
            if carry_reference is None:
                if close_last is not None and close_last["relative"] is not None:
                    carry_reference=close_last["relative"];carry_ref_source="Last actual close/close_hold trace grip.relative_bottle_palm"
                else:
                    value=report.get("pre_retreat_grip",{}).get("actual",{}).get("relative_bottle_palm")
                    if value is not None:carry_reference=np.asarray(value,dtype=float);carry_ref_source="report.pre_retreat_grip.actual.relative_bottle_palm"
            relative=relative_pose(row)
            if relative is not None and carry_reference is not None:
                carry_count+=1;max_carry_delta=max(max_carry_delta,float(np.linalg.norm(relative[:3,3]-carry_reference[:3,3])))
                max_carry_rotation=max(max_carry_rotation,rotation_difference_deg(carry_reference[:3,:3],relative[:3,:3]))
            else:carry_missing+=1
        previous=row;previous_q=q;last_time=time_s
    if not samples:raise ValueError("Empty physics trace")
    expected_steps=report.get("physical_steps")
    complete_endpoint=expected_steps is None or int(expected_steps)==int(previous["step"])
    if max_clock_error>1e-6:raise ValueError("Recorded times disagree with exact step*dt")
    for item in finger:
        item["max_abs_joint_delta_deg"]=math.degrees(item["max_abs_joint_delta_rad"])
        if close_started:
            item["actual_end_minus_start_deg"]=math.degrees(item["final_joint_rad"]-item["initial_joint_rad"])
            item["nominal_close_sdk"]=report.get("grasp_setting",{}).get("close_sdk",[None]*6)[TOKENS.index(item["channel"])]
    report_engine=report.get("dynamic_hand_audit",{}).get("max_engine_qvel_rad_s")
    if max_engine is None and report_engine is not None:max_engine=float(report_engine)
    carry_report=report.get("carry_audit")
    carry={"executed":after_samples>0,"after_grasp_samples":after_samples,"relative_reference_source":carry_ref_source,
           "trace_relative_samples":carry_count,"trace_relative_missing_samples":carry_missing,
           "trace_recomputed_max_translation_drift_m":max_carry_delta if carry_count else None,
           "trace_recomputed_max_rotation_drift_deg":max_carry_rotation if carry_count else None,
           "report_carry_audit":carry_report if after_samples else None,
           "limitation":None if carry_count==after_samples and after_samples else "Missing direct palm-relative trace; report-only carry evidence" if after_samples else "No physical carry executed"}
    unchanged=all(before[str(p)]==(p.stat().st_size,p.stat().st_mtime_ns) for p in paths)
    guards={"trace_full_physics_coverage":full_coverage,"complete_reported_endpoint":complete_endpoint,
            "source_files_stable_while_reading":unchanged,"close_no_push_global_initial_2mm_2deg":close_started and max_close_global_fp<=.002 and max_close_global_rotation<=2.,
            "five_primary_flexions_measured_moved":all(x["first_measured_motion_step"] is not None for x in finger),
            "true_nonexempt_self_contacts_zero":self_active_count==0 and self_unknown_count==0,
            "recorded_hand_actual_FD_source_ratio_le_1_05":max_recomputed_fd_ratio<=1.05,
            "recorded_mimic_residual_le_0_01rad":max_mimic<=.01,
            "main3_stable_contact_observed":first_main3 is not None}
    if carry_count:
        guards["direct_carry_slip_le_2mm_2deg"]=max_carry_delta<=.002 and max_carry_rotation<=2.
        guards["direct_carry_relative_trace_complete"]=carry_count==after_samples
    elif after_samples:
        guards["report_only_carry_slip_guard_available"]=bool(isinstance(carry_report,dict) and
             isinstance(carry_report.get("max_translation_drift_m"),(int,float)) and
             isinstance(carry_report.get("max_rotation_drift_deg"),(int,float)) and
             0<=carry_report["max_translation_drift_m"]<=.002 and 0<=carry_report["max_rotation_drift_deg"]<=2)
    guards={key:bool(value) for key,value in guards.items()}
    accepted=report.get("accepted") is True and report.get("result") in EXPECTED_RESULTS
    return {"scope":"Independent pure CPU recorded timing/normal-force/state QA; does not redo physics, mesh or dataset audits",
            "elapsed_s":time.monotonic()-started,"report_result":report.get("result"),"report_accepted":report.get("accepted"),
            "full_success_verified":bool(accepted and after_samples>0 and all(guards.values())),
            "accepted_report_is_required":True,"independent_recorded_checks":guards,
            "physics_clock":{"dt_s":dt,"source":dt_source,"samples":samples,"first_step":first_step,"last_step":int(previous["step"]),
                             "duration_s":int(previous["step"])*dt,"max_step_time_error_s":max_clock_error},
            "phase_sample_counts":counts,"motion_start_thresholds":{"measured_primary_delta_rad":measured_start_threshold_rad,
              "command_primary_sdk_delta_units":command_start_threshold_sdk,
              "interpretation":"Threshold crossing times are response metrics; one common profile does not imply equal first detectable joint response"},
            "fingers":finger,"closing":{"executed":close_started,"first_sample":close_first,
              "last_sample":None if close_last is None else {k:close_last[k] for k in ("step","time_s","force_N")},
              "duration_s":None if not close_started else (close_last["step"]-close_first["step"]+1)*dt,
              "reference_preclose_step":close_ref_step,"max_functional_point_shift_from_preclose_m":max_close_fp if close_started else None,
              "max_body_origin_shift_from_preclose_m":max_close_origin if close_started else None,
              "max_body_rotation_from_preclose_deg":max_close_rotation if close_started else None,
              "max_world_functional_point_delta_from_initial_m":max_close_global_fp if close_started else None,
              "max_world_rotation_delta_from_initial_deg":max_close_global_rotation if close_started else None,
              "report_continuous_profile":report.get("continuous_hand_profile"),
              "first_stable_main3":first_main3,"maximum_stable_main3_streak_samples":max_streak},
            "first_postgrasp_arm_action":first_after,"first_lift_action":first_lift,"first_observed_bottle_lift":first_observed_lift,
            "time_from_main3_first_valid_to_lift_action_s":None if first_main3 is None or first_lift is None else first_lift["action_boundary_time_s"]-first_main3["run_start"]["time_s"],
            "time_from_main3_confirmation_to_lift_action_s":None if first_main3 is None or first_lift is None else first_lift["action_boundary_time_s"]-first_main3["confirmed"]["time_s"],
            "dynamic_hand":{"trace_recorded_max_actual_FD_rad_s":max_recorded_fd,"trace_recomputed_max_actual_FD_rad_s":max_recomputed_fd,
              "trace_recomputed_max_actual_FD_source_velocity_ratio":max_recomputed_fd_ratio,"engine_qvel_max_rad_s":max_engine,
              "max_mimic_residual_rad":max_mimic,"engine_velocity_is_separate_from_FD":True,"report_dynamic_audit":report.get("dynamic_hand_audit")},
            "actual_self":{"active_steps":self_active_steps,"active_contact_records":self_active_count,
              "inactive_proximity_shell_records":self_inactive_count,"unknown_activity_records":self_unknown_count,
              "normal_impulse_default_threshold_Ns":1e-7,"reported_full36_audit":{k:report.get("actual36_geometry_audit",{}).get(k) for k in ("passed","historical_actual_self_active_count")}},
            "carry":carry,"report_source_acceptance":{"canonical_sources_unchanged":report.get("canonical_sources_unchanged"),
              "runtime_sources_unchanged":report.get("runtime_sources_unchanged")},
            "source_sha256":{str(p):sha256_file(p) for p in paths},"helper_sha256":sha256_file(__file__)}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_dir",type=Path);parser.add_argument("--manifest",required=True,type=Path)
    parser.add_argument("--report",type=Path);parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--dt-s",type=float);parser.add_argument("--first-lift-dense-call",type=int)
    parser.add_argument("--require-accepted",action="store_true")
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError("Refusing to replace QA output")
    report=args.report or next((p for p in (args.episode_dir/"replay_report.json",args.episode_dir/"attempt_report.json") if p.exists()),None)
    if report is None:raise FileNotFoundError("Missing published attempt/replay report")
    result=analyze_episode(args.episode_dir/"physics_trace.json",report,args.manifest,dt_s=args.dt_s,
                           first_lift_dense_call=args.first_lift_dense_call)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"output":str(args.output),"report_accepted":result["report_accepted"],
          "full_success_verified":result["full_success_verified"],"seconds":result["physics_clock"]["duration_s"],
          "checks":result["independent_recorded_checks"],"source_sha256":result["source_sha256"]},indent=2))
    return 0 if not args.require_accepted or result["full_success_verified"] else 2

if __name__=="__main__":raise SystemExit(main())