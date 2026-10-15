"""Bounded CPU correction of accepted-video QA labels; no physics/full audit rerun."""
from __future__ import annotations
import argparse
import copy
import csv
import json
from pathlib import Path
import shutil
import time
import numpy as np
from accepted_video_qa import _sha,_iter_json_array,_normal_forces,_clock,_rotation_angle_deg


def correct_qa(replay_dir):
    started=time.monotonic();root=Path(replay_dir).resolve()
    base=root/"video_qa_20261003";out=root/"video_qa_20261003_corrected"
    if out.exists():raise FileExistsError("Refusing to overwrite corrected QA")
    base_file=base/"video_qa_summary.json";original=json.loads(base_file.read_text(encoding="utf-8"))
    report_file=root/"replay_report.json";report=json.loads(report_file.read_text(encoding="utf-8"))
    audit=report["actual36_geometry_audit"]
    if not report["accepted"] or not audit["passed"] or not audit["full_temporal_coverage"]:
        raise ValueError("Correction requires the original independently accepted complete actual audit")
    if _sha(report_file)!=original["source_sha256"][str(report_file)]:
        raise ValueError("Published report differs from full-decode QA source")
    result=copy.deepcopy(original);rows=result["key9_contacts"]
    for row in rows:
        shells=row["actual_self_contacts"]
        if any("active" not in x for x in shells):raise ValueError("Missing actual contact activation flag")
        row["actual_self_contacts"]=[x for x in shells if x["active"] is True]
        row["inactive_self_contact_shells"]=len(shells)-len(row["actual_self_contacts"])
    erroneous=result.pop("trace_actual_self_contact_steps")
    result["trace_actual_active_self_contact_steps"]=int(audit["historical_actual_self_active_count"])
    result["active_self_count_source"]="Published actual36_geometry_audit.historical_actual_self_active_count; full prior audit, not a new scan"
    meta_file=root/"head_realtime_video.json";meta=json.loads(meta_file.read_text(encoding="utf-8"))
    if _sha(meta_file)!=original["source_sha256"][str(meta_file)]:
        raise ValueError("Head source clock differs from original QA")
    clock,source_times,steps,ages=_clock(meta,"head",30.)
    row=next(x for x in rows if x["camera"]=="head" and x["event"]=="first_main3_contact")
    frame_index=int(np.flatnonzero(steps>=row["requested_event_step"])[0]);target_step=int(steps[frame_index])
    actual=None;prefix_count=0
    for record in _iter_json_array(root/"physics_trace.json"):
        prefix_count+=1
        if record["step"]==target_step:actual=record;break
        if record["step"]>target_step:break
    if actual is None or min(_normal_forces(actual))<=.05:
        raise ValueError("Corrected source does not satisfy genuine actual three-contact event")
    out.mkdir()
    import cv2
    head_video=Path(meta["video"]);cap=cv2.VideoCapture(str(head_video))
    cap.set(cv2.CAP_PROP_POS_FRAMES,frame_index);ok,frame=cap.read();cap.release()
    if not ok or frame.shape!=(240,320,3):raise ValueError("Corrected head frame decode failed")
    png=out/"head_03_first_main3_contact.png"
    if not cv2.imwrite(str(png),frame):raise RuntimeError("Cannot save corrected native keyframe")
    previous={"frame_index":row["frame_index"],"source_physics_step":row["source_physics_step"],
              "main3_normal_N":row["main3_normal_N"]}
    row.update(frame_index=frame_index,output_time_s=float(clock[frame_index]),
       source_observation_time_s=float(source_times[frame_index]),source_physics_step=target_step,
       matched_trace_step=actual["step"],trace_source_alignment_error_s=abs(actual["sim_time_s"]-source_times[frame_index]),
       event_time_error_s=abs(float(clock[frame_index])-row["requested_event_step"]*report["dataset"]["physical_step_s"]),
       source_age_s=float(ages[frame_index]),phase=actual["phase"],main3_normal_N=_normal_forces(actual),
       upright_angle_deg=actual["bottle"]["upright_angle_deg"],official_success=actual["bottle"]["official_success"],
       actual_self_contacts=[x for x in actual["actual_self_contacts"] if x["active"] is True],
       inactive_self_contact_shells=sum(x["active"] is False for x in actual["actual_self_contacts"]),
       contact_points=actual["grip"]["contacts"],bottle_pose_world=actual["bottle"]["pose"],
       bottle_functional_point0=actual["bottle"]["functional_point0"],
       grip_measured={k:v for k,v in actual["grip"].items() if k=="contacts" or k=="relative_bottle_palm"},
       object_delta_from_initial_m=actual["fp_delta_m"],object_rotation_delta_from_initial_deg=actual["rotation_delta_deg"],
       relative_bottle_palm=actual["grip"]["relative_bottle_palm"],decoded_png=str(png))
    ref=np.asarray(audit["historical_reference_validation"]["relative_bottle_palm"])
    for item in rows:
        relative=np.asarray(item["relative_bottle_palm"])
        after=item["phase"]["name"]=="after_grasp"
        item["relative_drift_reference_source"]="Published actual36 historical_reference_validation.relative_bottle_palm"
        item["relative_translation_drift_from_pre_carry_m"]=float(np.linalg.norm(relative[:3,3]-ref[:3,3])) if after else None
        item["relative_rotation_drift_from_pre_carry_deg"]=_rotation_angle_deg(relative,ref) if after else None
    for kind in ("external","head"):
        if kind=="external":shutil.copyfile(base/"external_key9_contact_sheet.png",out/"external_key9_contact_sheet.png")
        else:
            cards=[]
            for i,item in enumerate(x for x in rows if x["camera"]=="head"):
                image=cv2.imread(item["decoded_png"])
                if image is None:raise ValueError("Prior native keyframe missing")
                thumb=cv2.resize(image,(480,360),interpolation=cv2.INTER_AREA)
                card=np.zeros((405,480,3),dtype=np.uint8);card[:360]=thumb
                cv2.putText(card,f"{i+1} {item['event']} t={item['output_time_s']:.3f}s",(8,378),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
                cv2.putText(card,"Fn m/i/t="+"/".join(f"{x:.3f}" for x in item["main3_normal_N"]),(8,398),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
                cards.append(card)
            sheet=np.vstack([np.hstack(cards[j:j+3]) for j in range(0,9,3)])
            if not cv2.imwrite(str(out/"head_key9_contact_sheet.png"),sheet):raise RuntimeError("Cannot save corrected sheet")
        next(x for x in result["videos"] if x["camera"]==kind)["key9_contact_sheet"]=str(out/f"{kind}_key9_contact_sheet.png")
    result["result"]="completed_decoded_video_timing_and_contact_QA_corrected"
    result["self_contact_scope"]="active=True contacts only; inactive zero-impulse PhysX shells do not count as collisions"
    result["correction"]={"base_qa_summary_path":str(base_file),"base_qa_summary_sha256":_sha(base_file),
       "base_qa_source_sha256":original["qa_source_sha256"],"correction_source_sha256":_sha(Path(__file__)),
       "correction_imported_qa_source_sha256":_sha(Path(__file__).with_name("accepted_video_qa.py")),
       "removed_erroneous_nonempty_shell_step_count":erroneous,"head_first_contact_before":previous,
       "head_first_contact_after":{"frame_index":frame_index,"source_physics_step":target_step,"main3_normal_N":row["main3_normal_N"]},
       "trace_prefix_records_read":prefix_count,"full_decode_and_clock_proof_retained":True,
       "no_physics_or_full_mesh_audit_repeated":True,"elapsed_s":time.monotonic()-started}
    result["qa_correction_output_sha256"]={str(p):_sha(p) for p in out.glob("*.png")}
    (out/"video_qa_summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    with (out/"key9_contacts.csv").open("w",newline="",encoding="utf-8-sig") as stream:
        writer=csv.writer(stream);writer.writerow(["camera","event","frame","source_step","middle_N","index_N","thumb_N","upright_deg","active_self_contacts","slip_m","slip_deg","decoded_png"])
        for r in rows:writer.writerow([r["camera"],r["event"],r["frame_index"],r["source_physics_step"],*r["main3_normal_N"],r["upright_angle_deg"],len(r["actual_self_contacts"]),r["relative_translation_drift_from_pre_carry_m"],r["relative_rotation_drift_from_pre_carry_deg"],r["decoded_png"]])
    (out/"correction_note.md").write_text(
       "# Accepted video QA correction\n\nThe prior QA used a nonempty PhysX contact-shell list as an active-contact flag. "
       "It is replaced by the independently completed actual36 audit's true active count (0), and all 9-frame records filter active=True. "
       f"The head first-contact image is advanced from source step {previous['source_physics_step']} to {target_step}, "
       "so its exact source observation is after the genuine three-contact threshold. "
       "Full decode/FPS/count/time proofs are retained from the original QA; no physics or full mesh audit is repeated. "
       "Other endpoint images retain their original transparent source-clock alignment.\n",encoding="utf-8")
    return {"result":result["result"],"output":str(out),"active_self_contact_steps":result["trace_actual_active_self_contact_steps"],
            "head_source_step":target_step,"main3_normal_N":row["main3_normal_N"],
            "summary_sha256":_sha(out/"video_qa_summary.json"),"elapsed_s":result["correction"]["elapsed_s"]}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("replay_dir");args=parser.parse_args()
    print(json.dumps(correct_qa(args.replay_dir),indent=2))
