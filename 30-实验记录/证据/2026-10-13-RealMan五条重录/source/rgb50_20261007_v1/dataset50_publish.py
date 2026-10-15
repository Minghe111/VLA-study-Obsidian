"""Additive atomic publisher. A fresh50 manifest is complete only after CPU audit.

No source trajectory, old20, or official asset is edited. Every raw N+1 cache and
proof is retained under episodes/episode_NNNNNNN. Default copies independent
bytes; --hardlink is an explicit storage-saving choice for frozen source trials.
"""
from __future__ import annotations
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import uuid

from dataset50_audit import ROOT, accepted_paths, audit_dataset, inventory, read_json, require, sha


def transfer(source, target, *, hardlink=False):
    target.parent.mkdir(parents=True,exist_ok=True)
    require(not target.exists(),"Refusing overwrite: "+str(target))
    if hardlink:
        try: os.link(source,target)
        except OSError: shutil.copy2(source,target)
    else: shutil.copy2(source,target)


def publish(paths, output, *, codec_path=None, hardlink=False, audit_output=None, allow_synthetic=False):
    output=Path(output).resolve()
    require(not output.exists(),"Output must be a new nonexistent dataset directory")
    require("rgb20_final_20261004" not in str(output),"Old20 publication path forbidden")
    paths=[Path(path).resolve() for path in paths]
    require(all(not output.is_relative_to(path) and not path.is_relative_to(output) for path in paths),"Source/output paths overlap")
    audit=audit_dataset(paths,codec_path=codec_path,allow_synthetic=allow_synthetic)
    if audit_output:
        audit_output=Path(audit_output).resolve()
        require(all(not audit_output.is_relative_to(path) for path in paths),"Audit output overlaps source trial")
        audit_output.parent.mkdir(parents=True,exist_ok=True)
        audit_output.write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    if not audit["complete"]:
        return {"complete":False,"status":"not_published_insufficient_or_invalid50","output":str(output),
                "accepted_count":audit["accepted_count"],"input_count":audit["input_count"],
                "actual_counts":audit["actual_counts"],"failures":[{"trial_dir":r["trial_dir"],"error":r.get("error")} for r in audit["episodes"] if not r["passed"]]}
    old_candidates=[ROOT.parent/name/"dataset/manifest.json" for name in ("rgb20_final_20261004","realman_rgb20_final_20261004")]
    old_manifest=next((p for p in old_candidates if p.is_file()),None)
    require(old_manifest is not None,"Original old20 manifest unavailable for nonmutation guard")
    old_sha=sha(old_manifest)
    require(old_sha=="97b33109a740ae2d9b99155dbf228e5d2a19761b323e60fc622c8998aec0e577","Original old20 manifest changed")
    require(not output.is_relative_to(old_manifest.parent) and not old_manifest.parent.is_relative_to(output),"Output overlaps original20")
    # Spec index fixes output identity regardless of filesystem discovery order.
    rows=sorted(audit["episodes"],key=lambda row:row["identity"]["spec_index"])
    staging=output.parent/("."+output.name+".staging-"+uuid.uuid4().hex)
    staging.mkdir(parents=True)
    manifest={"schema_version":1,"complete":False,"status":"staging_not_complete","task":"adjust_bottle",
        "count":50,"actual_counts":audit["actual_counts"],"target_counts":audit["target_counts"],
        "created_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "old20_manifest":str(old_manifest),"old20_manifest_sha256":old_sha,
        "hardware_io":False,"old20_preserved":True,"source_policy":"50 fresh successful physics episodes; no old20 relabeling",
        "transfer_mode":"hardlink_with_copy_fallback" if hardlink else "independent_copy",
        "synthetic_fixture_only":audit["synthetic_fixture_only"],
        "joint_ordering":"left arm6 rad,left primary hand6 SDK,right arm6 rad,right primary hand6 SDK",
        "training_cameras":["cam_head","cam_left_wrist","cam_right_wrist"],"rgb_only":True,
        "physics_dt_s":rows[0]["physical_dt_s"],"observation_dt_s":rows[0]["observation_dt_s"],"sdk_mapping_calibrated":False,
        "publisher_source_sha256":sha(Path(__file__)),"validator_source_sha256":audit["validator_sha256"],"episodes":[]}
    try:
        scene_info={}
        for index,row in enumerate(rows):
            source=Path(row["trial_dir"]); relative_episode=Path("episodes")/f"episode_{index:07d}"
            # Preserve every source file, including raw N+1 camera caches and all physical proofs.
            for rel,receipt in row["source_files"].items():
                origin=source/rel;require(origin.is_file() and not origin.is_symlink(),"Trial source disappeared")
                dest=staging/relative_episode/rel
                transfer(origin,dest,hardlink=hardlink)
                require(sha(dest)==receipt["sha256"],"Copied proof hash differs: "+str(dest))
            mapping={"data/episode_0000000.hdf5":f"data/episode_{index:07d}.hdf5",
                     "instruction/episode_0000000.json":f"instruction/episode_{index:07d}.json",
                     "_traj_data/episode0.pkl":f"_traj_data/episode{index}.pkl",
                     "video/episode_0000000.mp4":f"video/episode_{index:07d}.mp4"}
            for rel in row["source_files"]:
                if rel.startswith("video/") and rel.endswith("_realtime.mp4"):
                    base=Path(rel).stem
                    # Current runner head_realtime/left_wrist_realtime/right_wrist_realtime.
                    if base.startswith("episode_0000000_"):base=base[len("episode_0000000_"):]
                    mapping[rel]=f"video/episode_{index:07d}_{base}.mp4"
            top_files={}
            for original,published in mapping.items():
                # These link only to the frozen copied episode proof inside this new dataset.
                transfer(staging/relative_episode/original,staging/published,hardlink=True)
                top_files[published]={"sha256":sha(staging/published),"bytes":(staging/published).stat().st_size}
            if "scene_info.json" in row["source_files"]:
                value=read_json(source/"scene_info.json")
                scene_info[f"episode_{index}"]=value.get("episode_0",value)
            proof=dict(row);proof.pop("source_files",None)
            proof_path=relative_episode/"independent_dataset50_audit.json"
            require(not (staging/proof_path).exists(),"Source contains reserved publisher proof name")
            (staging/proof_path).write_text(json.dumps(proof,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
            manifest["episodes"].append({"index":index,**row["identity"],"source_trial_dir":str(source),
                "state_action_pairs":row["state_action_pairs"],"raw_observations":row["raw_observations"],
                "physical_steps":row["physical_steps"],"source_files":row["source_files"],
                "episode_proof_dir":str(relative_episode),"raw_cache_dir":str(relative_episode/".cache/episode0"),
                "published_files":top_files,"independent_audit":str(proof_path),
                "independent_audit_sha256":sha(staging/proof_path),"physical_success":True})
            require(inventory(source)==row["source_files"],"Source changed during publication")
        (staging/"scene_info.json").write_text(json.dumps(scene_info,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        (staging/"independent_readback.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        manifest["independent_readback_sha256"]=sha(staging/"independent_readback.json")
        manifest["total_state_action_pairs"]=sum(row["state_action_pairs"] for row in rows)
        manifest["total_raw_observations"]=sum(row["raw_observations"] for row in rows)
        manifest["total_physical_steps"]=sum(row["physical_steps"] for row in rows)
        # All copied files are read/hashes checked before any complete manifest exists.
        require(len(manifest["episodes"])==50,"Publication count differs")
        for ep in manifest["episodes"]:
            for rel,receipt in ep["published_files"].items():
                require(sha(staging/rel)==receipt["sha256"],"Published file changed")
        require(sha(old_manifest)==old_sha,"Original20 changed during publication")
        manifest.update(complete=True,status="synthetic_fixture50_test_only" if audit["synthetic_fixture_only"] else "complete_independently_validated50")
        (staging/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        require(not output.exists(),"Output appeared during atomic publication")
        staging.rename(output)
        return {"complete":True,"status":manifest["status"],"output":str(output),"count":50,
                "actual_counts":manifest["actual_counts"],"manifest_sha256":sha(output/"manifest.json"),
                "total_state_action_pairs":manifest["total_state_action_pairs"],"old20_preserved":True}
    except Exception as error:
        # Preserve reviewable staging evidence. Never delete source or partial output automatically.
        manifest.update(complete=False,status="publication_failed_staging_not_complete")
        (staging/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        (staging/"publication_error.json").write_text(json.dumps({"complete":False,"error":str(error)},indent=2)+"\n",encoding="utf-8")
        raise


def main():
    ap=argparse.ArgumentParser();group=ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--accepted-list",type=Path);group.add_argument("--trials-root",type=Path)
    ap.add_argument("--output",type=Path,required=True);ap.add_argument("--codec",type=Path)
    ap.add_argument("--audit-output",type=Path);ap.add_argument("--hardlink",action="store_true");ap.add_argument("--allow-synthetic-fixtures",action="store_true")
    args=ap.parse_args();paths=accepted_paths(accepted_list=args.accepted_list,trials_root=args.trials_root)
    result=publish(paths,args.output,codec_path=args.codec,hardlink=args.hardlink,audit_output=args.audit_output,allow_synthetic=args.allow_synthetic_fixtures)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0 if result["complete"] else 2

if __name__=="__main__": raise SystemExit(main())
