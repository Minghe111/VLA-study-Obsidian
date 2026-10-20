"""Materialize and validate the user-confirmed subset on server166."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import copy
import getpass
import hashlib
import json
import os
import shutil
import socket
import sys
import traceback

ROOT = Path(__file__).resolve().parent.parent
SOURCE = Path("/bigdata2/liminghe/VLA-benchmark/deploy/realman_gen4/rgb50_final_20261007/dataset")
DEST = ROOT / "dataset"
SOURCE_SHA = "2339d5749b650761011e3975cbeb8d42ceb31dbff4f1f2aba6d71cac9a3edf80"
EXCLUDED = [1, 17, 18, 20, 29, 34, 47]
RETAINED = [i for i in range(50) if i not in EXCLUDED]


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for b in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def event(phase, **kwargs):
    print(json.dumps(dict(phase=phase, **kwargs), ensure_ascii=False), flush=True)


def source_path(relative):
    rel = Path(relative)
    require(not rel.is_absolute() and ".." not in rel.parts, "Unsafe source path")
    path = SOURCE / rel
    require(path.resolve().is_relative_to(SOURCE), "Source path escapes publication")
    require(path.is_file(), "Missing published file: " + str(path))
    return path


def main():
    require(socket.gethostname() == "user-4U-GPU-Server" and getpass.getuser() == "liminghe"
            and os.getuid() == 1014, "Unexpected server/account")
    require(ROOT == Path("/bigdata2/liminghe/VLA-benchmark/deploy/realman_gen4/rgb50_clean43_20261019_v1")
            and ROOT.resolve() == ROOT and ROOT.stat().st_uid == os.getuid(), "Unexpected destination")
    require(SOURCE.resolve() == SOURCE and SOURCE.stat().st_uid == os.getuid(), "Unexpected source")
    require(not DEST.exists(), "Refusing to overwrite a dataset")
    require(sha(SOURCE / "manifest.json") == SOURCE_SHA, "Original source manifest changed")
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    require(manifest["count"] == 50 and manifest["complete"] is True, "Source is not complete50")
    rows = sorted(manifest["episodes"], key=lambda e: e["index"])
    require([r["index"] for r in rows] == list(range(50)), "Original ID coverage differs")
    signature = {}
    for count, row in enumerate(rows, 1):
        relative = "data/episode_%07d.hdf5" % row["index"]
        path = source_path(relative)
        payload = row["published_files"][relative]
        require(path.stat().st_size == payload["bytes"] and sha(path) == payload["sha256"],
                "Source HDF mismatch: " + relative)
        st = path.stat()
        signature[str(path)] = [st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]
        if count % 10 == 0:
            event("source50_hashes", completed=count, total=50)
    write_new(ROOT / "evidence/source50_precheck.json", {
        "passed": True, "host": socket.gethostname(), "user": getpass.getuser(),
        "source_manifest_sha256": SOURCE_SHA, "all50_hdf_hashes_match": True,
        "source_hdf_stat_signatures": signature})
    shutil.copyfile(SOURCE / "manifest.json", ROOT / "evidence/original50_manifest.json")
    shutil.copyfile(SOURCE / "scene_info.json", ROOT / "evidence/original50_scene_info.json")
    for name in ["data", "instruction", "episodes", "video", "_traj_data"]:
        (DEST / name).mkdir(parents=True, exist_ok=False)
    retained_rows = []
    copied = []
    for row in rows:
        i = row["index"]
        if i in EXCLUDED:
            continue
        for relative, payload in row["published_files"].items():
            old = source_path(relative)
            target = DEST / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            require(not target.exists(), "Unexpected repeated output path")
            if relative.startswith(("data/", "instruction/")):
                shutil.copyfile(old, target)
                require(target.stat().st_size == payload["bytes"] and sha(target) == payload["sha256"],
                        "Independent output copy mismatch")
                require(target.stat().st_ino != old.stat().st_ino, "Output copy unexpectedly shares inode")
                copied.append({"relative_path": relative, "bytes": payload["bytes"],
                               "sha256": payload["sha256"]})
            else:
                require(old.stat().st_size == payload["bytes"], "Auxiliary file size mismatch")
                target.symlink_to(old)
        old_proof = SOURCE / row["episode_proof_dir"]
        require(old_proof.is_dir() and old_proof.resolve().is_relative_to(SOURCE), "Unsafe proof location")
        (DEST / row["episode_proof_dir"]).symlink_to(old_proof, target_is_directory=True)
        retained_rows.append(copy.deepcopy(row))
        event("copied_retained", completed=len(retained_rows), total=43, original_episode=i)
    scenes = json.loads((SOURCE / "scene_info.json").read_text())
    write_new(DEST / "scene_info.json", {"episode_%d" % i: scenes["episode_%d" % i] for i in RETAINED})
    clean = copy.deepcopy(manifest)
    clean.update({"schema_version": 4, "complete": True,
                  "status": "clean43_membership_complete_data_validation_separate",
                  "count": 43, "episodes": retained_rows,
                  "source_dataset": str(SOURCE), "source_manifest_sha256": SOURCE_SHA,
                  "excluded_original_indices": EXCLUDED, "retained_original_indices": RETAINED,
                  "created_utc": datetime.now(timezone.utc).isoformat(),
                  "source_target_counts": manifest["target_counts"],
                  "actual_counts": {k: dict(Counter(str(r[k]) for r in retained_rows))
                                    for k in ["arm", "model_id", "pose_class"]},
                  "total_state_action_pairs": 9255, "total_raw_observations": 9298,
                  "total_physical_steps": 138825,
                  "transfer_mode": "independent_hdf_instruction_copy_with_original_proof_links",
                  "source_policy": "User-confirmed removal of seven full-turn episodes; retained original bytes",
                  "global_motion_quality_gate_applied": False,
                  "proof_and_video_links_are_original_historical_evidence": True})
    clean.pop("independent_readback_sha256", None)
    clean["target_counts"] = clean["actual_counts"]
    write_new(DEST / "manifest.json", clean)
    write_new(ROOT / "episode_mapping.json", {
        "filenames_keep_original_indices": True,
        "training_order_to_original_index": RETAINED,
        "excluded_original_indices": EXCLUDED})
    write_new(ROOT / "excluded/manifest.json", {
        "training_eligible": False, "storage_mode": "references_to_retained_historical_original50",
        "files_physically_purged": False,
        "reason": "User-confirmed active J6 actual coordinate range above360 degrees",
        "excluded_original_indices": EXCLUDED,
        "source_dataset": str(SOURCE), "source_manifest_sha256": SOURCE_SHA,
        "episodes": [{"original_index": r["index"], "source_hdf": str(SOURCE / ("data/episode_%07d.hdf5" % r["index"])),
                      "sha256": r["published_files"]["data/episode_%07d.hdf5" % r["index"]]["sha256"]}
                     for r in rows if r["index"] in EXCLUDED]})
    parameters = {"retained_original_indices": RETAINED, "original_manifest_sha256": SOURCE_SHA,
                  "clean_manifest_sha256": sha(DEST / "manifest.json")}
    write_new(ROOT / "tools/selection_parameters.json", parameters)
    sys.path.insert(0, str(ROOT / "tools"))
    import data_adapter_clean43 as adapter
    parent = Path("/bigdata2/liminghe/VLA-benchmark/RoboTwin")
    report = adapter.prepare(DEST, ROOT / "data43_v1", xpolicylab_parent=parent)
    dataset = adapter.RealmanNativeDataset(ROOT / "data43_v1", verify_hashes=True)
    require(len(dataset) == 8610, "Reader window count differs")
    covered = set()
    membership = set()
    boundary_items = {}
    for position in range(len(dataset)):
        entry, offset, source_rows = dataset._resolve(position)
        i = entry["index"]
        membership.add(i)
        covered.update((i, int(k)) for k in source_rows)
        boundary_items.setdefault(i, [position, position])[1] = position
    require(membership == set(RETAINED) and not membership.intersection(EXCLUDED)
            and len(covered) == 9255, "Reader membership or target coverage differs")
    import numpy as np
    checked = 0
    for i, boundaries in boundary_items.items():
        for position in boundaries:
            item = dataset[position]
            transformed = adapter.RealmanInputs()(item)
            require(transformed["state"].shape == (24,) and transformed["actions"].shape == (16, 24),
                    "Actual reader dimensions differ")
            delta = adapter.base.delta_arms(transformed["actions"], transformed["state"])
            restored = adapter.base.restore_arms(np.pad(delta, ((0, 0), (0, 8))),
                                               np.pad(transformed["state"], (0, 8)))
            require(np.allclose(restored[:, :24], transformed["actions"], atol=1e-6, rtol=0),
                    "Anchor delta roundtrip differs")
            checked += 1
    require(sorted(x.name for x in (DEST / "data").iterdir()) ==
            ["episode_%07d.hdf5" % i for i in RETAINED], "Unexpected HDF payload membership")
    require(sha(SOURCE / "manifest.json") == SOURCE_SHA, "Historical manifest changed")
    for filename, before in signature.items():
        st = Path(filename).stat()
        require([st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns] == before,
                "Historical source HDF changed during curation")
    require(sum(r["RGB_frames_decoded"] for r in report["per_episode_readbacks"]) == 27765,
            "Full RGB decode count differs")
    validation = {"passed": True, "host": socket.gethostname(), "user": getpass.getuser(),
                  "utc": datetime.now(timezone.utc).isoformat(), "dataset": str(DEST),
                  "excluded_original_indices": EXCLUDED, "retained_episode_count": 43,
                  "state_action_pairs": 9255, "h16_full_windows": 8610,
                  "all_RGB_frames_decoded": 27765, "actual_reader_boundary_items_checked": checked,
                  "all_window_episode_membership_checked": True,
                  "normalization_recomputed_from_retained_only": True,
                  "source50_manifest_unchanged": True, "source50_hdf_stat_signatures_unchanged": True,
                  "original_historical_files_physically_purged": False,
                  "bad_episodes_present_in_clean_training_index": False,
                  "new_global_motion_quality_pass_claimed": False,
                  "training_started": False, "GPU_used": False,
                  "copied_training_files": copied,
                  "clean_manifest_sha256": sha(DEST / "manifest.json"),
                  "native_index_sha256": sha(ROOT / "data43_v1/native_index.json"),
                  "norm_stats_sha256": report["norm_stats_sha256"]}
    write_new(ROOT / "evidence/final_validation.json", validation)
    write_new(ROOT / "TRAINING_DATA.json", {
        "dataset_root": str(DEST), "data_root": str(ROOT / "data43_v1"),
        "adapter_module": str(ROOT / "tools/data_adapter_clean43.py"),
        "asset_id": adapter.ASSET_ID,
        "norm_stats": str(ROOT / "data43_v1/assets" / adapter.ASSET_ID / "norm_stats.json"),
        "repo_id": "realman-rgb43-clean43-native-v1/train", "train_only": True,
        "episode_count": 43, "state_action_pairs": 9255, "horizon": 16,
        "full_windows": 8610, "physical_dimension": 24, "model_dimension": 32,
        "control_clock_seconds": clean["observation_dt_s"],
        "old_all50_adapter_is_not_the_clean43_entrypoint": True,
        "excluded_original_indices": EXCLUDED})
    (ROOT / "README.md").write_text(
        "# RGB43 数据清理版\n\n"
        "从原50条中剔除原编号1、17、18、20、29、34、47，保留43条、9255对状态动作。\n\n"
        "训练入口见 TRAINING_DATA.json；H16完整窗口8610个。文件保留原编号，不假定编号连续。\n"
        "必须使用 tools/data_adapter_clean43.py 和 data43_v1 的新归一化统计；原all50读取器固定要求50条。\n\n"
        "HDF和指令为独立复制；视频、逐物理步与原采集证明链接到历史发布集，未重新仿真或重录。\n"
        "excluded/manifest.json记录7条排除项；原50条以历史版本保留，没有永久清除原始证据。\n\n"
        "本版本只完成用户指定的整圈轨迹剔除，未宣称余下43条全面通过新增运动效率门禁。\n"
        "原编号9、16、28、35、39、44、45、46仍有>=300度行程需进一步评估，其他绕行指标见逐条审计。\n"
        "本次未启动训练、未修改已有训练作业或权重。\n")
    write_new(ROOT / "READY.json", {
        "passed": True, "dataset_root": str(DEST), "episode_count": 43,
        "validation_sha256": sha(ROOT / "evidence/final_validation.json"),
        "training_entry_sha256": sha(ROOT / "TRAINING_DATA.json")})
    event("COMPLETE", retained=43, pairs=9255, windows=8610, rgb_frames=27765,
          final_validation_sha256=sha(ROOT / "evidence/final_validation.json"))


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        failure = {"passed": False, "error_type": type(error).__name__, "error": str(error),
                   "traceback": traceback.format_exc(), "original_source_modified_by_this_tool": False}
        path = ROOT / "FAILED.json"
        if ROOT.exists() and not path.exists():
            write_new(path, failure)
        raise
