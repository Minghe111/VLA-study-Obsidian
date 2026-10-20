"""43-episode revision of the frozen native RGB50 reader; CPU data work only."""
from pathlib import Path
import importlib.util
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
PARAMS = json.loads((HERE / "selection_parameters.json").read_text())
spec = importlib.util.spec_from_file_location("_clean43_frozen_base", HERE / "frozen_rgb50_adapter.py")
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)
INDICES = PARAMS["retained_original_indices"]
EXCLUDED = [1, 17, 18, 20, 29, 34, 47]
assert INDICES == [i for i in range(50) if i not in EXCLUDED]
base.MANIFEST_SHA256 = PARAMS["clean_manifest_sha256"]
base.TRAIN_INDICES = INDICES
base.PAIR_COUNT = 9255
base.ANCHOR_COUNT = 8610
base.ASSET_ID = "realman_rgb43_clean_20261019"
base.REPO_IDS = {"realman-rgb43-clean43-native-v1/train": "train"}


def canonical(source, manifest_name="manifest.json", *, verify_hdfs=True):
    source = Path(source).resolve()
    path = base.safe_child(source, manifest_name)
    base.require(base.sha(path) == base.MANIFEST_SHA256, "Clean43 manifest identity mismatch")
    document = base.load(path)
    base.require(document["complete"] is True and document["count"] == 43
                 and document["source_manifest_sha256"] == PARAMS["original_manifest_sha256"]
                 and document["excluded_original_indices"] == EXCLUDED
                 and document["retained_original_indices"] == INDICES
                 and document["total_state_action_pairs"] == 9255
                 and document["total_raw_observations"] == 9298
                 and document["total_physical_steps"] == 138825
                 and document["rgb_only"] is True
                 and document["synthetic_fixture_only"] is False,
                 "Clean43 selection or data contract mismatch")
    rows = sorted(document["episodes"], key=lambda row: int(row["index"]))
    base.require([int(row["index"]) for row in rows] == INDICES, "Unexpected episode membership")
    base.require(len({row["seed"] for row in rows}) == 43, "Repeated source seed")
    for row in rows:
        i = int(row["index"])
        relative = "data/episode_%07d.hdf5" % i
        n = int(row["state_action_pairs"])
        base.require(row["physical_success"] is True and row["boundary_used"] is False
                     and n >= 16 and row["raw_observations"] == n + 1
                     and row["physical_steps"] == n * 15, "Invalid retained episode")
        payload = row["published_files"][relative]
        hdf = base.safe_child(source, relative)
        if verify_hdfs:
            base.require(hdf.is_file() and hdf.stat().st_size == payload["bytes"]
                         and base.sha(hdf) == payload["sha256"], "Retained HDF mismatch: %d" % i)
    return document


base.canonical = canonical
RealmanInputs = base.RealmanInputs
RealmanOutputs = base.RealmanOutputs
make_data_transforms = base.make_data_transforms
build_anchors = base.build_anchors
CAMERAS = base.CAMERAS
ARM_MASK = base.ARM_MASK
ASSET_ID = base.ASSET_ID
REPO_IDS = base.REPO_IDS


class RealmanNativeDataset(base.RealmanNativeDataset):
    def __init__(self, data_root, *args, **kwargs):
        index = base.load(Path(data_root) / "native_index.json")
        base.require(index["subset_adapter_sha256"] == base.sha(__file__)
                     and index["dataset_variant"] == "rgb43_clean43_train_only_native_v1",
                     "Clean43 wrapper identity mismatch")
        super().__init__(data_root, *args, **kwargs)


def prepare(source, output, *, xpolicylab_parent):
    source, output = Path(source).resolve(), Path(output).resolve()
    base.require(not output.exists(), "Output must be a new directory")
    document = canonical(source)
    decoder = base.shared_codec(xpolicylab_parent)
    episodes, readbacks, states, targets = [], [], {}, {}
    for row in sorted(document["episodes"], key=lambda e: int(e["index"])):
        i = int(row["index"])
        relative = "data/episode_%07d.hdf5" % i
        state, action, t, tn, prompt, readback = base.read_checked_episode(
            base.safe_child(source, relative), row, decoder, decode_all=True)
        episodes.append({"index": i, "spec_index": row["spec_index"], "seed": row["seed"],
                         "model_id": row["model_id"], "brand": row["brand"],
                         "arm": row["arm"], "pose_class": row["pose_class"], "split": "train",
                         "source_trial_dir_provenance_only": row["source_trial_dir"],
                         "source_hdf_relative_path": relative,
                         "hdf5_sha256": row["published_files"][relative]["sha256"],
                         "hdf5_bytes": row["published_files"][relative]["bytes"],
                         "prompt": prompt, "original_pairs": len(t), "zero_time_rows_removed": [],
                         "positive_rows": list(range(len(t))), "sample_time_s": t.tolist(),
                         "next_sample_time_s": tn.tolist(), "duration_s": float(tn[-1])})
        states[i], targets[i] = base.physical24(state), base.physical24(action)
        readbacks.append(readback)
        print(json.dumps({"phase": "all_rgb_decode", "completed": len(episodes),
                          "total": 43, "original_episode": i}, ensure_ascii=False), flush=True)
    anchors = build_anchors(episodes, "train", 16)
    base.require(len(anchors) == 8610, "Clean43 H16 window count mismatch")
    nstates, nactions, covered = [], [], set()
    for i, offset in anchors:
        anchor = base.scale_sdk(states[i][offset])
        nstates.append(anchor)
        nactions.append(base.delta_arms(base.scale_sdk(targets[i][offset:offset + 16]), anchor))
        covered.update((i, k) for k in range(offset, offset + 16))
    base.require(covered == {(e["index"], k) for e in episodes for k in e["positive_rows"]}
                 and len(covered) == 9255, "Label coverage mismatch")
    snorm, sfallback = base.norm_stats(nstates)
    anorm, afallback = base.norm_stats(nactions)
    output.mkdir(parents=True, exist_ok=False)
    counts = {"original_pairs": 9255, "positive_pairs": 9255, "zero_time_duplicates_removed": 0,
              "train_positive_pairs": 9255, "validation_positive_pairs": 0,
              "train_full_chunks": 8610, "validation_full_chunks": 0,
              "all_unique_targets_covered": 9255, "raw_observations": 9298}
    index = {"schema_version": 1, "dataset_variant": "rgb43_clean43_train_only_native_v1",
             "source_dataset": str(source), "manifest_name": "manifest.json",
             "source_manifest_sha256": base.MANIFEST_SHA256,
             "original50_manifest_sha256": PARAMS["original_manifest_sha256"],
             "adapter_sha256": base.sha(base.__file__), "subset_adapter_sha256": base.sha(__file__),
             "openpi_commit": base.OPENPI_COMMIT, "xpolicylab_parent": str(Path(xpolicylab_parent).resolve()),
             "physical_dimension": 24, "model_dimension": 32, "action_horizon": 16,
             "asset_id": ASSET_ID, "state_field": base.STATE_FIELD, "action_field": base.ACTION_FIELD,
             "state_order": "Larm6rad,Lhand6SDK,Rarm6rad,Rhand6SDK", "arm_delta_mask": ARM_MASK.tolist(),
             "sdk_divisor": 1000, "sdk_mapping_calibrated": False, "camera_mapping": CAMERAS,
             "input_metadata_excludes_arm_brand_pose_seed_and_success": True,
             "split": {"train_indices": INDICES, "validation_indices": []}, "counts": counts,
             "time_resampled": False, "terminal_padding": False, "dimension_mask_is_loss_mask": False,
             "full_chunks_only": True, "source_dataset_modified": False, "episodes": episodes}
    base.write_new(output / "native_index.json", index)
    normpath = output / "assets" / ASSET_ID / "norm_stats.json"
    base.write_new(normpath, {"norm_stats": {"state": snorm, "actions": anorm}})
    provenance = {"train_only": True, "all43_train": True, "train_indices": INDICES,
                  "excluded_original_indices": EXCLUDED, "state_anchor_rows": 8610,
                  "action_vectors_after_SDK_scale_and_anchor_delta": 8610 * 16,
                  "unique_targets_covered": len(covered), "state_fallback_dimensions": sfallback,
                  "action_fallback_dimensions": afallback,
                  "constant_dimension_rule": "q99-q01<1e-4 => median +/-0.001; std floor0.001",
                  "norm_stats_sha256": base.sha(normpath), "reused_original50_statistics": False}
    base.write_new(output / "normalization_provenance.json", provenance)
    report = {"passed": True, "counts": counts, "source_manifest_sha256": base.MANIFEST_SHA256,
              "per_episode_readbacks": readbacks, "all_shared_codec_RGB_frames_decoded": 27765,
              "train_only_normalization": True, "source_dataset_modified": False,
              "pickle_deserialized": False, "hardware_io": False, "GPU_used": False,
              "training_started": False, "new_physical_episodes_run": False,
              "global_motion_quality_gate_applied": False,
              "selection_rule": "User-confirmed exclusion of original episodes 1,17,18,20,29,34,47",
              "native_index_sha256": base.sha(output / "native_index.json"),
              "norm_stats_sha256": base.sha(normpath)}
    base.write_new(output / "conversion_validation.json", report)
    return report
