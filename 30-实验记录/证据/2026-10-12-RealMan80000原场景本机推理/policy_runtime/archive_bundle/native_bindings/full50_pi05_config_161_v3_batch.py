"""Additive full50 pi05 profile v2: exact official metadata wrapper map. Import is stdlib-only; it never launches training."""
from __future__ import annotations
import dataclasses
import hashlib
import importlib.util
import importlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys

OPENPI_COMMIT = "215abfb217dbac7d5f1273282331b9b1866c0479"
OFFICIAL_TRAIN_SHA256 = "22f98a06aaa6e68e0f725d727ab95398238d1cae8a46bb94e829f330758a9baf"
OFFICIAL_REVISION = "3dc3b798668feb99ac61cc9086d84cbcc3d79186"
OFFICIAL_ARCHIVE_SHA256 = "957b07d4319746c70ed5173c2b0ae11b96872410d139931de603d9b4c974ee1c"
OFFICIAL_ARCHIVE_BYTES = 12443100901
OFFICIAL_SOURCE_PATH = "cotrain_ckpt/Pi_05/RoboTwin-lerobot_v30-aloha_agilex-joint-0.tar.gz"
EXPECTED_FULL_PARAMETERS = 3353433872
EXPECTED_LEAVES = 51
EXPECTED_CHECKPOINT_RECEIPT_SHA256 = "8e9934b7bd2e5da85bcb2a477d729370fa0dd1d5e88727afe9a9ade96322e846"
KEY_MAP_SHA256 = "a52a1464430f98762e368645f0b28be00d4c530a94e50c9708702b31e90ad903"
ACTUAL_DIAGNOSTIC_SHA256 = "7bbce4626cbad94a539e2c062529f65b4ef5243f8628fb74fcc6a93bd2aa5ced"
KEY_MAP_PATH = Path(__file__).resolve().parent / "official_29999_metadata_key_map_v2.json"
MANIFEST_SHA256 = "2339d5749b650761011e3975cbeb8d42ceb31dbff4f1f2aba6d71cac9a3edf80"
ADAPTER_SHA256 = "4aaee44efff42054e66d7ec4e5ad4fc418d66d980b3c548d8c744a93c577947c"
INDEX_SHA256 = "9b21feef1c660900196e35ae3615ec47570d02a34990c3e17df94208b43038df"
NORM_SHA256 = "34cbdd09c831e04549d6c4a4e2eb321fb975bdea13bb816ec8aa3a2332d1ccc5"
CONVERSION_SHA256 = "c7e7e8a77027585d8a4501e51b134cc818c20bf808cb67fb162f85b6a9a1a1fa"
PROJECT_ROOT = Path("/home/admin123/liminghe/vla-benchmark")
GEN4 = PROJECT_ROOT / "deploy/realman_gen4"
BASE_ROOT = GEN4 / "pi05_rgb20_20261004"
TASK_ROOT = GEN4 / "pi05_rgb50_full50_100k_b8_4gpu_161_20261007_v1"
CONDA_PREFIX = Path("/home/admin123/liminghe/miniconda3/envs/robotwin")
ACTUAL_DIAGNOSTIC_PATH = TASK_ROOT / "status_recovery_v2/actual_cpu_restore_v2/cpu_actual_metadata_canonical_restore_diagnostic_v2.json"
DATA_ROOT = TASK_ROOT / "data50_v1"
SOURCE_ROOT = TASK_ROOT / "source_training_files"
ADAPTER_PATH = TASK_ROOT / "data_pipeline_v1/data_adapter_rgb50_v1.py"
NORM_PATH = DATA_ROOT / "assets/realman_rgb50_full50/norm_stats.json"
AUDIT_PATH = Path(__file__).resolve().parent / "train_step_data_gradient_audit_v3_batch.py"
NAME = "pi05_realman_rgb50_all50_full_v1"
REPO_ID = "realman-rgb50-full50-native-v1/train"
ASSET_ID = "realman_rgb50_full50"
DEPENDENCY_VERSIONS = {"jax": "0.5.3", "jaxlib": "0.5.3", "flax": "0.10.2",
    "numpy": "1.26.4", "optax": "0.2.4", "orbax-checkpoint": "0.11.13", "torch": "2.7.1+cpu"}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    path = Path(path)
    require(stat.S_ISREG(path.lstat().st_mode), "Expected a regular JSON file: " + str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(path, value):
    path = Path(path)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def fresh_directory(path, root=TASK_ROOT):
    path, root = Path(path).absolute(), Path(root).absolute()
    path.relative_to(root)
    require(not os.path.lexists(path), "A fresh directory is required: " + str(path))
    for parent in path.parents:
        if parent == root.parent:
            break
        if os.path.lexists(parent):
            require(stat.S_ISDIR(parent.lstat().st_mode), "A directory parent is a symlink or non-directory")


def official_identity():
    repo = BASE_ROOT / "openpi"
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0", GIT_TERMINAL_PROMPT="0")
    def git(*args):
        return subprocess.check_output(["git", "-c", "core.fsmonitor=false", "-C", str(repo), *args],
            env=env, text=True).strip()
    identity = {"commit": git("rev-parse", "HEAD"), "worktree_clean": not git("status", "--porcelain")}
    require(identity == {"commit": OPENPI_COMMIT, "worktree_clean": True}, "Official OpenPI must be pinned and clean")
    require(sha(repo / "scripts/train.py") == OFFICIAL_TRAIN_SHA256, "Official train.py bytes differ")
    return identity


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def training_contract(steps, batch_size):
    require(steps in (5, 100000), "Only separate 5-step audit or formal 100000 steps are permitted")
    require(type(batch_size) is int and batch_size in (32, 64), "Explicit global batch32 or64 is mandatory")
    return {"steps": steps, "formal_steps": 100000, "global_batch_size": batch_size, "per_device_batch_size": batch_size // 4, "fsdp_devices": 4,
        "physical_gpus": ["4", "5", "6", "7"], "pi05": True, "action_dim": 32, "action_horizon": 16,
        "model_variants": ["gemma_2b", "gemma_300m"], "full_parameter_count": EXPECTED_FULL_PARAMETERS,
        "all_parameter_leaves": 51, "frozen_parameter_count": 0, "lora_parameter_count": 0,
        "train_indices": list(range(50)), "validation_indices": [], "train_full_chunks": 10348,
        "positive_labels": 11098, "lr_warmup_steps": 1000, "lr_peak": 1e-4,
        "lr_decay_steps": 100000, "lr_final": 1e-5, "resume": False, "overwrite": False, "save_interval": 20000, "keep_period": 20000,
        "initialization": "official_RoboTwin_pi05_cotrain_29999_params_only_fresh_optimizer",
        "optimizer_initial_step": 0, "source_checkpoint_step": 29999}


def explicit_metadata_key_map():
    """Only the pinned 51 pairs proven by actual official restore are accepted."""
    require(sha(KEY_MAP_PATH) == KEY_MAP_SHA256, "Explicit official metadata mapping file bytes differ")
    document = read_json(KEY_MAP_PATH)
    expected = {"schema_version": 2, "status": "EXPLICIT_OFFICIAL_RESTORE_PARAMS_CANONICAL_MAPPING",
        "checkpoint_receipt_sha256": EXPECTED_CHECKPOINT_RECEIPT_SHA256,
        "source_archive_sha256": OFFICIAL_ARCHIVE_SHA256, "source_revision": OFFICIAL_REVISION,
        "official_commit": OPENPI_COMMIT, "actual_diagnostic_path": str(ACTUAL_DIAGNOSTIC_PATH),
        "actual_diagnostic_sha256": ACTUAL_DIAGNOSTIC_SHA256, "leaf_count": 51,
        "parameter_elements": EXPECTED_FULL_PARAMETERS}
    for field, value in expected.items():
        require(document.get(field) == value, "Explicit actual-restore mapping provenance differs: " + field)
    result, canonical = {}, set()
    for row in document["entries"]:
        raw_name, name = row["raw_metadata_path"], row["canonical_path"]
        require(raw_name not in result and name not in canonical and row["actual_restored_finite"] is True,
            "Explicit mapping duplicate or missing finite evidence")
        require(all(part not in ("", ".", "..") for part in raw_name.split("/"))
            and all(part not in ("", ".", "..") for part in name.split("/")), "Unsafe explicit mapping name")
        canonical.add(name)
        result[raw_name] = row
    require(len(result) == len(canonical) == EXPECTED_LEAVES
        and sum(math.prod(row["shape"]) for row in result.values()) == EXPECTED_FULL_PARAMETERS,
        "Explicit mapping must cover all 51 full parameter leaves")
    return result


def logical_metadata(receipt):
    rows = receipt["parameter_metadata"]["logical_parameter_leaves"]
    mapping = explicit_metadata_key_map()
    require(len(rows) == 51 and len({row["path"] for row in rows}) == 51
        and {row["path"] for row in rows} == set(mapping),
        "Actual Orbax metadata raw keys must exactly match all 51 proven mapping entries")
    result = {}
    for row in rows:
        raw_name = row["path"]
        expected = mapping[raw_name]
        # Exact dictionary lookup preserves internal attention/value branches.
        # There is no startswith clipping, suffix heuristic, or missing-leaf merge.
        name = expected["canonical_path"]
        shape = row["shape"]
        require("lora" not in name.lower() and name not in result, "Duplicate or LoRA canonical key")
        require(isinstance(shape, list) and all(type(d) is int and d > 0 for d in shape)
            and shape == expected["shape"] and row["dtype"] == expected["source_dtype"],
            "Actual Orbax shape/dtype differs from the proven exact canonical mapping")
        result[name] = {"shape": shape, "dtype": row["dtype"], "elements": math.prod(shape),
            "raw_metadata_path": raw_name}
    require(len(result) == EXPECTED_LEAVES and sum(r["elements"] for r in result.values()) == EXPECTED_FULL_PARAMETERS,
        "Canonical metadata must describe exactly 51 full parameter leaves")
    return result


def checkpoint_proof(receipt_path, params_path=None, *, verify_files=False, verify_archive=False):
    receipt_path = Path(receipt_path).absolute()
    receipt_path.relative_to(TASK_ROOT / "official_checkpoint_v1")
    require(sha(receipt_path) == EXPECTED_CHECKPOINT_RECEIPT_SHA256, "Completed official receipt bytes differ")
    receipt = read_json(receipt_path)
    expected = {"complete": True, "status": "verified_official_robotwin_pi05_jax_checkpoint",
        "repository": "TianxingChen/RoboTwin2.0", "repository_type": "dataset", "revision": OFFICIAL_REVISION,
        "official_step": 29999, "archive_size_bytes": OFFICIAL_ARCHIVE_BYTES,
        "archive_sha256": OFFICIAL_ARCHIVE_SHA256, "weight_loader_class": "CheckpointWeightLoader",
        "train_state_extracted": False, "links_allowed": False}
    for field, value in expected.items():
        require(receipt.get(field) == value, "Official checkpoint receipt mismatch: " + field)
    require(receipt.get("source_path") == OFFICIAL_SOURCE_PATH, "Official archive source differs")
    require(receipt["parameter_metadata"]["logical_tree_leaf_count"] == EXPECTED_LEAVES, "Logical leaf count differs")
    metadata = logical_metadata(receipt)
    extracted = Path(receipt["extracted_root"]).absolute()
    extracted.relative_to(TASK_ROOT / "official_checkpoint_v1")
    selected_params = Path(receipt["weight_loader_params_path"]).absolute()
    selected_params.relative_to(extracted)
    require(selected_params == extracted / "RoboTwin-lerobot_v30-aloha_agilex-joint-0/29999/params",
        "Only the receipt's unique official 29999/params subtree is allowed")
    if params_path is not None:
        require(Path(params_path).absolute() == selected_params, "CLI params path differs from official receipt")
    inventory = receipt["selected_file_inventory"]
    require(receipt["selected_file_count"] == len(inventory) and len(inventory) > 0, "Selected inventory count differs")
    seen, total = set(), 0
    for row in inventory:
        rel = PurePosixPath(row["relative_path"])
        require(not rel.is_absolute() and all(part not in ("", ".", "..") for part in rel.parts)
            and "\\" not in str(rel), "Unsafe selected inventory path")
        require(str(rel) not in seen, "Duplicate selected inventory path")
        seen.add(str(rel))
        target = extracted.joinpath(*rel.parts)
        target.relative_to(extracted)
        require(type(row["size_bytes"]) is int and row["size_bytes"] >= 0, "Invalid inventory byte count")
        require(isinstance(row["sha256"], str) and len(row["sha256"]) == 64, "Invalid inventory SHA")
        total += row["size_bytes"]
        require("train_state" not in rel.parts, "Optimizer/train_state extraction is forbidden")
        if verify_files:
            require(stat.S_ISDIR(extracted.lstat().st_mode), "Extracted root must be a real directory")
            for ancestor in target.parents:
                if ancestor == extracted.parent:
                    break
                require(stat.S_ISDIR(ancestor.lstat().st_mode), "Inventory parent must be a real directory")
            require(stat.S_ISREG(target.lstat().st_mode) and target.stat().st_size == row["size_bytes"],
                "Selected checkpoint file type/size differs: " + str(target))
            require(sha(target) == row["sha256"], "Selected checkpoint file SHA differs: " + str(target))
    require(receipt["selected_bytes"] == total, "Selected inventory byte sum differs")
    require(any(selected_params.relative_to(extracted).as_posix() + "/" in name for name in seen),
        "Receipt inventory does not cover params")
    if verify_archive:
        archive = Path(receipt["archive_path"]).absolute()
        archive.relative_to(TASK_ROOT / "official_checkpoint_v1")
        require(stat.S_ISREG(archive.lstat().st_mode) and archive.stat().st_size == OFFICIAL_ARCHIVE_BYTES,
            "Official archive actual size/type differs")
        require(sha(archive) == OFFICIAL_ARCHIVE_SHA256, "Official archive actual SHA differs")
    return receipt, selected_params, metadata


def data_proof(preflight_path):
    report = read_json(preflight_path)
    for field in ("passed", "official_transform_roundtrip_passed", "official_loader_actual_batch_passed",
                  "all50_train", "all11098_targets_covered", "no_time_resampling", "padded_dimensions_in_official_loss"):
        require(report.get(field) is True, "Passed official full50 data preflight required: " + field)
    for field in ("terminal_padding", "dimension_metadata_is_loss_mask", "training_started", "model_restored",
                  "GPU_used", "physical_episode_run", "hardware_io", "pickle_deserialized"):
        require(report.get(field) is False, "Data preflight scope differs: " + field)
    for field, value in {"train_indices": list(range(50)), "validation_indices": [],
                        "actual_H16_train_length": 10348, "actual_positive_labels": 11098,
                        "source_manifest_sha256": MANIFEST_SHA256, "adapter_sha256": ADAPTER_SHA256,
                        "native_index_sha256": INDEX_SHA256, "norm_stats_sha256": NORM_SHA256}.items():
        require(report.get(field) == value, "Full50 data preflight contract differs: " + field)
    require(report["official_identity"] == {"commit": OPENPI_COMMIT, "clean": True},
        "Data preflight official source differs")
    require(len(report["official_100_endpoint_samples"]) == 100 and len(report["all50_HDF_time_label_readbacks"]) == 50,
        "Data preflight must contain actual 100 endpoints and 50 clock/label readbacks")
    batches = report["actual_batches"]
    require(sorted(row["workers"] for row in batches) == [0, 2], "Official loader workers0/2 actual checks required")
    require(all(row["action_shape"] == [8, 16, 32] and row["state_shape"] == [8, 32]
        and row["actual_first8_values_exact"] is True and row["official_loader_used"] is True
        and row["CPU_only"] is True for row in batches), "Official actual batch8 readback differs")
    require(report["inputs_sha_before"] == report["inputs_sha_after"], "Data preflight inputs changed")
    bindings = dict(report["inputs_sha_after"])
    for file, expected in ((ADAPTER_PATH, ADAPTER_SHA256), (DATA_ROOT / "native_index.json", INDEX_SHA256),
        (NORM_PATH, NORM_SHA256), (DATA_ROOT / "conversion_validation.json", CONVERSION_SHA256),
        (SOURCE_ROOT / "manifest.json", MANIFEST_SHA256)):
        require(sha(file) == expected, "Pinned full50 input bytes differ: " + str(file))
        bindings[str(file)] = expected
    for file, expected in bindings.items():
        require(sha(file) == expected, "Preflight input changed: " + file)
    index = read_json(DATA_ROOT / "native_index.json")
    require(index["split"] == {"train_indices": list(range(50)), "validation_indices": []},
        "Native index must contain all50/train and no validation")
    require(index["counts"]["train_full_chunks"] == 10348 and index["counts"]["validation_full_chunks"] == 0
        and index["action_horizon"] == 16 and index["model_dimension"] == 32 and index["physical_dimension"] == 24
        and len(index["episodes"]) == 50 and sorted(row["index"] for row in index["episodes"]) == list(range(50)),
        "Native index dimensions/counts differ")
    bindings[str(Path(preflight_path).absolute())] = sha(preflight_path)
    return report, bindings


def source_bindings(data_bindings, receipt_path):
    paths = [Path(__file__), Path(__file__).parent / "run_full50_finetune_161_v3_batch.py",
        Path(__file__).parent / "cpu_full50_cotrain_restore_v3_batch.py", AUDIT_PATH,
        BASE_ROOT / "openpi/scripts/train.py", KEY_MAP_PATH, ACTUAL_DIAGNOSTIC_PATH, Path(receipt_path).absolute()]
    require(sha(ACTUAL_DIAGNOSTIC_PATH) == ACTUAL_DIAGNOSTIC_SHA256, "Actual CPU canonical restore diagnostic bytes differ")
    return dict(data_bindings, **{str(path): sha(path) for path in paths})


def bind_environment():
    os.environ["OPENPI_DATA_HOME"] = str(BASE_ROOT / "weights")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_DISABLED"] = "true"



def verify_restored_flat_tree(flat, target, metadata):
    """Verify actual arrays. No missing/random leaf merge is allowed."""
    import numpy as np
    require(set(flat) == set(target) == set(metadata), "Exact full tree coverage failed; missing/random leaves forbidden")
    require(len(flat) == EXPECTED_LEAVES, "Actual restored leaf count differs")
    rows = []
    for name in sorted(flat):
        value = flat[name]
        require("lora" not in name.lower() and list(value.shape) == list(target[name].shape) == metadata[name]["shape"],
            "Actual full shape/noLoRA check failed: " + name)
        require(str(value.dtype) == metadata[name]["dtype"], "Actual source dtype differs from Orbax metadata: " + name)
        # Chunking avoids a full-size boolean tensor for the largest embedding.
        array = value.reshape(-1)
        require(all(bool(np.isfinite(array[start:start + 1048576]).all())
            for start in range(0, array.size, 1048576)), "Nonfinite source parameter: " + name)
        rows.append({"name": name, "shape": list(value.shape), "source_dtype": str(value.dtype),
            "reference_dtype": str(target[name].dtype), "elements": int(value.size), "source_finite": True,
            "raw_metadata_path": metadata[name]["raw_metadata_path"]})
    require(sum(row["elements"] for row in rows) == EXPECTED_FULL_PARAMETERS, "Actual full element count differs")
    return rows


_RUNTIME = None
def runtime_classes():
    global _RUNTIME
    if _RUNTIME is not None:
        return _RUNTIME
    import flax.traverse_util
    import numpy as np
    from openpi.models import model as model_api
    from openpi.shared import array_typing as at
    from openpi.training import config as official_config, weight_loaders
    require(sha(ADAPTER_PATH) == ADAPTER_SHA256, "Native adapter bytes differ")
    sys.path.insert(0, str(ADAPTER_PATH.parent))
    adapter = importlib.import_module("data_adapter_rgb50_v1")
    require(Path(adapter.__file__).resolve() == ADAPTER_PATH.resolve(), "Unexpected native adapter import path")

    @dataclasses.dataclass(frozen=True)
    class RealmanFull50DataConfig(official_config.DataConfigFactory):
        repo_id: str = REPO_ID
        assets: official_config.AssetsConfig = dataclasses.field(default_factory=lambda:
            official_config.AssetsConfig(assets_dir=str(DATA_ROOT / "assets"), asset_id=ASSET_ID))
        def create(self, assets_dirs, model_config):
            base = self.create_base_config(assets_dirs, model_config)
            require(base.norm_stats is not None, "New full50 native normalization is mandatory")
            return dataclasses.replace(base, data_transforms=adapter.make_data_transforms(),
                model_transforms=official_config.ModelTransformFactory()(model_config), prompt_from_task=False)

    @dataclasses.dataclass(frozen=True)
    class StrictOfficialCotrainWeightLoader(weight_loaders.CheckpointWeightLoader):
        receipt_path: str
        readback_path: str
        def load(self, reference):
            receipt, params, metadata = checkpoint_proof(self.receipt_path, self.params_path,
                verify_files=True, verify_archive=True)
            restored = model_api.restore_params(str(params), restore_type=np.ndarray)
            flat = flax.traverse_util.flatten_dict(restored, sep="/")
            target = flax.traverse_util.flatten_dict(reference, sep="/")
            rows = verify_restored_flat_tree(flat, target, metadata)
            # Exact coverage has already been checked. No missing key can be randomized.
            merged = weight_loaders._merge_params(restored, reference, missing_regex="a^")
            at.check_pytree_equality(expected=reference, got=merged, check_shapes=True, check_dtypes=True)
            cast_flat = flax.traverse_util.flatten_dict(merged, sep="/")
            for row in rows:
                value = cast_flat[row["name"]]
                array = value.reshape(-1)
                require(all(bool(np.isfinite(array[start:start + 1048576]).all())
                    for start in range(0, array.size, 1048576)), "Nonfinite parameter after official dtype cast")
                row.update({"loaded_dtype": str(value.dtype), "loaded_finite": True,
                    "official_reference_dtype_cast": row["source_dtype"] != str(value.dtype)})
            proof = {"passed": True, "actual_restore_performed": True, "exact_tree_keys_shapes_finite_verified": True,
                "logical_parameter_leaves": 51, "parameter_elements": EXPECTED_FULL_PARAMETERS,
                "source_repository": receipt["repository"], "source_revision": receipt["revision"],
                "source_archive_sha256": receipt["archive_sha256"], "archive_actual_sha_verified": True,
                "selected_file_actual_sha_verified": True, "receipt_path": self.receipt_path,
                "receipt_sha256": sha(self.receipt_path), "params_path": str(params), "source_step": 29999,
                "initialization_scope": "params_only", "optimizer_state_loaded": False,
                "random_replacement_leaves": 0, "missing_regex": "a^", "parameter_rows": rows,
                "metadata_mapping_scope": "explicit_51_actual_official_restore_params_pairs",
                "metadata_key_map_path": str(KEY_MAP_PATH), "metadata_key_map_sha256": sha(KEY_MAP_PATH),
                "actual_canonical_diagnostic_path": str(ACTUAL_DIAGNOSTIC_PATH),
                "actual_canonical_diagnostic_sha256": ACTUAL_DIAGNOSTIC_SHA256}
            write_new(self.readback_path, proof)
            return merged

    # Stable import names allow framework introspection/serialization without changing official code.
    for cls in (RealmanFull50DataConfig, StrictOfficialCotrainWeightLoader):
        cls.__qualname__ = cls.__name__
        globals()[cls.__name__] = cls
    _RUNTIME = (RealmanFull50DataConfig, StrictOfficialCotrainWeightLoader, adapter)
    return _RUNTIME


def make_config(exp_name, *, steps, batch_size, checkpoint_receipt, params_path=None, weight_readback):
    training_contract(steps, batch_size)
    receipt, params, metadata = checkpoint_proof(checkpoint_receipt, params_path)
    from openpi.models.pi0_config import Pi0Config
    from openpi.training import config as official_config, optimizer
    DataConfig, Loader, _ = runtime_classes()
    model = Pi0Config(pi05=True, dtype="bfloat16", action_dim=32, action_horizon=16,
        paligemma_variant="gemma_2b", action_expert_variant="gemma_300m")
    return official_config.TrainConfig(name=NAME, exp_name=exp_name, model=model,
        weight_loader=Loader(str(params), str(checkpoint_receipt), str(weight_readback)), data=DataConfig(),
        freeze_filter=model.get_freeze_filter(), ema_decay=None,
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=1000, peak_lr=1e-4,
            decay_steps=100000, decay_lr=1e-5), optimizer=optimizer.AdamW(),
        assets_base_dir=str(TASK_ROOT / "training_assets"),
        checkpoint_base_dir=str(TASK_ROOT / "checkpoints"), seed=42, batch_size=batch_size, num_workers=2,
        num_train_steps=steps, log_interval=1 if steps == 5 else 50, save_interval=20000, keep_period=20000,
        overwrite=False, resume=False, wandb_enabled=False, fsdp_devices=4,
        policy_metadata={"physical_action_dimension": 24, "model_action_dimension": 32, "action_horizon": 16,
            "physical_action_order": ["left_arm_rad6", "left_primary_SDK6", "right_arm_rad6", "right_primary_SDK6"],
            "arm_delta_dimensions": list(range(6)) + list(range(12, 18)), "hand_absolute": True, "sdk_scale": 1000.,
            "three_real_RGB": True, "hardware_calibrated": False, "sample_order": "native_positive_clock",
            "padded_dimensions_in_official_loss": True, "all50_train": True, "validation_split": None,
            "initialization": "official_RoboTwin_pi05_cotrain_29999_params_only_fresh_optimizer",
            "source_revision": OFFICIAL_REVISION, "source_archive_sha256": OFFICIAL_ARCHIVE_SHA256,
            "parameter_update_scope": "all_nnx_Param", "lora_enabled": False})


def validate_runtime_config(cfg, steps, batch_size):
    training_contract(steps, batch_size)
    require(cfg.batch_size == batch_size, "Actual global batch differs from explicit requested profile")
    import flax.nnx as nnx
    require(cfg.freeze_filter is nnx.Nothing and cfg.model.pi05 is True
        and cfg.model.paligemma_variant == "gemma_2b" and cfg.model.action_expert_variant == "gemma_300m"
        and cfg.model.action_dim == 32 and cfg.model.action_horizon == 16 and cfg.batch_size == batch_size
        and cfg.fsdp_devices == 4 and cfg.num_train_steps == steps and cfg.ema_decay is None
        and cfg.save_interval == 20000 and cfg.keep_period == 20000
        and cfg.resume is False and cfg.overwrite is False, "Full50 full-parameter runtime configuration differs")


def install_native_dataset_factory():
    from openpi.training import data_loader
    _, _, adapter = runtime_classes()
    original = data_loader.create_torch_dataset
    def only_full50(data_config, action_horizon, model_config):
        if data_config.repo_id in adapter.REPO_IDS:
            require(action_horizon == 16 and model_config.action_dim == 32, "Native H16/action32 contract differs")
            return adapter.RealmanNativeDataset(DATA_ROOT, split="train", action_horizon=16, verify_hashes=True,
                source_override=SOURCE_ROOT, xpolicylab_parent=PROJECT_ROOT / "RoboTwin")
        return original(data_config, action_horizon, model_config)
    data_loader.create_torch_dataset = only_full50
    return original
