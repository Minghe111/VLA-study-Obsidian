"""Same-input local retention metrics; eight time points are not task SR."""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path

import torch


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, report):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".staging")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _finite_pair(actual, reference):
    actual, reference = actual.detach().float().cpu(), reference.detach().float().cpu()
    if actual.shape != reference.shape or not torch.isfinite(actual).all() or not torch.isfinite(reference).all():
        raise ValueError("Retention tensors must have identical shapes and finite values")
    return actual, reference


def tensor_error(actual, reference):
    actual, reference = _finite_pair(actual, reference)
    if not actual.numel():
        raise ValueError("Retention metrics require at least one element")
    difference = actual - reference
    return {
        "elements": difference.numel(),
        "rms": float(difference.square().mean().sqrt()),
        "max_abs": float(difference.abs().max()),
        "exact": bool(torch.equal(actual, reference)),
    }


def action_metrics(actions, reference_actions, model_actions, reference_model_actions):
    """Keep normalized physical dimensions and SDK units as separate metrics."""
    if tuple(actions.shape) != (1, 16, 24) or tuple(model_actions.shape) != (1, 16, 32):
        raise ValueError("RM65-B retention requires SDK[1,16,24] and normalized[1,16,32]")
    actions, reference_actions = _finite_pair(actions, reference_actions)
    model_actions, reference_model_actions = _finite_pair(model_actions, reference_model_actions)
    # Existing RM65-B transforms pad the 24 physical dimensions to model dim32.
    # Padded dimensions 24:32 are recorded in the snapshot but excluded here.
    arms = list(range(0, 6)) + list(range(12, 18))
    hands = list(range(6, 12)) + list(range(18, 24))
    return {
        "normalized_physical24": tensor_error(model_actions[..., :24], reference_model_actions[..., :24]),
        "sdk_arm_rad": tensor_error(actions[..., arms], reference_actions[..., arms]),
        "sdk_hand_units": tensor_error(actions[..., hands], reference_actions[..., hands]),
        "sdk_actions_exact": bool(torch.equal(actions, reference_actions)),
        "normalized_internal32_exact": bool(torch.equal(model_actions, reference_model_actions)),
    }


def kv_metrics(cache, reference_cache, mask, reference_mask):
    """Pool valid-token errors over every language layer and both K/V tensors."""
    mask, reference_mask = mask.detach().bool().cpu(), reference_mask.detach().bool().cpu()
    if mask.ndim != 2 or not torch.equal(mask, reference_mask) or not mask.any():
        raise ValueError("K/V comparison requires equal, nonempty [B,P] prefix masks")
    if not cache or len(cache) != len(reference_cache):
        raise ValueError("K/V comparison requires all matching language layers")
    numerator = denominator = 0.0
    elements = 0
    maximum = 0.0
    all_exact = True
    valid_exact = True
    layers = []
    for index, (current_layer, target_layer) in enumerate(zip(cache, reference_cache, strict=True)):
        if len(current_layer) != 2 or len(target_layer) != 2:
            raise ValueError("Each language cache layer must contain K and V")
        layer_metrics = {"layer": index}
        for label, current, target in zip(("key", "value"), current_layer, target_layer, strict=True):
            current, target = _finite_pair(current, target)
            if current.ndim != 4 or tuple(current.shape[:2]) != tuple(mask.shape):
                raise ValueError("K/V tensors must have [B,P,heads,head_dim] shape")
            selected = mask[..., None, None].expand_as(current)
            actual_valid, target_valid = current[selected], target[selected]
            difference = actual_valid - target_valid
            squared_error = float(difference.double().square().sum())
            squared_target = float(target_valid.double().square().sum())
            count = difference.numel()
            numerator += squared_error
            denominator += squared_target
            elements += count
            maximum = max(maximum, float(difference.abs().max()))
            all_exact = all_exact and torch.equal(current, target)
            valid_exact = valid_exact and torch.equal(actual_valid, target_valid)
            layer_metrics[label] = {
                "elements": count,
                "rms": (squared_error / count) ** 0.5,
                "relative_rms": (squared_error / max(squared_target, 1e-24)) ** 0.5,
                "max_abs": float(difference.abs().max()),
                "valid_exact": bool(torch.equal(actual_valid, target_valid)),
            }
        layers.append(layer_metrics)
    return {
        "definition": "sqrt(sum_valid_all_layers_KV(error^2) / sum_valid_all_layers_KV(teacher^2)); denominator floor=1e-24",
        "layers": layers,
        "valid_elements": elements,
        "rms": (numerator / elements) ** 0.5,
        "relative_rms": (numerator / max(denominator, 1e-24)) ** 0.5,
        "max_abs": maximum,
        "valid_exact": bool(valid_exact),
        "all_positions_exact": bool(all_exact),
    }


def evaluate_retention(actor, manifest_path, output_dir, baseline_path=None):
    """Save a disabled-adapter baseline, or compare configured student to it.

    Baseline output is ``snapshot.pt``. Pass that file as ``baseline_path`` to
    subsequent calls. No simulator step, parameter update or generation repair
    occurs. Same observation/instruction/noise hashes are mandatory.
    """
    import numpy as np
    from rlinf.data.datasets.recap.hierarchical_dataset import load_observation
    from rlinf.models.embodiment.openpi.modules import model as model_lib
    from rlinf.models.embodiment.openpi.recap_lora import legacy_prefix_context

    manifest_path, output_dir = Path(manifest_path).resolve(), Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text())
    cases = manifest.get("cases", [])
    if manifest.get("format") != "rm65b_k1_retention_anchor_v1" or len(cases) != 8 or len({item["case_id"] for item in cases}) != 8:
        raise ValueError("Retention requires the audited eight-case RM65-B manifest")
    manifest_sha = _sha256(manifest_path)
    baseline = torch.load(baseline_path, map_location="cpu", weights_only=True) if baseline_path else None
    if baseline is not None and (baseline.get("format") != "rm65b_fit_retention_v1" or baseline.get("manifest_sha256") != manifest_sha or baseline.get("all_adapters_disabled") is not True):
        raise ValueError("Baseline format or retention manifest differs")
    if baseline is not None and set(baseline.get("cases", {})) != {item["case_id"] for item in cases}:
        raise ValueError("Baseline cases differ from the audited manifest")
    model = actor.model
    prior_training = model.training
    report = {
        "format": "rm65b_fit_retention_v1",
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "teacher_mode": "all_adapters_disabled" if baseline is None else "stored_all_adapters_disabled_baseline",
        "student_mode": "not_evaluated_baseline_only" if baseline is None else "configured_stage_adapters_enabled_MEM_bypassed",
        "training_stage": getattr(model, "recap_training_stage", "legacy"),
        "scope": "Eight fixed observation time points from one expert episode; same three RGB cameras, state, original instruction and stored noise. Local retention only, not rollout SR.",
        "source_episode": manifest.get("source_episode"),
        "control_contract": {"action_dim": 24, "camera_count": 3, "nominal_control_seconds": 0.06},
        "thresholds": {"normalized_physical24_rms_max": 0.02, "masked_all_layer_kv_relative_rms_max": 0.05, "scope": "engineering local gates; not validated task-level safety or success thresholds"},
        "cases": [],
        "snapshot_path": str(output_dir / "snapshot.pt"),
    }
    snapshot = {"format": report["format"], "manifest_sha256": manifest_sha, "all_adapters_disabled": baseline is None, "cases": {}}
    model.eval()
    try:
        for item in cases:
            observation_path = manifest_path.parent / item["observation"]["npz_path"]
            noise_path = manifest_path.parent / item["noise"]["npz_path"]
            if _sha256(observation_path) != item["observation_sha256"] or _sha256(noise_path) != item["noise_sha256"]:
                raise ValueError("Retention observation/noise differs from the manifest")
            native = load_observation({"npz_path": str(observation_path)})
            with np.load(noise_path, allow_pickle=False) as archive:
                noise_array = archive[item["noise"].get("key", "noise")].copy()
            if noise_array.dtype != np.float32 or noise_array.shape != (1, 16, 32) or not np.isfinite(noise_array).all():
                raise ValueError("Stored flow noise must be finite float32[1,16,32]")
            noise = torch.as_tensor(noise_array, device=model.device)
            observation = model.env_obs_to_observation(actor.env_io(native, item["instruction"]))
            prepared = model_lib.preprocess_observation(observation, train=False)
            prepared = model_lib._observation_to_dtype(prepared, model.embed_dtype)
            with torch.no_grad(), legacy_prefix_context(model, disable_adapters=baseline is None):
                prefix_output, mask, cache = model.build_prefix_cache(prepared)
                del prefix_output
                actions, extra = model._predict_eval(observation, noise=noise)
            internal = extra.get("model_actions")
            if internal is None:
                raise RuntimeError("The retained sampler must expose its normalized model_actions")
            if tuple(actions.shape) != (1, 16, 24) or tuple(internal.shape) != (1, 16, 32) or not torch.isfinite(actions).all() or not torch.isfinite(internal).all():
                raise RuntimeError("Invalid retained RM65-B action chunk")
            current = {
                "observation_sha256": item["observation_sha256"],
                "instruction": item["instruction"],
                "noise_sha256": item["noise_sha256"],
                "mask": mask.detach().cpu().clone(),
                "tokenized_prompt": observation.tokenized_prompt.detach().cpu().clone(),
                "normalized_state": observation.state.detach().cpu().clone(),
                "cache": [[tensor.detach().cpu().clone() for tensor in layer] for layer in cache],
                "actions": actions.detach().cpu().clone(),
                "model_actions": internal.detach().cpu().clone(),
            }
            snapshot["cases"][item["case_id"]] = current
            case_report = {
                "case_id": item["case_id"], "control_step": item["source"]["control_step"],
                "sdk_action_shape": list(actions.shape), "normalized_internal_shape": list(internal.shape),
                "valid_prefix_tokens": int(current["mask"].sum()),
            }
            if baseline is not None:
                target = baseline["cases"][item["case_id"]]
                if any(current[key] != target[key] for key in ("observation_sha256", "instruction", "noise_sha256")):
                    raise ValueError("Retention source, instruction or noise changed")
                context_exact = all(torch.equal(current[key], target[key]) for key in ("mask", "tokenized_prompt", "normalized_state"))
                if not context_exact:
                    raise ValueError("Retention prefix mask or original prompt changed")
                action_error = action_metrics(current["actions"], target["actions"], current["model_actions"], target["model_actions"])
                cache_error = kv_metrics(current["cache"], target["cache"], current["mask"], target["mask"])
                case_report.update(context_exact=context_exact, action_metrics=action_error, kv_metrics=cache_error)
                case_report["passed"] = action_error["normalized_physical24"]["rms"] <= 0.02 and cache_error["relative_rms"] <= 0.05
            report["cases"].append(case_report)
            _write_json(output_dir / "report.json", report)
            del cache, mask, actions, extra, internal, observation, prepared, noise, noise_array, native
            gc.collect()
    finally:
        model.train(prior_training)
    torch.save(snapshot, output_dir / "snapshot.pt")
    if baseline is not None:
        report["passed"] = all(case["passed"] for case in report["cases"])
        report["summary"] = {
            "count": len(cases), "passed": sum(case["passed"] for case in report["cases"]),
            "max_case_normalized_physical24_rms": max(case["action_metrics"]["normalized_physical24"]["rms"] for case in report["cases"]),
            "max_case_masked_all_layer_kv_relative_rms": max(case["kv_metrics"]["relative_rms"] for case in report["cases"]),
            "all_sdk_actions_exact": all(case["action_metrics"]["sdk_actions_exact"] for case in report["cases"]),
            "all_kv_positions_exact": all(case["kv_metrics"]["all_positions_exact"] for case in report["cases"]),
        }
    else:
        report["baseline_created"] = True
    _write_json(output_dir / "report.json", report)
    return report
