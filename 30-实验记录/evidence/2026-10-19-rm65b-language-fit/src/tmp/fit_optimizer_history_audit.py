"""Read-only CPU audit of optimizer moments and existing retention receipts.

Loads trainer_state.pt only, never full_weights.pt or a model.  Optimizer moment
history is evidence that a parameter received gradients; it is not a receipt
for the latest individual backward call.  Outputs are constrained to /tmp.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import torch


DEFAULT_ROOT = Path("/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/recap_memory_20261017/experiments/protocol_fit_20261019_v1")
ORIGINAL_MANIFEST = Path("/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/recap_memory_20261017/experiments/protocol_sft_20261018_v1/data/retention_anchor/manifest.json")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def scope_receipt(path):
    path = Path(path)
    if not path.exists():
        return {"status": "not_available", "manifest_path": str(path)}
    data = read_json(path)
    return {
        "status": "manifest_present_not_a_new_model_evaluation",
        "manifest_path": str(path), "manifest_sha256": sha256(path),
        "split": data.get("split"), "purpose": data.get("purpose"),
        "source_episode": data.get("source_episode"),
        "count": len(data.get("cases", [])),
        "control_steps": [case["source"]["control_step"] for case in data.get("cases", [])],
        "case_ids": [case["case_id"] for case in data.get("cases", [])],
        "limitations": "Eight temporal samples from one episode; disjoint times are not independent episodes or task SR.",
    }


def retention_receipt(path):
    path = Path(path)
    if not path.exists():
        return {"status": "not_evaluated_or_report_not_yet_written", "report_path": str(path)}
    data = read_json(path)
    result = {
        "status": "reported", "report_path": str(path), "report_sha256": sha256(path),
        "manifest_path": data.get("manifest_path"), "manifest_sha256": data.get("manifest_sha256"),
        "scope": data.get("scope"), "summary": data.get("summary"),
        "passed": data.get("passed"), "thresholds": data.get("thresholds"),
    }
    cases = data.get("cases", [])
    if cases and all("action_metrics" in case for case in cases):
        result["sdk_max_abs_separate_units"] = {
            key: max(case["action_metrics"][key]["max_abs"] for case in cases)
            for key in ("sdk_arm_rad", "sdk_hand_units")
        }
    return result


def audit_run(run_dir):
    report_path = run_dir / "report.json"
    data = read_json(report_path)
    trainer_path = run_dir / "checkpoint/trainer_state.pt"
    result = {"name": run_dir.name, "run_status": data.get("status"), "report_path": str(report_path)}
    if data.get("status") != "completed" or not trainer_path.exists():
        result["audit_status"] = "skipped_incomplete_run"
        return result
    trainer_hash = sha256(trainer_path)
    trainer = torch.load(trainer_path, map_location="cpu", weights_only=False)
    manifest = trainer["training_manifest"]
    parameters = manifest["trainable_parameters"]
    groups = trainer["optimizer"]["param_groups"]
    ids = [identifier for group in groups for identifier in group["params"]]
    if len(parameters) != len(ids) or len(set(ids)) != len(ids):
        raise ValueError("Optimizer and ordered training manifest differ")
    # The experiment constructs AdamW from model.parameters() filtered by
    # requires_grad, the same enumeration order used for the saved manifest.
    moments = []
    for metadata, identifier in zip(parameters, ids, strict=True):
        state = trainer["optimizer"]["state"][identifier]
        first, second = state["exp_avg"], state["exp_avg_sq"]
        if list(first.shape) != metadata["shape"] or list(second.shape) != metadata["shape"]:
            raise ValueError("Optimizer moment shape does not match the named parameter")
        match = re.fullmatch(r"llm\.layers\.(\d+)\.attn\.(q_proj|k_proj|v_proj|o_proj)\.0\.lora_([AB])", metadata["name"])
        if not match:
            raise ValueError("Unexpected trainable tensor outside language attention LoRA")
        moments.append({
            "name": metadata["name"], "layer": int(match[1]), "projection": match[2], "matrix": match[3],
            "shape": metadata["shape"], "numel": metadata["numel"], "step": int(state["step"]),
            "exp_avg_l2": float(first.norm()), "exp_avg_sq_sum": float(second.sum()),
            "finite": bool(torch.isfinite(first).all() and torch.isfinite(second).all()),
            "gradient_history_nonzero": bool(torch.count_nonzero(first) or torch.count_nonzero(second)),
        })
    stage = manifest["training_stage"]
    expected = (144, 3_686_400) if stage == "language_warmup" else (4, 131_072) if stage == "protocol" else None
    actual = (len(moments), sum(item["numel"] for item in moments))
    if actual != expected:
        raise ValueError(f"Stage tensor count/numel differs: {stage} {actual} vs {expected}")
    if sha256(trainer_path) != trainer_hash:
        raise RuntimeError("Optimizer state changed while being read")
    result.update({
        "audit_status": "completed_CPU_read_only", "trainer_state_path": str(trainer_path),
        "trainer_state_sha256": trainer_hash, "training_stage": stage,
        "global_step": trainer["global_step"], "runtime_mem_enabled": manifest["runtime_mem_enabled"],
        "trainable_tensors": actual[0], "trainable_numel": actual[1],
        "layers": sorted({item["layer"] for item in moments}),
        "all_moments_finite": all(item["finite"] for item in moments),
        "nonzero_gradient_history_tensors": sum(item["gradient_history_nonzero"] for item in moments),
        "early_late_moments": [item for item in moments if item["layer"] in {0, 17}],
        "all_moments": moments,
        "gradient_receipt_from_training_report": data.get("gradient_audit"),
        "frozen_parameters_exact_reported": data.get("frozen_parameters_exact"),
        "changed_trainable_tensors_reported": data.get("changed_trainable_tensors"),
        "evaluation": data.get("evaluation"), "memory": data.get("memory"),
        "retention_original8": retention_receipt(run_dir / "retention/report.json"),
        "retention_heldout8": retention_receipt(run_dir / "retention_holdout/report.json"),
        "sampler_before": data.get("sampler_before"), "sampler_after": data.get("sampler_after"),
    })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=Path("/tmp/fit_optimizer_history_audit.json"))
    parser.add_argument("--runs", nargs="*", help="Optional run directory names; default all existing reports")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.parent != Path("/tmp"):
        parser.error("Audit output must be a direct file under /tmp")
    comparison = args.root.resolve() / "reports/comparison_v1"
    run_dirs = [comparison / name for name in args.runs] if args.runs else sorted(path.parent for path in comparison.glob("*/report.json") if (path.parent / "checkpoint/trainer_state.pt").exists())
    original = scope_receipt(ORIGINAL_MANIFEST)
    heldout = scope_receipt(args.root / "data/retention_holdout/manifest.json")
    receipt = {
        "format": "rm65b_optimizer_history_CPU_audit_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_path": str(Path(__file__).resolve()), "script_sha256": sha256(__file__),
        "loads_model": False, "loads_full_weights": False, "uses_GPU": False, "writes_only_tmp": True,
        "original8_scope": original, "heldout8_scope": heldout,
        "timepoints_disjoint": not bool(set(original.get("control_steps", [])) & set(heldout.get("control_steps", []))),
        "same_source_episode": original.get("source_episode") == heldout.get("source_episode"),
        "interpretation": [
            "Adam moments measure gradient history, not a particular current backward call.",
            "B-zero initialization normally gives zero first-step A gradient and nonzero B gradient.",
            "Frozen-weight equality and gradient counts are existing run-report receipts, not recomputed full-model checks.",
            "Original eight states can be used for KV anchoring; heldout eight are evaluation-only times in the same expert episode.",
            "Neither local retention gate nor text fitting measures robot success or visual grounding generalization.",
        ],
        "runs": [audit_run(path) for path in run_dirs],
    }
    output.write_text(json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), "audited_runs": len(receipt["runs"]), "nonzero_moments": {run["name"]: run.get("nonzero_gradient_history_tensors") for run in receipt["runs"]}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
