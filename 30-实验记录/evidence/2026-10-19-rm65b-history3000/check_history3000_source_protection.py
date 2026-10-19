"""Read-only source/asset protection audit. Never imports model or simulator code."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_STAGE = Path("/tmp/rm65b_protocol_history_3000_20261019_v1")
DEFAULT_NEW = Path("/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/recap_memory_20261017/experiments/protocol_history_3000_20261019_v1")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_file(path, expected, source, expected_bytes=None):
    path = Path(path)
    receipt = {"path": str(path), "expected_sha256": expected, "expected_sha_source": source}
    try:
        receipt.update(bytes=path.stat().st_size, actual_sha256=sha(path))
        receipt["sha256_matches"] = receipt["actual_sha256"] == expected
        receipt["passed"] = receipt["sha256_matches"] and (expected_bytes is None or receipt["bytes"] == expected_bytes)
    except OSError as exc:
        receipt.update(passed=False, error=f"{type(exc).__name__}: {exc}")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, default=DEFAULT_STAGE)
    parser.add_argument("--new-root", type=Path, default=DEFAULT_NEW)
    parser.add_argument("--output", type=Path, default=Path("/tmp/history3000_source_protection.json"))
    args = parser.parse_args(argv)
    started = time.monotonic()
    snapshot_path = args.stage / "source_snapshot_manifest.json"
    snapshot = json.loads(snapshot_path.read_text())
    source = Path(snapshot["source"])
    report = {"schema": "history3000_source_protection_v1", "checked_at": datetime.now(UTC).isoformat(),
              "scope": "read-only file hashes, no torch/model/tokenizer/renderer/simulator execution",
              "source": str(source), "new_root": str(args.new_root),
              "snapshot_manifest": {"path": str(snapshot_path), "sha256": sha(snapshot_path), "original_scope": snapshot["scope"]},
              "training_started_by_audit": False, "source_mutated_by_audit": False}
    source_checks = [check_file(source / relative, expected, f"{snapshot_path}:source_files[{relative}]")
                     for relative, expected in snapshot["source_files"].items()]
    mismatches = [entry for entry in source_checks if not entry["passed"]]
    symlinks = []
    for relative, expected_target in snapshot["links"].items():
        path = source / relative
        try:
            actual = str(path.resolve(strict=True))
            symlinks.append({"path": str(path), "expected_target": expected_target, "actual_target": actual,
                             "is_symlink": path.is_symlink(), "passed": path.is_symlink() and actual == str(Path(expected_target).resolve(strict=True))})
        except OSError as exc:
            symlinks.append({"path": str(path), "passed": False, "error": str(exc)})
    current = set()
    for directory in ("code", "configs", "scripts"):
        for root, directories, files in os.walk(source / directory, followlinks=False):
            directories[:] = [d for d in directories if d not in (".git", ".codegraph", "__pycache__") and not (Path(root) / d).is_symlink()]
            for name in files:
                path = Path(root) / name
                if not path.is_symlink() and path.suffix not in (".pyc", ".pyo"):
                    current.add(str(path.relative_to(source)))
    all_added = sorted(current - set(snapshot["source_files"]))
    generated_cache = [relative for relative in all_added if ".pytest_cache" in Path(relative).parts]
    added = [relative for relative in all_added if relative not in generated_cache]
    report["original_source"] = {"tracked_files": len(source_checks), "matched_files": len(source_checks) - len(mismatches),
                                 "tracked_bytes_read": sum(e.get("bytes", 0) for e in source_checks),
                                 "mismatches": mismatches, "symlinks": symlinks, "added_noncache_files": added,
                                 "untracked_generated_pytest_cache_files": generated_cache,
                                 "passed": not mismatches and all(e["passed"] for e in symlinks) and not added}

    c2200_report_path = source / "reports/continuation_v1/language_warmup_96_2200/report.json"
    c2200_report = json.loads(c2200_report_path.read_text())
    highlevel_report_path = source / "reports/postfit_validation_20261019_v1/highlevel_closedloop/report.json"
    highlevel_report = json.loads(highlevel_report_path.read_text())
    protected = {}
    for name in ("selected_weights", "checkpoint_version", "tokenizer", "norm_stats"):
        before = highlevel_report["source_files_before"][name]
        after = highlevel_report["source_files_after"][name]
        result = check_file(before["path"], before["sha256"], f"{highlevel_report_path}:source_files_before.{name}", before["bytes"])
        result["historical_before_after_receipt_agree"] = before == after
        result["passed"] &= before == after
        protected[name] = result
    checkpoint_dir = Path(c2200_report["checkpoint"])
    version = json.loads((checkpoint_dir / "version.json").read_text())
    report["c2200_metadata"] = {"checkpoint": str(checkpoint_dir), "version_global_step": version.get("global_step"),
                                "training_report_status": c2200_report["status"],
                                "training_report_formal_3000_training": c2200_report["formal_3000_training"],
                                "metadata_inspection_only_no_model_loaded": True,
                                "passed": version.get("global_step") == 2200}

    dataset_checks, observation_checks, rows = [], {}, []
    for item in c2200_report["data_audit"]["files"]:
        path = Path(item["path"])
        dataset_checks.append(check_file(path, item["sha256"], f"{c2200_report_path}:data_audit.files"))
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    for row in rows:
        for ref in row.get("evidence", {}).get("observation_refs", []):
            path, expected = ref["npz_path"], ref["npz_sha256"]
            if path in observation_checks and observation_checks[path]["expected_sha256"] != expected:
                raise ValueError("Original96 contains conflicting recorded observation hashes")
            if path not in observation_checks:
                observation_checks[path] = check_file(path, expected, "original fit96 evidence.observation_refs[].npz_sha256")
    modes = dict(Counter(row["mode"] for row in rows))
    report["original_fit96"] = {"dataset_files": dataset_checks, "rows": len(rows), "modes": modes,
                                "unique_actual_observation_files": len(observation_checks),
                                "observation_files": list(observation_checks.values()),
                                "passed": len(rows) == 96 and modes == {"READ": 32, "WRITE": 32, "SUBTASK": 32}
                                and all(e["passed"] for e in dataset_checks) and all(e["passed"] for e in observation_checks.values())}
    # Current capture's frozen v2 plan records pre-capture source hashes. This
    # supplements the original-code snapshot with the shared robot assets and
    # source expert commands; it does not read or change live simulator state.
    capture_path = args.new_root / "data/dynamic_capture_v2/manifest.json"
    if capture_path.is_file():
        capture = json.loads(capture_path.read_text())
        identities = capture["identities"]
        reference = capture["source_action_receipt"]
        protected["reference_expert_commands"] = check_file(reference["path"], reference["sha256"], f"{capture_path}:source_action_receipt")
        config = Path(capture["source_config"])
        # The retained robot location is explicitly tied to its frozen plan.
        # Avoid importing YAML/engineering code in this independent audit.
        asset = Path("/home/admin123/liminghe/vla-benchmark/RoboTwin/assets/embodiments/realman-rm65b-inspire-gen4")
        protected["retained_robot_urdf"] = check_file(asset / "robot_sapien.urdf", identities["command_urdf_sha256"], f"{capture_path}:identities.command_urdf_sha256")
        protected["retained_robot_source_manifest"] = check_file(asset / "source_manifest.json", identities["command_manifest_sha256"], f"{capture_path}:identities.command_manifest_sha256")
    report["protected_assets"] = protected
    report["explicit_limits"] = ["Original source protection covers all2196 manifest-tracked code/config/script files and2 symlinks, plus a noncache added-file scan.",
                                 "C2200 check hashes the entire6.968GB full_weights.pt without loading tensors; numerical inference capability is not retested.",
                                 "Original96 NPZ hashes are rechecked; this audit does not rerender images or rerun fitting.",
                                 "Text tokenizer and norm use recorded before/after SHA from the completed highlevel evaluation; no new tokenizer/model execution.",
                                 "FAST and value backbone assets are outside these selected receipts; this audit makes no full-cache or full-Gemma-asset unchanged claim.",
                                 "Live new capture may continue independently; its completion/SR is outside this source-protection result."]
    report["passed"] = report["original_source"]["passed"] and report["c2200_metadata"]["passed"] and report["original_fit96"]["passed"] and all(e["passed"] for e in protected.values())
    report["elapsed_s"] = time.monotonic() - started
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "source_matched": report["original_source"]["matched_files"],
                      "source_tracked": len(source_checks), "added_files": len(added), "source_mismatches": len(mismatches),
                      "fit96_observations": len(observation_checks), "protected_assets": {k: v["passed"] for k, v in protected.items()},
                      "report": str(args.output), "elapsed_s": report["elapsed_s"]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
