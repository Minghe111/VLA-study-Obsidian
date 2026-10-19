"""Build train-only 9 -> 96 fitting controls without inventing any labels.

This CPU audit recomputes actual tokenized-prefix/three-RGB input identities
from the original reviewed observations. It copies JSON records only; all NPZ
references stay absolute in the original permanent experiment directory.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import re
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "RLinf"))

from build_protocol_labels import (
    CAMERAS,
    _high_prefix,
    _load_frame,
    actor_input_digest,
)
from protocol_pilot import audit_records, check_output, read_records, sha256
from rlinf.models.embodiment.openpi.recap_tokenizer import RecapTokenizer

SEED = 20261019
MODES = ("READ", "WRITE", "SUBTASK")
WRITE_CLASSES = ("EMPTY", "WRITE", "KEEP", "CORRECT")
DEFAULT_SOURCE = Path(
    "/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/"
    "recap_memory_20261017/experiments/protocol_sft_20261018_v1/data/"
    "protocol_labels_reviewed_v1/train.shuffled.reviewed.jsonl"
)


def operation_class(record):
    if record["mode"] != "WRITE":
        return None
    operations = json.loads(record["target_text"])["operations"]
    if not operations:
        return "EMPTY"
    if len(operations) != 1:
        raise ValueError("This fitting control expects the reviewed one-slot labels")
    return operations[0]["op"]


def stratum(record):
    side = record["evaluation"]["target_side"]
    return (operation_class(record), side) if record["mode"] == "WRITE" else (side,)


def causal_label_audit(record):
    """Check the exact existing static-label rule, never assert motor success."""
    if record["split"] != "train" or record["supervision_kind"] != "verified_high_sft":
        raise ValueError("Only previously reviewed train labels are eligible")
    metadata, evidence, provenance = (
        record["metadata"], record["evidence"], record["provenance"]
    )
    if metadata.get("training_eligible") is not True:
        raise ValueError("Record is not training eligible")
    if metadata.get("motor_labels") is not False:
        raise ValueError("Static protocol labels cannot become motor labels")
    if provenance.get("review_status") != "approved":
        raise ValueError("Missing source review approval")
    if provenance.get("physical_behavior") != "static_hold_only_not_a_skill_demonstration":
        raise ValueError("Unexpected physical evidence scope")
    step = record["decision_step"]
    if type(step) is not int or step < 0:
        raise ValueError("Invalid decision time")
    if record["decision_physics_step"] != step * 15:
        raise ValueError("Decision clock no longer follows 15 x 4 ms = 60 ms")
    for timestamp in (record["observed_until_step"], evidence["observed_until_step"]):
        if type(timestamp) is not int or not 0 <= timestamp <= step:
            raise ValueError("Label uses future observations")
    context = record["context"]
    if context["visible_history"] != []:
        raise ValueError("Static controls cannot claim an executed subtask history")
    sides = re.findall(r"\b(left|right)\b", context["goal"].lower())
    side = record["evaluation"]["target_side"]
    if not sides or sides[0] != side:
        raise ValueError("Target side does not bind the first requested goal role")
    fact = f"The first requested sample is initially on the {side}."
    if evidence.get("candidate_fact") != fact:
        raise ValueError("Evidence fact disagrees with the target role")
    if set(evidence["does_not_establish"]) != {
        "grasp", "contact", "dwell", "verification", "return", "success"
    }:
        raise ValueError("Physical-success exclusions changed")
    if record["mode"] == "READ":
        expected = {"read_ids": ["sample_identity"]}
    elif record["mode"] == "SUBTASK":
        expected_text = f"Pick up and hold the sample initially on the {side}."
        if record["target_text"] != expected_text:
            raise ValueError("Subtask label differs from the reviewed static instruction")
        return
    else:
        selected = context["selected_memory"]
        if len(selected) != 1 or selected[0]["slot_id"] != "sample_identity":
            raise ValueError("Unexpected teacher memory slot")
        slot = selected[0]
        if slot["status"] != "active":
            operation = {"confirmed": True, "op": "WRITE", "slot_id": "sample_identity", "value": fact}
        elif slot["value"] != fact:
            if not record.get("counterfactual_memory_corruption"):
                raise ValueError("Wrong-role memory must be marked as authored counterfactual")
            operation = {"confirmed": True, "op": "CORRECT", "slot_id": "sample_identity", "value": fact}
        elif slot["confirmed"] is not True:
            if not record.get("counterfactual_memory_confidence"):
                raise ValueError("Confidence perturbation lacks counterfactual provenance")
            operation = {"confirmed": True, "op": "KEEP", "slot_id": "sample_identity"}
        else:
            operation = None
        expected = {"operations": [] if operation is None else [operation]}
    if json.loads(record["target_text"]) != expected:
        raise ValueError("Existing target is not justified by actor-visible teacher memory")


def unique_actor_records(records):
    unique = {}
    for record in records:
        digest = record["actor_input_digest"]
        previous = unique.get(digest)
        if previous is not None and previous["target_text"] != record["target_text"]:
            raise ValueError("Conflicting targets for identical actual actor input")
        unique.setdefault(digest, record)
    return list(unique.values())


def choose_subsets(records):
    """Balance actual unique inputs, with identical data for both LoRA arms."""
    unique = unique_actor_records(sorted(records, key=lambda row: row["sample_id"]))
    groups = defaultdict(list)
    for record in unique:
        groups[(record["mode"], stratum(record))].append(record)
    rng = random.Random(SEED)
    for key in sorted(groups, key=str):
        rng.shuffle(groups[key])

    # Three WRITE rows cannot cover four operations. The 9-row fitting test
    # deliberately covers all three nonempty operations; EMPTY enters 96/12.
    nine_keys = [
        ("READ", ("left",)), ("READ", ("right",)), ("READ", ("left",)),
        ("WRITE", ("WRITE", "left")),
        ("WRITE", ("KEEP", "right")),
        ("WRITE", ("CORRECT", "left")),
        ("SUBTASK", ("left",)), ("SUBTASK", ("right",)), ("SUBTASK", ("right",)),
    ]
    positions = Counter()
    nine = []
    for key in nine_keys:
        nine.append(groups[key][positions[key]])
        positions[key] += 1

    ninety_six = list(nine)
    selected = {row["actor_input_digest"] for row in nine}
    for key in sorted(groups, key=str):
        mode, _ = key
        quota = 4 if mode == "WRITE" else 16
        have = sum((row["mode"], stratum(row)) == key for row in ninety_six)
        for row in groups[key]:
            if have == quota:
                break
            if row["actor_input_digest"] not in selected:
                ninety_six.append(row)
                selected.add(row["actor_input_digest"])
                have += 1
        if have != quota:
            raise ValueError(f"Insufficient unique actual inputs for {key}: {have}/{quota}")

    quick = []
    quick_requirements = {
        "READ": [("left",), ("right",), ("left",), ("right",)],
        "WRITE": [(op, "left" if i % 2 == 0 else "right") for i, op in enumerate(WRITE_CLASSES)],
        "SUBTASK": [("left",), ("right",), ("left",), ("right",)],
    }
    quick_seen = set()
    for mode in MODES:
        for group in quick_requirements[mode]:
            row = next(row for row in ninety_six if row["mode"] == mode
                       and stratum(row) == group and row["actor_input_digest"] not in quick_seen)
            quick.append(row)
            quick_seen.add(row["actor_input_digest"])
    outputs = {"fit9.train": nine, "fit96.train": ninety_six, "fit96.quick12.train": quick}
    for index, rows in enumerate(outputs.values()):
        random.Random(SEED + index).shuffle(rows)
    return outputs


def check_isolation(train, heldout):
    details = {}
    for key in ("sample_id", "episode_id", "family_id", "seed", "actor_input_digest"):
        overlap = {row[key] for row in train} & {row[key] for row in heldout}
        if overlap:
            raise ValueError(f"Training/heldout {key} overlap")
        details[key] = {"overlap": 0}
    overlap = {row["observation"]["npz_path"] for row in train} & {
        row["observation"]["npz_path"] for row in heldout
    }
    if overlap:
        raise ValueError("Training/heldout observation paths overlap")
    details["observation_paths"] = {"overlap": 0}
    return details


def summarize(records, tokenizer):
    by_mode = {}
    for mode in MODES:
        rows = [row for row in records if row["mode"] == mode]
        lengths = [len(tokenizer.encode_target(row["target_text"])) for row in rows]
        by_mode[mode] = {
            "records": len(rows),
            "unique_actor_inputs": len({row["actor_input_digest"] for row in rows}),
            "unique_target_strings": len({row["target_text"] for row in rows}),
            "supervised_tokens_including_eos": sum(lengths),
            "target_tokens_min": min(lengths), "target_tokens_max": max(lengths),
            "sides": dict(Counter(row["evaluation"]["target_side"] for row in rows)),
        }
    return {
        "records": len(records), "by_mode": by_mode,
        "unique_actor_inputs": len({row["actor_input_digest"] for row in records}),
        "unique_target_strings": len({row["target_text"] for row in records}),
        "supervised_tokens_including_eos": sum(item["supervised_tokens_including_eos"] for item in by_mode.values()),
        "json_denominator": sum(row["mode"] in {"READ", "WRITE"} for row in records),
        "eos_denominator": len(records),
        "write_operations": dict(Counter(operation_class(row) for row in records if row["mode"] == "WRITE")),
        "families": dict(Counter(row["family_id"] for row in records)),
        "unique_observation_files": len({row["observation"]["npz_path"] for row in records}),
        "sample_ids": [row["sample_id"] for row in records],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/rm65b_protocol_sft_pilot.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "data/fit_subsets_v1")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/fit_subsets_audit.json")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    tokenizer = RecapTokenizer.from_local_paths(config["tokenizer"]["text_tokenizer_path"],
        max_prefix_tokens=config["tokenizer"]["max_prefix_tokens"], max_target_tokens=512)
    stats = json.loads(Path(config["model"]["openpi_data"]["norm_stats_path"]).read_text())["norm_stats"]["state"]
    records, loader_audit = audit_records([args.source], config, training=True)
    observation_cache, original_hashes, max_target_tokens = {}, {}, 0
    for row in records:
        causal_label_audit(row)
        path = Path(row["observation"]["npz_path"])
        if not path.is_absolute() or not path.is_file():
            raise ValueError("NPZ references must point to existing absolute original files")
        if path not in observation_cache:
            observation_cache[path] = _load_frame({"npz_path": str(path)}, stats)
        state, hashes = observation_cache[path]
        digest = actor_input_digest(tokenizer, row, state, hashes["camera_pixel_sha256"])
        if digest != row["actor_input_digest"]:
            raise ValueError("Actual tokenized-prefix/three-image digest changed")
        evidence_ref = row["evidence"]["observation_refs"][0]
        if evidence_ref["npz_sha256"] != hashes["npz_sha256"] or evidence_ref["camera_pixel_sha256"] != hashes["camera_pixel_sha256"]:
            raise ValueError("Source observation/camera evidence hash changed")
        prefix, target = tokenizer.encode_prefix(_high_prefix(row, state)), tokenizer.encode_target(row["target_text"])
        if int(target[-1]) != tokenizer.eos_token_id or len(target) > 96 or len(prefix) + len(target) > 2048:
            raise ValueError("Target does not actually include EOS within the generation budget")
        if tokenizer.decode_text(target) != row["target_text"]:
            raise ValueError("Actual target tokenizer round trip differs")
        max_target_tokens = max(max_target_tokens, len(target))
        original_hashes[str(path)] = hashes["npz_sha256"]

    heldout_paths = [args.source.parent / f"{split}.reviewed.jsonl" for split in ("val", "test")]
    heldout = read_records(heldout_paths)
    isolation = check_isolation(records, heldout)
    outputs = choose_subsets(records)
    args.output.mkdir(parents=True, exist_ok=True)
    subset_reports = {}
    for name, rows in outputs.items():
        clean = [{key: deepcopy(value) for key, value in row.items() if key != "_source"} for row in rows]
        path = args.output / f"{name}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in clean))
        verified, audit = audit_records([path], config, training=True)
        for row in verified:
            causal_label_audit(row)
            if not check_output(row, row["target_text"], True)["passed"]:
                raise ValueError("Reference answer fails the actual protocol evaluator")
        subset_reports[name] = {**summarize(clean, tokenizer), "path": str(path.resolve()), "sha256": sha256(path), "loader_audit": audit}
    nine_ids = set(subset_reports["fit9.train"]["sample_ids"])
    full_ids = set(subset_reports["fit96.train"]["sample_ids"])
    quick_ids = set(subset_reports["fit96.quick12.train"]["sample_ids"])
    assert nine_ids <= full_ids and quick_ids <= full_ids
    report = {
        "status": "passed_cpu_data_gate_no_training", "seed": SEED,
        "source": {"path": str(args.source.resolve()), "sha256": sha256(args.source), **summarize(records, tokenizer)},
        "subsets": subset_reports,
        "source_loader_audit": loader_audit,
        "heldout_isolation": isolation,
        "heldout_files_read_for_exclusion_only": [str(path.resolve()) for path in heldout_paths],
        "nine_is_subset_of_ninety_six": True, "quick_twelve_is_subset_of_ninety_six": True,
        "actor_input_digest_recomputed_for_all_source_records": True,
        "actor_input_digest_definition": "mode + actual tokenized normalized-state prefix + all three native RGB pixel SHA256s",
        "all_actual_targets_include_eos": True, "eos_token_id": tokenizer.eos_token_id,
        "max_actual_target_tokens_including_eos": max_target_tokens,
        "max_new_tokens": 96,
        "raw_observations_copied": False,
        "source_observation_sha256": original_hashes,
        "limitations": [
            "Nine rows allocate three WRITE rows, covering WRITE/KEEP/CORRECT; EMPTY is covered by 12/96 only.",
            "All fitting-evaluation rows are train-only; fitting is not heldout generalization.",
            "Labels retain prior agent visual review; this CPU audit is not a new human or visual review.",
            "Static hold observations and authored counterfactual memory only; no physical success, executed skill, or MEM video evidence.",
            "READ/WRITE JSON denominator only; EOS denominator covers all three modes.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "report": str(args.report),
        "subsets": {name: {key: value for key, value in summary.items() if key not in {"sample_ids", "loader_audit"}} for name, summary in subset_reports.items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
