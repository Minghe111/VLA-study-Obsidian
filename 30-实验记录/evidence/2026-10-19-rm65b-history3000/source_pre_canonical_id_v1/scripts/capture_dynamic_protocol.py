"""Fresh SDK24 reference attempts for causal protocol history, never motor labels.

The old single-bottle expert supplies commands only. New dual-bottle observations,
clocks and monitor outcomes are recorded from a new simulator episode. In
particular, neither a source expert phase nor its success is a new-scene label.
Default is plan-only. --execute is the sole simulator entry point; no actor,
optimizer, checkpoint or tokenizer is loaded by this program.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

DEFAULT_EXPERIMENT = Path(
    "/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/recap_memory_20261017/"
    "experiments/protocol_repair_20261019_v2"
)
DEFAULT_ACTIONS = Path(
    "/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/"
    "acob_fullsim_5090_rm65b_v1/data/expert_demo_v7/rows.jsonl"
)
CAMERAS = ("full_image", "left_wrist_image", "right_wrist_image")
PHYSICS_DT = 0.004
TICKS_PER_TARGET = 15
HIGH_WINDOW = 32
CONTRACT = {
    "physical_action_dim": 24,
    "action_order": ["left_arm6_rad", "left_hand6_sdk", "right_arm6_rad", "right_hand6_sdk"],
    "camera_keys": list(CAMERAS),
    "native_rgb_shape": [480, 640, 3],
    "physics_dt_s": PHYSICS_DT,
    "physics_steps_per_action": TICKS_PER_TARGET,
    "nominal_control_seconds": 0.060,
    "rpc_action_chunk": 16,
    "high_level_window_targets": HIGH_WINDOW,
    "policy_loaded": False,
    "training_enabled": False,
    "motor_supervision_eligible": False,
}


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def digest(value):
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(jsonable(data), indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def append_json(stream, data):
    stream.write(json.dumps(jsonable(data), allow_nan=False) + "\n")
    stream.flush()


def load_reference(path):
    """Use the contiguous full-clock prefix; never skip an intermediate target."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    full = []
    partial_seen = False
    for index, row in enumerate(rows):
        if row.get("index") != index:
            raise ValueError("Reference row indices must be consecutive from zero")
        a = np.asarray(row["actions"], dtype=np.float32)
        if a.shape != (1, 24) or not np.isfinite(a).all():
            raise ValueError("Reference action must be finite [1,24]")
        ticks = row["physics_ticks"]
        if ticks == TICKS_PER_TARGET and not partial_seen:
            full.append(a[0].copy())
        else:
            partial_seen = True
            if index != len(rows) - 1 or not 0 < ticks < TICKS_PER_TARGET:
                raise ValueError("Only the final reference row may have a partial clock")
    if len(full) != 244:
        raise ValueError(f"Expected this audited source's 244 full targets, got {len(full)}")
    actions = np.stack(full)
    receipt = {
        "path": str(Path(path).resolve()), "sha256": sha(path),
        "raw_rows": len(rows), "full_clock_rows": len(full),
        "excluded_partial_final_rows": len(rows) - len(full),
        "source_task": "single_bottle_adjust_bottle_upright_hold",
        "source_active_arm": "right", "observations_imported": False,
        "source_success_or_phase_reused_as_label": False,
        "action_units": "absolute_SDK24_arm_rad_hand_SDK",
        "scope": "reference_commands_only_not_dual_bottle_expert",
    }
    return actions, receipt


def make_families(train=16, val=6, test=6, seed_base=41000):
    """Split physical families before goals; each family owns both task orders."""
    if min(train, val, test) < 1 or max(train, val, test) > 32:
        raise ValueError("Every split needs 1..32 families")
    families = []
    for split, count in (("train", train), ("val", val), ("test", test)):
        for _ in range(count):
            i = len(families)
            # Offsets differ from the previous eight static layouts; all 28
            # tuples are explicit and distinct, independent of RNG/goal order.
            homes = [
                [-0.087 - 0.0013 * i, -0.112 + 0.003 * (i % 5)],
                [0.092 + 0.0011 * i, -0.119 + 0.004 * (i % 4)],
            ]
            families.append({
                "family_id": f"dynamic_reference_layout_{i:02d}", "split": split,
                "home_xy": homes, "model_ids": [4, 13] if i % 2 == 0 else [13, 4],
                "seed_left": seed_base + i * 100,
                "seed_right": seed_base + i * 100 + 10,
            })
    if len({digest(f["home_xy"]) for f in families}) != len(families):
        raise AssertionError("Physical family layouts overlap")
    return families


def episode_specs(families):
    episodes = []
    for family in families:
        for side in ("left", "right"):
            episodes.append({
                "episode_id": f"fresh_{family['family_id']}_{side}_{family['seed_' + side]}",
                "family_id": family["family_id"], "split": family["split"],
                "seed": family["seed_" + side], "target_side": side,
                "layout": {"home_xy": family["home_xy"], "model_ids": family["model_ids"],
                           "execution_order": ["sample_a", "sample_b"] if side == "left" else ["sample_b", "sample_a"]},
            })
    return episodes


def replay_actions(reference, initial_state, side, controls=256):
    if side not in ("left", "right") or not 1 <= controls <= 256:
        raise ValueError("Invalid replay side or target budget")
    initial = np.asarray(initial_state, dtype=np.float32)
    if initial.shape != (24,) or not np.isfinite(initial).all():
        raise ValueError("Actual initial measured SDK24 must be finite [24]")
    sources = np.minimum(np.arange(controls), len(reference) - 1)
    out = np.repeat(initial[None], controls, axis=0)
    if side == "right":
        out[:, 12:24] = reference[sources, 12:24]
        mapping = "right_reference_active_right_inactive_left_initial_measured"
    else:
        # Explicitly unvalidated kinematic mirror. It is an action perturbation
        # experiment, never an expert/motor-supervision/grasp-success assertion.
        out[:, :6] = -reference[sources, 12:18]
        out[:, 6:12] = reference[sources, 18:24]
        mapping = "mirror_reference_motion_not_validated_expert"
    return out, sources.tolist(), mapping


def issue_event(step, side):
    if step % HIGH_WINDOW:
        raise ValueError("Subtasks are issued only on 32-target boundaries")
    return {
        "type": "issued_subtask", "event_id": f"issued_{step:04d}",
        "text": f"Pick up and hold the sample initially on the {side}.",
        "issued_at_step": step, "issued_at_physics_step": step * TICKS_PER_TARGET,
        "source": "authored_subtask_for_reference_attempt",
        "execution_semantics": "instruction_issued_not_physical_completion",
        "planned_window_end_step": min(step + HIGH_WINDOW, 256),
    }


def history_at(events, observed_step, physics_step):
    """Expose past issued commands only after actual execution has begun."""
    visible = []
    for original in events:
        begin = original["issued_at_step"]
        if begin >= observed_step:
            continue
        event = copy.deepcopy(original)
        event["executed_until_step"] = min(observed_step, event["planned_window_end_step"])
        event["observed_until_step"] = observed_step
        event["observed_until_physics_step"] = physics_step
        visible.append(event)
    return visible


def history_text(events):
    return "\n".join(event["text"] for event in events[-2:])


def save_observation(obs, folder, step, physics_step, events):
    arrays = {k: np.asarray(obs[k]) for k in (*CAMERAS, "state")}
    if arrays["state"].shape != (24,) or not np.isfinite(arrays["state"]).all():
        raise ValueError("Invalid actual SDK24 observation")
    cameras = {}
    for key in CAMERAS:
        image = arrays[key]
        if image.dtype != np.uint8 or image.shape != (480, 640, 3) or image.std() <= 1:
            raise ValueError(f"Invalid actual camera image: {key}")
        cameras[key] = {"sha256": hashlib.sha256(image.tobytes()).hexdigest(), "std": float(image.std())}
    path = Path(folder) / f"obs_{step:04d}.npz"
    np.savez_compressed(path, **arrays, instruction=np.asarray(obs["instruction"]))
    visible = history_at(events, step, physics_step)
    return {"step": step, "physics_step": physics_step, "sim_time_s": physics_step * PHYSICS_DT,
            "npz_path": str(path), "sha256": sha(path), "cameras": cameras,
            "state_sha256": hashlib.sha256(arrays["state"].tobytes()).hexdigest(),
            "visible_history": visible, "history_text": history_text(visible),
            "observed_until_step": step, "observation_source": "fresh_actual_simulator_capture"}


def validate_chunk(info, sent, previous_controls, previous_ticks):
    """Check actual masks and every target's clock, including early-stop tails."""
    count = int(info["executed_action_count"])
    mask = np.asarray(info["valid_action_mask"], dtype=bool)
    if not 0 < count <= len(sent) or mask.shape != (len(sent),):
        raise AssertionError("Invalid actual chunk count/mask")
    if not np.array_equal(mask, np.arange(len(sent)) < count):
        raise AssertionError("Executed mask must be a contiguous prefix")
    records = info["action_trace"]
    if len(records) != count or not info["observation_valid"]:
        raise AssertionError("Missing target clock receipt or valid actual observation")
    cursor = previous_ticks
    for index, record in enumerate(records):
        if not np.array_equal(np.asarray(record["raw_model_action24"], dtype=np.float32), sent[index]):
            raise AssertionError("Native SDK command differs from issued reference")
        ticks = int(record["executed_physics_steps"])
        if record["physics_step_start"] != cursor or record["physics_step_end"] != cursor + ticks:
            raise AssertionError("Native physics clock is discontinuous")
        if record["requested_physics_steps"] != TICKS_PER_TARGET or not 0 < ticks <= TICKS_PER_TARGET:
            raise AssertionError("Native target clock violates 60ms contract")
        if ticks < TICKS_PER_TARGET and not (index == count - 1 and info["episode_terminated"]):
            raise AssertionError("Only a real terminal target may have a partial tick clock")
        if ticks == TICKS_PER_TARGET:
            if not record["endpoint_command_written"] or not np.allclose(
                np.asarray(record["actual_final_drive_command24"]), sent[index], atol=2e-5, rtol=0
            ):
                raise AssertionError("Full native target did not reach its drive-command endpoint")
        cursor += ticks
    if info["step_count"] != previous_controls + count or info["physics_step_count"] != cursor:
        raise AssertionError("Chunk/target cumulative counters disagree")
    if abs(info["sim_time_s"] - cursor * PHYSICS_DT) > 1e-8:
        raise AssertionError("Simulator seconds disagree with actual physics ticks")
    if count < len(sent) and not (info["episode_terminated"] or info["episode_truncated"]):
        raise AssertionError("Unexecuted chunk tail without a terminal boundary")
    return count, cursor


def motion_summary(states, snapshots):
    state = np.stack(states)
    delta = np.max(np.abs(state - state[0]), axis=0)
    camera_counts = {key: len({s["cameras"][key]["sha256"] for s in snapshots}) for key in CAMERAS}
    arm_max = float(delta[np.r_[0:6, 12:18]].max())
    hand_max = float(delta[np.r_[6:12, 18:24]].max())
    return {"measured_state_samples": len(state), "measured_state_sampling": "initial_and_post_RPC_chunk",
            "max_abs_arm_motion_rad": arm_max, "max_abs_hand_motion_sdk": hand_max,
            "max_abs_motion_per_dim": delta.tolist(), "unique_saved_rgb_hashes": camera_counts,
            "dynamic_state_and_rgb_observed": bool(arm_max > 0.01 and camera_counts["full_image"] > 1),
            "physical_task_progress_inferred_from_motion": False}


def plan_for(args, source_receipt):
    families = make_families(args.train_families, args.val_families, args.test_families, args.seed_base)
    episodes = episode_specs(families)
    cfg_sha = sha(args.config)
    runtime_sources = {
        name: sha(args.robotwin_root / "integrations/realman_rm65b_gen4" / name)
        for name in ("rl_env.py", "rl_contract.py", "rl_factory.py", "action_clock.py", "sample_verification.py")
    }
    identities = {"script_sha256": sha(__file__), "config_sha256": cfg_sha,
                  "robotwin_root": str(args.robotwin_root.resolve()), "source_sha256": source_receipt["sha256"],
                  "runtime_source_sha256": runtime_sources,
                  "families": families, "controls": args.controls,
                  "snapshot_stride_train": 32, "snapshot_stride_eval": 16, "contract": CONTRACT}
    return {"schema": "rm65b_dynamic_protocol_capture_v1", "capture_kind": "fresh_simulation_reference_replay",
            "identities": identities, "plan_identity_sha256": digest(identities),
            "source_config": str(args.config.resolve()), "source_action_receipt": source_receipt,
            "contract": CONTRACT, "families": families, "planned_episodes": episodes,
            "expected_full_episodes": len(episodes),
            "expected_targets_per_full_episode": args.controls,
            "expected_physics_ticks_per_full_episode": args.controls * TICKS_PER_TARGET,
            "expected_sim_time_s_per_full_episode": args.controls * 0.06,
            "expected_train_physical_contexts": args.train_families * 2 * (args.controls // 32 + 1),
            "expected_val_physical_contexts": args.val_families * 2 * (args.controls // 16 + 1),
            "expected_test_physical_contexts": args.test_families * 2 * (args.controls // 16 + 1),
            "history_semantics": "typed issued_subtask only; last-two texts newline-joined; no completion inferred",
            "left_mapping": "mirror_reference_motion_not_validated_expert",
            "physical_skill_success_claimed": False, "motor_supervision_eligible": False,
            "policy_loaded": False, "training_enabled": False,
            "data_leakage_boundary": "fresh images/states only; source commands only; old heldout/diagnostic observations excluded",
            "label_limit": "Monitor truth stays private. Source teacher phase/success never labels new physical progress.",
            "independence_limit": "Sibling snapshots and authored history variants are correlated; bank-only changes do not create unique READ inputs."}


def capture_episode(args, base, spec, reference, env_class, verification_spec, folder):
    folder.mkdir(parents=True, exist_ok=False)
    specification = verification_spec.from_mapping(spec["layout"])
    goal = specification.canonical_instruction
    task = {**base, "memory_task": spec["layout"], "instruction": goal,
            "runtime_dir": str(args.output / "runtime"), "max_episode_steps": args.controls + 16,
            "rpc_timeout_s": 240.0}
    record = {**copy.deepcopy(spec), "goal": goal, "env_id": "rm65b_dynamic_reference_env0",
              "status": "in_progress", "completed": False, "collection_purpose": "high_protocol_observation",
              "capture_kind": "fresh_simulation_reference_replay", "motor_supervision_eligible": False,
              "physical_skill_success_claimed": False, "policy_loaded": False,
              "observations": [], "executed_controls": 0, "physics_ticks": 0,
              "issued_subtasks_path": str(folder / "issued_subtasks.jsonl"),
              "transitions_path": str(folder / "transitions.jsonl"),
              "private_monitor_path": str(folder / "private_monitor.jsonl"),
              "monitor_sampling": "post_RPC_chunk only; event physics_step values recorded by native 250Hz monitor",
              "event_source": "authored_issued_commands_not_policy_output", "interventions": 0}
    env = None
    states, events = [], []
    final_info = None
    error = None
    started = time.monotonic()
    try:
        env = env_class(task, 1, [spec["seed"]])
        obs = env.get_obs()[0]
        if obs["instruction"] != goal:
            raise AssertionError("Actual environment instruction differs from authored goal")
        actions, source_indices, mapping = replay_actions(reference, obs["state"], spec["target_side"], args.controls)
        record.update(action_mapping=mapping, goal_action_alignment_validated=False,
                      source_reference_targets=min(args.controls, len(reference)),
                      tail_hold_targets=max(0, args.controls - len(reference)),
                      inactive_arm_initial_state24=np.asarray(obs["state"]).tolist())
        states.append(np.asarray(obs["state"]).copy())
        record["observations"].append(save_observation(obs, folder, 0, 0, events))
        atomic_json(folder / "episode.json", record)
        stride = 32 if spec["split"] == "train" else 16
        with (folder / "issued_subtasks.jsonl").open("w") as issued, (folder / "transitions.jsonl").open("w") as trace, (folder / "private_monitor.jsonl").open("w") as monitor:
            while record["executed_controls"] < args.controls:
                start = record["executed_controls"]
                if start % HIGH_WINDOW == 0:
                    event = issue_event(start, spec["target_side"])
                    events.append(event)
                    append_json(issued, event)
                # RPC chunks never cross an issue or snapshot boundary.
                length = min(16, args.controls - start, HIGH_WINDOW - start % HIGH_WINDOW, stride - start % stride)
                sent = actions[start:start + length].copy()
                observations, rewards, terminated, truncated, infos = env.step(sent[None])
                info = infos[0]
                count, ticks = validate_chunk(info, sent, start, record["physics_ticks"])
                final_info = info
                obs = observations[0]
                if obs["instruction"] != goal:
                    raise AssertionError("Actual environment instruction changed during attempt")
                stop = start + count
                record.update(executed_controls=stop, physics_ticks=ticks, sim_time_s=ticks * PHYSICS_DT)
                states.append(np.asarray(obs["state"]).copy())
                append_json(trace, {
                    "query_index": start // 16, "control_start": start, "control_end": stop,
                    "physics_step_end": ticks, "source_action_indices": source_indices[start:stop],
                    "tail_hold_mask": [i >= len(reference) for i in range(start, stop)],
                    "issued_action_chunk": sent[None], "executed_action_count": count,
                    "valid_action_mask": info["valid_action_mask"], "sdk_target_receipts": info["action_trace"],
                    "measured_state24_post_chunk": np.asarray(obs["state"]),
                    "chunk_rewards": rewards, "chunk_terminated": terminated, "chunk_truncated": truncated,
                    "observation_valid": info["observation_valid"], "failure_reason": info["failure_reason"],
                    "action_mapping": mapping, "motor_supervision_eligible": False,
                })
                append_json(monitor, {"private_actor_input_forbidden": True, "step": stop,
                                      "physics_step": ticks, "episode_metrics": info["episode_metrics"],
                                      "episode_contract": info["episode_contract"]})
                terminal = bool(info["episode_terminated"] or info["episode_truncated"])
                if stop % stride == 0 or terminal or stop == args.controls:
                    record["observations"].append(save_observation(obs, folder, stop, ticks, events))
                atomic_json(folder / "episode.json", record)
                if terminal:
                    break
        record.update(status="completed", completed=True, capture_budget_reached=record["executed_controls"] == args.controls,
                      episode_terminated=bool(final_info and final_info["episode_terminated"]),
                      episode_truncated=bool(final_info and final_info["episode_truncated"]),
                      failure_reason=final_info["failure_reason"] if final_info else None,
                      motion=motion_summary(states, record["observations"]),
                      actual_monitor_summary={"private_actor_input_forbidden": True, **(final_info["episode_metrics"] if final_info else {})})
        if record["capture_budget_reached"] and record["physics_ticks"] != args.controls * TICKS_PER_TARGET:
            raise AssertionError("Full capture budget must have an exact 15-tick clock per target")
    except Exception as exc:
        error = exc
        record.update(status="failed", completed=False, error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    finally:
        if env is not None:
            try:
                env.close(clear_cache=True)
                record["worker_shutdown_confirmed"] = all(not p.is_alive() for p in env._processes)
            except Exception as exc:
                record["worker_shutdown_confirmed"] = False
                record["close_error"] = f"{type(exc).__name__}: {exc}"
                error = error or exc
        else:
            record["worker_shutdown_confirmed"] = True
        record["elapsed_s"] = time.monotonic() - started
        if not record["worker_shutdown_confirmed"]:
            record.update(status="failed", completed=False)
        record["artifact_sha256"] = {
            name: sha(folder / name)
            for name in ("issued_subtasks.jsonl", "transitions.jsonl", "private_monitor.jsonl")
            if (folder / name).is_file()
        }
        atomic_json(folder / "episode.json", record)
    if error is not None:
        raise RuntimeError(f"Capture failed; preserved evidence at {folder}") from error
    return record


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--execute", action="store_true")
    p.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT)
    p.add_argument("--config", type=Path)
    p.add_argument("--robotwin-root", type=Path)
    p.add_argument("--source-actions", type=Path, default=DEFAULT_ACTIONS)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", action="store_true", help="Skip completed matching episodes; restart incomplete attempts in new folders")
    p.add_argument("--max-episodes", type=int, default=None, help="Maximum new episodes this invocation; 2 gives the paired-side pilot")
    p.add_argument("--controls", type=int, default=256)
    p.add_argument("--train-families", type=int, default=16)
    p.add_argument("--val-families", type=int, default=6)
    p.add_argument("--test-families", type=int, default=6)
    p.add_argument("--seed-base", type=int, default=41000)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    args.config = (args.config or args.experiment_root / "configs/rm65b_mem_protocol.yaml").resolve()
    args.robotwin_root = (args.robotwin_root or args.experiment_root / "code/RoboTwin").resolve()
    args.output = args.output.resolve()
    if not 1 <= args.controls <= 256 or args.controls % 16:
        raise ValueError("Target budget must be a positive multiple of 16, <=256")
    if args.max_episodes is not None and args.max_episodes < 1:
        raise ValueError("--max-episodes must be positive")
    reference, receipt = load_reference(args.source_actions)
    plan = plan_for(args, receipt)
    if not args.execute:
        print(json.dumps({**plan, "mode": "plan_only", "output": str(args.output), "max_new_episodes": args.max_episodes}, indent=2))
        return 0

    # The only branch that imports the simulator. Spawn children inherit this
    # script's path setup. There is deliberately no actor/model import.
    sys.path.insert(0, str(args.robotwin_root))
    os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    import yaml
    from integrations.realman_rm65b_gen4.rl_env import RealManVectorEnv
    from integrations.realman_rm65b_gen4.sample_verification import VerificationSpec

    base = yaml.safe_load(args.config.read_text())["environment"]["task_config"]
    if base.get("task_name") != "sample_verify_return" or base.get("profile") != "full50_sdk24":
        raise ValueError("This capture requires the existing dual-bottle SDK24 environment")
    urdf = Path(base["asset_root"]) / "embodiments/realman-rm65b-inspire-gen4/robot_sapien.urdf"
    urdf_before = sha(urdf)
    path = args.output / "manifest.json"
    if args.output.exists():
        if not args.resume or not path.is_file():
            raise FileExistsError("Existing output requires --resume and a matching manifest")
        manifest = json.loads(path.read_text())
        if manifest.get("plan_identity_sha256") != plan["plan_identity_sha256"]:
            raise ValueError("Resume source/code/config/split contract differs from stored plan")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        manifest = {**plan, "created_at": datetime.now(UTC).isoformat(), "episodes": [], "attempts": [], "completed": False}
    manifest.update(status="in_progress", last_started_at=datetime.now(UTC).isoformat(), vendor_urdf_sha256=urdf_before)
    atomic_json(path, manifest)
    captured = 0
    try:
        for spec in plan["planned_episodes"]:
            previous = next((x for x in manifest["episodes"] if x["episode_id"] == spec["episode_id"]), None)
            if previous and previous.get("completed") and previous.get("worker_shutdown_confirmed"):
                episode_file = Path(previous["episode_json_path"])
                if not episode_file.is_file() or sha(episode_file) != previous["episode_json_sha256"]:
                    raise ValueError("Completed episode receipt was altered; refusing resume")
                if any(not Path(x["npz_path"]).is_file() or sha(x["npz_path"]) != x["sha256"] for x in previous["observations"]):
                    raise ValueError("Completed observation receipt was altered; refusing resume")
                if any(not (episode_file.parent / name).is_file() or sha(episode_file.parent / name) != expected
                       for name, expected in previous["artifact_sha256"].items()):
                    raise ValueError("Completed command/history/monitor trace was altered; refusing resume")
                continue
            if args.max_episodes is not None and captured >= args.max_episodes:
                break
            episode_base = args.output / "episodes" / spec["episode_id"]
            attempt = 0
            folder = episode_base
            while folder.exists():
                attempt += 1
                folder = episode_base.with_name(episode_base.name + f"_attempt{attempt:03d}")
            attempt_receipt = {"episode_id": spec["episode_id"], "folder": str(folder), "attempt": attempt}
            manifest["attempts"].append(attempt_receipt)
            atomic_json(path, manifest)
            entry = capture_episode(args, base, spec, reference, RealManVectorEnv, VerificationSpec, folder)
            entry.update(episode_json_path=str(folder / "episode.json"), episode_json_sha256=sha(folder / "episode.json"))
            manifest["episodes"] = [x for x in manifest["episodes"] if x["episode_id"] != spec["episode_id"]] + [entry]
            captured += 1
            atomic_json(path, manifest)
            print(json.dumps({"captured": spec["episode_id"], "split": spec["split"], "targets": entry["executed_controls"],
                              "ticks": entry["physics_ticks"], "dynamic": entry["motion"]["dynamic_state_and_rgb_observed"],
                              "completed_episodes": len(manifest["episodes"])}, separators=(",", ":")), flush=True)
        manifest["completed"] = len(manifest["episodes"]) == plan["expected_full_episodes"]
        manifest["status"] = "completed" if manifest["completed"] else "pilot_complete"
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        manifest["vendor_urdf_unchanged"] = sha(urdf) == urdf_before
        manifest["source_actions_unchanged"] = sha(args.source_actions) == receipt["sha256"]
        manifest["last_finished_at"] = datetime.now(UTC).isoformat()
        if not manifest["vendor_urdf_unchanged"] or not manifest["source_actions_unchanged"]:
            manifest.update(status="failed", completed=False, integrity_error="Original URDF or reference actions changed during capture")
        atomic_json(path, manifest)
    if manifest.get("integrity_error"):
        raise RuntimeError(manifest["integrity_error"])
    print(json.dumps({"status": manifest["status"], "episodes": len(manifest["episodes"]), "manifest": str(path)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
