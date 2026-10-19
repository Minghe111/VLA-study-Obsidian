# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Serial, versioned RM65-B RECAP rounds; construction performs no execution."""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import sys
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Protocol


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def asset_versions(config: dict) -> dict[str, str]:
    """Identify small normalization/tokenizer assets without loading a model."""
    from rlinf.data.schema.hierarchical_types import SCHEMA_VERSION
    from rlinf.data.storage.episode_memory import MEMORY_VERSION

    paths = {
        "normalization": config["model"]["openpi_data"]["norm_stats_path"],
        "text_tokenizer": config["tokenizer"]["text_tokenizer_path"],
    }
    from rlinf.models.embodiment.openpi.recap_tokenizer import (
        PROMPT_VERSION,
        TOKENIZER_VERSION,
    )

    result = {
        "schema": SCHEMA_VERSION,
        "memory": MEMORY_VERSION,
        "prompt": PROMPT_VERSION,
        "tokenizer": TOKENIZER_VERSION,
    }
    if config.get("memory", {}).get("strict_protocol", False):
        from rlinf.data.storage.episode_memory import STRICT_MEMORY_VERSION

        result["memory"] = STRICT_MEMORY_VERSION
    if config["model"]["openpi"].get("mem_enabled", False):
        from rlinf.data.storage.observation_history import HISTORY_VERSION

        result["observation_history"] = HISTORY_VERSION
    for name, raw in paths.items():
        path = Path(raw) if raw else None
        result[name + "_sha256"] = (
            hashlib.sha256(path.read_bytes()).hexdigest()
            if path is not None and path.is_file()
            else "unavailable"
        )
    raw_fast = config["tokenizer"].get("fast_processor_path")
    fast_root = Path(raw_fast) if raw_fast else None
    if fast_root is None or not fast_root.is_dir():
        result["fast_processor_sha256"] = "unavailable"
        return result
    files = sorted(
        path
        for path in fast_root.rglob("*")
        if path.is_file() and path.suffix in {".json", ".py", ".model", ".txt"}
    )
    if not files:
        result["fast_processor_sha256"] = "unavailable"
        return result
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(fast_root)).encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    result["fast_processor_sha256"] = digest.hexdigest()
    return result


def audit_config(config: dict) -> dict:
    """Inspect configuration and local asset paths without loading an asset."""
    contract = config["contract"]
    expected = {
        "physical_action_dim": 24,
        "model_action_dim": 32,
        "action_horizon": 16,
        "execute_actions": 1,
        "physics_steps_per_action": 15,
        "physics_dt_s": 0.004,
        "camera_keys": ["full_image", "left_wrist_image", "right_wrist_image"],
        "native_rgb_shape": [480, 640, 3],
    }
    for field, value in expected.items():
        if contract.get(field) != value:
            raise ValueError(
                f"Expected contract.{field}={value!r}, got {contract.get(field)!r}"
            )
    window = int(config["rollout"]["high_level_window"])
    if (
        window < contract["action_horizon"]
        and float(config["actor_training"]["loss_weights"]["fast_ce"]) > 0
    ):
        raise ValueError(
            "High-level window must cover H16 when FAST supervision is enabled"
        )
    if config["model"]["openpi"].get("task") != "recap_hier":
        raise ValueError("The text and motor actor must use openpi.task=recap_hier")
    if config["model"].get("is_lora", False):
        raise ValueError(
            "Use recap_vlm_mode in the native task; generic model.is_lora is unsupported"
        )
    mem = config["model"]["openpi"]
    if mem.get("mem_enabled", False):
        if int(mem.get("mem_num_frames", 4)) < 1:
            raise ValueError("MEM must retain at least the current observation")
        stride = config["rollout"].get("history_stride_steps", 8)
        if type(stride) is not int or stride < 1:
            raise ValueError("rollout.history_stride_steps must be a positive integer")
        if (
            config["rollout"]["goal"]
            != config["environment"]["task_config"]["instruction"]
        ):
            raise ValueError("MEM goal must match the environment task instruction")
    train_seeds, eval_seeds = (
        config["rollout"]["train_seeds"],
        config["rollout"]["eval_seeds"],
    )
    if set(train_seeds) & set(eval_seeds):
        raise ValueError("Training and held-out evaluation seeds must be disjoint")
    if not train_seeds or not eval_seeds:
        raise ValueError("Explicit nonempty training and evaluation seeds are required")
    required = {
        "model": config["model"]["model_path"],
        "norm_stats": config["model"]["openpi_data"]["norm_stats_path"],
        "text_tokenizer": config["tokenizer"]["text_tokenizer_path"],
        "fast_processor": config["tokenizer"]["fast_processor_path"],
        "robotwin_root": config["environment"]["robotwin_root"],
        "robot_assets": config["environment"]["task_config"]["asset_root"],
        "value_siglip": config["value"]["model"]["siglip_path"],
        "value_gemma": config["value"]["model"]["gemma3_path"],
        "value_tokenizer": config["value"]["model"]["tokenizer_path"],
    }
    assets = {}
    for name, raw in required.items():
        path = Path(raw).expanduser() if raw else None
        assets[name] = {
            "path": str(path) if path else None,
            "exists": path is not None and path.exists(),
            "absolute": path is not None and path.is_absolute(),
        }
    packages = {
        name: importlib.util.find_spec(name) is not None
        for name in (
            "torch",
            "numpy",
            "openpi",
            "omegaconf",
            "sentencepiece",
            "transformers",
            "safetensors",
        )
    }
    assets_ready = all(
        item["exists"] and item["absolute"] for item in assets.values()
    ) and all(packages.values())
    return {
        "config_valid": True,
        "assets": assets,
        "packages": packages,
        "ready_for_asset_loading": assets_ready,
        "ready_for_execution": assets_ready,
        "readiness_scope": "local_asset_paths_and_package_discovery_only",
        "model_capabilities_checked": False,
        "training_enabled": False,
        "execution_performed": False,
    }


class StageBackend(Protocol):
    """Each stage owns and releases its actor or independent value model."""

    def collect(self, destination: Path, *, evaluate: bool = False) -> list[Path]: ...

    def fit_value(
        self, episodes: list[Path], destination: Path, *, enable_training: bool
    ) -> dict: ...

    def label(self, episodes: list[Path], destination: Path) -> list[Path]: ...

    def fit_actor(
        self, episodes: list[Path], destination: Path, *, enable_training: bool
    ) -> dict: ...


class HierarchicalReCapRunner:
    """Advance complete rounds only after explicit training opt-in."""

    def __init__(self, backend: StageBackend, output_root: Path):
        self.backend, self.output_root = backend, Path(output_root)

    def run_round(self, round_index: int, *, enable_training: bool = False) -> dict:
        if not enable_training:
            raise PermissionError(
                "A full round includes optimizer steps; explicit enable_training is required"
            )
        root = self.output_root / f"round_{round_index:04d}"
        root.mkdir(parents=True, exist_ok=False)
        stages = {}
        try:
            episodes = self.backend.collect(root / "collect")
            stages["collect"] = [str(path) for path in episodes]
            _write_json(
                root / "stages" / "01_collect.json",
                {"episode_paths": stages["collect"]},
            )
            stages["value"] = self.backend.fit_value(
                episodes, root / "value", enable_training=True
            )
            _write_json(root / "stages" / "02_value.json", stages["value"])
            labelled = self.backend.label(episodes, root / "labelled")
            stages["labelled"] = [str(path) for path in labelled]
            _write_json(
                root / "stages" / "03_labelled.json",
                {"episode_paths": stages["labelled"]},
            )
            stages["actor"] = self.backend.fit_actor(
                labelled, root / "actor", enable_training=True
            )
            _write_json(root / "stages" / "04_actor.json", stages["actor"])
            evaluation = self.backend.collect(root / "evaluation", evaluate=True)
            stages["evaluation"] = [str(path) for path in evaluation]
            _write_json(root / "complete.json", stages)
            return stages
        except Exception as exc:
            _write_json(
                root / "failure.json",
                {"completed_stages": stages, "error": f"{type(exc).__name__}: {exc}"},
            )
            raise


def _move(value: Any, device: str):
    import torch

    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    return value


def _batches(examples: list, batch_size: int, updates: int) -> Iterable[list]:
    if not examples or batch_size < 1 or updates < 1:
        raise ValueError(
            "Batches require nonempty examples, positive batch size and update budget"
        )
    for update in range(updates):
        yield [
            examples[(update * batch_size + offset) % len(examples)]
            for offset in range(batch_size)
        ]



def actor_data_fingerprint(streams: dict[str, list]) -> str:
    """Hash ordered sample contents and referenced NPZ bytes, not only row counts."""
    import numpy as np

    digest = hashlib.sha256()
    reference_hashes = {}

    def update(value, key=None):
        if isinstance(value, dict):
            digest.update(b"dict{")
            for name in sorted(value):
                update(str(name))
                update(value[name], name)
            digest.update(b"}")
        elif isinstance(value, (list, tuple)):
            digest.update(b"list[")
            for item in value:
                update(item)
            digest.update(b"]")
        elif isinstance(value, np.ndarray):
            digest.update(b"array" + str(value.dtype).encode())
            digest.update(str(value.shape).encode())
            digest.update(np.ascontiguousarray(value).tobytes())
        elif hasattr(value, "detach") and hasattr(value, "cpu"):
            update(value.detach().cpu().numpy())
        elif isinstance(value, Path):
            update(str(value), key)
        elif isinstance(value, (str, int, float, bool, type(None))):
            digest.update(json.dumps(value, ensure_ascii=False, allow_nan=False).encode())
            if key == "npz_path":
                path = Path(value)
                if not path.is_file():
                    raise FileNotFoundError(f"Sampler reference is missing: {path}")
                if str(path) not in reference_hashes:
                    reference_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
                digest.update(reference_hashes[str(path)].encode())
        elif isinstance(value, np.generic):
            update(value.item(), key)
        else:
            raise TypeError(f"Cannot fingerprint sampler data type {type(value).__name__}")

    update(streams)
    return digest.hexdigest()


class ActorMicrobatchSampler:
    """Resumable high/action/anchor sampler with private per-stream RNGs.

    Defaults preserve the former cyclic order, including advancing primary
    streams on an anchor microbatch. A saved cursor counts consumed examples,
    rather than optimizer updates; epochs may wrap within a microbatch.
    """

    VERSION = 1

    def __init__(self, high, actions, anchor_high, anchor_actions, *,
                 batch_size: int, gradient_accumulation: int, anchor_interval: int,
                 training_mode: str, loss_weights: dict, shuffle: bool = False,
                 seed: int = 0):
        import random

        if min(batch_size, gradient_accumulation) < 1 or anchor_interval < 2:
            raise ValueError("Invalid actor sampler batch/accumulation/anchor contract")
        if type(shuffle) is not bool or type(seed) is not int:
            raise ValueError("Sampler shuffle must be boolean and seed must be integer")
        self.streams = {"high": high, "actions": actions,
                        "anchor_high": anchor_high, "anchor_actions": anchor_actions}
        self.contract = {"batch_size": batch_size,
                         "gradient_accumulation": gradient_accumulation,
                         "anchor_interval": anchor_interval,
                         "training_mode": training_mode,
                         "loss_weights": dict(loss_weights), "shuffle": shuffle,
                         "seed": seed}
        self.fingerprint = actor_data_fingerprint(self.streams)
        self.microbatch_cursor = 0
        self.states, self.rngs = {}, {}
        for index, (name, data) in enumerate(self.streams.items()):
            rng = random.Random(seed + 104729 * index)
            order = list(range(len(data)))
            if shuffle:
                rng.shuffle(order)
            self.states[name] = {"order": order, "cursor": 0, "epoch": 0, "draws": 0}
            self.rngs[name] = rng

    def _next(self, name):
        data, state = self.streams[name], self.states[name]
        if not data:
            return []
        result = []
        for _ in range(self.contract["batch_size"]):
            if state["cursor"] == len(data):
                state["cursor"] = 0
                state["epoch"] += 1
                if self.contract["shuffle"]:
                    self.rngs[name].shuffle(state["order"])
            result.append(data[state["order"][state["cursor"]]])
            state["cursor"] += 1
            state["draws"] += 1
        return result

    def next_batch(self):
        high, actions = self._next("high"), self._next("actions")
        anchor = bool(self.streams["anchor_high"] or self.streams["anchor_actions"]) and (
            self.microbatch_cursor % self.contract["anchor_interval"]
            == self.contract["anchor_interval"] - 1
        )
        if anchor:
            high, actions = self._next("anchor_high"), self._next("anchor_actions")
        self.microbatch_cursor += 1
        return high, actions, anchor

    def state_dict(self):
        return {"version": self.VERSION, "contract": deepcopy(self.contract),
                "data_fingerprint": self.fingerprint,
                "microbatch_cursor": self.microbatch_cursor,
                "streams": deepcopy(self.states),
                "stream_rng": {name: rng.getstate() for name, rng in self.rngs.items()}}

    def load_state_dict(self, state):
        import random

        if type(state.get("version")) is not int or state["version"] != self.VERSION:
            raise ValueError("Unsupported actor sampler checkpoint version")
        if state.get("contract") != self.contract:
            raise ValueError("Actor sampler batch/shuffle/training contract differs")
        if state.get("data_fingerprint") != self.fingerprint:
            raise ValueError("Actor sampler ordered data fingerprint differs")
        if type(state.get("microbatch_cursor")) is not int or state["microbatch_cursor"] < 0:
            raise ValueError("Invalid actor sampler microbatch cursor")
        streams, rngs = state.get("streams", {}), state.get("stream_rng", {})
        if set(streams) != set(self.streams) or set(rngs) != set(self.streams):
            raise ValueError("Actor sampler streams differ")
        restored_rngs = {}
        for name, data in self.streams.items():
            item = streams[name]
            if (sorted(item["order"]) != list(range(len(data)))
                or any(type(item[key]) is not int or item[key] < 0
                       for key in ("cursor", "epoch", "draws"))
                or item["cursor"] > len(data)):
                raise ValueError("Invalid actor sampler stream cursor/order")
            expected_batches = state["microbatch_cursor"]
            if name.startswith("anchor_"):
                expected_batches //= self.contract["anchor_interval"]
            expected_draws = expected_batches * self.contract["batch_size"] if data else 0
            if (item["draws"] != expected_draws
                or item["draws"] != item["epoch"] * len(data) + item["cursor"]
                or (not data and item["epoch"] != 0)):
                raise ValueError("Actor sampler draws/epoch/microbatch cursor are inconsistent")
            rng = random.Random()
            rng.setstate(rngs[name])
            restored_rngs[name] = rng
        self.microbatch_cursor = state["microbatch_cursor"]
        self.states, self.rngs = deepcopy(streams), restored_rngs


def select_actor_action_samples(actions: list[dict], loss_weights: dict) -> list[dict]:
    """Use complete behavior windows when FAST is the sole action objective."""
    if loss_weights.get("fast_ce", 0) > 0 and loss_weights.get("fm", 0) <= 0:
        selected = [sample for sample in actions if sample.get("fast_eligible", False)]
        if not selected:
            raise ValueError(
                "FAST-only action sampling requires a complete behavior window"
            )
        return selected
    return actions


def balanced_value_batches(
    action_examples: list, high_examples: list, *, batch_size: int, updates: int
) -> Iterable[list]:
    """Visit both value phases even in the minimum permitted fitting budget."""
    if (
        not action_examples
        or not high_examples
        or min(batch_size, updates) < 1
        or batch_size * updates < 2
    ):
        raise ValueError(
            "Value fitting requires both phases and at least two total examples"
        )
    cursors, sources = [0, 0], [action_examples, high_examples]
    for update in range(updates):
        batch = []
        for offset in range(batch_size):
            phase = (update * batch_size + offset) % 2
            batch.append(sources[phase][cursors[phase] % len(sources[phase])])
            cursors[phase] += 1
        yield batch


def validate_training_record(
    metadata: dict,
    *,
    eval_seeds: Iterable[int] = (),
    excluded_families: Iterable[str] = (),
) -> None:
    """Reject held-out, diagnostic and explicitly excluded evidence consistently."""
    if (
        metadata.get("split")
        in {
            "heldout_eval",
            "eval",
            "validation",
            "val",
            "test",
            "diagnostic",
            "excluded",
        }
        or metadata.get("exclude_from_training", False)
        or metadata.get("excluded_from_training", False)
        or metadata.get("excluded", False)
        or metadata.get("diagnostic", False)
        or metadata.get("is_diagnostic", False)
        or metadata.get("training_eligible") is False
        or metadata.get("seed") in set(eval_seeds)
        or metadata.get("family_id") in set(excluded_families)
    ):
        raise ValueError(
            "Held-out evaluation, diagnostic or excluded data cannot enter training"
        )


def validate_high_context(mode: str, context: dict) -> None:
    """Keep standalone labels within the memory view available at each stage."""
    if not isinstance(context, dict):
        raise ValueError("Standalone high SFT context must be a mapping")
    forbidden = {
        "critic_only",
        "critic_context",
        "entries",
        "full_bank",
        "fullbank",
        "full_memory",
        "bank",
        "memory_before",
        "memory_after",
        "read_context",
        "memory_version",
    }

    def reject_full_bank(value):
        if isinstance(value, dict):
            if set(value) & forbidden:
                raise ValueError(
                    "Standalone actor context cannot expose critic or full-bank fields"
                )
            for item in value.values():
                reject_full_bank(item)
        elif isinstance(value, list):
            for item in value:
                reject_full_bank(item)

    reject_full_bank(context)
    if mode == "READ":
        if any(
            context.get(key)
            for key in (
                "actor_memory",
                "selected_memory",
                "visible_memory",
                "current_writes",
            )
        ):
            raise ValueError(
                "READ context cannot reveal memory contents before selection"
            )
    elif mode == "WRITE":
        if any(
            key in context
            for key in ("actor_memory", "current_writes", "visible_memory")
        ):
            raise ValueError("WRITE context accepts only pre-write selected_memory")
        read_ids = context.get("read_ids")
        selected = context.get("selected_memory")
        if (
            not isinstance(read_ids, list)
            or any(not isinstance(slot, str) or not slot for slot in read_ids)
            or len(read_ids) != len(set(read_ids))
            or not isinstance(selected, list)
            or any(not isinstance(entry, dict) for entry in selected)
            or [entry.get("slot_id") for entry in selected] != read_ids
        ):
            raise ValueError(
                "WRITE selected_memory must match its explicit pre-write read_ids"
            )
    elif mode == "SUBTASK":
        actor_memory = context.get("actor_memory", {})
        if not isinstance(actor_memory, dict) or set(actor_memory) - {
            "selected",
            "current_writes",
        }:
            raise ValueError("SUBTASK requires the resolved actor memory view")


def load_verified_high_samples(
    paths: Iterable[Path],
    *,
    eval_seeds: Iterable[int] = (),
    excluded_families: Iterable[str] = (),
) -> list[dict]:
    """Read verified high-level JSON/JSONL without manufacturing motor labels."""
    from rlinf.data.datasets.recap.hierarchical_collator import training_advantage
    from rlinf.data.schema.history_events import history_from_sample

    result = []
    for raw_path in paths:
        path = Path(raw_path)
        text = path.read_text(encoding="utf-8")
        records = (
            [json.loads(line) for line in text.splitlines() if line.strip()]
            if path.suffix == ".jsonl"
            else json.loads(text)
        )
        records = records if isinstance(records, list) else [records]
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("Verified high SFT records must be mappings")
            metadata = {
                **record.get("metadata", {}),
                **{
                    key: record[key]
                    for key in (
                        "split",
                        "seed",
                        "family_id",
                        "diagnostic",
                        "is_diagnostic",
                        "training_eligible",
                        "excluded",
                        "exclude_from_training",
                        "excluded_from_training",
                    )
                    if key in record
                },
            }
            validate_training_record(
                metadata, eval_seeds=eval_seeds, excluded_families=excluded_families
            )
            if metadata.get("split") not in {"train", "training"}:
                raise ValueError(
                    "Independent high SFT requires an explicit train split"
                )
            if record.get("supervision_kind") != "verified_high_sft" or not record.get(
                "evidence"
            ):
                raise ValueError(
                    "High-only SFT requires verified_high_sft and visible evidence"
                )
            evidence = record["evidence"]
            history_from_sample(record)
            decision_step = record.get("decision_step", metadata.get("decision_step"))
            evidence_steps = [
                record.get("observed_until_step"),
                metadata.get("observed_until_step"),
            ]
            if isinstance(evidence, dict):
                evidence_steps.append(evidence.get("observed_until_step"))
            if decision_step is not None and any(
                step is not None and step > decision_step for step in evidence_steps
            ):
                raise ValueError(
                    "High SFT evidence cannot extend past its decision step"
                )
            training_advantage(record, "IH", training_mode="verified_sft")
            for key in ("observation", "context", "mode", "target_text"):
                if key not in record:
                    raise ValueError(f"Verified high SFT record lacks {key}")
            if (
                record["mode"] not in {"READ", "WRITE", "SUBTASK"}
                or not isinstance(record["target_text"], str)
                or not record["target_text"].strip()
            ):
                raise ValueError(
                    "Verified high SFT requires a READ/WRITE/SUBTASK text target"
                )
            validate_high_context(record["mode"], record["context"])

            def absolute_references(value):
                if isinstance(value, dict):
                    return {
                        key: str((path.parent / item).resolve())
                        if key == "npz_path" and not Path(item).is_absolute()
                        else absolute_references(item)
                        for key, item in value.items()
                    }
                if isinstance(value, list):
                    return [absolute_references(item) for item in value]
                return value

            record = absolute_references(deepcopy(record))
            history = record["observation"].get("mem_history")
            if history is not None:
                from rlinf.data.storage.observation_history import (
                    validate_history_reference,
                )

                if type(decision_step) is not int or decision_step < 0:
                    raise ValueError(
                        "Video high SFT labels require a nonnegative integer decision step"
                    )
                expected_physics_step = decision_step * 15
                if (
                    record.get("decision_physics_step", expected_physics_step)
                    != expected_physics_step
                ):
                    raise ValueError(
                        "Video decision physics time must follow the 15-tick control contract"
                    )
                if record["observation"].get("physics_step") != expected_physics_step:
                    raise ValueError(
                        "Video observation does not match its decision physics time"
                    )
                validate_history_reference(
                    history,
                    current_physics_step=expected_physics_step,
                    persisted=True,
                )
                for key in ("env_id", "episode_id"):
                    if not record.get(key) or record[key] != history[key]:
                        raise ValueError(
                            "High SFT history belongs to another environment/episode"
                        )
                    if record["observation"].get(key) != record[key]:
                        raise ValueError(
                            "High SFT observation identity differs from its record"
                        )
            result.append(record)
    return result


def prepare_actor_samples(
    episodes: list,
    *,
    training_mode: str,
    loss_weights: dict,
    eval_seeds: Iterable[int] = (),
    excluded_families: Iterable[str] = (),
) -> tuple[list[dict], list[dict]]:
    """Validate verified SFT or reward-labelled RECAP before native model loading."""
    from rlinf.data.datasets.recap.hierarchical_collator import training_advantage
    from rlinf.data.datasets.recap.hierarchical_dataset import (
        build_action_samples,
        build_high_level_samples,
    )

    if training_mode not in {"recap", "verified_sft"}:
        raise ValueError("training_mode must be recap or verified_sft")
    need_high = float(loss_weights.get("high_ce", 0)) > 0
    need_action = any(
        float(loss_weights.get(key, 0)) > 0 for key in ("fast_ce", "fm", "kv_anchor")
    )
    if not (need_high or need_action):
        raise ValueError("Actor training requires an enabled objective")
    high, actions, action_scores, high_scores = [], [], [], []
    for episode in episodes:
        validate_training_record(
            episode.metadata, eval_seeds=eval_seeds, excluded_families=excluded_families
        )
        provenance = episode.metadata.get("provenance")
        if training_mode == "verified_sft":
            training_advantage(
                {
                    "supervision_kind": episode.metadata.get("supervision_kind"),
                    "provenance": provenance,
                },
                "IH",
                training_mode=training_mode,
            )
        labels = episode.metadata.get("hierarchical_advantages", {})
        action_scores.extend(labels.get("AA", []))
        high_scores.extend(labels.get("AH", []))
        new_high = (
            build_high_level_samples(episode, require_labels=training_mode == "recap")
            if need_high
            else []
        )
        new_actions = (
            build_action_samples(
                episode, horizon=16, require_labels=training_mode == "recap"
            )
            if need_action
            else []
        )
        if training_mode == "verified_sft":
            for sample in new_high + new_actions:
                sample["supervision_kind"], sample["provenance"] = (
                    "verified_sft",
                    provenance,
                )
        high.extend(new_high)
        actions.extend(new_actions)
    if (need_high and not high) or (need_action and not actions):
        raise ValueError(
            "Actor training requires recorded samples for every enabled objective"
        )
    if training_mode == "recap":
        if loss_weights.get("high_ce", 0) > 0 and (
            len(high_scores) < 2 or max(high_scores) - min(high_scores) <= 1e-8
        ):
            raise ValueError(
                "High-level advantage signal is constant or missing; actor updates are refused"
            )
        if (loss_weights.get("fast_ce", 0) > 0 or loss_weights.get("fm", 0) > 0) and (
            len(action_scores) < 2 or max(action_scores) - min(action_scores) <= 1e-8
        ):
            raise ValueError(
                "Action advantage signal is constant or missing; actor updates are refused"
            )
    if loss_weights.get("fast_ce", 0) > 0 and not any(
        sample["fast_eligible"] for sample in actions
    ):
        raise ValueError(
            "No same-condition H16 behavior window exists for FAST supervision"
        )
    return high, actions


class NativeHierarchicalBackend:
    """Lazy adapters to native Torch OpenPI, RECAP value and RoboTwin RM65-B."""

    def __init__(self, config: dict):
        self.config = deepcopy(config)
        self.actor_checkpoint = config["model"]["model_path"]
        self.actor_version = config["model"].get("version", "sft_initial")
        self.actor_training_mode = config["model"].get("training_mode", "verified_sft")
        manifest = Path(self.actor_checkpoint) / "version.json"
        if manifest.is_file():
            saved = json.loads(manifest.read_text(encoding="utf-8"))
            self.actor_training_mode = saved.get("metadata", {}).get(
                "training_mode", self.actor_training_mode
            )
        self.value_checkpoint = config["value"].get("checkpoint")
        self.value_version = config["value"].get("version", "unfitted")
        self.replay_episode_paths = [
            Path(path) for path in config.get("data", {}).get("prior_episode_paths", [])
        ]
        self.verified_sft_paths = [
            Path(path)
            for path in config.get("data", {}).get("verified_sft_episode_paths", [])
        ]

    def _replay_paths(self, fresh: list[Path]) -> list[Path]:
        from rlinf.data.schema.hierarchical_types import load_episode_json

        paths = list(
            dict.fromkeys(path.resolve() for path in self.replay_episode_paths + fresh)
        )
        identities = set()
        for path in paths:
            episode = load_episode_json(path)
            identity = (episode.env_id, episode.episode_id)
            if identity in identities:
                raise ValueError(
                    "Replay contains conflicting files for the same env_id and episode_id"
                )
            identities.add(identity)
            validate_training_record(
                episode.metadata,
                eval_seeds=self.config["rollout"]["eval_seeds"],
                excluded_families=self.config.get("data", {}).get(
                    "excluded_family_ids", []
                )
                + self.config.get("data", {}).get("eval_family_ids", []),
            )
            if episode.metadata.get("supervision_kind") == "verified_sft":
                raise ValueError(
                    "Verified SFT anchors belong in their separate buffer, not rollout replay"
                )
        self.replay_episode_paths = paths
        return paths

    def _require_assets(self, stage: str):
        report = audit_config(self.config)
        fields = {
            "actor": ("model", "norm_stats", "text_tokenizer"),
            "fast": ("fast_processor",),
            "collect": (
                "model",
                "norm_stats",
                "text_tokenizer",
                "robotwin_root",
                "robot_assets",
            ),
            "value": ("value_siglip", "value_gemma", "value_tokenizer"),
        }[stage]
        if not all(
            report["assets"][field]["exists"] and report["assets"][field]["absolute"]
            for field in fields
        ) or not all(report["packages"].values()):
            raise RuntimeError(
                f"Native {stage} assets/dependencies are not ready; inspect audit-config output"
            )

    def _actor(self):
        self._require_assets("actor")
        from omegaconf import OmegaConf

        from rlinf.models.embodiment.openpi import get_model
        from rlinf.models.embodiment.openpi.recap_tokenizer import RecapTokenizer
        from rlinf.workers.rollout.hf.hierarchical_recap_worker import NativePi05Actor

        model_cfg = deepcopy(self.config["model"])
        model_cfg["model_path"] = self.actor_checkpoint
        model = get_model(OmegaConf.create(model_cfg)).to(
            self.config["runtime"]["device"]
        )
        token_cfg = self.config["tokenizer"]
        tokenizer = RecapTokenizer.from_local_paths(
            token_cfg["text_tokenizer_path"],
            fast_processor_path=token_cfg["fast_processor_path"],
            max_prefix_tokens=token_cfg["max_prefix_tokens"],
            max_target_tokens=token_cfg["max_target_tokens"],
        )
        return NativePi05Actor(
            model,
            tokenizer,
            version=self.actor_version,
            max_new_tokens=self.config["rollout"]["max_new_tokens"],
            eos_token_id=tokenizer.eos_token_id,
            advantage_condition=self.actor_training_mode == "recap",
            versions=asset_versions(self.config),
        )

    def collect(self, destination: Path, *, evaluate: bool = False) -> list[Path]:
        from rlinf.workers.rollout.hf.hierarchical_recap_worker import (
            HierarchicalReCapRolloutWorker,
        )

        self._require_assets("collect")
        if not evaluate:
            self._replay_paths([])
        destination.mkdir(parents=True, exist_ok=False)
        actor = self._actor()
        env = None
        try:
            env_cfg = self.config["environment"]
            root = str(Path(env_cfg["robotwin_root"]).resolve())
            if root not in sys.path:
                sys.path.insert(0, root)
            from integrations.realman_rm65b_gen4.rl_env import RealManVectorEnv

            seeds = self.config["rollout"]["eval_seeds" if evaluate else "train_seeds"]
            env = RealManVectorEnv(
                env_cfg["task_config"], n_envs=1, env_seeds=[seeds[0]]
            )
            controller = HierarchicalReCapRolloutWorker(
                actor,
                env,
                self.config["memory"]["catalogue"],
                high_level_window=self.config["rollout"]["high_level_window"],
                max_steps=self.config["rollout"]["max_steps"],
                history_num_frames=(
                    int(self.config["model"]["openpi"].get("mem_num_frames", 4))
                    if self.config["model"]["openpi"].get("mem_enabled", False)
                    else 1
                ),
                history_stride_steps=self.config["rollout"].get(
                    "history_stride_steps", 8
                ),
                strict_memory_protocol=self.config["memory"].get(
                    "strict_protocol", False
                ),
            )
            result = []
            for index, seed in enumerate(seeds):
                name = f"episode_{index:04d}_seed_{seed}_{uuid.uuid4().hex}"
                controller.collect_episode(
                    destination / name,
                    episode_id=name,
                    goal=self.config["rollout"]["goal"],
                    seed=seed,
                    evaluate=evaluate,
                )
                result.append(destination / name / "episode.json")
            return result if evaluate else self._replay_paths(result)
        finally:
            if env is not None:
                env.close()
            del actor
            gc.collect()

    def _value(self):
        self._require_assets("value")
        from omegaconf import OmegaConf

        from rlinf.data.datasets.recap.hierarchical_value import StrictValueCollator
        from rlinf.models.embodiment.value_model.recap import get_model
        from rlinf.models.embodiment.value_model.recap.processing import ValueProcessor

        cfg = deepcopy(self.config["value"]["model"])
        # Always initialize the independent architecture from explicit local backbones.
        cfg.pop("model_path", None)
        model = get_model(OmegaConf.create(cfg)).to(self.config["runtime"]["device"])
        if self.value_checkpoint:
            import torch

            state = torch.load(
                self.value_checkpoint, map_location="cpu", weights_only=True
            )
            model.load_state_dict(state, strict=True)
        processor = ValueProcessor(
            max_token_len=cfg["max_token_len"],
            tokenizer_name_or_path=cfg["tokenizer_path"],
            do_augment=False,
        )
        return model, StrictValueCollator(
            processor, max_length=cfg["max_token_len"], train=False
        )

    def _returns(self, episode):
        from rlinf.algorithms.offline.process.hierarchical import (
            compute_episode_returns,
        )

        return compute_episode_returns(
            episode,
            gamma=self.config["labeling"]["gamma"],
            reward_field=self.config["labeling"]["reward_field"],
            truncation_mode=self.config["labeling"]["truncation_mode"],
        )

    def fit_value(
        self, episodes: list[Path], destination: Path, *, enable_training: bool = False
    ) -> dict:
        if not enable_training:
            raise PermissionError("Value fitting requires --enable-training")
        import torch

        from rlinf.data.datasets.recap.hierarchical_value import episode_value_examples
        from rlinf.data.schema.hierarchical_types import load_episode_json
        from rlinf.workers.sft.fsdp_hierarchical_recap_worker import (
            IndependentValueTrainer,
        )

        examples, action_examples, high_examples = [], [], []
        for path in episodes:
            episode = load_episode_json(path)
            actions, high = episode_value_examples(episode, self._returns(episode))
            examples.extend(actions + high)
            action_examples.extend(actions)
            high_examples.extend(high)
        if not examples:
            raise ValueError("Value fitting requires actual recorded episodes")
        cfg = self.config["value"]
        lower, upper = cfg["model"]["v_min"], cfg["model"]["v_max"]
        if any(not lower <= sample["target_values"] <= upper for sample in examples):
            raise ValueError(
                "Value atom support does not cover observed returns; clipping is refused"
            )
        if cfg["batch_size"] * cfg["max_updates"] < 2:
            raise ValueError(
                "Value fitting needs both action and pre_read phases; increase batch_size or max_updates"
            )
        model, collator = self._value()
        try:
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=cfg["learning_rate"],
            )
            worker = IndependentValueTrainer(model, optimizer, enable_training=True)
            batches = (
                _move(collator(batch), self.config["runtime"]["device"])
                for batch in balanced_value_batches(
                    action_examples,
                    high_examples,
                    batch_size=cfg["batch_size"],
                    updates=cfg["max_updates"],
                )
            )
            metrics = worker.fit(batches, max_updates=cfg["max_updates"])
            destination.mkdir(parents=True, exist_ok=False)
            path = destination / "full_weights.pt"
            torch.save(model.state_dict(), path)
            self.value_checkpoint, self.value_version = (
                str(path),
                destination.parent.name + "/value",
            )
            return {
                **metrics,
                "checkpoint": str(path),
                "version": self.value_version,
                "independent_parameters": True,
                "asset_versions": asset_versions(self.config),
            }
        finally:
            del model
            gc.collect()

    def label(self, episodes: list[Path], destination: Path) -> list[Path]:
        if self.value_version == "unfitted" or not self.value_checkpoint:
            raise ValueError(
                "Advantage labels require an explicitly fitted value checkpoint"
            )
        import torch

        from rlinf.algorithms.offline.process.hierarchical import (
            compute_episode_advantages,
            quantile_threshold,
        )
        from rlinf.data.datasets.recap.hierarchical_value import episode_value_examples
        from rlinf.data.schema.hierarchical_types import (
            load_episode_json,
            save_episode_json,
        )

        model, collator = self._value()
        cfg = self.config["labeling"]
        predictions = []
        try:
            model.eval()

            def predict(examples, value_model):
                values = []
                for example in examples:
                    batch = _move(collator([example]), self.config["runtime"]["device"])
                    with torch.no_grad():
                        output = value_model(observation=batch["observation"])
                    values.append(
                        float(output.predicted_values.detach().float().reshape(-1)[0])
                    )
                return values

            for path in episodes:
                episode = load_episode_json(path)
                actions, high = episode_value_examples(episode)
                values_a, values_h = predict(actions, model), predict(high, model)
                labels = compute_episode_advantages(
                    episode,
                    values_a,
                    values_h,
                    gamma=cfg["gamma"],
                    action_horizon=cfg["action_horizon"],
                    reward_field=cfg["reward_field"],
                    truncation_mode=cfg["truncation_mode"],
                    action_threshold=0.0,
                    high_level_threshold=0.0,
                    value_version=self.value_version,
                )
                predictions.append((episode, labels, values_a, values_h))
            # Thresholds belong to the collected round, never individual episodes.
            boundary_a = quantile_threshold(
                [score for _, labels, _, _ in predictions for score in labels["AA"]],
                cfg["action_positive_fraction"],
            )
            boundary_h = quantile_threshold(
                [score for _, labels, _, _ in predictions for score in labels["AH"]],
                cfg["high_positive_fraction"],
            )
            destination.mkdir(parents=True, exist_ok=False)
            result = []
            for index, (episode, _, values_a, values_h) in enumerate(predictions):
                labels = compute_episode_advantages(
                    episode,
                    values_a,
                    values_h,
                    gamma=cfg["gamma"],
                    action_horizon=cfg["action_horizon"],
                    reward_field=cfg["reward_field"],
                    truncation_mode=cfg["truncation_mode"],
                    action_threshold=boundary_a,
                    high_level_threshold=boundary_h,
                    action_positive_fraction=cfg["action_positive_fraction"],
                    high_level_positive_fraction=cfg["high_positive_fraction"],
                    value_version=self.value_version,
                )
                episode.metadata["hierarchical_advantages"] = labels
                episode.metadata["return_targets"] = self._returns(episode)
                path = destination / f"episode_{index:04d}.json"
                save_episode_json(path, episode)
                result.append(path)
            return result
        finally:
            del model
            gc.collect()

    def fit_actor(
        self,
        episodes: list[Path],
        destination: Path,
        *,
        enable_training: bool = False,
        training_mode: str = "recap",
    ) -> dict:
        if not enable_training:
            raise PermissionError("Joint actor fitting requires --enable-training")
        import torch

        from rlinf.data.datasets.recap.hierarchical_collator import NativeJointCollator
        from rlinf.data.schema.hierarchical_types import load_episode_json
        from rlinf.workers.sft.fsdp_hierarchical_recap_worker import (
            FSDPHierarchicalReCapWorker,
        )

        cfg = self.config["actor_training"]
        stage = self.config["model"]["openpi"].get("recap_training_stage", "legacy")
        if stage in {"protocol", "video", "language_warmup"} and cfg["loss_weights"].get("fm", 0) > 0:
            raise ValueError(
                "Protocol/video stages freeze the action expert; disable FM."
            )
        data_cfg = self.config.get("data", {})
        excluded_families = data_cfg.get("excluded_family_ids", []) + data_cfg.get(
            "eval_family_ids", []
        )
        high_paths = [
            Path(path) for path in data_cfg.get("verified_high_sft_paths", [])
        ]
        if high_paths and training_mode != "verified_sft":
            raise ValueError(
                "High-only verified SFT sidecars cannot be relabelled as RECAP rollouts"
            )
        extra_high = load_verified_high_samples(
            high_paths,
            eval_seeds=self.config["rollout"]["eval_seeds"],
            excluded_families=excluded_families,
        )
        if episodes:
            high, actions = prepare_actor_samples(
                [load_episode_json(path) for path in episodes],
                training_mode=training_mode,
                loss_weights=cfg["loss_weights"],
                eval_seeds=self.config["rollout"]["eval_seeds"],
                excluded_families=excluded_families,
            )
            actions = select_actor_action_samples(actions, cfg["loss_weights"])
            high.extend(extra_high)
        else:
            high, actions = extra_high, []
            if (
                not high
                or cfg["loss_weights"].get("high_ce", 0) <= 0
                or any(cfg["loss_weights"].get(key, 0) > 0 for key in ("fast_ce", "fm"))
            ):
                raise ValueError("Independent high SFT sidecars support high_ce only")
        anchor_high, anchor_actions = [], []
        if self.verified_sft_paths:
            anchor_high, anchor_actions = prepare_actor_samples(
                [load_episode_json(path) for path in self.verified_sft_paths],
                training_mode="verified_sft",
                loss_weights=cfg["loss_weights"],
                eval_seeds=self.config["rollout"]["eval_seeds"],
                excluded_families=excluded_families,
            )
            anchor_actions = select_actor_action_samples(
                anchor_actions, cfg["loss_weights"]
            )
        if cfg["loss_weights"].get("kv_anchor", 0) > 0 and not anchor_actions:
            raise ValueError(
                "KV preservation requires a separate verified original-skill anchor buffer"
            )
        anchor_interval = int(cfg.get("anchor_interval", 4))
        if anchor_interval < 2:
            raise ValueError(
                "anchor_interval must be at least two to retain RECAP micro-batches"
            )
        if cfg["loss_weights"].get("fast_ce", 0) > 0:
            self._require_assets("fast")
        actor = self._actor()
        try:
            if cfg.get("gradient_checkpointing", False):
                actor.model.gradient_checkpointing_enable()
            optimizer = torch.optim.AdamW(
                [p for p in actor.model.parameters() if p.requires_grad],
                lr=cfg["learning_rate"],
                weight_decay=cfg["weight_decay"],
            )
            sampler_cfg = cfg.get("sampler", {})
            sampler = ActorMicrobatchSampler(
                high, actions, anchor_high, anchor_actions,
                batch_size=cfg["micro_batch_size"],
                gradient_accumulation=cfg["gradient_accumulation"],
                anchor_interval=anchor_interval, training_mode=training_mode,
                loss_weights=cfg["loss_weights"],
                shuffle=sampler_cfg.get("shuffle", False),
                seed=sampler_cfg.get("seed", 0),
            )
            trainer = FSDPHierarchicalReCapWorker(
                actor.model,
                optimizer,
                gradient_accumulation=cfg["gradient_accumulation"],
                max_grad_norm=cfg["max_grad_norm"],
                enable_training=True,
                sequential_objectives=cfg.get(
                    "sequential_objectives", stage != "legacy"
                ),
                sampler=sampler,
                strict_resume=cfg.get("strict_resume", False),
            )
            resume_path = cfg.get("resume_trainer_path")
            if resume_path:
                if Path(resume_path).resolve() != Path(self.actor_checkpoint).resolve():
                    raise ValueError(
                        "Optimizer resume must use the strictly loaded actor checkpoint"
                    )
                trainer.restore_trainer_state(Path(resume_path))
            collator = NativeJointCollator(
                actor.model,
                actor.tokenizer,
                loss_weights={**cfg["loss_weights"], "kv_anchor": 0.0},
                training_mode=training_mode,
            )
            micro_steps = cfg["max_updates"] * cfg["gradient_accumulation"]
            anchor_collator = NativeJointCollator(
                actor.model,
                actor.tokenizer,
                loss_weights=cfg["loss_weights"],
                training_mode="verified_sft",
            )

            def joint_batches():
                for _ in range(micro_steps):
                    h, a, is_anchor = sampler.next_batch()
                    if is_anchor:
                        batch = anchor_collator(
                            h, a,
                            anchor_samples=a
                            if cfg["loss_weights"].get("kv_anchor", 0) > 0
                            else None,
                        )
                        batch["supervision_source"] = "verified_sft_anchor"
                    else:
                        batch = collator(h, a)
                        if cfg["loss_weights"].get("kv_anchor", 0) > 0:
                            batch["loss_weights"] = {
                                **batch["loss_weights"], "kv_anchor": 0.0,
                            }
                        batch["supervision_source"] = training_mode
                    yield batch

            metrics = trainer.train(joint_batches(), max_updates=cfg["max_updates"])
            version = destination.parent.name + "/actor"
            trainer.save_checkpoint(
                destination,
                version=version,
                metadata={
                    "episodes": [str(path) for path in episodes],
                    "value_version": self.value_version
                    if training_mode == "recap"
                    else None,
                    "training_mode": training_mode,
                    "verified_high_sft_paths": [str(path) for path in high_paths],
                    "verified_sft_anchors": [
                        str(path) for path in self.verified_sft_paths
                    ],
                    "asset_versions": asset_versions(self.config),
                },
            )
            self.actor_checkpoint, self.actor_version = str(destination), version
            self.actor_training_mode = training_mode
            return {**metrics, "checkpoint": str(destination), "version": version,
                    "resume": trainer.resume_status,
                    "sampler_microbatch_cursor": sampler.microbatch_cursor,
                    "sampler_data_fingerprint": sampler.fingerprint}
        finally:
            del actor
            gc.collect()
