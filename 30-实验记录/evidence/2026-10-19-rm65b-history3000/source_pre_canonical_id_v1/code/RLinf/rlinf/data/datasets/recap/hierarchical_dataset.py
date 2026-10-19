# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Causal training samples from raw observations and actual SDK24 execution."""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from rlinf.data.schema.hierarchical_types import (
    ACTION_DIM,
    CAMERA_KEYS,
    HierarchicalEpisode,
    load_episode_json,
)
from rlinf.data.storage.episode_memory import EpisodeMemory
from rlinf.data.schema.history_events import (
    HISTORY_VERSION,
    history_from_decisions,
    normalize_history,
)


def load_observation(
    reference: dict[str, Any],
    *,
    base_dir: str | Path | None = None,
    verify_contract: bool = True,
) -> dict[str, Any]:
    """Materialize an NPZ reference and verify native RGB8/state24 data.

    NPZ stores the three native camera keys and ``state``. It contains raw
    observations, never a latent cache, and is read with object loading disabled.
    """
    import numpy as np

    if "npz_path" in reference:
        path = Path(reference["npz_path"])
        if not path.is_absolute():
            if base_dir is None:
                raise ValueError(
                    "Relative observation path requires an explicit base_dir"
                )
            path = Path(base_dir) / path
        with np.load(path, allow_pickle=False) as archive:
            observation = {key: archive[key].copy() for key in archive.files}
    else:
        observation = dict(reference)
    if verify_contract:
        for key in CAMERA_KEYS:
            if key not in observation:
                raise ValueError(f"Missing actual native camera {key}")
            image = np.asarray(observation[key])
            if image.dtype != np.uint8 or image.shape != (480, 640, 3):
                raise ValueError(
                    f"{key} must be native 480x640 RGB8; found {image.shape}/{image.dtype}"
                )
        state = np.asarray(observation.get("state"))
        if state.shape != (ACTION_DIM,) or not np.isfinite(state).all():
            raise ValueError(
                "Observation state must contain 24 finite native SDK values"
            )
    return observation


def _labels(episode: HierarchicalEpisode, require_labels: bool) -> dict[str, Any]:
    labels = episode.metadata.get("hierarchical_advantages")
    if labels is None:
        if require_labels:
            raise ValueError(
                "Unlabelled rollout cannot silently become positive RECAP training data"
            )
        return {
            "IH": [None] * len(episode.decisions),
            "IA": [None] * len(episode.steps),
            "value_version": None,
        }
    if len(labels["IH"]) != len(episode.decisions) or len(labels["IA"]) != len(
        episode.steps
    ):
        raise ValueError("High/action labels do not align with the episode")
    if not labels.get("value_version"):
        raise ValueError("RECAP labels need explicit value-model provenance")
    for field in ("IH", "IA"):
        if any(not isinstance(item, bool) for item in labels[field]):
            raise ValueError(f"{field} labels must be booleans")
    return labels


def _catalogue(episode: HierarchicalEpisode, decision: Any) -> list[dict[str, str]]:
    catalogue = decision.tool_inputs.get("catalogue", episode.metadata.get("catalogue"))
    if not isinstance(catalogue, list) or not catalogue:
        raise ValueError("Record the fixed catalogue to reconstruct READ context")
    if any(set(item) != {"slot_id", "description"} for item in catalogue):
        raise ValueError("Actor catalogue must not reveal bank status or progress")
    return deepcopy(catalogue)


def validate_memory_trace(episode: HierarchicalEpisode) -> None:
    """Replay actual generated tools and reject fabricated actor-visible memory."""
    first_catalogue = _catalogue(episode, episode.decisions[0])
    bank = EpisodeMemory(
        first_catalogue,
        episode.env_id,
        episode.episode_id,
        strict_protocol=episode.metadata.get("strict_memory_protocol", False),
    )
    for decision in episode.decisions:
        if _catalogue(episode, decision) != first_catalogue:
            raise ValueError("The fixed catalogue changed inside an episode")
        if decision.memory_before != bank.critic_view():
            raise ValueError("Recorded pre-READ bank disagrees with causal replay")
        read = json.loads(decision.generated_read_text)
        if set(read) != {"read_ids"} or read["read_ids"] != decision.read_ids:
            raise ValueError("Recorded READ IDs disagree with actual generated text")
        selected = bank.read(decision.read_ids)
        if (
            selected != decision.selected_memory
            or decision.tool_outputs.get("selected_memory") != selected
        ):
            raise ValueError("Selected memory differs from the actual READ response")
        write = json.loads(decision.generated_write_text)
        if set(write) != {"operations"} or not isinstance(write["operations"], list):
            raise ValueError("Generated WRITE must contain an operations list")
        if len(write["operations"]) != len(decision.current_writes):
            raise ValueError("Memory operation count disagrees with recorded responses")
        written_ids = []
        for operation, response in zip(write["operations"], decision.current_writes):
            source = response.get("source")
            if bank.strict_protocol:
                if (
                    not isinstance(source, dict)
                    or type(source.get("step")) is not int
                    or source["step"] != decision.start_step
                ):
                    raise ValueError(
                        "Strict memory evidence step must equal the current decision start"
                    )
                observed_until = source.get("observed_until_step", source["step"])
                if (
                    type(observed_until) is not int
                    or not 0 <= observed_until <= decision.start_step
                ):
                    raise ValueError(
                        "Strict memory evidence cannot use a future observation"
                    )
                if (
                    not isinstance(source.get("policy_version"), str)
                    or not source["policy_version"]
                    or source["policy_version"] != decision.policy_version
                ):
                    raise ValueError(
                        "Strict memory source policy_version differs from its decision"
                    )
            result = bank.apply(operation, source=source)
            if result != response:
                raise ValueError("Recorded memory write differs from causal replay")
            if operation["slot_id"] not in written_ids:
                written_ids.append(operation["slot_id"])
        if decision.memory_after != bank.critic_view():
            raise ValueError("Recorded post-WRITE bank disagrees with causal replay")
        if decision.actor_memory != bank.actor_view(decision.read_ids, written_ids):
            raise ValueError(
                "Actor view contains unselected or fabricated current memory"
            )


def _visible_memory(decision: Any) -> dict[str, Any]:
    if any(
        item.get("slot_id") not in decision.read_ids
        for item in decision.selected_memory
    ):
        raise ValueError("Actor READ response contains an unselected memory slot")
    return deepcopy(decision.actor_memory)


def validate_history_trace(episode: HierarchicalEpisode) -> None:
    """Reject future history; new traces must exactly match actual prior windows."""
    strict = episode.metadata.get("history_version") == HISTORY_VERSION
    for index, decision in enumerate(episode.decisions):
        recorded = normalize_history(
            decision.tool_inputs.get("visible_history", []),
            observed_until_step=decision.start_step,
            allow_legacy=not strict,
        )
        if strict and recorded != history_from_decisions(
            episode.decisions[:index], observed_until_step=decision.start_step
        ):
            raise ValueError("Recorded history differs from prior actual decision windows")


def build_high_level_samples(
    episode: HierarchicalEpisode, *, require_labels: bool = True
) -> list[dict[str, Any]]:
    """Construct three generated-text targets per decision; tool bodies are inputs."""
    episode.validate()
    validate_memory_trace(episode)
    validate_history_trace(episode)
    labels = _labels(episode, require_labels)
    samples = []
    for index, decision in enumerate(episode.decisions):
        catalogue = _catalogue(episode, decision)
        visible = _visible_memory(decision)
        history = normalize_history(
            decision.tool_inputs.get("visible_history", []),
            observed_until_step=decision.start_step,
        )
        common = {
            "env_id": episode.env_id,
            "episode_id": episode.episode_id,
            "decision_id": decision.decision_id,
            "goal": episode.goal,
            "observation": deepcopy(episode.steps[decision.start_step].observation),
            "catalogue": catalogue,
            "visible_history": history,
            "observed_until_step": decision.start_step,
            "IH": labels["IH"][index],
        }
        stages = [
            ("READ", decision.generated_read_text, decision.read_token_ids, {}),
            (
                "WRITE",
                decision.generated_write_text,
                decision.write_token_ids,
                {
                    "read_ids": list(decision.read_ids),
                    "selected_memory": deepcopy(decision.selected_memory),
                },
            ),
            (
                "SUBTASK",
                decision.generated_subtask,
                decision.generated_token_ids,
                {"read_ids": list(decision.read_ids), "actor_memory": visible},
            ),
        ]
        for mode, text, tokens, additions in stages:
            if not text.strip():
                raise ValueError(
                    f"Missing actual generated {mode} text; do not invent a target"
                )
            context = {**deepcopy(common), "mode": mode, **deepcopy(additions)}
            samples.append(
                {
                    "sample_type": "high_level",
                    "mode": mode,
                    "context": context,
                    "observation": deepcopy(common["observation"]),
                    "target_text": text,
                    "target_token_ids": list(tokens),
                    "IH": labels["IH"][index],
                    "decision_index": index,
                    "value_version": labels["value_version"],
                    "critic_context": {
                        "critic_only": True,
                        "memory_before": deepcopy(decision.memory_before),
                    },
                    "versions": dict(episode.versions),
                }
            )
    return samples


def build_action_samples(
    episode: HierarchicalEpisode, *, horizon: int = 16, require_labels: bool = True
) -> list[dict[str, Any]]:
    """Reconstruct K1 behavior windows; proposal tails never become FAST labels.

    Short windows retain a valid FM mask and have no FAST target. No window
    crosses a subtask, memory-condition, policy-version or episode boundary.
    """
    episode.validate()
    validate_memory_trace(episode)
    if horizon <= 0:
        raise ValueError("Action horizon must be positive")
    labels = _labels(episode, require_labels)
    decision_map = {
        item.decision_id: (index, item) for index, item in enumerate(episode.decisions)
    }
    samples = []
    for index, first in enumerate(episode.steps):
        decision_index, decision = decision_map[first.decision_id]
        key = (
            first.decision_id,
            first.condition_id,
            first.policy_version,
            first.versions,
        )
        actual = []
        for step in episode.steps[index : index + horizon]:
            if (
                step.decision_id,
                step.condition_id,
                step.policy_version,
                step.versions,
            ) != key:
                break
            actual.append(list(step.executed_action))
            if step.terminated or step.truncated:
                break
        valid = len(actual)
        padded = actual + [[0.0] * ACTION_DIM for _ in range(horizon - valid)]
        mask = [[True] * ACTION_DIM for _ in range(valid)] + [
            [False] * ACTION_DIM for _ in range(horizon - valid)
        ]
        context = {
            "goal": episode.goal,
            "generated_subtask": decision.generated_subtask,
            "actor_memory": _visible_memory(decision),
            "IA": labels["IA"][index],
        }
        samples.append(
            {
                "sample_type": "action",
                "env_id": episode.env_id,
                "episode_id": episode.episode_id,
                "decision_id": first.decision_id,
                "step_index": index,
                "observation": deepcopy(first.observation),
                "context": context,
                "generated_subtask": decision.generated_subtask,
                "actor_memory": deepcopy(context["actor_memory"]),
                "actions": padded,
                "fm_valid_mask": mask,
                "valid_actions": valid,
                "fast_actions": deepcopy(actual) if valid == horizon else None,
                "fast_eligible": valid == horizon,
                "IA": labels["IA"][index],
                "IH": labels["IH"][decision_index],
                "value_version": labels["value_version"],
                "behavior_source": "actual_execution",
                "proposal_actions": deepcopy(first.proposal_actions),
                "physics_steps": [
                    step.physics_steps for step in episode.steps[index : index + valid]
                ],
                "critic_context": {
                    "critic_only": True,
                    "memory_after": deepcopy(decision.memory_after),
                },
                "versions": dict(episode.versions),
                "policy_version": first.policy_version,
            }
        )
    return samples


class HierarchicalDataset:
    """Small direct dataset of causal high-level and actual-action samples."""

    def __init__(
        self,
        episodes: list[HierarchicalEpisode],
        *,
        horizon: int = 16,
        require_labels: bool = True,
    ):
        self.episodes = episodes
        self.samples = []
        for episode in episodes:
            self.samples.extend(
                build_high_level_samples(episode, require_labels=require_labels)
            )
            self.samples.extend(
                build_action_samples(
                    episode, horizon=horizon, require_labels=require_labels
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return deepcopy(self.samples[index])

    @classmethod
    def from_paths(
        cls, paths: list[str | Path], **kwargs: Any
    ) -> "HierarchicalDataset":
        """Read standalone episode JSON or one episode per JSONL line."""
        episodes = []
        for path in paths:
            path = Path(path)
            if path.suffix == ".jsonl":
                with path.open(encoding="utf-8") as stream:
                    episodes.extend(
                        HierarchicalEpisode.from_dict(json.loads(line))
                        for line in stream
                        if line.strip()
                    )
            else:
                episodes.append(load_episode_json(path))
        return cls(episodes, **kwargs)
