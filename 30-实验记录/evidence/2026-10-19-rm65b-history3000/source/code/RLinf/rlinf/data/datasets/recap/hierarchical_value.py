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

"""Three-camera RM65-B examples for the independent categorical value model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_native_observation(reference: dict) -> dict:
    """Load native images and measured state, rejecting object/pickle archives."""
    if "npz_path" in reference:
        with np.load(Path(reference["npz_path"]), allow_pickle=False) as archive:
            result = {
                key: archive[key].copy()
                for key in (
                    "full_image",
                    "left_wrist_image",
                    "right_wrist_image",
                    "state",
                )
            }
    else:
        result = dict(reference)
    for key in ("full_image", "left_wrist_image", "right_wrist_image"):
        array = np.asarray(result[key])
        if array.shape != (480, 640, 3) or array.dtype != np.uint8:
            raise ValueError(
                f"Expected native RGB8 {key}[480,640,3], got {array.shape}/{array.dtype}"
            )
    state = np.asarray(result["state"], dtype=np.float32)
    if state.shape != (24,) or not np.isfinite(state).all():
        raise ValueError("Value observations require measured finite SDK24 state")
    result["state"] = state
    return result


def value_example(
    observation: dict,
    *,
    goal: str,
    phase: str,
    memory: dict,
    subtask: str = "",
    target: float | None = None,
    visible_history: Any = None,
    actor_memory: dict | None = None,
    observed_until_step: int | None = None,
) -> dict:
    """Encode phase and measured state in text; the value net is actor-disjoint."""
    if phase not in {"pre_read", "action"}:
        raise ValueError("Value phase must distinguish pre_read and action")
    native = load_native_observation(observation)
    context = {
        "goal": goal,
        "phase": phase,
        "memory": memory,
        "state": native["state"].tolist(),
    }
    if phase == "action":
        if not subtask:
            raise ValueError("Action value requires the recorded generated subtask")
        context["subtask"] = subtask
        context["actor_memory"] = actor_memory or {}
    else:
        from rlinf.data.schema.history_events import normalize_history

        context["visible_history"] = normalize_history(
            visible_history, observed_until_step=observed_until_step
        )
    images = {
        "base_0_rgb": native["full_image"],
        "left_wrist_0_rgb": native["left_wrist_image"],
        "right_wrist_0_rgb": native["right_wrist_image"],
    }
    result = {
        "images": images,
        "image_masks": dict.fromkeys(images, True),
        "prompt": json.dumps(context, ensure_ascii=False, separators=(",", ":")),
        "actions": None,
        "value_phase": phase,
    }
    if target is not None:
        if not np.isfinite(target):
            raise ValueError("Value target must be finite")
        result["target_values"] = float(target)
    return result


def episode_value_examples(
    episode: Any, returns: dict | None = None
) -> tuple[list[dict], list[dict]]:
    """Return separately aligned low-level and pre-READ value examples."""
    from rlinf.data.datasets.recap.hierarchical_dataset import (
        validate_history_trace,
        validate_memory_trace,
    )

    episode.validate()
    validate_memory_trace(episode)
    validate_history_trace(episode)
    decisions = {item.decision_id: item for item in episode.decisions}
    action_examples, high_examples = [], []
    for index, step in enumerate(episode.steps):
        decision = decisions[step.decision_id]
        target = None if returns is None else returns["action_returns"][index]
        action_examples.append(
            value_example(
                step.observation,
                goal=episode.goal,
                phase="action",
                memory=decision.memory_after,
                subtask=decision.generated_subtask,
                target=target,
                actor_memory=decision.actor_memory,
            )
        )
    for index, decision in enumerate(episode.decisions):
        target = (
            None if returns is None else returns["high_level_returns"][index]["return"]
        )
        high_examples.append(
            value_example(
                episode.steps[decision.start_step].observation,
                goal=episode.goal,
                phase="pre_read",
                memory=decision.memory_before,
                target=target,
                visible_history=decision.tool_inputs.get("visible_history", []),
                observed_until_step=decision.start_step,
            )
        )
    return action_examples, high_examples


class StrictValueCollator:
    """Refuse hidden context truncation before using the existing collator."""

    def __init__(self, processor: Any, max_length: int = 1024, *, train: bool = False):
        from rlinf.models.embodiment.value_model.recap.data_collator import (
            ValueDataCollator,
        )

        self.processor, self.max_length = processor, max_length
        self.collator = ValueDataCollator(processor, max_length=max_length, train=train)

    def __call__(self, examples: list[dict]) -> dict:
        for example in examples:
            text = self.processor._build_prefix_text(example["prompt"])
            length = len(self.processor.tokenizer.encode(text, add_special_tokens=True))
            if length > self.max_length:
                raise ValueError(
                    f"Value context has {length} tokens; budget is {self.max_length}; truncation is disabled"
                )
        return self.collator(examples)
