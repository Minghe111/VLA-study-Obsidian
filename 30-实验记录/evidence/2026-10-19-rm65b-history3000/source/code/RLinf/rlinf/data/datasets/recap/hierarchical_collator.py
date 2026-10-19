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

"""Native joint-actor batching from causal high and actually executed actions."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from rlinf.data.datasets.recap.hierarchical_dataset import load_observation
from rlinf.data.schema.history_events import history_from_sample


def _json(value: Any) -> str:
    from rlinf.models.embodiment.openpi.recap_tokenizer import canonical_json

    return canonical_json(value)


def _advantage(value: bool) -> str:
    if not isinstance(value, bool):
        raise ValueError(
            "A training sample requires an explicit boolean advantage label"
        )
    return "positive" if value else "negative"


def training_advantage(sample: dict, label: str, *, training_mode: str) -> str | None:
    """Only explicitly verified SFT may omit reward-derived conditioning."""
    if training_mode == "verified_sft":
        provenance = sample.get("provenance")
        if (
            sample.get("supervision_kind")
            not in (
                {"verified_sft", "verified_high_sft"}
                if label == "IH"
                else {"verified_sft"}
            )
            or not isinstance(provenance, (str, dict))
            or not provenance
            or (isinstance(provenance, str) and not provenance.strip())
        ):
            raise ValueError(
                "Verified SFT samples require supervision_kind=verified_sft and nonempty provenance"
            )
        return None
    if training_mode != "recap":
        raise ValueError("training_mode must be recap or verified_sft")
    return _advantage(sample.get(label))


class NativeJointCollator:
    """Use the actor's original SDK24 transforms and separate answer streams."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        loss_weights: dict | None = None,
        training_mode: str = "recap",
    ):
        if training_mode not in {"recap", "verified_sft"}:
            raise ValueError("training_mode must be recap or verified_sft")
        self.training_mode = training_mode
        self.model, self.tokenizer = model, tokenizer
        self.loss_weights = (
            dict(loss_weights)
            if loss_weights is not None
            else {"high_ce": 1.0, "fast_ce": 1.0, "fm": 1.0}
        )
        if any(float(value) < 0 for value in self.loss_weights.values()):
            raise ValueError("Joint loss weights must be nonnegative")

    def _processed(self, reference: dict, actions: Any = None):
        native = load_observation(reference)
        from rlinf.models.embodiment.openpi.mem_observation import (
            attach_mem_history,
            native_transform_input,
        )

        processed = self.model.input_transform(
            native_transform_input(native, actions), transpose=False
        )
        observation = self.model._observation_dict_to_device(processed)
        if getattr(self.model, "mem_enabled", False):
            observation = attach_mem_history(self.model, observation, reference)
        return native, observation, processed.get("actions")

    def _observations(
        self, bases: list, prompts: list[str], targets: list[np.ndarray] | None = None
    ):
        import torch

        ids = [self.tokenizer.encode_prefix(text) for text in prompts]
        for index, tokens in enumerate(ids):
            total = len(tokens) + (len(targets[index]) if targets is not None else 0)
            if total > self.model.max_token_len:
                raise ValueError(
                    f"Joint text needs {total} tokens; max_token_len is {self.model.max_token_len}; no truncation allowed"
                )
        size = max(map(len, ids))
        tokens = torch.zeros(len(ids), size, dtype=torch.long, device=self.model.device)
        mask = torch.zeros_like(tokens, dtype=torch.bool)
        for index, values in enumerate(ids):
            tokens[index, : len(values)] = torch.as_tensor(
                values, device=self.model.device
            )
            mask[index, : len(values)] = True

        def stack(values):
            if all(value is None for value in values):
                return None
            if any(value is None for value in values):
                raise ValueError(
                    "A microbatch must use one consistent MEM history schema"
                )
            if isinstance(values[0], dict):
                return {
                    key: stack([value[key] for value in values]) for key in values[0]
                }
            return torch.cat(values, dim=0)

        first = bases[0]
        fields = {
            field.name: stack([getattr(base, field.name) for base in bases])
            for field in dataclasses.fields(first)
            if field.name
            not in {
                "tokenized_prompt",
                "tokenized_prompt_mask",
                "token_ar_mask",
                "token_loss_mask",
            }
        }
        return dataclasses.replace(
            first,
            **fields,
            tokenized_prompt=tokens,
            tokenized_prompt_mask=mask,
            token_ar_mask=None,
            token_loss_mask=None,
        )

    def _targets(self, targets: list[np.ndarray], valid: list[bool] | None = None):
        import torch

        count = max(map(len, targets))
        result = torch.full(
            (len(targets), count),
            self.tokenizer.eos_token_id,
            dtype=torch.long,
            device=self.model.device,
        )
        token_mask = torch.zeros_like(result, dtype=torch.bool)
        loss_mask = torch.zeros_like(result, dtype=torch.bool)
        for index, value in enumerate(targets):
            result[index, : len(value)] = torch.as_tensor(
                value, device=self.model.device
            )
            token_mask[index, : len(value)] = True
            if valid is None or valid[index]:
                loss_mask[index, : len(value)] = True
        return result, loss_mask, token_mask

    def __call__(
        self,
        high_samples: list[dict] | None = None,
        action_samples: list[dict] | None = None,
        *,
        anchor_samples: list[dict] | None = None,
    ) -> dict:
        """Materialize enabled/present branches without touching disabled codecs."""
        import torch

        from rlinf.models.embodiment.openpi.recap_tokenizer import (
            build_action_prefix,
            build_high_prefix,
        )

        enabled = {
            key: float(self.loss_weights.get(key, 0.0)) > 0
            for key in ("high_ce", "fast_ce", "fm")
        }
        result = {"loss_weights": dict(self.loss_weights)}
        high_bases, high_prompts, high_targets = [], [], []
        if enabled["high_ce"]:
            for sample in high_samples or []:
                _, base, _ = self._processed(sample["observation"])
                context, mode = sample["context"], sample["mode"]
                visible = context.get(
                    "actor_memory", context.get("selected_memory", {})
                )
                high_prompts.append(
                    build_high_prefix(
                        mode,
                        task=context["goal"],
                        state=base.state[0].detach().float().cpu().numpy(),
                        visible_memory=_json(visible),
                        memory_index=_json(context["catalogue"]),
                        history=history_from_sample(sample),
                        advantage=training_advantage(
                            sample, "IH", training_mode=self.training_mode
                        ),
                    )
                )
                high_bases.append(base)
                high_targets.append(self.tokenizer.encode_target(sample["target_text"]))
        if high_bases:
            ids, loss_mask, mask = self._targets(high_targets)
            result.update(
                high_observation=self._observations(
                    high_bases, high_prompts, high_targets
                ),
                high_target_ids=ids,
                high_loss_mask=loss_mask,
                high_target_mask=mask,
            )

        action_bases, action_prompts, actions, valid_masks = [], [], [], []
        fast_bases, fast_prompts, fast_targets = [], [], []
        if enabled["fm"] or enabled["fast_ce"]:
            for sample in action_samples or []:
                eligible = enabled["fast_ce"] and bool(
                    sample.get("fast_eligible", False)
                )
                if not enabled["fm"] and not eligible:
                    continue
                if sample.get("behavior_source") != "actual_execution":
                    raise ValueError(
                        "Proposal tails cannot be used as actual behavior labels"
                    )
                _, base, transformed = self._processed(
                    sample["observation"], sample["actions"]
                )
                context = sample["context"]
                prompt = build_action_prefix(
                    task=context["goal"],
                    subtask=context["generated_subtask"],
                    state=base.state[0].detach().float().cpu().numpy(),
                    visible_memory=_json(context["actor_memory"]),
                    advantage=training_advantage(
                        sample, "IA", training_mode=self.training_mode
                    ),
                )
                if transformed is None:
                    raise ValueError(
                        "Original action normalization pipeline did not return actions"
                    )
                transformed = transformed.to(
                    device=self.model.device, dtype=torch.float32
                )
                if enabled["fm"]:
                    physical_mask = np.asarray(sample["fm_valid_mask"], dtype=bool)
                    if physical_mask.shape != (16, 24):
                        raise ValueError("RM65-B requires fm_valid_mask[16,24]")
                    mask = torch.zeros_like(transformed, dtype=torch.bool)
                    mask[..., :24] = torch.as_tensor(
                        physical_mask, device=self.model.device
                    )
                    action_bases.append(base)
                    action_prompts.append(prompt)
                    actions.append(transformed)
                    valid_masks.append(mask)
                if eligible:
                    fast_bases.append(base)
                    fast_prompts.append(prompt)
                    fast_targets.append(
                        self.tokenizer.encode_fast(
                            transformed[0].detach().cpu().numpy()
                        )
                    )
        if action_bases:
            result.update(
                action_observation=self._observations(action_bases, action_prompts),
                actions=torch.cat(actions),
                action_valid_mask=torch.cat(valid_masks),
            )
        if fast_bases:
            ids, loss_mask, mask = self._targets(fast_targets)
            result.update(
                fast_observation=self._observations(
                    fast_bases, fast_prompts, fast_targets
                ),
                fast_target_ids=ids,
                fast_loss_mask=loss_mask,
                fast_target_mask=mask,
                fast_sample_weights=torch.ones(
                    len(fast_bases), device=self.model.device
                ),
            )
        if float(self.loss_weights.get("kv_anchor", 0.0)) > 0 and anchor_samples:
            anchor_bases, anchor_prompts = [], []
            for sample in anchor_samples:
                if sample.get("behavior_source") != "actual_execution":
                    raise ValueError(
                        "Retention anchors require actual executed skill provenance"
                    )
                training_advantage(sample, "IA", training_mode="verified_sft")
                _, base, _ = self._processed(sample["observation"])
                context = sample["context"]
                anchor_bases.append(base)
                anchor_prompts.append(
                    build_action_prefix(
                        task=context["goal"],
                        subtask=context["generated_subtask"],
                        state=base.state[0].detach().float().cpu().numpy(),
                        visible_memory=_json(context["actor_memory"]),
                        advantage=None,
                    )
                )
            result["anchor_observation"] = self._observations(
                anchor_bases, anchor_prompts
            )
        if len(result) == 1:
            raise ValueError("Batch has no samples for any enabled loss branch")
        for objective, field in (
            ("high_ce", "high_observation"),
            ("fast_ce", "fast_observation"),
            ("fm", "action_observation"),
            ("kv_anchor", "anchor_observation"),
        ):
            if field not in result:
                result["loss_weights"][objective] = 0.0
        return result
