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

"""Native attention adapters with unchanged base projection checkpoint keys."""

from __future__ import annotations

import math
import re
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class AttentionLoRALinear(nn.Linear):
    """Linear projection preserving ``weight`` while adding a low-rank delta."""

    def __init__(self, original: nn.Linear, rank: int, alpha: float):
        if rank < 1 or alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive.")
        super().__init__(
            original.in_features,
            original.out_features,
            bias=original.bias is not None,
            device=original.weight.device,
            dtype=original.weight.dtype,
        )
        self.enabled = True
        self.weight = original.weight
        self.bias = original.bias
        self.register_buffer("lora_scaling", original.weight.new_tensor(alpha / rank))
        self.lora_A = nn.Parameter(
            original.weight.new_empty(rank, original.in_features)
        )
        self.lora_B = nn.Parameter(
            original.weight.new_zeros(original.out_features, rank)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        result = F.linear(inputs, self.weight, self.bias)
        if not self.enabled:
            return result
        delta = F.linear(
            F.linear(inputs, self.lora_A.to(inputs.dtype)), self.lora_B.to(inputs.dtype)
        )
        return result + delta * self.lora_scaling.to(inputs.dtype)


def is_adapter_key(name: str) -> bool:
    return name.endswith(("lora_A", "lora_B", "lora_scaling"))


def is_recap_extension_key(name: str) -> bool:
    """Identify every newly initialized tensor needed in a compact save."""
    return is_adapter_key(name) or name in {
        "history_state_proj.weight",
        "history_state_proj.bias",
    }


def configure_recap_parameters(
    model,
    *,
    vlm_mode: str = "lora",
    rank: int = 16,
    alpha: float = 16.0,
    training_stage: str = "legacy",
    expert_rank: int = 8,
    expert_alpha: float = 8.0,
) -> dict[str, list[str]]:
    """Select shared VLM adapters and the full continuous action expert."""
    if training_stage != "legacy":
        return configure_mem_parameters(
            model,
            stage=training_stage,
            rank=rank,
            alpha=alpha,
            expert_rank=expert_rank,
            expert_alpha=expert_alpha,
        )
    if vlm_mode not in {"lora", "full", "frozen"}:
        raise ValueError(f"Unsupported RECAP VLM mode: {vlm_mode!r}")
    if vlm_mode == "lora":
        for layer in model.llm.layers:
            for attr in ("q_proj", "k_proj", "v_proj", "o_proj"):
                projections = getattr(layer.attn, attr)
                if projections[0] is not None and not isinstance(
                    projections[0], AttentionLoRALinear
                ):
                    projections[0] = AttentionLoRALinear(projections[0], rank, alpha)
    groups = {"vlm": [], "action_expert": [], "frozen": []}
    expert_pattern = re.compile(
        r"^llm\.(?:layers\.\d+\.(?:attn\.(?:q_proj|k_proj|v_proj|o_proj)|pre_attention_norms|pre_ffw_norms|mlps)|final_norms)\.1\."
    )
    for name, parameter in model.named_parameters():
        is_expert = bool(expert_pattern.match(name)) or name.startswith(
            (
                "action_in_proj.",
                "action_out_proj.",
                "time_mlp_in.",
                "time_mlp_out.",
                "state_proj.",
                "action_time_mlp_in.",
                "action_time_mlp_out.",
            )
        )
        is_vlm = not is_expert and name.startswith(("llm.", "img."))
        train_vlm = is_vlm and (
            vlm_mode == "full" or (vlm_mode == "lora" and is_adapter_key(name))
        )
        parameter.requires_grad_(is_expert or train_vlm)
        group = "action_expert" if is_expert else "vlm" if train_vlm else "frozen"
        groups[group].append(name)
    if not groups["action_expert"] or (vlm_mode != "frozen" and not groups["vlm"]):
        raise RuntimeError(
            "RECAP parameter selection produced an empty trainable group."
        )
    model.recap_parameter_groups = groups
    model.recap_vlm_mode = vlm_mode
    return groups


def configure_mem_parameters(
    model,
    *,
    stage: str,
    rank: int = 16,
    alpha: float = 16.0,
    expert_rank: int = 8,
    expert_alpha: float = 8.0,
) -> dict[str, list[str]]:
    """Select stage-specific adapters while preserving every pretrained tensor."""
    if stage not in {"protocol", "language_warmup", "video", "joint"}:
        raise ValueError(f"Unsupported MEM training stage: {stage!r}")
    # Both text-only stages bypass MEM at runtime.  The old protocol stage
    # retains its final-layer q/o ablation; language_warmup trains q/k/v/o in
    # every language layer without unfreezing vision, history or the expert.
    language_only = stage in {"protocol", "language_warmup"}
    architecture_mem = bool(
        getattr(
            getattr(model, "_config", None),
            "mem_enabled",
            getattr(model, "mem_enabled", False),
        )
    )
    if not language_only and not architecture_mem:
        raise ValueError("Video and joint stages require a MEM-enabled model.")
    for layer in model.llm.layers:
        for attr in ("q_proj", "k_proj", "v_proj", "o_proj"):
            projections = getattr(layer.attn, attr)
            for index, adapter_rank, adapter_alpha in (
                (0, rank, alpha),
                (1, expert_rank, expert_alpha),
            ):
                if projections[index] is not None and not isinstance(
                    projections[index], AttentionLoRALinear
                ):
                    projections[index] = AttentionLoRALinear(
                        projections[index], adapter_rank, adapter_alpha
                    )
    groups = {
        "vlm": [],
        "vision": [],
        "history_state": [],
        "action_expert": [],
        "frozen": [],
    }
    final_layer = len(model.llm.layers) - 1
    for name, parameter in model.named_parameters():
        group = "frozen"
        if is_adapter_key(name):
            language = bool(
                re.match(
                    r"^llm\.layers\.\d+\.attn\.(?:q_proj|k_proj|v_proj|o_proj)\.0\.",
                    name,
                )
            )
            expert = bool(
                re.match(
                    r"^llm\.layers\.\d+\.attn\.(?:q_proj|k_proj|v_proj|o_proj)\.1\.",
                    name,
                )
            )
            last_readout = any(
                name.startswith(f"llm.layers.{final_layer}.attn.{attr}.0.")
                for attr in ("q_proj", "o_proj")
            )
            if language and (stage != "protocol" or last_readout):
                group = "vlm"
            elif expert and stage == "joint":
                group = "action_expert"
            elif name.startswith("img.") and not language_only:
                group = "vision"
        elif name.startswith("history_state_proj.") and not language_only:
            group = "history_state"
        parameter.requires_grad_(group != "frozen")
        groups[group].append(name)
    if not groups["vlm"] or (stage == "joint" and not groups["action_expert"]):
        raise RuntimeError("MEM parameter selection produced an empty required group.")
    if not language_only and (not groups["vision"] or not groups["history_state"]):
        raise RuntimeError(
            "MEM video training requires visual adapters and history_state_proj."
        )
    model.recap_parameter_groups = groups
    model.recap_vlm_mode = "lora"
    model.recap_training_stage = stage
    model.mem_enabled = architecture_mem and not language_only
    return groups


@contextmanager
def legacy_prefix_context(model, *, disable_adapters: bool):
    """Temporarily bypass video/state tokens and optionally all adapter deltas."""
    adapters = []
    for name, module in model.named_modules():
        if isinstance(module, AttentionLoRALinear) or name.endswith(".mem_lora"):
            adapters.append((module, module.enabled))
    prior_mem = getattr(model, "mem_enabled", None)
    try:
        if prior_mem is not None:
            model.mem_enabled = False
        if disable_adapters:
            for module, _ in adapters:
                module.enabled = False
        yield
    finally:
        if prior_mem is not None:
            model.mem_enabled = prior_mem
        for module, enabled in adapters:
            module.enabled = enabled


def recap_training_manifest(model) -> dict:
    """Describe the exact trainable tensors required for optimizer resume."""
    return {
        "training_stage": getattr(model, "recap_training_stage", "legacy"),
        "vlm_mode": getattr(model, "recap_vlm_mode", None),
        "runtime_mem_enabled": bool(getattr(model, "mem_enabled", False)),
        "trainable_parameters": [
            {"name": name, "shape": list(parameter.shape), "numel": parameter.numel()}
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ],
        "mem_config": {
            key: getattr(model._config, key)
            for key in (
                "mem_enabled",
                "mem_num_frames",
                "mem_temporal_interval",
                "mem_drop_after",
                "mem_vision_lora_rank",
                "mem_vision_lora_alpha",
            )
            if hasattr(model, "_config") and hasattr(model._config, key)
        },
    }


def adapter_state_dict(model) -> dict[str, torch.Tensor]:
    """Save every adapter and the learned history-state projection together."""
    return {
        key: tensor.detach().cpu()
        for key, tensor in model.state_dict().items()
        if is_recap_extension_key(key)
    }


def load_adapter_weights(model, path: str | Path) -> None:
    """Restore every adapter tensor, rejecting missing or unrelated keys."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    expected = {key for key in model.state_dict() if is_recap_extension_key(key)}
    if not isinstance(state, dict) or set(state) != expected or not expected:
        actual = set(state) if isinstance(state, dict) else set()
        raise RuntimeError(
            f"Adapter checkpoint mismatch: missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys or any(
        is_recap_extension_key(k) for k in incompatible.missing_keys
    ):
        raise RuntimeError("Adapter checkpoint did not restore every adapter tensor.")
