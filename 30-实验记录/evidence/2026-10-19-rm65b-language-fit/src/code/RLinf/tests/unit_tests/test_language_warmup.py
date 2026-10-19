"""CPU checks for the all-language-attention protocol fitting ablation."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from rlinf.models.embodiment.openpi import _apply_openpi_param_dtypes
from rlinf.models.embodiment.openpi.modules import gemma
from rlinf.models.embodiment.openpi.modules.utils import set_torch_compile
from rlinf.models.embodiment.openpi.pi0_config import Pi0Config
from rlinf.models.embodiment.openpi.recap_lora import (
    AttentionLoRALinear,
    configure_recap_parameters,
    recap_training_manifest,
)
from rlinf.models.embodiment.openpi.tasks.recap_hier import Pi0RecapHier


def small_actor(*, mem):
    torch.set_num_threads(2)
    torch.manual_seed(73)
    set_torch_compile(False)
    return Pi0RecapHier(
        Pi0Config(
            pi05=True,
            dtype="float32",
            vision_variant="mu/224",
            vocabulary_size=128,
            paligemma_variant="dummy",
            action_expert_variant="dummy",
            action_horizon=2,
            action_dim=4,
            max_token_len=4,
            mem_enabled=mem,
            mem_temporal_interval=1,
            mem_drop_after=1,
        )
    )


@pytest.mark.parametrize("mem", [False, True])
def test_language_warmup_only_trains_every_language_attention_adapter(mem):
    model = small_actor(mem=mem)
    groups = configure_recap_parameters(
        model, training_stage="language_warmup", rank=16, alpha=16
    )
    parameters = dict(model.named_parameters())
    expected = {
        f"llm.layers.{index}.attn.{projection}.0.lora_{matrix}"
        for index in range(len(model.llm.layers))
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj")
        for matrix in ("A", "B")
    }
    assert set(groups["vlm"]) == expected
    assert {name for name, parameter in parameters.items() if parameter.requires_grad} == expected
    assert not any(groups[key] for key in ("vision", "history_state", "action_expert"))
    assert not model.mem_enabled
    assert all(not parameter.requires_grad for parameter in model.llm.embedder.parameters())
    assert all(not parameters[name].requires_grad for name in groups["frozen"])


def test_stage_switch_keeps_checkpoint_layout_and_old_protocol_selection():
    model = small_actor(mem=True)
    layouts = []
    for stage in ("protocol", "language_warmup", "video", "joint", "protocol"):
        groups = configure_recap_parameters(model, training_stage=stage, rank=16, alpha=16)
        assert model.mem_enabled == (stage in {"video", "joint"})
        if stage == "protocol":
            assert len(groups["vlm"]) == 4
            final = len(model.llm.layers) - 1
            assert all(name.startswith(f"llm.layers.{final}.") for name in groups["vlm"])
        layouts.append(set(model.state_dict()))
    assert all(layout == layouts[0] for layout in layouts)


def test_zero_init_keeps_text_output_and_ce_gradient_only_reaches_language_lora():
    model = small_actor(mem=False)
    observation = model._config.fake_obs()
    target = torch.tensor([[5, 6, 1]])
    mask = torch.ones_like(target, dtype=torch.bool)
    model.eval()
    with torch.no_grad():
        original = model.text_loss(observation, target, mask)
    groups = configure_recap_parameters(model, training_stage="language_warmup", rank=2)
    with torch.no_grad():
        torch.testing.assert_close(model.text_loss(observation, target, mask), original, rtol=0, atol=0)
    model.train()
    model.text_loss(observation, target, mask).mean().backward()
    parameters = dict(model.named_parameters())
    assert all(parameters[name].grad is None for name in groups["frozen"])
    # B starts at zero: A gradients may be zero on the first backward, while
    # early and final language attention B tensors must receive actual signal.
    for layer in (0, len(model.llm.layers) - 1):
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            gradient = parameters[f"llm.layers.{layer}.attn.{projection}.0.lora_B"].grad
            assert gradient is not None and torch.isfinite(gradient).all()
            assert gradient.abs().sum().item() > 0


def test_language_stage_preserves_fp32_adapter_values_after_base_bf16_cast():
    model = small_actor(mem=True)
    groups = configure_recap_parameters(model, training_stage="language_warmup", rank=2)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in groups["vlm"]:
                parameter.fill_(0.1234567)
    _apply_openpi_param_dtypes(model, torch.bfloat16)
    parameters = dict(model.named_parameters())
    assert all(parameters[name].dtype == torch.float32 for name in groups["vlm"])
    assert all(torch.equal(parameters[name], torch.full_like(parameters[name], 0.1234567)) for name in groups["vlm"])
    assert model.llm.layers[0].attn.q_proj[0].weight.dtype == torch.bfloat16
    assert not model.mem_enabled


def test_production_gemma_parameter_count_is_measured_on_meta_not_estimated():
    # Build the real Gemma layer shapes without allocating multi-GB weights.
    with torch.device("meta"):
        model = nn.Module()
        model.llm = gemma.Module(
            [gemma.get_config("gemma_2b"), gemma.get_config("gemma_300m")],
            vocab_size=257152,
        )
        model._config = SimpleNamespace(mem_enabled=False)
        configure_recap_parameters(model, training_stage="language_warmup", rank=16, alpha=16)
    manifest = recap_training_manifest(model)
    assert manifest["training_stage"] == "language_warmup"
    assert not manifest["runtime_mem_enabled"]
    assert len(manifest["trainable_parameters"]) == 18 * 4 * 2
    assert sum(item["numel"] for item in manifest["trainable_parameters"]) == 3_686_400
    assert sum(isinstance(module, AttentionLoRALinear) for module in model.modules()) == 18 * 4 * 2
