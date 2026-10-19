"""CPU metric checks: masks, padding, shapes and distinct physical units."""

import importlib.util
from pathlib import Path

import pytest
import torch

SPEC = importlib.util.spec_from_file_location("fit_retention_test_module", Path(__file__).with_name("fit_retention.py"))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_sdk_arm_and_hand_units_are_never_averaged_together():
    reference = torch.zeros(1, 16, 24)
    actual = reference.clone()
    actual[..., :6] = 0.1
    actual[..., 12:18] = 0.1
    actual[..., 6:12] = 100.0
    actual[..., 18:24] = 100.0
    normalized_reference = torch.zeros(1, 16, 32)
    normalized = normalized_reference.clone()
    normalized[..., :24] = 0.01
    normalized[..., 24:] = 1000.0
    result = MODULE.action_metrics(actual, reference, normalized, normalized_reference)
    assert result["sdk_arm_rad"]["rms"] == pytest.approx(0.1)
    assert result["sdk_hand_units"]["rms"] == pytest.approx(100.0)
    assert result["normalized_physical24"]["rms"] == pytest.approx(0.01)
    assert result["normalized_physical24"]["elements"] == 16 * 24
    assert result["normalized_physical24"]["max_abs"] == pytest.approx(0.01)


def test_kv_metric_ignores_masked_tokens_but_records_full_exactness():
    mask = torch.tensor([[True, False, True]])
    target = torch.ones(1, 3, 1, 2)
    actual = target.clone()
    actual[:, 1] = 1e6
    result = MODULE.kv_metrics([(actual, actual)], [(target, target)], mask, mask)
    assert result["relative_rms"] == 0.0
    assert result["valid_exact"]
    assert not result["all_positions_exact"]
    assert result["valid_elements"] == 2 * 2 * 2


def test_kv_relative_rms_pools_every_layer_key_and_value():
    mask = torch.tensor([[True, False, True]])
    target = torch.full((1, 3, 1, 2), 2.0)
    actual = target + 0.1
    actual[:, 1] = 1e6
    result = MODULE.kv_metrics([(actual, actual), (actual, actual)], [(target, target), (target, target)], mask, mask)
    assert len(result["layers"]) == 2
    assert result["rms"] == pytest.approx(0.1, rel=1e-5)
    assert result["relative_rms"] == pytest.approx(0.05, rel=1e-5)
    assert result["valid_elements"] == 16


def test_retention_shape_finite_and_mask_mismatch_fail_closed():
    with pytest.raises(ValueError, match="SDK"):
        MODULE.action_metrics(torch.zeros(1, 1, 24), torch.zeros(1, 1, 24), torch.zeros(1, 16, 32), torch.zeros(1, 16, 32))
    mask = torch.tensor([[True, False]])
    kv = torch.ones(1, 2, 1, 2)
    with pytest.raises(ValueError, match="prefix masks"):
        MODULE.kv_metrics([(kv, kv)], [(kv, kv)], mask, ~mask)
    with pytest.raises(ValueError, match="finite"):
        MODULE.tensor_error(torch.tensor([float("nan")]), torch.zeros(1))


def test_exact_zero_metric_retains_diagnostic_exact_flags():
    action = torch.randn(1, 16, 24)
    internal = torch.randn(1, 16, 32)
    metrics = MODULE.action_metrics(action, action, internal, internal)
    assert metrics["sdk_actions_exact"] and metrics["normalized_internal32_exact"]
    assert all(metrics[key]["rms"] == 0.0 for key in ("normalized_physical24", "sdk_arm_rad", "sdk_hand_units"))
