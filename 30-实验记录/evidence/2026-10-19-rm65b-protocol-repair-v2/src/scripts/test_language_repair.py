"""CPU checks for explicit longer budgets and honest weights-only branches."""
from copy import deepcopy
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

import fit_comparison as fit
import run_language_repair as repair


def test_dry_run_does_not_create_output_or_update(tmp_path):
    destination = tmp_path / 'absent'
    result = fit.main(['fit', '--size', '96', '--total-updates', '600',
        '--max-fit-updates', '2000', '--output', str(destination)])
    assert result['status'] == 'planned'
    assert result['optimizer_updates_executed'] == 0
    assert not destination.exists()


@pytest.mark.parametrize('extra', [
    ['--total-updates', '301'],
    ['--size', '9', '--max-fit-updates', '2000'],
    ['--size', '96', '--max-fit-updates', '2401'],
    ['--size', '96', '--total-updates', '2001', '--max-fit-updates', '2000'],
    ['--learning-rate', 'nan'], ['--learning-rate', '-1'],
    ['--kv-weight', '-1'], ['--kv-weight', 'inf'],
    ['--resume', '/a', '--fork-from', '/b'],
    ['--fork-from', '/a', '--checkpoint', '/b'],
])
def test_bad_budgets_and_ambiguous_weight_sources_rejected(extra):
    with pytest.raises(SystemExit):
        fit.parse_arguments(['fit', '--output', '/unused', *extra])


def test_explicit_96_cap_accepted_default_stays_300():
    assert fit.parse_arguments(['fit', '--output', '/unused']).max_fit_updates == 300
    args = fit.parse_arguments(['fit', '--output', '/unused', '--size', '96',
        '--total-updates', '2000', '--max-fit-updates', '2000'])
    assert args.max_fit_updates == args.total_updates == 2000
    config = fit.build_config('language_warmup', 96, max_fit_updates=2000)
    assert config['actor_training']['max_updates'] == 2000
    assert config['experiment']['max_updates_per_run'] == 2000


def test_fork_metadata_requires_same_data_and_reports_source_step():
    checkpoint = {'global_step': 300, 'metadata': {'fit_dataset_size': 96, 'training_label_sha256': 'same'}}
    assert fit.validate_fork_checkpoint(checkpoint, size=96, labels_sha='same') == 300
    for size, digest in ((9, 'same'), (96, 'different')):
        with pytest.raises(ValueError, match='same audited fit labels'):
            fit.validate_fork_checkpoint(checkpoint, size=size, labels_sha=digest)


def test_branch_seed_is_common_and_independent_of_loading_randomness():
    fit.reset_branch_rng()
    expected = random.random(), float(np.random.random()), torch.rand(4)
    random.random(); np.random.random(); torch.rand(100)
    fit.reset_branch_rng()
    actual = random.random(), float(np.random.random()), torch.rand(4)
    assert actual[0] == expected[0] and actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def retention_receipt(passed=False, *, count=None, kv=.06, action=.01):
    if count is None:
        count = 8 if passed else 4
    return {'passed': passed, 'summary': {'count': 8, 'passed': count,
        'max_case_masked_all_layer_kv_relative_rms': kv,
        'max_case_normalized_physical24_rms': action}}


def report(ce=1, fit_count=0, retain=False, holdout=False):
    return {'evaluation': {'count': 96, 'teacher_forced_ce': ce,
        'all_fit_passed': fit_count == 96, 'fit_passed': {'passed': fit_count, 'eligible': 96}},
        'retention': retention_receipt(retain), 'retention_holdout': retention_receipt(holdout)}


def test_targets_c300_to_cap2000_no_overshoot():
    assert repair.planned_targets(300, 2000) == [600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
    assert repair.planned_targets(300, 650) == [600, 650]
    with pytest.raises(ValueError):
        repair.planned_targets(300, 2401)


def test_fit_only_cannot_stop():
    reports = [report(fit_count=96), report(fit_count=96)]
    assert repair.decide_stop(reports, 800, 2000) is None
    reports[-1]['retention'] = retention_receipt(True)
    reports[-1]['retention_holdout'] = retention_receipt(True)
    assert repair.decide_stop(reports, 800, 2000) is None
    reports[-2]['retention'] = retention_receipt(True)
    reports[-2]['retention_holdout'] = retention_receipt(True)
    assert repair.decide_stop(reports, 800, 2000) == 'converged_all96_and_local_retention_twice'


def test_plateau_never_before_1000_and_requires_ce_and_fit_stagnation():
    reports = [report(1), report(.99), report(.98)]
    assert repair.decide_stop(reports, 800, 2000) is None
    assert repair.decide_stop(reports, 1000, 2000) == 'plateau_ready_for_controlled_lr_kv_comparison'
    reports[-1] = report(.9)
    assert repair.decide_stop(reports, 1000, 2000) is None
    reports[-1] = report(.98, fit_count=1)
    assert repair.decide_stop(reports, 1000, 2000) is None


def test_cap_does_not_imply_convergence_or_capacity_failure():
    reports = [report(1), report(.8), report(.6)]
    assert repair.decide_stop(reports, 2000, 2000) == 'budget_exhausted_not_converged'


def test_orchestrator_dry_run_has_zero_effect(tmp_path):
    source = tmp_path / 'c300'
    source.mkdir()
    labels = tmp_path / 'fit96.train.jsonl'
    labels.write_text('[]')
    (tmp_path / 'effective_config.yaml').write_text('data:\n  verified_high_sft_paths:\n  - ' + str(labels) + '\n')
    (source / 'version.json').write_text(json.dumps({'global_step': 300,
        'metadata': {'fit_dataset_size': 96, 'training_label_sha256': fit.sha256(labels)}, 'training_manifest': {'training_stage': 'language_warmup'}}))
    destination = tmp_path / 'not-created'
    result = repair.main(['--resume', str(source), '--baseline', '/not-needed-in-dry-run', '--output', str(destination)])
    assert result['status'] == 'planned'
    assert result['additional_optimizer_update_cap'] == 1700
    assert not destination.exists()


def test_resume_rejects_version_trainer_step_disagreement():
    state = {'global_step': 299, 'optimizer': {'param_groups': [{'lr': 5e-5, 'weight_decay': 0.0}]}}
    with pytest.raises(ValueError, match='global_step differ'):
        fit.check_resume_lr(state, 5e-5, expected_step=300)
    fit.check_resume_lr(state, 5e-5, expected_step=299)


@pytest.mark.parametrize('pool', ['retention', 'retention_holdout'])
@pytest.mark.parametrize('metric', ['max_case_masked_all_layer_kv_relative_rms', 'max_case_normalized_physical24_rms'])
def test_retention_error_improvement_prevents_false_plateau(pool, metric):
    reports = [report(1, fit_count=96), report(.99, fit_count=96), report(.98, fit_count=96)]
    first = reports[0][pool]['summary'][metric]
    reports[1][pool]['summary'][metric] = first * .94
    reports[2][pool]['summary'][metric] = first * .93
    assert repair.decide_stop(reports, 1000, 2000) is None


def test_retention_pass_count_improvement_prevents_false_plateau():
    reports = [report(1, fit_count=96), report(.99, fit_count=96), report(.98, fit_count=96)]
    reports[-1]['retention_holdout']['summary']['passed'] = 5
    assert repair.decide_stop(reports, 1000, 2000) is None


@pytest.mark.parametrize('change', ['missing_summary', 'missing_metric', 'nan_metric', 'inconsistent_boolean', 'wrong_count'])
def test_incomplete_retention_receipts_never_establish_plateau_or_convergence(change):
    reports = [report(1, fit_count=96, retain=True, holdout=True) for _ in range(3)]
    for receipt in reports:
        target = receipt['retention_holdout']
        if change == 'missing_summary':
            target.pop('summary')
        elif change == 'missing_metric':
            target['summary'].pop('max_case_masked_all_layer_kv_relative_rms')
        elif change == 'nan_metric':
            target['summary']['max_case_masked_all_layer_kv_relative_rms'] = float('nan')
        elif change == 'inconsistent_boolean':
            target['summary']['passed'] = 7
        else:
            target['summary']['count'] = 7
    assert repair.decide_stop(reports, 1000, 2000) is None
    assert repair.decide_stop(reports, 2000, 2000) == 'budget_exhausted_not_converged'


def test_middle_checkpoint_progress_also_prevents_plateau_after_regression():
    reports = [report(1), report(.99), report(.98)]
    reports[1]['retention_holdout']['summary']['max_case_masked_all_layer_kv_relative_rms'] = .05
    assert repair.decide_stop(reports, 1000, 2000) is None


@pytest.mark.parametrize('mode', ['resume', 'fork'])
def test_cloned_config_reuses_source_label_path_without_stripping_provenance(tmp_path, mode):
    source = tmp_path / 'original' / 'checkpoint'
    source.mkdir(parents=True)
    original_labels = tmp_path / 'original' / 'fit96.train.jsonl'
    cloned_labels = tmp_path / 'clone' / 'fit96.train.jsonl'
    cloned_labels.parent.mkdir()
    original_labels.write_text('{"sample_id":"same","observation":{}}\n')
    cloned_labels.write_bytes(original_labels.read_bytes())
    (source.parent / 'effective_config.yaml').write_text('data:\n  verified_high_sft_paths:\n  - ' + str(original_labels) + '\n')
    (source / 'version.json').write_text(json.dumps({'metadata': {'training_label_sha256': fit.sha256(original_labels)}}))
    config = {'data': {'verified_high_sft_paths': [str(cloned_labels)]}}
    old = fit.read_records([original_labels])
    wrong = fit.read_records([cloned_labels])
    from rlinf.runners.hierarchical_recap_runner import actor_data_fingerprint
    assert actor_data_fingerprint({'high': old}) != actor_data_fingerprint({'high': wrong})
    selected, proof = fit.select_training_labels(config, checkpoint=source)
    restored = fit.read_records([selected])
    assert selected == original_labels
    assert config['data']['verified_high_sft_paths'] == [str(original_labels)]
    assert restored[0]['_source']['path'] == str(original_labels)
    assert actor_data_fingerprint({'high': old}) == actor_data_fingerprint({'high': restored})
    assert proof['checkpoint_provenance_preserved'] is True
    with pytest.raises(ValueError, match='source-path provenance'):
        fit.select_training_labels(config, checkpoint=source, explicit=cloned_labels)
    original_labels.write_text('changed')
    with pytest.raises(ValueError, match='metadata SHA'):
        fit.select_training_labels(config, checkpoint=source)
