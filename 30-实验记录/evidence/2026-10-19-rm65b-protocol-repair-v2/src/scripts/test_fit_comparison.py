"""CPU-only tests of bounded CLI, stage configuration and evaluator accounting."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import fit_comparison as fit


def test_config_stage_scope_and_no_formal_dataset():
    narrow = fit.build_config('protocol', 9)
    broad = fit.build_config('language_warmup', 96)
    assert narrow['model']['model_path'] == broad['model']['model_path']
    for config, size in ((narrow, 9), (broad, 96)):
        assert config['actor_training']['strict_resume'] is True
        assert config['actor_training']['gradient_accumulation'] == 4
        assert config['actor_training']['max_updates'] == 300
        assert config['actor_training']['sampler'] == {'shuffle': True, 'seed': 20261019}
        assert config['actor_training']['loss_weights']['fast_ce'] == 0
        assert config['actor_training']['loss_weights']['fm'] == 0
        assert config['data']['verified_high_sft_paths'] == [str(fit.ROOT / f'data/fit_subsets_v1/fit{size}.train.jsonl')]
        assert config['contract']['physical_action_dim'] == 24
        assert config['contract']['physics_steps_per_action'] * config['contract']['physics_dt_s'] == .06
        assert len(config['contract']['camera_keys']) == 3
    # Last-layer q/o cannot affect the KV cache and must not backpropagate a constant.
    assert narrow['actor_training']['loss_weights']['kv_anchor'] == 0
    assert broad['actor_training']['loss_weights']['kv_anchor'] == 1


@pytest.mark.parametrize('arguments', [
    ['fit', '--execute'],
    ['fit', '--execute', '--enable-training', '--total-updates', '0'],
    ['fit', '--execute', '--enable-training', '--total-updates', '301'],
    ['fit', '--execute', '--enable-training', '--size', '3000'],
    ['fit', '--execute', '--enable-training', '--checkpoint', '/unused-warmstart'],
    ['eval', '--execute', '--resume', '/unused-a', '--checkpoint', '/unused-b'],
])
def test_cli_refuses_unapproved_or_out_of_bounds_before_runtime(arguments, tmp_path):
    destination = tmp_path / 'must_not_exist'
    with pytest.raises(SystemExit):
        fit.main([*arguments, '--output', str(destination)])
    assert not destination.exists()


@pytest.mark.parametrize('name,size', [
    ('fit9.train', 9), ('fit96.train', 96), ('fit96.quick12.train', 12),
])
def test_actual_audited_subset_passes_runtime_guard(name, size):
    rows = fit.read_records([fit.ROOT / f'data/fit_subsets_v1/{name}.jsonl'])
    fit.validate_fit_records(rows, size)
    audit = json.loads((fit.ROOT / 'reports/fit_subsets_audit.json').read_text())
    assert fit.sha256(fit.ROOT / f'data/fit_subsets_v1/{name}.jsonl') == audit['subsets'][name]['sha256']


@pytest.mark.parametrize('change', ['count', 'mode_balance', 'duplicate_digest', 'missing_digest', 'short_digest'])
def test_changed_fit_subset_rejected(change):
    rows = fit.read_records([fit.ROOT / 'data/fit_subsets_v1/fit9.train.jsonl'])
    if change == 'count':
        rows.extend(deepcopy(rows))
    elif change == 'mode_balance':
        rows[0]['mode'] = 'READ' if rows[0]['mode'] != 'READ' else 'WRITE'
    elif change == 'duplicate_digest':
        rows[0]['actor_input_digest'] = rows[1]['actor_input_digest']
    elif change == 'missing_digest':
        rows[0].pop('actor_input_digest')
    else:
        rows[0]['actor_input_digest'] = 'bad'
    with pytest.raises(ValueError, match='Fit'):
        fit.validate_fit_records(rows, 9)


def test_resume_optimizer_scope_matches_config():
    state = {'optimizer': {'param_groups': [{'lr': 5e-5, 'weight_decay': 0.0}]}}
    fit.check_resume_lr(state, 5e-5)


@pytest.mark.parametrize('groups', [
    [{'lr': 1e-4, 'weight_decay': 0.0}],
    [{'lr': 5e-5, 'weight_decay': 0.1}],
    [{'lr': 5e-5, 'weight_decay': 0.0}, {'lr': 1e-4, 'weight_decay': 0.0}],
])
def test_resume_optimizer_changed_lr_or_decay_rejected(groups):
    with pytest.raises(ValueError, match='LR/weight decay'):
        fit.check_resume_lr({'optimizer': {'param_groups': groups}}, 5e-5)


class FakeTokenizer:
    eos_token_id = 1

    @staticmethod
    def encode_target(text):
        return [ord(char) + 2 for char in text] + [1]


class FakeModel:
    def __init__(self):
        def decode(hidden):
            ids = hidden[:, 0].long()
            logits = torch.full((len(ids), 256), -30.0)
            logits[torch.arange(len(ids)), ids] = 30.0
            return logits
        self.llm = SimpleNamespace(embedder=SimpleNamespace(decode=decode))

    def eval(self):
        return self

    def __call__(self, *, batch):
        # Match production's selected-target-token chunking, including a
        # short final chunk and genuine EOS as the final supervised token.
        ids = batch['high_target_ids'][batch['high_loss_mask']]
        for chunk in ids.split(8):
            self.llm.embedder.decode(chunk[:, None].float())
        return {'high_ce': torch.tensor(0.0)}


class FakeCollator:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, high, actions):
        assert not actions and len(high) == 1
        ids = torch.tensor([FakeTokenizer.encode_target(high[0]['target_text'])])
        return {'high_target_ids': ids, 'high_loss_mask': torch.ones_like(ids, dtype=torch.bool)}


def sample_records():
    context = {'goal': 'Pick the left sample.', 'catalogue': {'sample_identity': 'Observed identity.'},
               'visible_history': [], 'read_ids': ['sample_identity'],
               'selected_memory': [{'slot_id': 'sample_identity', 'status': 'active',
                                    'confirmed': True, 'value': 'Left sample.'}]}
    return [
        {'sample_id': 'read', 'mode': 'READ', 'target_text': '{"read_ids":["sample_identity"]}',
         'context': deepcopy(context), 'observation': {}},
        {'sample_id': 'write', 'mode': 'WRITE', 'target_text': '{"operations":[]}',
         'context': deepcopy(context), 'observation': {}},
        {'sample_id': 'subtask', 'mode': 'SUBTASK', 'target_text': 'Pick up and hold the sample on the left.',
         'context': deepcopy(context), 'observation': {},
         'evaluation': {'target_side': 'left', 'required_terms': ['pick', 'hold']}},
    ]


def run_fake_evaluation(monkeypatch, tmp_path, *, missing_eos=False, subtask_variant=False):
    import rlinf.data.datasets.recap.hierarchical_collator as collator_module
    import rlinf.data.datasets.recap.hierarchical_dataset as dataset_module
    rows = sample_records()
    model = FakeModel()
    original_decode = model.llm.embedder.decode
    def generate(observation, mode, context):
        row = next(row for row in rows if row['mode'] == mode)
        text = row['target_text']
        if mode == 'SUBTASK' and subtask_variant:
            text = 'Pick and hold the left sample.'
        return SimpleNamespace(text=text, token_ids=FakeTokenizer.encode_target(text), ended_with_eos=not missing_eos)
    monkeypatch.setattr(fit, 'read_records', lambda paths: rows)
    monkeypatch.setattr(collator_module, 'NativeJointCollator', FakeCollator)
    monkeypatch.setattr(dataset_module, 'load_observation', lambda reference: {})
    actor = SimpleNamespace(model=model, tokenizer=FakeTokenizer(), generate=generate)
    path = tmp_path / 'evaluation.json'
    summary = fit.evaluate(actor, Path('/not_read'), path)
    # Production monkeypatch of decode must always be restored.
    assert model.llm.embedder.decode is original_decode
    report = json.loads(path.read_text())
    return rows, summary, report


def test_evaluator_json_denominator_and_chunked_teacher_forcing(monkeypatch, tmp_path):
    rows, summary, report = run_fake_evaluation(monkeypatch, tmp_path)
    assert summary['count'] == 3
    assert summary['json'] == {'passed': 2, 'eligible': 2}
    assert summary['schema_legal'] == {'passed': 2, 'eligible': 2}
    assert summary['eos'] == {'passed': 3, 'eligible': 3}
    assert summary['all_fit_passed'] is True
    expected_tokens = sum(len(FakeTokenizer.encode_target(row['target_text'])) for row in rows)
    assert summary['gold_correct'] == summary['gold_tokens'] == expected_tokens
    assert all(case['gold_eos_probability'] > .999 for case in report['cases'])


def test_evaluator_missing_eos_fails_despite_correct_json(monkeypatch, tmp_path):
    _, summary, _ = run_fake_evaluation(monkeypatch, tmp_path, missing_eos=True)
    assert summary['json']['passed'] == 2
    assert summary['eos'] == {'passed': 0, 'eligible': 3}
    assert summary['fit_passed']['passed'] == 0
    assert summary['all_fit_passed'] is False


def test_small_fit_subtask_exactness_is_stricter_than_lexical_proxy(monkeypatch, tmp_path):
    _, summary, report = run_fake_evaluation(monkeypatch, tmp_path, subtask_variant=True)
    case = next(case for case in report['cases'] if case['mode'] == 'SUBTASK')
    assert case['checks']['passed'] is True
    assert case['checks']['text_exact'] is False
    assert case['checks']['fit_passed'] is False
    assert summary['all_fit_passed'] is False
