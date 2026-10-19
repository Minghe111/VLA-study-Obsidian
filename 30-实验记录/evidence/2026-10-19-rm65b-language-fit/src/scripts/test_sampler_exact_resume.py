"""CPU regression: actual optimizer/RNG continuation and strict sample contracts."""
import copy
import importlib
import random
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code' / 'RLinf'))
runner = importlib.import_module('rlinf.runners.hierarchical_recap_runner')
worker_module = importlib.import_module('rlinf.workers.sft.fsdp_hierarchical_recap_worker')
Sampler = runner.ActorMicrobatchSampler
Worker = worker_module.FSDPHierarchicalReCapWorker


def sampler(*, shuffle=True, seed=7, batch_size=2, high=None):
    return Sampler(list(range(9)) if high is None else high, list(range(5)),
                   list(range(3)), list(range(4)), batch_size=batch_size,
                   gradient_accumulation=2, anchor_interval=4,
                   training_mode='verified_sft', loss_weights={'high_ce': 1.},
                   shuffle=shuffle, seed=seed)


@pytest.mark.parametrize('shuffle', [False, True])
@pytest.mark.parametrize('split', [0, 1, 4, 17])
def test_same_microbatch_sequence_across_resume(shuffle, split):
    whole = sampler(shuffle=shuffle)
    expected = [whole.next_batch() for _ in range(42)]
    first = sampler(shuffle=shuffle)
    before = [first.next_batch() for _ in range(split)]
    second = sampler(shuffle=shuffle)
    second.load_state_dict(first.state_dict())
    after = [second.next_batch() for _ in range(42 - split)]
    assert before + after == expected
    assert second.state_dict() == whole.state_dict()


def test_default_sequence_preserves_old_cyclic_and_anchor_order():
    subject = sampler(shuffle=False)
    values = [subject.next_batch() for _ in range(5)]
    assert values == [([0, 1], [0, 1], False), ([2, 3], [2, 3], False),
                      ([4, 5], [4, 0], False), ([0, 1], [0, 1], True),
                      ([8, 0], [3, 4], False)]


@pytest.mark.parametrize('change', ['data', 'order', 'batch', 'seed'])
def test_changed_contract_rejected(change):
    subject = sampler()
    subject.next_batch()
    arguments = {'high': list(range(9))}
    if change == 'data':
        arguments['high'][4] = 99
    elif change == 'order':
        arguments['high'].reverse()
    elif change == 'batch':
        arguments['batch_size'] = 1
    elif change == 'seed':
        arguments['seed'] = 8
    with pytest.raises(ValueError, match='differs'):
        sampler(**arguments).load_state_dict(subject.state_dict())


def test_reference_file_content_is_fingerprinted(tmp_path):
    path = tmp_path / 'observation.npz'
    path.write_bytes(b'original')
    data = [{'observation': {'npz_path': str(path)}}]
    first = sampler(high=data)
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='fingerprint'):
        sampler(high=data).load_state_dict(first.state_dict())


def test_rng_roundtrip_weights_only_serialization(tmp_path):
    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    np.random.normal()  # Include NumPy's cached Gaussian state.
    state = worker_module.capture_training_rng()
    torch.save(state, tmp_path / 'rng.pt')
    expected = (random.random(), np.random.normal(), torch.rand(5))
    random.seed(23)
    np.random.seed(24)
    torch.manual_seed(25)
    worker_module.restore_training_rng(torch.load(tmp_path / 'rng.pt', weights_only=True))
    assert random.random() == expected[0]
    assert np.random.normal() == expected[1]
    assert torch.equal(torch.rand(5), expected[2])


@pytest.fixture
def manifest_stub(monkeypatch):
    # Avoid requiring a production VLM in a CPU optimizer continuity test.
    module = types.ModuleType('rlinf.models.embodiment.openpi.recap_lora')
    module.recap_training_manifest = lambda model: {'training_stage': 'legacy', 'parameters': ['weight']}
    monkeypatch.setitem(sys.modules, module.__name__, module)


class StochasticModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(1))

    def forward(self, batch):
        jitter = random.random() + np.random.random() + torch.rand(())
        loss = (self.weight * batch['value'] + jitter).square().mean()
        return {'loss': loss, 'high_ce': loss}


def make_worker(*, strict=True, accumulation=2):
    model = StochasticModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    subject = Worker(model, optimizer, sampler=sampler(), strict_resume=strict,
                     gradient_accumulation=accumulation, enable_training=True)
    return subject


def batches(subject, updates):
    for _ in range(updates * subject.gradient_accumulation):
        high, _, _ = subject.sampler.next_batch()
        yield {'value': torch.tensor(high, dtype=torch.float32) + .1}


def test_real_optimizer_save_resume_matches_uninterrupted(manifest_stub, tmp_path):
    random.seed(55)
    np.random.seed(56)
    torch.manual_seed(57)
    subject = make_worker()
    subject.train(batches(subject, 3), max_updates=3)
    checkpoint = subject.save_checkpoint(tmp_path / 'saved', version='test')
    subject.train(batches(subject, 4), max_updates=4)
    expected = subject.model.weight.detach().clone()
    expected_sampler = subject.sampler.state_dict()
    expected_rng = worker_module.capture_training_rng()
    resumed = make_worker()
    resumed.model.load_state_dict(torch.load(checkpoint / 'full_weights.pt', weights_only=True))
    status = resumed.restore_trainer_state(checkpoint)
    assert status['exact_resume'] is True
    resumed.train(batches(resumed, 4), max_updates=4)
    assert torch.equal(resumed.model.weight, expected)
    assert resumed.sampler.state_dict() == expected_sampler
    assert resumed.global_step == 7
    actual_rng = worker_module.capture_training_rng()
    assert torch.equal(actual_rng['torch_cpu'], expected_rng['torch_cpu'])
    assert actual_rng['python'] == expected_rng['python']
    assert actual_rng['numpy'] == expected_rng['numpy']


def test_legacy_resume_strict_refuses_and_permissive_warns(manifest_stub, tmp_path):
    subject = make_worker()
    checkpoint = subject.save_checkpoint(tmp_path / 'legacy', version='legacy')
    state = torch.load(checkpoint / 'trainer_state.pt', weights_only=True)
    for field in ('rng_state', 'sampler_state', 'batch_contract', 'resume_state_version'):
        state.pop(field)
    torch.save(state, checkpoint / 'trainer_state.pt')
    with pytest.raises(RuntimeError, match='Exact resume refused'):
        make_worker().restore_trainer_state(checkpoint)
    with pytest.warns(RuntimeWarning, match='NON-EXACT RESUME'):
        status = make_worker(strict=False).restore_trainer_state(checkpoint)
    assert status['exact_resume'] is False
    assert set(status['missing']) == {'rng_state', 'sampler_state', 'batch_contract', 'resume_state_version'}


def test_worker_accumulation_contract_refused(manifest_stub, tmp_path):
    subject = make_worker()
    checkpoint = subject.save_checkpoint(tmp_path / 'saved', version='test')
    with pytest.raises(RuntimeError, match='accumulation'):
        make_worker(accumulation=1).restore_trainer_state(checkpoint)


@pytest.mark.parametrize('version', [None, 0, 2])
def test_resume_schema_version_validated(manifest_stub, tmp_path, version):
    subject = make_worker()
    checkpoint = subject.save_checkpoint(tmp_path / 'saved', version='test')
    state = torch.load(checkpoint / 'trainer_state.pt', weights_only=True)
    if version is None:
        state.pop('resume_state_version')
    else:
        state['resume_state_version'] = version
    torch.save(state, checkpoint / 'trainer_state.pt')
    with pytest.raises(RuntimeError, match='resume_state_version'):
        make_worker().restore_trainer_state(checkpoint)
    if version is None:
        with pytest.warns(RuntimeWarning, match='NON-EXACT RESUME'):
            status = make_worker(strict=False).restore_trainer_state(checkpoint)
        assert not status['exact_resume']
    else:
        with pytest.raises(RuntimeError, match='resume_state_version'):
            make_worker(strict=False).restore_trainer_state(checkpoint)


def randomized_batches(subject, updates):
    for _ in range(updates * subject.gradient_accumulation):
        high, _, _ = subject.sampler.next_batch()
        # Mimic actual random collation before trainer prefetch, not just model RNG.
        jitter = random.random() + np.random.random() + float(torch.rand(()))
        yield {'value': torch.tensor(high, dtype=torch.float32) + jitter}


def test_randomized_collation_resume_equals_continuous(manifest_stub, tmp_path):
    random.seed(75)
    np.random.seed(76)
    torch.manual_seed(77)
    subject = make_worker()
    subject.train(randomized_batches(subject, 3), max_updates=3)
    checkpoint = subject.save_checkpoint(tmp_path / 'saved', version='test')
    subject.train(randomized_batches(subject, 4), max_updates=4)
    expected = subject.model.weight.detach().clone()
    expected_sampler = subject.sampler.state_dict()
    expected_rng = worker_module.capture_training_rng()
    resumed = make_worker()
    resumed.model.load_state_dict(torch.load(checkpoint / 'full_weights.pt', weights_only=True))
    resumed.restore_trainer_state(checkpoint)
    resumed.train(randomized_batches(resumed, 4), max_updates=4)
    assert torch.equal(resumed.model.weight, expected)
    assert resumed.sampler.state_dict() == expected_sampler
    actual_rng = worker_module.capture_training_rng()
    assert torch.equal(actual_rng['torch_cpu'], expected_rng['torch_cpu'])
    assert actual_rng['python'] == expected_rng['python']
    assert actual_rng['numpy'] == expected_rng['numpy']


@pytest.mark.parametrize('failure', ['collation', 'forward', 'backward'])
def test_training_failure_forbids_checkpoint(manifest_stub, tmp_path, failure):
    subject = make_worker()
    if failure == 'collation':
        def invalid_batches():
            yield from batches(subject, 1)
            subject.sampler.next_batch()
            random.random()
            raise RuntimeError('collation failed')
        source = invalid_batches()
    elif failure == 'forward':
        def invalid_forward(batch):
            raise RuntimeError('forward failed')
        subject.model.forward = invalid_forward
        source = batches(subject, 2)
    else:
        subject.model.weight.register_hook(lambda grad: (_ for _ in ()).throw(RuntimeError('backward failed')))
        source = batches(subject, 2)
    with pytest.raises(RuntimeError, match='failed'):
        subject.train(source, max_updates=2)
    with pytest.raises(RuntimeError, match='Cannot save'):
        subject.save_checkpoint(tmp_path / 'invalid', version='bad')
    assert not (tmp_path / 'invalid').exists()
    with pytest.raises(RuntimeError, match='requires restoring'):
        subject.train(batches(subject, 1), max_updates=1)


@pytest.mark.parametrize('field', ['microbatch_cursor', 'draws', 'epoch', 'cursor'])
def test_corrupt_sampler_cursor_consistency_refused(field):
    subject = sampler()
    for _ in range(5):
        subject.next_batch()
    state = subject.state_dict()
    if field == 'microbatch_cursor':
        state[field] += 1
    else:
        state['streams']['high'][field] += 1
    with pytest.raises(ValueError, match='inconsistent'):
        sampler().load_state_dict(state)


def test_valid_restore_recovers_failed_worker(manifest_stub, tmp_path):
    subject = make_worker()
    checkpoint = subject.save_checkpoint(tmp_path / 'valid', version='test')
    def invalid_batches():
        subject.sampler.next_batch()
        raise RuntimeError('collation failed')
        yield
    with pytest.raises(RuntimeError):
        subject.train(invalid_batches(), max_updates=1)
    subject.model.load_state_dict(torch.load(checkpoint / 'full_weights.pt', weights_only=True))
    subject.restore_trainer_state(checkpoint)
    subject.train(batches(subject, 1), max_updates=1)
    subject.save_checkpoint(tmp_path / 'recovered', version='recovered')
