"""Extract eight unused time points for evaluation-only local action retention.

The original successful expert episode is read-only. These are unused times
from the SAME episode, not an independent task-success or generalization set.
No observation, action, success label, or physical event is synthesized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import prepare_retention_anchor as anchor

ROOT = Path(__file__).resolve().parents[1]
HOLDOUT_STEPS = (16, 48, 80, 112, 144, 176, 208, 240)
TRAIN_ANCHOR_DEFAULT = Path(
    "/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/"
    "recap_memory_20261017/experiments/protocol_sft_20261018_v1/"
    "data/retention_anchor/manifest.json"
)


def actual_input_digest(raw):
    digest = hashlib.sha256()
    for key in anchor.KEY_MAP.values():
        array = np.ascontiguousarray(raw[key])
        digest.update(key.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(tuple(array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def build(source, output, train_manifest):
    source, output, train_manifest = source.resolve(strict=True), output.resolve(), train_manifest.resolve(strict=True)
    train = json.loads(train_manifest.read_text())
    if train['format'] != 'rm65b_k1_retention_anchor_v1' or train['source_episode'] != str(source):
        raise ValueError('Expected the existing training anchor from this same expert episode')
    train_steps = {case['source']['control_step'] for case in train['cases']}
    if set(HOLDOUT_STEPS) & train_steps or len(train_steps) != 8:
        raise ValueError('Evaluation-only times must be disjoint from eight training-anchor times')
    train_sha_before = anchor.sha256(train_manifest)

    # Reuse the original checked read_raw/audit_source/prepare implementation.
    # Only the in-process tuple changes; the original script bytes and assets
    # remain untouched. prepare refuses any pre-existing output directory.
    prior_steps = anchor.CASE_STEPS
    try:
        anchor.CASE_STEPS = HOLDOUT_STEPS
        summary = anchor.prepare(source, output)
    finally:
        anchor.CASE_STEPS = prior_steps
    manifest_path = output / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    instruction = manifest['cases'][0]['instruction']
    train_digests = set()
    for case in train['cases']:
        raw = anchor.read_raw(Path(case['source']['observation_npz']), instruction)
        train_digests.add(actual_input_digest(raw))
    holdout_digests = []
    case_audits = []
    for case in manifest['cases']:
        step = case['source']['control_step']
        case['case_id'] = f'legacy_single_bottle_heldout_step_{step:06d}'
        case['evaluation_only'] = True
        case['training_eligible'] = False
        original = anchor.read_raw(Path(case['source']['observation_npz']), instruction)
        with np.load(output / case['observation']['npz_path'], allow_pickle=False) as archive:
            stored = {key: archive[key].copy() for key in archive.files}
        if set(stored) != set(anchor.KEY_MAP.values()) or not all(np.array_equal(original[key], stored[key]) for key in original):
            raise ValueError('Prepared holdout must preserve every original model input value')
        digest = actual_input_digest(stored)
        if digest in train_digests or digest in holdout_digests:
            raise ValueError('Heldout actual three-camera/state input duplicates an anchor or another case')
        holdout_digests.append(digest)
        with np.load(output / case['noise']['npz_path'], allow_pickle=False) as archive:
            noise = archive['noise'].copy()
        expected = np.random.default_rng(20261018 + step).standard_normal((1, 16, 32)).astype(np.float32)
        if not np.array_equal(noise, expected):
            raise ValueError('Stored fixed noise differs from the original preparation seed recipe')
        case_audits.append({'case_id': case['case_id'], 'control_step': step,
            'actual_input_sha256': digest, 'original_and_prepared_arrays_identical': True,
            'fixed_noise_reproducible': True,
            'source_sha256': case['source']['sha256'],
            'observation_sha256': case['observation_sha256'], 'noise_sha256': case['noise_sha256']})
    manifest.update(split='heldout_retention', purpose='evaluation_only_unseen_timepoints_local_retention',
        training_eligible=False, used_for_KV_training=False,
        independent_episode=False, independent_task_success_evaluation=False,
        disjoint_training_anchor_steps=sorted(train_steps),
        training_anchor_manifest=str(train_manifest), training_anchor_manifest_sha256=train_sha_before)
    manifest['limitations'].insert(0, 'Heldout times from the SAME expert episode; disjoint KV training observations do not establish independent episode or task SR.')
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    if anchor.sha256(train_manifest) != train_sha_before:
        raise ValueError('Original training anchor manifest changed')
    summary.update(format='rm65b_retention_holdout_preparation_v1',
        split='heldout_retention', status='CPU_prepared_evaluation_only_not_model_evaluated',
        training_eligible=False, used_for_KV_training=False,
        selected_control_steps=list(HOLDOUT_STEPS),
        disjoint_training_anchor_steps=sorted(train_steps),
        actual_inputs_unique_and_disjoint=True,
        source_episode_independent=False, task_SR_measured=False,
        original_training_manifest_unchanged=True,
        training_anchor_manifest_sha256=train_sha_before,
        manifest_sha256=anchor.sha256(manifest_path),
        cases=case_audits)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (output / 'holdout_audit.json').write_text(json.dumps({
        'status': 'passed_CPU_extraction_and_input_disjointness_gate',
        'source_episode': str(source), 'case_count': len(case_audits),
        'source_readonly': True, 'original_arrays_preserved': True,
        'train_anchor_manifest_unchanged': True, 'no_GPU': True,
        'no_model_loaded': True, 'no_simulator_loaded': True,
        'KV_training_uses_these_cases': False,
        'heldout_actual_input_sha256': holdout_digests,
        'training_actual_input_sha256': sorted(train_digests),
        'cases': case_audits,
    }, indent=2) + '\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=anchor.SOURCE_DEFAULT)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/retention_holdout')
    parser.add_argument('--training-anchor-manifest', type=Path, default=TRAIN_ANCHOR_DEFAULT)
    args = parser.parse_args()
    summary = build(args.source, args.output, args.training_anchor_manifest)
    print(json.dumps({key: value for key, value in summary.items() if key != 'cases'}, indent=2))
