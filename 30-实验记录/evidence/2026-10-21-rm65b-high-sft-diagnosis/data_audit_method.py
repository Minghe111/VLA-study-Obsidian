#!/usr/bin/env python3
"""Reproduce the read-only CPU history3000 data/case audit.

This script reads existing JSON/JSONL reports; it does not load a model, read
NPZ/image contents, tokenize text, train, or modify source data. Pixel diversity
uses recorded camera SHA metadata, and token totals use recorded token_lengths.
The earlier analysis was performed in temporary Python snippets. This saved
script consolidates those calculations and is run separately to reproduce them.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


DEFAULT_DATA = Path('/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/recap_memory_20261017/experiments/protocol_history_3000_20261019_v1')
DEFAULT_CASES = Path('/tmp/166_failure_analysis_20261021')
DEFAULT_VAULT = Path('/home/admin123/liminghe/VLA-WAM-RL')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def receipt(path):
    path = Path(path)
    return {'path': str(path.resolve()), 'sha256': sha256(path), 'bytes': path.stat().st_size}


def load_json(path):
    return json.loads(Path(path).read_text())


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def counts(values):
    return dict(sorted(Counter(values).items(), key=lambda item: str(item[0])))


def ratio(passed, eligible):
    return {'passed': passed, 'eligible': eligible}


def pixel_metadata_key(row):
    return json.dumps(row['evidence']['camera_pixel_sha256'], sort_keys=True)


def write_kind(text):
    try:
        operations = json.loads(text)['operations']
        return 'EMPTY' if not operations else '+'.join(op['op'] + ':' + op['slot_id'] for op in operations)
    except (KeyError, TypeError, ValueError):
        return 'INVALID_JSON_OR_KEY'


def has_identity_conflict(case, slot='sample_identity'):
    prefix = 'WRITE:input_conflict:' + slot + ':confirmed:'
    return case['mode'] == 'WRITE' and any(s.startswith(prefix) and ':side:' not in s for s in case['strata'])


def visible_write_oracle(case):
    """Derive gold from actual text prefix under the declared protocol rules.

    This does not claim an unconstrained VLM already knows those conventions.
    No evidence.truth_facts, target_text, pixels or source-state metadata is read.
    """
    prefix = case['actual_prefix']
    side = re.search(r'sample initially on the (left|right)', prefix).group(1)
    memory = json.loads(re.search(r'\nMemory: (.*?)\nHistory:', prefix, re.S).group(1))
    history = json.loads(re.search(r'\nHistory: (.*?)\nMode:', prefix, re.S).group(1))
    total = sum(event['executed_until_step'] - event['issued_at_step'] for event in history)
    truth = {'sample_identity': f'The first requested sample is initially on the {side}.'}
    if history:
        truth['current_progress'] = (
            f"Last issued {history[-1]['decision_id']} for initial-{side}; "
            f'executed_control_steps={total}; physical completion unverified.'
        )
    operations = []
    for entry in memory:
        slot = entry['slot_id']
        if slot not in truth:
            continue
        if entry['status'] == 'empty':
            operations.append(dict(op='WRITE', slot_id=slot, value=truth[slot], confirmed=True))
        elif entry['value'] != truth[slot]:
            operations.append(dict(op='CORRECT', slot_id=slot, value=truth[slot], confirmed=True))
        elif not entry['confirmed']:
            operations.append(dict(op='KEEP', slot_id=slot, confirmed=True))
    return {'operations': operations}


def dataset_stats(rows):
    input_targets = defaultdict(set)
    for row in rows:
        input_targets[row['actor_input_digest']].add(row['target_text'])
    correct = [r for r in rows if r['mode'] == 'WRITE' and r['metadata']['memory_condition'] == 'CORRECT']
    identity = [r for r in correct if r['evidence']['memory_trace']['controlled_target_slot'] == 'sample_identity']
    identity_cells = Counter()
    for row in identity:
        slot = next(m for m in row['context']['selected_memory'] if m['slot_id'] == 'sample_identity')
        cell = f"side:{row['metadata']['target_side']}:revision:{row['metadata']['active_revision']}:confirmed:{slot['confirmed']}"
        identity_cells[cell] += 1
    goal_history_conflicts = []
    for row in rows:
        side = re.search(r'sample initially on the (left|right)', row['context']['goal']).group(1)
        for event in row['context']['visible_history']:
            event_side = re.search(r'sample initially on the (left|right)', event['text']).group(1)
            if side != event_side:
                goal_history_conflicts.append(row['sample_id'])
                break
    return {
        'rows': len(rows),
        'modes': counts(r['mode'] for r in rows),
        'families': len({r['family_id'] for r in rows}),
        'episodes': len({r['episode_id'] for r in rows}),
        'npz_paths': len({r['observation']['npz_path'] for r in rows}),
        'physical_context_ids': len({r['metadata']['physical_context_id'] for r in rows}),
        'unique_recorded_raw_camera_hash_triplets': len({pixel_metadata_key(r) for r in rows}),
        'base_contexts': len({r['metadata']['base_context_id'] for r in rows}),
        'actor_input_digests': len(input_targets),
        'conflicting_actor_input_digests': sum(len(v) > 1 for v in input_targets.values()),
        'unique_target_strings': len({r['target_text'] for r in rows}),
        'unique_target_strings_per_mode': {m: len({r['target_text'] for r in rows if r['mode'] == m}) for m in ['READ', 'WRITE', 'SUBTASK']},
        'goal_template_counts': counts(str(r['metadata']['goal_template']) for r in rows),
        'side_counts': counts(r['metadata']['target_side'] for r in rows),
        'history_event_counts': counts(str(r['metadata']['text_history_events']) for r in rows),
        'control_observation_steps': sorted({r['decision_step'] for r in rows}),
        'nonempty_history_rows': sum(bool(r['context']['visible_history']) for r in rows),
        'subtask_without_writer_repair_rows': sum(r['mode'] == 'SUBTASK' and r['metadata']['stress_without_writer_repair'] for r in rows),
        'write_condition_counts': counts(r['metadata']['memory_condition'] for r in rows if r['mode'] == 'WRITE'),
        'write_correct_controlled_slots': counts(r['evidence']['memory_trace']['controlled_target_slot'] for r in correct),
        'identity_correct_rows': len(identity),
        'identity_correct_side_revision_confirmed_cells': dict(sorted(identity_cells.items())),
        'target_tokens_including_eos_recorded': sum(r['token_lengths']['target'] for r in rows),
        'target_tokens_by_mode_recorded': {m: sum(r['token_lengths']['target'] for r in rows if r['mode'] == m) for m in ['READ', 'WRITE', 'SUBTASK']},
        'goal_history_side_conflict_rows': goal_history_conflicts,
        'rows_declaring_motor_labels_false': sum(r['metadata']['motor_labels'] is False for r in rows),
    }


def suite_stats(report):
    cases = report['cases']
    output = {'rows': len(cases), 'strict_by_mode': {}}
    for mode in ['READ', 'WRITE', 'SUBTASK']:
        subset = [c for c in cases if c['mode'] == mode]
        if subset:
            output['strict_by_mode'][mode] = ratio(sum(c['checks']['strict_passed'] for c in subset), len(subset))
    output['overall_strict'] = ratio(sum(c['checks']['strict_passed'] for c in cases), len(cases))
    output['reported_overall_metrics'] = report['summary']['overall']
    output['conflicting_memory_cases'] = {}
    for slot in ['sample_identity', 'current_progress']:
        subset = [c for c in cases if has_identity_conflict(c, slot)]
        output['conflicting_memory_cases'][slot] = {
            'strict': ratio(sum(c['checks']['strict_passed'] for c in subset), len(subset)),
            'predicted_op_slot_counts': counts(write_kind(c['raw_text']) for c in subset),
            'actual_eos': ratio(sum(c['ended_with_eos'] for c in subset), len(subset)),
        }
    confusion = defaultdict(Counter)
    for case in cases:
        if case['mode'] == 'WRITE':
            confusion[write_kind(case['target_text'])][write_kind(case['raw_text'])] += 1
    output['write_op_slot_confusion'] = {k: dict(v) for k, v in confusion.items()}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--cases-dir', type=Path, default=DEFAULT_CASES)
    parser.add_argument('--vault-root', type=Path, default=DEFAULT_VAULT)
    parser.add_argument('--out', type=Path, default=DEFAULT_CASES / 'data_audit.json')
    args = parser.parse_args()
    started = datetime.now(timezone.utc).isoformat()
    label_paths = {s: args.data_root / 'data/history3000_labels_v1' / (s + '.jsonl') for s in ['train', 'val', 'test']}
    rows = {s: load_jsonl(p) for s, p in label_paths.items()}
    case_paths = {s: args.cases_dir / (s + '.json') for s in ['val180', 'train_quick96', 'revision_only40', 'history_ablation16']}
    reports = {s: load_json(p) for s, p in case_paths.items()}
    source_paths = {
        **{'labels_' + s: p for s, p in label_paths.items()},
        **{'final_cases_' + s: p for s, p in case_paths.items()},
        'dataset_manifest': args.data_root / 'data/history3000_labels_v1/manifest.json',
        'capture_manifest': args.data_root / 'data/dynamic_capture_v2/manifest.json',
        'independent_acceptance': args.data_root / 'reports/history3000_acceptance_v1/report.json',
        'final_eval_summary': args.vault_root / '30-实验记录/evidence/2026-10-19-rm65b-history3000-migrate166/status_20261021_final/eval_s024000_summary.json',
        'builder_snapshot_statically_reviewed': args.vault_root / '30-实验记录/evidence/2026-10-19-rm65b-history3000/source/scripts/build_history3000.py',
        'this_method': Path(__file__),
    }
    result = {
        'format': 'history3000_readonly_cpu_data_audit_v1',
        'execution': {
            'started_utc': started,
            'method': 'consolidated_saved_script_reproducing_prior_temporary_python_calculations',
            'scope': 'CPU JSON/JSONL data and previously generated final checkpoint cases only',
            'model_inference_executed': False, 'training_executed': False,
            'npz_or_image_contents_read': False, 'tokenizer_executed': False,
            'source_data_modified': False,
            'limitations': [
                'Image diversity uses camera SHA values already recorded in labels; image contents are not rehashed.',
                'Token totals use recorded token_lengths, not a new SentencePiece run.',
                'Case strict checks and EOS are inspected from preserved model-generation reports, not regenerated.',
                'Visible-prefix oracle assumes the declared memory rules; it is not evidence of natural-language model understanding.',
                'No causal attribution to dataset size, optimization, anchoring, or model capacity is established by this audit.',
            ],
        },
        'sources': {k: receipt(p) for k, p in source_paths.items()},
        'dataset_splits': {s: dataset_stats(rr) for s, rr in rows.items()},
        'cross_split_overlap': {},
        'final_checkpoint_suites': {s: suite_stats(d) for s, d in reports.items()},
    }
    field_functions = {
        'family_ids': lambda r: r['family_id'], 'episode_ids': lambda r: r['episode_id'],
        'npz_paths': lambda r: r['observation']['npz_path'],
        'recorded_raw_camera_hash_triplets': pixel_metadata_key,
        'base_context_ids': lambda r: r['metadata']['base_context_id'],
        'actor_input_digests': lambda r: r['actor_input_digest'],
    }
    for left, right in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        result['cross_split_overlap'][left + ':' + right] = {
            key: len({fn(r) for r in rows[left]} & {fn(r) for r in rows[right]})
            for key, fn in field_functions.items()
        }
    target_shift = {}
    for split in ['val', 'test']:
        target_shift[split] = {}
        for mode in ['READ', 'WRITE', 'SUBTASK']:
            train_targets = {r['target_text'] for r in rows['train'] if r['mode'] == mode}
            unseen = [r for r in rows[split] if r['mode'] == mode and r['target_text'] not in train_targets]
            target_shift[split][mode] = {
                'unseen_target_strings': len({r['target_text'] for r in unseen}),
                'unseen_target_rows': len(unseen),
                'unseen_by_condition': counts(r['metadata']['memory_condition'] for r in unseen),
                'examples': sorted({r['target_text'] for r in unseen})[:3],
            }
    result['target_distribution_shift_relative_to_train'] = target_shift
    oracle = {}; all_passed = 0; all_eligible = 0
    for suite in ['val180', 'train_quick96', 'revision_only40']:
        eligible = [c for c in reports[suite]['cases'] if c['mode'] == 'WRITE']
        mismatches = [c['sample_id'] for c in eligible if visible_write_oracle(c) != json.loads(c['target_text'])]
        oracle[suite] = {**ratio(len(eligible) - len(mismatches), len(eligible)), 'mismatch_sample_ids': mismatches}
        all_passed += len(eligible) - len(mismatches); all_eligible += len(eligible)
    result['visible_prefix_write_gold_oracle'] = {'combined': ratio(all_passed, all_eligible), 'suites': oracle}
    revision_groups = defaultdict(list)
    for case in reports['revision_only40']['cases']:
        revision_groups[case['sample_id'].split(':revision_only:')[0]].append(case)
    result['revision_only_structure'] = {
        'distinct_base_case_ids': len(revision_groups),
        'variants_per_base_case': counts(str(len(group)) for group in revision_groups.values()),
        'identity_correct_groups': [
            {'base_sample_id': key, 'variants': len(group), 'predictions': counts(c['raw_text'] for c in group),
             'all_have_eos': all(c['ended_with_eos'] for c in group)}
            for key, group in revision_groups.items() if has_identity_conflict(group[0])
        ],
        'warning': '40 rows are 8 fixed base cases times 5 revision substitutions, not 40 independent scenarios.',
    }
    example_requests = [
        ('val180', '2aa725ad48e9ea16', 'CORRECT', 'Goal/history left, confirmed wrong identity right; generated EMPTY.'),
        ('train_quick96', '49168d476215c165', 'CORRECT', 'Training case: goal/history left, unconfirmed wrong identity right; generated KEEP.'),
        ('val180', '86b0c66a52fbc18a', 'CORRECT', 'Wrong identity slot ignored; correct progress 80 replaced by progress 96.'),
        ('revision_only40', 'c96896b3f6c62044', 'WRITE', 'Midwindow progress 80 replaced by 96, invariant over all five revision variants.'),
    ]
    examples = []
    for suite, fragment, operation, description in example_requests:
        case = next(c for c in reports[suite]['cases'] if fragment in c['sample_id'] and ':WRITE:' + operation + ':' in c['sample_id'])
        file_lines = case_paths[suite].read_text().splitlines()
        line = next(i for i, text in enumerate(file_lines, 1) if case['sample_id'] in text)
        examples.append({
            'description': description, 'suite': suite, 'case_index_zero_based': reports[suite]['cases'].index(case),
            'source_path': str(case_paths[suite]), 'source_sample_id_line': line,
            **{key: case[key] for key in ['sample_id', 'mode', 'actual_prefix', 'actual_prefix_sha256', 'target_text', 'raw_text', 'ended_with_eos', 'strata']},
            'generated_tokens_including_eos_if_present': len(case['token_ids']),
            'oracle_target': visible_write_oracle(case),
        })
    result['examples'] = examples
    result['static_review_observations'] = [
        {
            'claim': 'Builder truth_facts are goal-role and issued-control facts; no physical completion is supervised.',
            'basis': 'Previously inspected CodeGraph current source; not automatically re-proven by this script.',
            'source_key': 'builder_snapshot_statically_reviewed', 'lines': [95, 108],
        },
        {
            'claim': 'Builder uses GOALS[0:4] for train, GOALS[4] for val, GOALS[5] for test; rows remain correlated augmentations.',
            'basis': 'Previously inspected CodeGraph source; template distributions and physical reuse are independently recomputed above.',
            'source_key': 'builder_snapshot_statically_reviewed', 'lines': [176, 216],
        },
        {
            'claim': 'Gold WRITE operations are reconstructable from visible Task/Memory/History under the declared rules; this audit found no hidden-truth requirement in the 132 final WRITE cases.',
            'basis': 'CPU script recomputation above, not model inference.',
        },
    ]
    assert result['dataset_splits']['train']['target_tokens_including_eos_recorded'] == 55707
    assert result['visible_prefix_write_gold_oracle']['combined'] == ratio(132, 132)
    result['execution']['finished_utc'] = datetime.now(timezone.utc).isoformat()
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output': str(args.out), 'bytes': args.out.stat().st_size,
                      'train_target_tokens': 55707, 'visible_oracle': result['visible_prefix_write_gold_oracle']['combined'],
                      'sources': len(result['sources'])}, ensure_ascii=False))


if __name__ == '__main__':
    main()
