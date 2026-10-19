"""Build high-only protocol labels from fresh execution and native memory replay.

This program labels explicit goal roles and factual issued/control history. It
does not label physical skill success, learn from privileged monitor outcomes,
run a model, or invoke any optimizer. Existing captures/labels are never edited.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code/RLinf'))
from rlinf.data.schema.history_events import HISTORY_VERSION, control_decision_id, normalize_history
from rlinf.data.storage.episode_memory import EpisodeMemory
from rlinf.models.embodiment.openpi.recap_tokenizer import (
    PROMPT_VERSION, RecapTokenizer, build_high_prefix, canonical_json,
)
from build_protocol_labels import CATALOGUE, normalized_state24

VERSION = 'rm65b_dynamic_history_protocol_3000_v1'
CAMERAS = ('full_image', 'left_wrist_image', 'right_wrist_image')
REVISIONS = (1, 2, 4, 7, 10)
OPERATIONS = ('WRITE', 'KEEP', 'CORRECT', 'EMPTY')
GOALS = (
    'Verify the sample initially on the {first} and return it home, then verify the sample initially on the {second} and return it home.',
    'First verify the sample initially on the {first} and put it back home; afterward verify the sample initially on the {second} and return it home.',
    'Start with the sample initially on the {first}: verify it and return it home. Then verify the sample initially on the {second} and return it home.',
    'Process the sample initially on the {first} before the sample initially on the {second}; verify each sample and return each one home.',
    'Please verify and return home the sample initially on the {first} first, followed by the sample initially on the {second}.',
    'Do these in order: verify the sample initially on the {first} and return it home; then do the same for the sample initially on the {second}.',
)
SUBTASKS = (
    'Pick up and hold the sample initially on the {first}.',
    'Grasp and hold the sample initially on the {first}.',
    'Pick up the sample initially on the {first} and hold it.',
    'Lift and hold the sample initially on the {first}.',
    'Pick up and keep holding the sample initially on the {first}.',
    'Grasp the sample initially on the {first} and hold it.',
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def source_history(episode, observation):
    """Rebuild all actually started windows from the issued log, never last-two text."""
    step = observation['step']
    issued = jsonl(episode['issued_subtasks_path'])
    history = []
    for event in issued:
        start = event['issued_at_step']
        if start >= step:
            continue
        history.append(dict(
            type='issued_subtask', decision_id=event['event_id'], text=event['text'],
            issued_at_step=start,
            executed_until_step=min(step, event['planned_window_end_step']),
        ))
    history = normalize_history(history, observed_until_step=step, allow_legacy=False)
    raw_visible = observation['visible_history']
    projected = [{key: event['event_id'] if key == 'decision_id' else event[key]
                  for key in ('type', 'decision_id', 'text', 'issued_at_step', 'executed_until_step')}
                 for event in raw_visible]
    if history != projected:
        raise ValueError('Snapshot history differs from the complete causal issued log')
    if sum(event['executed_until_step']-event['issued_at_step'] for event in history) != step:
        raise ValueError('Observed issued windows do not cover actual executed controls')
    # Source IDs stay intact in the immutable capture. Model-visible IDs use
    # the real controller's control-step naming, after raw trace equality.
    return normalize_history([
        {**event, 'decision_id': control_decision_id(event['issued_at_step'])}
        for event in history
    ], observed_until_step=step, allow_legacy=False)


def truth_facts(side, history):
    """Only language-role and executed-control facts; no privileged physical outcome."""
    facts = {'sample_identity': f'The first requested sample is initially on the {side}.'}
    if history:
        total = sum(event['executed_until_step']-event['issued_at_step'] for event in history)
        facts['current_progress'] = (
            f"Last issued {history[-1]['decision_id']} for initial-{side}; "
            f'executed_control_steps={total}; physical completion unverified.'
        )
    return facts


def read_ids_for(history):
    return ['sample_identity', 'current_progress'] if history else ['sample_identity']


def source_for(base, kind='authored_counterfactual_memory_condition'):
    return dict(kind=kind, step=base['step'], observed_until_step=base['step'],
                policy_version=VERSION, teacher_version=VERSION,
                observation_refs=[{'npz_path': base['observation']['npz_path']}])


def wrong_fact(slot, facts, side, history):
    if slot == 'sample_identity':
        other = 'right' if side == 'left' else 'left'
        return f'The first requested sample is initially on the {other}.'
    # It contradicts the current, real control history, never invents a physical event.
    total = sum(event['executed_until_step']-event['issued_at_step'] for event in history)
    return (f"Last issued {history[-1]['decision_id']} for initial-{side}; "
            f'executed_control_steps={max(0, total-16)}; physical completion unverified.')


def memory_case(base, condition, revision, target_slot, *, stress=False, bad_confirmed=False):
    """Use the actual strict tool, including its latest-value selected semantics."""
    if condition not in OPERATIONS or revision not in REVISIONS:
        raise ValueError('Unknown controlled condition/revision')
    facts = truth_facts(base['side'], base['history'])
    if target_slot not in facts:
        raise ValueError('Target memory fact must be available from goal/history')
    ids = read_ids_for(base['history'])
    bank = EpisodeMemory(CATALOGUE, 'history3000_bank', base['episode']['episode_id'], strict_protocol=True)
    seed_rounds = []
    src = source_for(base)

    def seed(slot, value, confirmed, repetitions):
        for i in range(repetitions):
            bank.read([slot])
            operation = dict(op='WRITE' if i == 0 else 'CORRECT', slot_id=slot,
                             value=value, confirmed=confirmed)
            result = bank.apply(operation, source=src)
            seed_rounds.append(dict(read_ids=[slot], operation=operation,
                                    source=deepcopy(src), response=result))

    # All active conditions share exactly the same target revision r. The
    # nontarget selected slot is identical across a four-condition comparison.
    if condition != 'WRITE':
        value = wrong_fact(target_slot, facts, base['side'], base['history']) if condition == 'CORRECT' else facts[target_slot]
        seed(target_slot, value, condition == 'EMPTY' or (condition == 'CORRECT' and bad_confirmed), revision)
    other_slots = [slot for slot in ids if slot != target_slot]
    for slot in other_slots:
        repetitions = 1 if condition != 'WRITE' else revision+1
        seed(slot, facts[slot], True, repetitions)
    selected = bank.read(ids)
    if condition == 'EMPTY':
        operations = []
    elif condition == 'KEEP':
        operations = [dict(op='KEEP', slot_id=target_slot, confirmed=True)]
    else:
        operations = [dict(op=condition, slot_id=target_slot, value=facts[target_slot], confirmed=True)]
    if stress:
        operations = []  # explicit conditional context: writer did not repair the bank
    responses = [bank.apply(operation, source=source_for(base, 'formal_goal_and_issued_control_history'))
                 for operation in operations]
    actor = bank.actor_view(ids, [operation['slot_id'] for operation in operations])
    trace = dict(seed_rounds=seed_rounds, read_ids=ids, selected_memory=selected,
                 operations=operations, operation_responses=responses, actor_memory=actor,
                 native_strict_protocol=True, condition=condition, controlled_target_slot=target_slot,
                 controlled_active_revision=revision, stress_without_writer_repair=stress)
    return selected, {'operations': operations}, actor, trace


def load_bases(capture_path, *, preview=False):
    capture = json.loads(Path(capture_path).read_text())
    if capture.get('capture_kind') != 'fresh_simulation_reference_replay':
        raise ValueError('Require new, fresh reference execution, not old diagnostics')
    if not preview and (not capture.get('completed') or capture.get('status') != 'completed'):
        raise ValueError('Full source collection must pass before building train3000')
    bases = {split: [] for split in ('train', 'val', 'test')}
    episodes = capture['episodes']
    physical = {split: set() for split in bases}
    for episode in episodes:
        if not episode.get('completed') or not episode.get('worker_shutdown_confirmed'):
            raise ValueError('Source episode has not finished with a clean worker shutdown')
        if not episode['motion']['dynamic_state_and_rgb_observed']:
            raise ValueError('Reference episode did not produce measured motion and RGB change')
        if sha(episode['episode_json_path']) != episode['episode_json_sha256']:
            raise ValueError('Source episode receipt changed')
        for key in ('issued_subtasks', 'transitions'):
            path = episode[key+'_path']
            # Per-log receipts are independently checked by the source validator.
            if not Path(path).is_file():
                raise ValueError('Missing real execution log')
        split = episode['split']
        for observation in episode['observations']:
            step = observation['step']
            if observation['physics_step'] != step*15:
                continue  # a genuine partial terminal target is not a full 60ms sample
            if sha(observation['npz_path']) != observation['sha256']:
                raise ValueError('Actual observation changed')
            history = source_history(episode, observation)
            physical[split].add((episode['episode_id'], step))
            variants = range(4) if split == 'train' else (4 if split == 'val' else 5,)
            for template in variants:
                side = episode['target_side'];other = 'right' if side == 'left' else 'left'
                goal = GOALS[template].format(first=side, second=other)
                base = dict(episode=episode, observation=observation, step=step, history=history,
                            side=side, goal=goal, template=template,
                            subtask=SUBTASKS[template].format(first=side))
                base['base_context_id'] = digest(dict(episode=episode['episode_id'], step=step,
                                                      goal=goal, history=history))
                bases[split].append(base)
    return capture, bases, {split: len(values) for split, values in physical.items()}


def balanced_select(bases, count, seed):
    """Keep sides balanced without conditioning on model output or future reward."""
    if count % 2:
        raise ValueError('Balanced context selection needs an even count')
    rng = random.Random(seed)
    sides = {}
    for side in ('left', 'right'):
        choices = sorted((base for base in bases if base['side'] == side), key=lambda b: b['base_context_id'])
        rng.shuffle(choices)
        if len(choices) < count//2:
            raise ValueError(f'Insufficient distinct {side} contexts: {len(choices)} < {count//2}')
        sides[side] = choices[:count//2]
    return [base for pair in zip(sides['left'], sides['right']) for base in pair]


def model_prefix(record, state):
    context = record['context']
    return build_high_prefix(record['mode'], task=context['goal'], state=state,
                             memory_index=canonical_json(context['catalogue']),
                             visible_memory=canonical_json(context.get('actor_memory', context.get('selected_memory', {}))),
                             history=context['visible_history'])


def make_record(base, mode, target, *, trace=None, condition='none', revision=None, stress=False):
    episode = base['episode'];obs = base['observation'];step = base['step']
    normalize_history(base['history'], observed_until_step=step, allow_legacy=False)
    context = dict(mode=mode, goal=base['goal'], catalogue=deepcopy(CATALOGUE),
                   visible_history=deepcopy(base['history']), observed_until_step=step)
    if mode == 'WRITE':
        context.update(read_ids=trace['read_ids'], selected_memory=trace['selected_memory'])
    elif mode == 'SUBTASK':
        context.update(read_ids=trace['read_ids'], actor_memory=trace['actor_memory'])
    unique = f"{base['base_context_id']}:{mode}:{condition}:r{revision}:stress{int(stress)}"
    return dict(
        sample_id=VERSION+':'+unique, sample_type='high_level', mode=mode,
        split=episode['split'], seed=episode['seed'], family_id=episode['family_id'],
        env_id=episode['env_id'], episode_id=episode['episode_id'],
        decision_step=step, decision_physics_step=step*15, observed_until_step=step,
        history_version=HISTORY_VERSION, observation={'npz_path': obs['npz_path']},
        context=context, target_text=target, supervision_kind='verified_high_sft',
        provenance=dict(kind='formally_verified_high_protocol_sft', builder_version=VERSION,
                        capture_kind='fresh_simulation_reference_replay',
                        verification_scope=['goal_role', 'actual_issued_control_history', 'native_strict_memory_replay'],
                        not_verified=['visual_grounding', 'physical_skill_success', 'model_performance']),
        evidence=dict(kind='explicit_goal_and_causal_execution_history', observed_until_step=step,
                      truth_facts=truth_facts(base['side'], base['history']),
                      source_episode_json=episode['episode_json_path'],
                      source_episode_json_sha256=episode['episode_json_sha256'],
                      source_observation_sha256=obs['sha256'], camera_pixel_sha256=obs['cameras'],
                      source_issued_subtasks_path=episode['issued_subtasks_path'],
                      source_transitions_path=episode['transitions_path'],
                      source_history_id_map=[dict(source_event_id=event['event_id'],
                                                  decision_id=control_decision_id(event['issued_at_step']),
                                                  issued_at_step=event['issued_at_step'])
                                             for event in obs['visible_history']],
                      memory_trace=deepcopy(trace), physical_progress_claimed=False),
        metadata=dict(training_eligible=episode['split']=='train', motor_labels=False,
                      history_version=HISTORY_VERSION, history_id_encoding='runtime_control_step_h6',
                      label_builder_version=VERSION,
                      base_context_id=base['base_context_id'], physical_context_id=f"{episode['episode_id']}:s{step}",
                      goal_template=base['template'], memory_condition=condition,
                      active_revision=revision, target_side=base['side'],
                      stress_without_writer_repair=stress, text_history_events=len(base['history']),
                      behavior_source=episode['action_mapping'], goal_action_alignment_validated=False,
                      label_scope='protocol_rules_not_motor_success', MEM_ready=False),
    )


def build_records(capture, all_bases, tokenizer, state_stats):
    output = {};summaries = {};cache = {};seen = {}
    for split, count in (('train', 1000), ('val', 200), ('test', 200)):
        bases = balanced_select(all_bases[split], count, 8120+count+len(split))
        # Keep a few initial empty histories, but place them after the guaranteed
        # per-side dynamic stress grid; this uses observed history, never reward.
        sides = {side: sorted((base for base in bases if base['side'] == side),
                              key=lambda base: not bool(base['history']))
                 for side in ('left', 'right')}
        bases = [base for pair in zip(sides['left'], sides['right']) for base in pair]
        matched = balanced_select(bases, count//4, 9013+count)  # train250, eval50
        rows = []
        for base in bases:
            rows.append(make_record(base, 'READ', canonical_json({'read_ids': read_ids_for(base['history'])})))
        for index, base in enumerate(matched):
            revision = REVISIONS[index % len(REVISIONS)]
            target = 'current_progress' if base['history'] and index % 3 else 'sample_identity'
            for condition in OPERATIONS:
                # A wrong value requires correction even when an old writer
                # marked it confirmed; cross both flags with sides/revisions.
                bad_confirmed = (index//2 + index % 2) % 2 == 1
                _, target_json, _, trace = memory_case(base, condition, revision, target,
                                                       bad_confirmed=bad_confirmed)
                rows.append(make_record(base, 'WRITE', canonical_json(target_json), trace=trace,
                                        condition=condition, revision=revision))
        # One SUBTASK per distinct base. KEEP/CORRECT post-tool views can coincide
        # at the same base; never count them as distinct just by their sample IDs.
        for index, base in enumerate(bases):
            condition = OPERATIONS[(index//2) % 4]
            revision = REVISIONS[(index//8) % len(REVISIONS)]
            target = 'current_progress' if base['history'] and index % 3 else 'sample_identity'
            stress = index % 5 == 0
            if stress:
                per_side_stress_index = (index//5)//2
                condition = ('WRITE', 'KEEP', 'CORRECT')[per_side_stress_index % 3]
                target = ('current_progress' if base['history'] and (per_side_stress_index//3) % 2
                          else 'sample_identity')
            _, _, _, trace = memory_case(base, condition, revision, target, stress=stress,
                                         bad_confirmed=stress and (per_side_stress_index//6) % 2 == 1)
            rows.append(make_record(base, 'SUBTASK', base['subtask'], trace=trace,
                                    condition=condition, revision=revision, stress=stress))
        modes = Counter(row['mode'] for row in rows)
        assert modes == {'READ': count, 'WRITE': count, 'SUBTASK': count}
        operation_counts = Counter()
        for row in rows:
            path = row['observation']['npz_path']
            if path not in cache:
                with np.load(path, allow_pickle=False) as data:
                    state = np.asarray(normalized_state24(data['state'], state_stats), dtype=np.float32)
                    images = {camera: hashlib.sha256(data[camera].tobytes()).hexdigest() for camera in CAMERAS}
                cache[path] = state, images
            state, images = cache[path]
            prefix_ids = tokenizer.encode_prefix(model_prefix(row, state))
            target_ids = tokenizer.encode_target(row['target_text'])
            if len(prefix_ids)+len(target_ids) > 2048 or len(target_ids) > 96:
                raise ValueError('Actual rollout text budget would truncate this label')
            actor_digest = digest(dict(mode=row['mode'], prefix_ids=prefix_ids.tolist(), camera_pixel_sha256=images))
            if actor_digest in seen:
                raise ValueError('Duplicate actual actor input; row IDs cannot manufacture new examples: '+row['sample_id'])
            seen[actor_digest] = (split, row['target_text'])
            row['actor_input_digest'] = actor_digest
            row['token_lengths'] = dict(prefix=len(prefix_ids), target=len(target_ids), total=len(prefix_ids)+len(target_ids))
            if row['mode'] == 'WRITE':
                operation_counts[row['metadata']['memory_condition']] += 1
        assert operation_counts == {op: count//4 for op in OPERATIONS}
        random.Random(1776+count).shuffle(rows)
        output[split] = rows
        summaries[split] = dict(rows=len(rows), by_mode=dict(modes), write_conditions=dict(operation_counts),
                                distinct_actual_inputs=len(rows), distinct_base_contexts=len(bases),
                                selected_physical_contexts=len({row['metadata']['physical_context_id'] for row in rows}),
                                nonempty_history_rows=sum(bool(row['context']['visible_history']) for row in rows),
                                subtask_without_writer_repair=sum(row['metadata']['stress_without_writer_repair'] for row in rows),
                                unique_target_strings=len({row['target_text'] for row in rows}),
                                max_prefix_tokens=max(row['token_lengths']['prefix'] for row in rows),
                                max_target_tokens=max(row['token_lengths']['target'] for row in rows))
    return output, summaries


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture-manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--norm-stats', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--schema-preview', action='store_true')
    args = p.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    capture, bases, physical = load_bases(args.capture_manifest, preview=args.schema_preview)
    if args.schema_preview:
        first = next(base for group in bases.values() for base in group if base['history'])
        _, target, _, trace = memory_case(first, 'CORRECT', 2, 'sample_identity')
        rows = [make_record(first, 'READ', canonical_json({'read_ids': read_ids_for(first['history'])})),
                make_record(first, 'WRITE', canonical_json(target), trace=trace, condition='CORRECT', revision=2),
                make_record(first, 'SUBTASK', first['subtask'], trace=trace, condition='CORRECT', revision=2)]
        for row in rows:
            row['metadata'].update(training_eligible=False, documentation_preview=True)
            row['supervision_kind'] = 'protocol_candidate'
        args.output.mkdir(parents=True)
        (args.output/'schema_preview.jsonl').write_text(''.join(canonical_json(row)+'\n' for row in rows))
        print(json.dumps({'schema_preview': str(args.output), 'rows': len(rows), 'training_enabled': False}))
        return
    norm_file = json.loads(args.norm_stats.read_text());stats = norm_file.get('norm_stats', norm_file)['state']
    tokenizer = RecapTokenizer.from_local_paths(args.tokenizer, max_prefix_tokens=1400, max_target_tokens=96)
    labels, summary = build_records(capture, bases, tokenizer, stats)
    args.output.mkdir(parents=True)
    receipts = {}
    for split, rows in labels.items():
        path = args.output/(split+'.jsonl')
        path.write_text(''.join(canonical_json(row)+'\n' for row in rows))
        receipts[split] = dict(path=str(path.resolve()), sha256=sha(path), bytes=path.stat().st_size)
    report = dict(schema=VERSION, status='generated_waiting_independent_acceptance',
                  builder_script_sha256=sha(__file__), prompt_version=PROMPT_VERSION,
                  history_version=HISTORY_VERSION, capture_manifest=str(args.capture_manifest.resolve()),
                  capture_manifest_sha256=sha(args.capture_manifest), source_physical_context_counts=physical,
                  summary=summary, labels=receipts, training_enabled=False, optimizer_updates=0,
                  norm_stats=dict(path=str(args.norm_stats.resolve()), sha256=sha(args.norm_stats)),
                  tokenizer=dict(path=str(args.tokenizer.resolve()), sha256=sha(args.tokenizer)),
                  physical_skill_success_claimed=False, visual_grounding_verified=False,
                  MEM_ready=False, video_history_rows=0,
                  count_scope='Rows are correlated protocol contexts; reference execution is not policy rollout or motor supervision.')
    (args.output/'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'status': report['status'], 'summary': summary}, ensure_ascii=False))


if __name__ == '__main__':
    main()
