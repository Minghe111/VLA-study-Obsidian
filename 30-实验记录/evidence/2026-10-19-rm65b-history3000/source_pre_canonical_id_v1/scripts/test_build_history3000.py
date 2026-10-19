"""CPU semantic and causal regressions for the new protocol data builder."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from build_history3000 import (
    CATALOGUE, GOALS, REVISIONS, OPERATIONS, balanced_select, memory_case,
    source_history, truth_facts,
)


def base(side='left', step=64):
    history = [dict(type='issued_subtask', decision_id=f'issued_{start:04d}',
                    text=f'Pick up and hold the sample initially on the {side}.',
                    issued_at_step=start, executed_until_step=min(step, start+32))
               for start in range(0, step, 32)]
    return dict(step=step, side=side, history=history,
                episode={'episode_id': 'cpu_fixture'},
                observation={'npz_path': '/not_loaded_by_memory_fixture.npz'})


@pytest.mark.parametrize('side', ['left', 'right'])
def test_truth_records_control_count_and_no_success(side):
    item = base(side, 48)
    fact = truth_facts(side, item['history'])
    assert fact['sample_identity'].endswith(side+'.')
    assert 'executed_control_steps=48;' in fact['current_progress']
    assert 'physical completion unverified' in fact['current_progress']
    assert 'issued_0032' in fact['current_progress']
    assert 'current_progress' not in truth_facts(side, [])


@pytest.mark.parametrize('revision', REVISIONS)
@pytest.mark.parametrize('slot', ['sample_identity', 'current_progress'])
def test_four_conditions_share_base_without_revision_operation_shortcut(revision, slot):
    item = base()
    results = {op: memory_case(item, op, revision, slot) for op in OPERATIONS}
    active = {op: next(x for x in values[0] if x['slot_id'] == slot)
              for op, values in results.items() if op != 'WRITE'}
    assert all(row['revision'] == revision and row['status'] == 'active' for row in active.values())
    assert active['KEEP']['value'] == active['EMPTY']['value']
    assert active['CORRECT']['value'] != active['KEEP']['value']
    assert active['KEEP']['confirmed'] is False
    assert active['EMPTY']['confirmed'] is True
    assert next(x for x in results['WRITE'][0] if x['slot_id'] == slot)['revision'] == 0
    others = [[x for x in values[0] if x['slot_id'] != slot] for values in results.values()]
    assert all(row == others[0] for row in others)
    for op, (selected, target, actor, trace) in results.items():
        assert trace['native_strict_protocol'] is True
        actual = next(x for x in actor['selected'] if x['slot_id'] == slot)
        assert actual['value'] == truth_facts(item['side'], item['history'])[slot]
        assert actual['confirmed'] is True
        assert actor['current_writes'] == [] if op == 'EMPTY' else len(actor['current_writes']) == 1
        assert target['operations'] == trace['operations']


@pytest.mark.parametrize('condition', ['WRITE', 'KEEP', 'CORRECT'])
def test_stress_without_writer_repair_is_not_falsely_marked_correct(condition):
    item = base()
    selected, target, actor, trace = memory_case(item, condition, 4, 'sample_identity', stress=True, bad_confirmed=True)
    assert target == {'operations': []} and actor['current_writes'] == []
    assert actor['selected'] == selected
    entry = next(x for x in selected if x['slot_id'] == 'sample_identity')
    if condition == 'WRITE':
        assert entry['status'] == 'empty'
    elif condition == 'KEEP':
        assert entry['confirmed'] is False
    else:
        assert entry['value'] != truth_facts('left', item['history'])['sample_identity']
        assert entry['confirmed'] is True


def test_source_history_clips_all_real_windows_and_ignores_future(tmp_path):
    issued = [dict(type='issued_subtask', event_id=f'issued_{start:04d}',
                   text='actual old issued text', issued_at_step=start,
                   planned_window_end_step=start+32) for start in range(0, 256, 32)]
    path = tmp_path/'issued.jsonl'
    path.write_text(''.join(json.dumps(row)+'\n' for row in issued))
    visible = [{**row, 'executed_until_step': min(80, row['planned_window_end_step'])}
               for row in issued if row['issued_at_step'] < 80]
    episode = {'issued_subtasks_path': str(path)}
    observation = {'step': 80, 'visible_history': visible, 'history_text': 'deliberately only last text'}
    first = source_history(episode, observation)
    assert len(first) == 3 and first[-1]['executed_until_step'] == 80
    assert sum(x['executed_until_step']-x['issued_at_step'] for x in first) == 80
    issued[-1]['text'] = 'changed future text and invented future success'
    path.write_text(''.join(json.dumps(row)+'\n' for row in issued))
    assert source_history(episode, observation) == first
    bad = deepcopy(observation);bad['visible_history'] = bad['visible_history'][-2:]
    with pytest.raises(ValueError, match='complete causal'):
        source_history(episode, bad)


def test_balanced_select_does_not_depend_on_available_future_outcomes():
    rows = [dict(base_context_id=f'{side}_{i}', side=side, actual_success=False)
            for side in ['left', 'right'] for i in range(12)]
    first = balanced_select(rows, 10, 17)
    assert [row['side'] for row in first] == ['left', 'right']*5
    for row in rows:
        row['actual_success'] = True
    assert [row['base_context_id'] for row in balanced_select(rows, 10, 17)] == [row['base_context_id'] for row in first]
    with pytest.raises(ValueError, match='Insufficient'):
        balanced_select(rows, 26, 17)


@pytest.mark.parametrize('index', range(len(GOALS)))
def test_all_declared_paraphrases_keep_explicit_first_role(index):
    goal = GOALS[index].format(first='left', second='right')
    assert goal.index('sample initially on the left') < goal.index('sample initially on the right')
    assert 'return' in goal or 'put it back home' in goal
