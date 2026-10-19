"""CPU tool checks, using real local NPZ/norm/tokenizer and transport fixtures.

History combinations created here are explicit unit-test contexts, not new
training rows or claims of a successful physical episode.
"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[4]
SPEC=importlib.util.spec_from_file_location('history3000_interface_tool',ROOT/'scripts/check_history3000_interface.py')
TOOL=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(TOOL)
SOURCE=Path('/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/recap_memory_20261017/experiments/protocol_repair_20261019_v2')
CONFIG=SOURCE/'reports/continuation_v1/language_warmup_96_2200/effective_config.yaml'
DATA=SOURCE/'data/fit_subsets_v1/fit96.train.jsonl'


@pytest.fixture
def interface_rows():
    if not DATA.is_file() or not CONFIG.is_file():pytest.skip('Optional existing local assets absent; no download')
    rows=[json.loads(line) for line in DATA.read_text().splitlines() if line.strip()]
    chosen=[]
    for mode in ['READ','SUBTASK']:
        chosen.append(next(r for r in rows if r['mode']==mode))
    for op in ['WRITE','KEEP','CORRECT','EMPTY']:
        for r in rows:
            if r['mode']!='WRITE':continue
            operations=json.loads(r['target_text'])['operations']
            if (operations[0]['op'] if operations else 'EMPTY')==op:
                chosen.append(r);break
    return deepcopy(chosen)


def save_rows(tmp_path,rows):
    path=tmp_path/'unit_contexts_not_training.jsonl'
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    return path


def test_real_paths_all_modes_four_operations_and_target_isolation(tmp_path,interface_rows):
    from rlinf.data.schema.history_events import issued_subtask_event
    events=[issued_subtask_event(decision_id=f'fixture{i}',text=f'Issued CPU fixture {i}; no completion asserted',
        issued_at_step=i,executed_until_step=i+1) for i in range(3)]
    for i,row in enumerate(interface_rows):
        row['observed_until_step']=3
        row['context']['visible_history']=events if i%2 else json.dumps(events)
    labels=save_rows(tmp_path,interface_rows)
    report=TOOL.run_check([labels],CONFIG,tmp_path/'report.json',batch_size=4)
    assert report['status']=='passed' and report['checked_rows']==6
    assert report['coverage']=='full_dataset'
    assert report['mode_counts']=={'READ':1,'SUBTASK':1,'WRITE':4}
    assert report['WRITE_counts']=={'WRITE':1,'KEEP':1,'CORRECT':1,'EMPTY':1}
    assert report['history_event_count_distribution']=={'3':6}
    assert report['sources_unchanged']
    assert all(r['actual_prefix_bytes_equal'] and r['actual_prefix_ids_equal'] and r['target_mutation_prefix_invariant'] for r in report['row_receipts'])
    assert report['model_and_image_stub']['parameters_loaded']==0
    assert not report['model_and_image_stub']['synthetic_EOS_is_model_prediction']


def test_explicit_partial_limit_cannot_be_reported_full(tmp_path,interface_rows):
    labels=save_rows(tmp_path,interface_rows)
    report=TOOL.run_check([labels],CONFIG,tmp_path/'partial.json',limit=2,batch_size=2)
    assert report['total_dataset_rows']==6 and report['checked_rows']==2
    assert report['coverage']=='explicit_partial_limit'


@pytest.mark.parametrize('bad', ['label_field','future_history','wrong_actor_envelope','unbounded_history'])
def test_invalid_inputs_fail_before_neural_transport(tmp_path,interface_rows,bad):
    from rlinf.data.schema.history_events import issued_subtask_event
    row=next(r for r in interface_rows if r['mode']=='SUBTASK')
    if bad=='label_field':row['context']['target_text']=row['target_text']
    elif bad=='wrong_actor_envelope':row['context']['actor_memory']=row['context']['actor_memory']['selected']
    else:
        row['context']['visible_history']=[issued_subtask_event(decision_id='fixture',text='issued',issued_at_step=0,executed_until_step=999)]
        if bad=='unbounded_history':
            row.pop('observed_until_step',None);row.pop('decision_step',None)
            row['context'].pop('observed_until_step',None)
            row['context'].pop('decision_step',None)
    labels=save_rows(tmp_path,[row])
    with pytest.raises(ValueError):TOOL.run_check([labels],CONFIG,tmp_path/'rejected.json')
