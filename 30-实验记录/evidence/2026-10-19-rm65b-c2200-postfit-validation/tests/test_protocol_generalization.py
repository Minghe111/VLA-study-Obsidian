import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import torch

spec=importlib.util.spec_from_file_location('heldout_v1','/tmp/verify_protocol_generalization.py')
v1=importlib.util.module_from_spec(spec);spec.loader.exec_module(v1)
spec2=importlib.util.spec_from_file_location('heldout_v2','/tmp/verify_protocol_generalization_v2.py')
v2=importlib.util.module_from_spec(spec2);spec2.loader.exec_module(v2)


def test_actual_generation_call_is_no_grad_and_eos_is_unforced(monkeypatch,tmp_path):
    import rlinf.data.datasets.recap.hierarchical_dataset as module
    monkeypatch.setattr(module,'load_observation',lambda reference:{})
    row=v1.rows(v1.LABELS/'val.reviewed.jsonl')[0]
    class Actor:
        def generate(self,observation,mode,context):
            assert torch.is_grad_enabled() is False
            assert 'target_text' not in context
            return SimpleNamespace(text=row['target_text'],token_ids=[7,1],ended_with_eos=False)
    cases=v1.run_generation(Actor(),[row],tmp_path/'progress.json')
    assert cases[0]['checks']['json'] is True
    assert cases[0]['checks']['eos'] is False
    assert cases[0]['checks']['strict_passed'] is False


def test_counterfactual_goal_pairs_hold_memory_and_images_fixed():
    validation=v2.rows(v2.LABELS/'val.reviewed.jsonl')
    signatures={row['observation']['npz_path']:'same_image_state' for row in validation}
    data=v2.counterfactual_rows(validation,4,observation_signatures=signatures)
    assert len(data)==22  # One unique observation x2goals x11mode/memory cases.
    groups={}
    for row in data:
        assert v2.validate_target(row)
        groups.setdefault(row['counterfactual_group']['goal_pair'],[]).append(row)
    assert len(groups)==11
    for pair in groups.values():
        assert len(pair)==2 and pair[0]['observation']==pair[1]['observation']
        a={k:x for k,x in pair[0]['context'].items() if k!='goal'}
        b={k:x for k,x in pair[1]['context'].items() if k!='goal'}
        assert a==b
    assert all(row['metadata']['training_eligible'] is False for row in data)


def test_rule_labels_cover_empty_keep_correction_and_confidence():
    validation=v2.rows(v2.LABELS/'val.reviewed.jsonl')
    data=v2.counterfactual_rows(validation,1)
    operations=[]
    for row in data:
        if row['mode']=='WRITE':
            ops=json.loads(row['target_text'])['operations']
            operations.append('EMPTY' if not ops else ops[0]['op'])
    assert set(operations)=={'EMPTY','WRITE','KEEP','CORRECT'}
