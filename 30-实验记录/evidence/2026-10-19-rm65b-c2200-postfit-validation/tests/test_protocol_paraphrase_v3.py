import importlib.util
from collections import Counter, defaultdict
from copy import deepcopy

spec=importlib.util.spec_from_file_location('paraphrase_v3','/tmp/verify_protocol_generalization_v3.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_two_equivalent_goals_for_each_unique_original_input():
    original=module.rows(module.LABELS/'val.reviewed.jsonl')
    data=module.paraphrase_rows(original)
    grouped=defaultdict(list)
    for row in data:
        grouped[row['text_intervention']['original_actor_input_digest']].append(row)
    assert len(grouped)==28 and len(data)==56
    assert Counter(x['mode'] for x in data)=={'READ':16,'WRITE':20,'SUBTASK':20}
    for pair in grouped.values():
        assert len(pair)==2
        assert pair[0]['context']['goal']!=pair[1]['context']['goal']
        assert pair[0]['target_text']==pair[1]['target_text']
        assert module.validate_target(pair[0]) and module.validate_target(pair[1])


def test_only_actor_goal_changes_with_no_new_physical_or_training_claim():
    original=module.rows(module.LABELS/'val.reviewed.jsonl')
    before=deepcopy(original)
    by_digest={}
    for row in original:
        by_digest.setdefault(row['actor_input_digest'],row)
    for row in module.paraphrase_rows(original):
        base=by_digest[row['text_intervention']['original_actor_input_digest']]
        assert {k:v for k,v in row['context'].items() if k!='goal'}=={k:v for k,v in base['context'].items() if k!='goal'}
        assert row['observation']==base['observation']
        assert row['target_text']==base['target_text']
        assert row['metadata']['training_eligible'] is False
        assert row['metadata']['evaluation_only'] is True
        assert row['provenance']['new_physical_event_claimed'] is False
        assert row['text_intervention']['modified_actor_context_fields']==['goal']
    assert original==before


def test_original_executable_files_remain_frozen():
    assert module.sha('/tmp/verify_protocol_generalization.py')=='c115912d6a5f8b0eb67bc65e5bd70276bd1a43955dfa09f4877fc4ef69d7e428'
    assert module.sha('/tmp/verify_protocol_generalization_v2.py')=='a9284fb04b0680da021d3b9156f605642edb7363bd075c590a7fb9acd8c7b46f'
