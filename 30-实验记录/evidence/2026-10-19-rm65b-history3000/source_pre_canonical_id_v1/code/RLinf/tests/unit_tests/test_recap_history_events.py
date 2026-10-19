"""CPU contracts for causal history and exact native/collator prompt parity.

No actor weights, environment backend, GPU or optimizer are used. The native
actor and training collator entry points run with a CPU observation and an EOS
transport stub, so the actual prompt construction and real local tokenizer are
tested without claiming learned generation or physical execution.
"""

import dataclasses
from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rlinf.data.schema.history_events import (
    HISTORY_VERSION,
    history_from_context,
    history_from_decisions,
    history_from_sample,
    issued_subtask_event,
    normalize_history,
    serialize_history,
)
from rlinf.models.embodiment.openpi.recap_tokenizer import (
    PROMPT_VERSION,
    RecapTokenizer,
    build_high_prefix,
)


def event(start=0, end=32, name="h000000"):
    return issued_subtask_event(
        decision_id=name, text="Pick up and hold the sample initially on the left.",
        issued_at_step=start, executed_until_step=end,
    )


@pytest.mark.parametrize("value", [None, "", "  ", [], "[]", " [ ] "])
def test_empty_aliases_are_exactly_the_same_prefix(value):
    assert normalize_history(value)==[]
    assert serialize_history(value)=="[]"
    assert build_high_prefix("READ",task="task",state=np.zeros(32),history=value)==build_high_prefix(
        "READ",task="task",state=np.zeros(32),history=[]
    )


def test_typed_event_roundtrip_preserves_text_and_does_not_claim_completion():
    original=event();original["text"]="  command text stays exact\nnext line  "
    encoded=serialize_history([original],observed_until_step=32,allow_legacy=False)
    assert normalize_history(encoded,observed_until_step=32,allow_legacy=False)==[original]
    assert json.loads(encoded)==[original]
    assert set(original)=={"type","text","decision_id","issued_at_step","executed_until_step"}
    assert not {"completed","success","outcome"}.intersection(original)
    assert PROMPT_VERSION=="rm65b_recap_prompt_v3_history_events"
    assert HISTORY_VERSION=="rm65b_history_events_v1"


@pytest.mark.parametrize("value", ["previous issued text",["previous issued text"],
    [{"type":"legacy_text","text":"previous issued text"}]])
def test_legacy_text_is_marked_without_fabricated_clock(value):
    assert normalize_history(value)==[{"type":"legacy_text","text":"previous issued text"}]
    with pytest.raises(ValueError,match="legacy_text"):
        normalize_history(value,allow_legacy=False)


@pytest.mark.parametrize("history", [
    {}, 3, [3], [""], [{"type":"completed","text":"done"}],
    [{**event(),"success":True}],
    [{**event(),"issued_at_step":True}], [{**event(),"executed_until_step":-1}],
    [{**event(),"executed_until_step":0,"issued_at_step":1}],
    [event(),event(16,64,"h000016")], [event(),event(32,64)],
    [{"type":"legacy_text","text":"old","issued_at_step":0}],
    '[malformed]', '{"type":"issued_subtask"}',
])
def test_malformed_outcome_extra_reordered_overlap_and_duplicate_rejected(history):
    with pytest.raises(ValueError):normalize_history(history)


def test_future_clock_and_alias_conflicts_rejected():
    with pytest.raises(ValueError,match="current observation"):
        normalize_history([event()],observed_until_step=31)
    with pytest.raises(ValueError,match="disagree"):
        history_from_context({"history":[],"visible_history":[event()]},observed_until_step=32)
    assert history_from_context({"history":"[]","visible_history":[]})==[]
    assert history_from_context({"history":json.dumps([event()]),"visible_history":[event()]},observed_until_step=32)==[event()]
    with pytest.raises(ValueError,match="current observation"):
        history_from_context({"history":[event()],"observed_until_step":0})


def test_full_history_accumulates_without_silent_last_two_or_reset_leak():
    decisions=[SimpleNamespace(decision_id=f"h{i:06d}",generated_subtask=f"issued {i}",start_step=i,end_step=i+1) for i in range(6)]
    result=history_from_decisions(decisions,observed_until_step=6)
    assert len(result)==6
    assert [r["issued_at_step"] for r in result]==list(range(6))
    assert history_from_decisions([],observed_until_step=0)==[]
    result[0]["text"]="changed copy"
    assert decisions[0].generated_subtask=="issued 0"
    with pytest.raises(ValueError,match="current observation"):
        history_from_decisions(decisions,observed_until_step=5)


@dataclasses.dataclass
class CPUObservation:
    state: object
    tokenized_prompt: object=None
    tokenized_prompt_mask: object=None
    token_ar_mask: object=None
    token_loss_mask: object=None


@pytest.fixture(scope="module")
def local_tokenizer():
    fallback='/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/recap_memory_20261017/cache/openpi/big_vision/paligemma_tokenizer.model'
    path=Path(os.environ.get('RM65B_TEST_TOKENIZER',fallback))
    if not path.is_file():pytest.skip('Optional real local PaliGemma tokenizer unavailable; no download permitted')
    return RecapTokenizer.from_local_paths(path,max_prefix_tokens=1400,max_target_tokens=512)


@pytest.mark.parametrize("mode",["READ","WRITE","SUBTASK"])
@pytest.mark.parametrize("training_history,runtime_history",[
    ([],""),("[]",None),([event()],json.dumps([event()])),
    ("legacy issued command",["legacy issued command"]),
])
def test_actual_collator_and_native_actor_prefix_tokens_identical_cpu(
    local_tokenizer,mode,training_history,runtime_history,
):
    import torch
    from rlinf.data.datasets.recap.hierarchical_collator import NativeJointCollator
    from rlinf.workers.rollout.hf.hierarchical_recap_worker import NativePi05Actor

    base=CPUObservation(state=torch.linspace(-1,1,32)[None])
    class CPUTransport:
        device=torch.device('cpu')
        max_token_len=2048
        def eval(self):return self
        def generate_text_tokens(self,observation,**kwargs):
            assert not torch.is_grad_enabled()
            self.actual_prompt_tokens=observation.tokenized_prompt.detach().clone()
            return torch.tensor([[local_tokenizer.eos_token_id]],device='cpu')
    model=CPUTransport()
    collator=NativeJointCollator(model,local_tokenizer,loss_weights={'high_ce':1,'fm':0,'fast_ce':0},training_mode='verified_sft')
    collator._processed=lambda reference:(None,base,None)
    slot={'slot_id':'sample_identity','status':'active','value':'initial left','confirmed':True,'revision':1}
    context={'goal':'Pick up the left sample.','catalogue':[{'slot_id':'sample_identity','description':'sample identity'}],
             'visible_history':deepcopy(training_history)}
    visible={}
    if mode=='WRITE':context['selected_memory']=[slot];visible=[slot]
    if mode=='SUBTASK':context['actor_memory']={'selected':[slot],'current_writes':[]};visible=context['actor_memory']
    sample={'mode':mode,'context':context,'observation':{},'observed_until_step':32,
       'target_text':'answer','supervision_kind':'verified_high_sft','provenance':'CPU-only transport test'}
    actual_train=collator(high_samples=[sample])['high_observation'].tokenized_prompt
    native=NativePi05Actor(model,local_tokenizer,version='cpu_transport_no_weights',advantage_condition=False)
    native._base_observation=lambda observation:base
    native.generate({},mode,{'goal':context['goal'],'memory_index':context['catalogue'],
       'history':deepcopy(runtime_history),'visible_memory':deepcopy(visible),'observed_until_step':32})
    assert torch.equal(actual_train,model.actual_prompt_tokens)
    assert 'History: '+serialize_history(training_history)+'\n' in local_tokenizer.decode_text(actual_train[0].numpy())


def test_overlong_history_is_rejected_not_silently_truncated(local_tokenizer):
    very_long=[{**event(),"text":"issued word "*2000}]
    before=deepcopy(very_long)
    prompt=build_high_prefix('READ',task='task',state=np.zeros(32),history=very_long)
    with pytest.raises(ValueError,match='Truncation would hide'):
        local_tokenizer.encode_prefix(prompt)
    assert very_long==before


def strict_record():
    return {'supervision_kind':'verified_high_sft','provenance':{'reviewer':'CPU fixture'},
       'evidence':{'observed_until_step':32},'split':'train','mode':'READ',
       'metadata':{'history_version':HISTORY_VERSION},
       'decision_step':32,'observed_until_step':32,'observation':{'npz_path':'not_loaded.npz'},
       'context':{'goal':'task','visible_history':[event()]},
       'target_text':'{"read_ids":["sample_identity"]}'}


def test_actual_verified_loader_validates_without_rewriting_history(tmp_path):
    from rlinf.runners.hierarchical_recap_runner import load_verified_high_samples
    record=strict_record();record['context']['visible_history']=json.dumps([event()])
    before=deepcopy(record)
    path=tmp_path/'declared.jsonl';path.write_text(json.dumps(record)+'\n')
    loaded=load_verified_high_samples([path])
    assert loaded[0]['context']==before['context']
    assert history_from_sample(loaded[0])==[event()]
    assert record==before


@pytest.mark.parametrize('bad',['future','legacy','alias_conflict','missing_decision','missing_observation',
    'different_context_bound','unknown_version','boolean_clock','missing_history'])
def test_actual_verified_loader_rejects_new_protocol_history_violations(tmp_path,bad):
    from rlinf.runners.hierarchical_recap_runner import load_verified_high_samples
    row=strict_record()
    if bad=='future':row['context']['visible_history'][0]['executed_until_step']=33
    elif bad=='legacy':row['context']['visible_history']='undated earlier command'
    elif bad=='alias_conflict':row['context']['history']=[]
    elif bad=='missing_decision':row.pop('decision_step')
    elif bad=='missing_observation':row.pop('observed_until_step')
    elif bad=='different_context_bound':row['context']['observed_until_step']=31
    elif bad=='unknown_version':row['metadata']['history_version']='unknown'
    elif bad=='boolean_clock':row['decision_step']=True
    elif bad=='missing_history':row['context'].pop('visible_history')
    path=tmp_path/'bad_declared.jsonl';path.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError):load_verified_high_samples([path])


@pytest.mark.parametrize('bound',[None,31])
def test_native_actor_rejects_typed_history_without_actual_causal_bound(bound):
    from rlinf.workers.rollout.hf.hierarchical_recap_worker import NativePi05Actor
    native=NativePi05Actor(SimpleNamespace(),None,version='CPU_no_model')
    native._base_observation=lambda observation:object()
    context={'history':[event()]}
    if bound is not None:context['observed_until_step']=bound
    with pytest.raises(ValueError):native.generate({},'READ',context)
