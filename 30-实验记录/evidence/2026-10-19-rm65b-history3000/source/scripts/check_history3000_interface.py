"""CPU-only dataset-level native training/inference prefix parity.

Reads actual JSONL/NPZ, original SDK24 normalization and local SentencePiece.
Uses the real NativeJointCollator and NativePi05Actor.generate entry points.
Only image embedding/base text preprocessing and neural generation computation
are replaced: the transport returns a synthetic EOS, never a model prediction.
There is no network/model/optimizer construction, GPU tensor or simulator.
"""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time

import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code/RLinf'))


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):digest.update(block)
    return digest.hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def load_rows(paths,limit=0):
    result=[]
    for path in paths:
        path=Path(path).resolve()
        for line_number,line in enumerate(path.read_text().splitlines(),1):
            if not line.strip():continue
            row=json.loads(line)
            if not isinstance(row,dict):raise ValueError('Every JSONL row must be an object')
            row=deepcopy(row)
            reference=row.get('observation',{})
            if 'npz_path' not in reference:raise ValueError('Actual observation.npz_path required')
            observation=Path(reference['npz_path'])
            reference['npz_path']=str((path.parent/observation).resolve() if not observation.is_absolute() else observation.resolve())
            result.append((row,{'path':str(path),'line_1based':line_number}))
    total=len(result)
    if not total:raise ValueError('No dataset rows to check')
    return (result[:limit] if limit else result),total


def independent_state32(raw,stats):
    state=np.asarray(raw,dtype=np.float32).copy()
    if state.shape!=(24,) or not np.isfinite(state).all():raise ValueError('Finite measured SDK24 required')
    state[6:12]/=1000.;state[18:24]/=1000.
    q01=np.asarray(stats['q01']);q99=np.asarray(stats['q99'])
    return np.pad((state-q01)/(q99-q01+1e-6)*2.-1.,(0,8)).astype(np.float32)


def write_report(path,report):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')


class RecordingTokenizer:
    def __init__(self,base):self.base=base;self.prefixes=[]
    def __getattr__(self,name):return getattr(self.base,name)
    def encode_prefix(self,text):
        self.prefixes.append(text)
        return self.base.encode_prefix(text)
    def consume(self):
        current=self.prefixes;self.prefixes=[]
        return current


def run_check(labels,config,output,*,limit=0,batch_size=64):
    if limit<0 or batch_size<1:raise ValueError('limit must be nonnegative and batch_size positive')
    started=time.monotonic()
    labels=[Path(p).resolve() for p in labels];config=Path(config).resolve()
    cfg=yaml.safe_load(config.read_text())
    rows,total=load_rows(labels,limit)
    paths={'config':config,'norm':Path(cfg['model']['openpi_data']['norm_stats_path']),
           'tokenizer':Path(cfg['tokenizer']['text_tokenizer_path']),
           **{f'labels_{i}':p for i,p in enumerate(labels)}}
    before={name:sha(path) for name,path in paths.items()}

    import torch
    import openpi.transforms as transforms
    from openpi.shared.normalize import NormStats
    import rlinf.data.datasets.recap.hierarchical_collator as collator_module
    from rlinf.data.schema.history_events import HISTORY_VERSION,history_from_context
    from rlinf.models.embodiment.openpi.env_io import EnvIO
    from rlinf.models.embodiment.openpi.policies.realman_policy import (
        ARM_DELTA_MASK,RealManInputs,validate_norm_stats,
    )
    from rlinf.models.embodiment.openpi.recap_tokenizer import PROMPT_VERSION,RecapTokenizer
    from rlinf.workers.rollout.hf.hierarchical_recap_worker import NativePi05Actor

    torch.set_num_threads(2)
    stats_json=json.loads(paths['norm'].read_text())['norm_stats']
    stats={name:NormStats(**{key:np.asarray(value) for key,value in fields.items()}) for name,fields in stats_json.items()}
    validate_norm_stats(stats)
    tokenizer=RecordingTokenizer(RecapTokenizer.from_local_paths(paths['tokenizer'],
        max_prefix_tokens=cfg['tokenizer']['max_prefix_tokens'],max_target_tokens=cfg['tokenizer']['max_target_tokens']))

    class CPUBaseInputStub:
        def __call__(self,data):
            # RealManInputs has already validated all actual RGB pixels. Only
            # image/token preprocessing is stubbed; state is left untouched.
            result=dict(data);result.pop('prompt',None)
            result['image']={key:np.zeros((1,1,3),dtype=np.uint8) for key in result['image']}
            result['tokenized_prompt']=np.zeros(1,dtype=np.int64)
            result['tokenized_prompt_mask']=np.ones(1,dtype=bool)
            return result

    class CPUTransport(EnvIO):
        device=torch.device('cpu')
        state_indices=None
        mem_enabled=False
        config_name=cfg['model']['openpi']['config_name']
        max_token_len=cfg['model']['openpi']['max_token_len']
        def eval(self):return self
        def generate_text_tokens(self,observation,**kwargs):
            if torch.is_grad_enabled():raise AssertionError('Native actor no_grad guard missing')
            if observation.state.device.type!='cpu':raise AssertionError('Only CPU transport allowed')
            self.runtime_tokens=observation.tokenized_prompt.detach().clone()
            self.runtime_state=observation.state.detach().clone()
            return torch.tensor([[tokenizer.eos_token_id]],device='cpu')

    model=CPUTransport()
    model.setup_transforms([RealManInputs(),transforms.DeltaActions(ARM_DELTA_MASK),
        transforms.Normalize(stats,use_quantiles=True),transforms.PadStatesAndActions(32),CPUBaseInputStub()],[])
    collator=collator_module.NativeJointCollator(model,tokenizer,
        loss_weights={'high_ce':1.,'fm':0.,'fast_ce':0.},training_mode='verified_sft')
    actor=NativePi05Actor(model,tokenizer,version='CPU_prefix_transport_no_weights',
        max_new_tokens=cfg['rollout']['max_new_tokens'],advantage_condition=False)

    original_load=collator_module.load_observation
    observation_cache=OrderedDict();observation_audit={}
    def cached_load(reference):
        path=str(Path(reference['npz_path']).resolve())
        if path in observation_cache:
            native=observation_cache.pop(path);observation_cache[path]=native
            return native
        native=original_load(reference)  # actual native contract validator
        stat=Path(path).stat()
        receipt={'path':path,'npz_sha256':sha(path),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns,
            'state_sha256':array_sha(native['state']),
            'cameras':{name:{'shape':list(native[name].shape),'dtype':str(native[name].dtype),
                'pixel_sha256':array_sha(native[name])} for name in ('full_image','left_wrist_image','right_wrist_image')}}
        if path in observation_audit and receipt!=observation_audit[path]:raise ValueError('Source NPZ changed during read-only check')
        observation_audit[path]=receipt;observation_cache[path]=native
        if len(observation_cache)>8:observation_cache.popitem(last=False)
        return native

    mode_counts=Counter();write_counts=Counter();history_counts=Counter();receipts=[]
    collator_module.load_observation=cached_load
    try:
        for offset in range(0,len(rows),batch_size):
            batch=rows[offset:offset+batch_size]
            samples=[row for row,_ in batch]
            for row,_ in batch:
                if row['mode'] not in {'READ','WRITE','SUBTASK'}:raise ValueError('Unknown mode')
                context=row['context']
                if any(key in context for key in ('target_text','target_ids','labels','answer','loss','LOSS')):
                    raise ValueError('Label/loss fields must not enter actor context')
                bound=row.get('observed_until_step',row.get('decision_step',context.get('observed_until_step')))
                checked_history=history_from_context(context,observed_until_step=bound,allow_legacy=False)
                if checked_history and bound is None:raise ValueError('Nonempty typed history needs an observed control-step bound')
                if context.get('mode',row['mode'])!=row['mode']:raise ValueError('Context and row modes disagree')
                if row['mode']=='SUBTASK' and (not isinstance(context.get('actor_memory'),dict) or set(context['actor_memory'])!={'selected','current_writes'}):
                    raise ValueError('SUBTASK requires native actor_memory envelope')
            trained=collator(high_samples=samples)
            train_prompts=tokenizer.consume()
            if len(train_prompts)!=len(batch):raise AssertionError('Expected one actual training prefix per row')
            altered=[{**row,'target_text':f'label isolation sentinel {offset+i}'} for i,(row,_) in enumerate(batch)]
            counterfactual=collator(high_samples=altered)
            isolated_prompts=tokenizer.consume()
            target_invariance=(train_prompts==isolated_prompts and torch.equal(
                trained['high_observation'].tokenized_prompt,counterfactual['high_observation'].tokenized_prompt))
            if not target_invariance:raise AssertionError('Changing only supervised labels changed the actual prefix')
            if trained['high_observation'].token_loss_mask is not None:raise AssertionError('Loss targets leaked into prefix observation')
            for i,(row,location) in enumerate(batch):
                context=row['context'];native=cached_load(row['observation'])
                bound=row.get('observed_until_step',row.get('decision_step',context.get('observed_until_step')))
                history=history_from_context(context,observed_until_step=bound,allow_legacy=False)
                runtime={'goal':context['goal'],'memory_index':context['catalogue'],
                    'visible_memory':context.get('actor_memory',context.get('selected_memory',{})),
                    'history':history}
                if bound is not None:runtime['observed_until_step']=bound
                actor.generate(native,row['mode'],runtime)  # native entry, synthetic EOS computation only
                actual=tokenizer.consume()
                if actual!=[train_prompts[i]]:raise AssertionError('Native/train actual prefix bytes differ')
                obs=trained['high_observation']
                train_ids=obs.tokenized_prompt[i][obs.tokenized_prompt_mask[i]]
                if not torch.equal(train_ids,model.runtime_tokens[0]):raise AssertionError('Native/train actual prefix token ids differ')
                expected=independent_state32(native['state'],stats_json['state'])
                if not np.array_equal(obs.state[i].numpy(),expected) or not torch.equal(obs.state[i],model.runtime_state[0]):
                    raise AssertionError('Actual SDK24 state preprocessing differs between paths or original formula')
                if row['mode']=='WRITE':
                    operations=json.loads(row['target_text'])['operations']
                    operation=operations[0]['op'] if operations else 'EMPTY'
                    write_counts[operation]+=1
                else:operation=None
                mode_counts[row['mode']]+=1;history_counts[str(len(history))]+=1
                receipts.append({'sample_id':row['sample_id'],'source':location,'mode':row['mode'],
                    'write_branch':operation,'history_events':len(history),'prefix_tokens':len(train_ids),
                    'prefix_utf8_sha256':hashlib.sha256(actual[0].encode()).hexdigest(),
                    'prefix_int64_sha256':array_sha(train_ids.numpy()),'normalized_state32_sha256':array_sha(expected),
                    'actual_prefix_bytes_equal':True,'actual_prefix_ids_equal':True,
                    'target_mutation_prefix_invariant':True,'source_npz':row['observation']['npz_path']})
    finally:
        collator_module.load_observation=original_load
    after={name:sha(path) for name,path in paths.items()}
    if before!=after:raise AssertionError('Read-only config/data/assets changed')
    for path,receipt in observation_audit.items():
        stat=Path(path).stat()
        if (stat.st_size,stat.st_mtime_ns)!=(receipt['bytes'],receipt['mtime_ns']):raise AssertionError('NPZ changed during check')
    implementation={
        'NativeJointCollator':inspect.getfile(collator_module.NativeJointCollator),
        'NativePi05Actor':inspect.getfile(NativePi05Actor),'EnvIO':inspect.getfile(EnvIO),
        'RealManInputs':inspect.getfile(RealManInputs),'Normalize':inspect.getfile(transforms.Normalize)}
    report={'status':'passed','scope':'CPU actual native/collator prefix interface; no learned generation or visual embedding',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'elapsed_s':time.monotonic()-started,
        'total_dataset_rows':total,'checked_rows':len(receipts),'coverage':'full_dataset' if len(receipts)==total else 'explicit_partial_limit',
        'unique_npz_audited':len(observation_audit),'mode_counts':dict(mode_counts),'WRITE_counts':dict(write_counts),
        'history_event_count_distribution':dict(history_counts),'versions':{'prompt':PROMPT_VERSION,'history':HISTORY_VERSION},
        'read_only_source_sha_before':before,'read_only_source_sha_after':after,'sources_unchanged':True,
        'actual_implementation_files':{name:{'path':path,'sha256':sha(path)} for name,path in implementation.items()},
        'model_and_image_stub':{'parameters_loaded':0,'optimizer_created':False,'device':'cpu',
            'synthetic_EOS_is_model_prediction':False,'image_embeddings_computed':False,
            'real_RGB_checked_before_stub':True,'post_validation_transport_image_shape':[1,1,3],
            'actual_state_chain':'RealManInputs -> DeltaActions(no actions present) -> original quantile Normalize -> PadStatesAndActions(32) -> native float32 state',
            'base_text_tokenization_stubbed':'Base empty prompt only; actual high prefix uses real local SentencePiece',
            'video_MEM_scope':'Not evaluated; text-history prefix interface only',
            'raw_NPZ_cache':'Up to8 actual observations; source loader restored in finally'},
        'label_isolation':'Every row: changing only target_text leaves actual prefix bytes and ids unchanged; no loss/label fields accepted in context. A legal earlier issued text may equal the current label, so substring equality alone is not called leakage.',
        'observation_audit':list(observation_audit.values()),'row_receipts':receipts}
    write_report(output,report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels',type=Path,nargs='+',required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--limit',type=int,default=0,help='0 means all rows; nonzero is explicitly partial coverage')
    parser.add_argument('--batch-size',type=int,default=64)
    args=parser.parse_args()
    try:
        report=run_check(args.labels,args.config,args.output,limit=args.limit,batch_size=args.batch_size)
    except Exception as exc:
        write_report(args.output,{'status':'failed','scope':'CPU prefix interface check; no training',
            'error':f'{type(exc).__name__}: {exc}'})
        raise
    print(json.dumps({key:report[key] for key in ('status','checked_rows','total_dataset_rows','coverage','mode_counts','WRITE_counts','history_event_count_distribution','elapsed_s')},ensure_ascii=False))


if __name__=='__main__':main()
