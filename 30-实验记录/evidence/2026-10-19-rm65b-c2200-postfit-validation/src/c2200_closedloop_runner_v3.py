"""Read-only model inference in the existing RM65-B simulator; plans by default.

This is a 128-control wiring test, not a full verify-return skill benchmark.
No optimizer is constructed. Original engineering entry points are not edited.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback
def configure_process_environment(root=None, tokenizer_file=None):
    """Set task-local runtime/cache routing before NumPy/Torch or simulator spawn."""
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key]='4'
    os.environ['JAX_PLATFORMS']='cpu'
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    if root is not None:
        os.environ['PYTHONPATH']=str(Path(root)/'code/RLinf')+os.pathsep+str(Path(root)/'code/RoboTwin')
    if tokenizer_file is not None:
        tokenizer_file=Path(tokenizer_file)
        if not tokenizer_file.is_file():raise FileNotFoundError('Existing local PaliGemma tokenizer required')
        os.environ['OPENPI_DATA_HOME']=str(tokenizer_file.parent.parent)
    keys=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','JAX_PLATFORMS','XLA_PYTHON_CLIENT_PREALLOCATE','PYTHONDONTWRITEBYTECODE','OPENPI_DATA_HOME','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','PYTHONPATH')
    return {key:os.environ.get(key) for key in keys}

configure_process_environment()
import numpy as np

DEFAULT_ROOT=Path('/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/recap_memory_20261017/experiments/protocol_repair_20261019_v2')
FORBIDDEN_POLICY_KEYS={'episode_metrics','task_events','privileged_metrics','episode_success','success','failure_reason','sample_positions','verification_phase'}

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):digest.update(block)
    return digest.hexdigest()

def source_files_receipt(paths):
    return {name:{'path':str(path),'bytes':Path(path).stat().st_size,'sha256':sha(path)} for name,path in paths.items()}

def compare_adapter_tensors(trainable,source):
    import torch
    for name,param in trainable.items():
        if name not in source or not torch.equal(param.detach().cpu(),source[name]):
            raise ValueError('Actual loaded adapter differs from checkpoint: '+name)
    return len(trainable)

def check_loaded_adapters(model,checkpoint):
    """Verify all actual learned adapter tensors against mmap C2200 weights."""
    import torch
    from rlinf.models.embodiment.openpi.recap_lora import AttentionLoRALinear
    checkpoint=Path(checkpoint)
    modules={n:m for n,m in model.named_modules() if isinstance(m,AttentionLoRALinear) and n.startswith('llm.layers.') and n.endswith('.0')}
    if len(modules)!=72 or any(not x.enabled for x in modules.values()):raise ValueError('Expected72 enabled language attention adapters')
    trainable={n:p for n,p in model.named_parameters() if p.requires_grad}
    if len(trainable)!=144 or sum(p.numel() for p in trainable.values())!=3686400:raise ValueError('Expected full144 adapter parameter tensors')
    version=json.loads((checkpoint/'version.json').read_text())
    if version['global_step']!=2200 or version['training_manifest']['training_stage']!='language_warmup':raise ValueError('Expected actual C2200 source')
    source=torch.load(checkpoint/'full_weights.pt',map_location='cpu',weights_only=True,mmap=True)
    if set(source)!=set(model.state_dict()):raise ValueError('Full checkpoint keys differ; partial loading forbidden')
    count=compare_adapter_tensors(trainable,source)
    learned=sum(bool(x.lora_B.detach().abs().sum()) for x in modules.values())
    if learned!=72:raise ValueError('Expected learned nonzero B in all72 language projections')
    receipt={'strict_recap_full_load':True,'all144_loaded_adapters_exact':count==144,'nonzero_language_B_matrices':learned,'enabled_language_attention_adapters':len(modules),'source_global_step':version['global_step'],'full_state_key_count':len(source),'version_sha256':sha(checkpoint/'version.json')}
    del source
    return receipt

def seed_motor_query(action_calls,torch_api=None):
    if type(action_calls) is not int or action_calls<0:raise ValueError('Action query index must be nonnegative integer')
    if torch_api is None:
        import torch as torch_api
    seed=7+action_calls
    torch_api.manual_seed(seed)
    torch_api.cuda.manual_seed_all(seed)
    return seed

def policy_observation_receipt(observation):
    ensure_no_privileged(observation)
    state=np.asarray(observation['state'],dtype=np.float32)
    if state.shape!=(24,) or not np.isfinite(state).all():raise ValueError('Expected finite SDK24 measured state')
    pixels={}
    for key in ('full_image','left_wrist_image','right_wrist_image'):
        value=np.asarray(observation[key])
        if value.shape!=(480,640,3) or value.dtype!=np.uint8:raise ValueError('Expected three native RGB8 images')
        pixels[key]=hashlib.sha256(value.tobytes()).hexdigest()
    return {'state24':state.tolist(),'state24_sha256':hashlib.sha256(state.tobytes()).hexdigest(),'rgb_sha256':pixels}

def episode_evidence(root):
    """Read actual persisted metadata, including failures before a decision exists."""
    root=Path(root)
    for name in ('failed_episode.json','episode.json'):
        path=root/name
        if path.is_file():
            data=json.loads(path.read_text());steps=data.get('steps',[])
            last=steps[-1] if steps else None
            return {'path':str(path),'kind':name,'metadata':data.get('metadata',{}),
                'recorded_decision_count':len(data.get('decisions',[])),'recorded_step_count':len(steps),
                'last_step_terminal_fields':None if last is None else {key:last.get(key) for key in ('terminated','truncated','termination_reason','physics_steps','env_reward')}}
    return {'path':None,'kind':None,'metadata':{},'recorded_decision_count':0,'recorded_step_count':0,'last_step_terminal_fields':None}

def jsonable(x):
    if isinstance(x,dict):return {str(k):jsonable(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [jsonable(v) for v in x]
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,Path):return str(x)
    return x

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(jsonable(value),ensure_ascii=False,indent=2)+'\n');tmp.replace(path)

def append_json(path,value):
    with Path(path).open('a') as f:f.write(json.dumps(jsonable(value),ensure_ascii=False,separators=(',',':'))+'\n')

def cases(count=2):
    result=[]
    for i in range(count):
        first='left' if i%2==0 else 'right';second='right' if first=='left' else 'left';shift=.005 if i>=2 else 0
        result.append({'case_id':f'case{i:02d}_{first}_first','seed':[31600,31610,31700,31710][i],
            'first_side':first,'execution_order':['sample_a','sample_b'] if first=='left' else ['sample_b','sample_a'],
            'home_xy':[[-.10-shift,-.10-shift],[.10+shift,-.10-shift]],
            'goal':f'Verify the sample initially on the {first} and return it home, then verify the sample initially on the {second} and return it home.'})
    return result

def config_for_case(config,root,checkpoint,case,output,steps):
    cfg=deepcopy(config);cfg['model']['model_path']=str(Path(checkpoint).resolve());cfg['model']['training_mode']='verified_sft'
    cfg['model']['openpi']['recap_training_stage']='language_warmup'
    env=cfg['environment'];env['robotwin_root']=str(Path(root)/'code/RoboTwin');task=env['task_config']
    task.update(task_name='sample_verify_return',max_episode_steps=steps,instruction=case['goal'],runtime_dir=str(Path(output)/'sim_runtime'))
    memory=deepcopy(task.get('memory_task') or {});memory.update(execution_order=case['execution_order'],home_xy=case['home_xy']);task['memory_task']=memory
    cfg['rollout'].update(goal=case['goal'],eval_seeds=[case['seed']],max_steps=steps,high_level_window=32)
    cfg['memory']['strict_protocol']=True
    return cfg

def check_checkpoint(checkpoint):
    version=json.loads((Path(checkpoint)/'version.json').read_text());manifest=version['training_manifest']
    if manifest['training_stage']!='language_warmup' or manifest['runtime_mem_enabled'] is not False:
        raise ValueError('Requires frozen-MEM full-language attention checkpoint')
    numel=sum(x['numel'] for x in manifest['trainable_parameters'])
    if numel!=3686400:raise ValueError(f'Unexpected language adapter count {numel}')
    if version['metadata'].get('training_mode')!='verified_sft':raise ValueError('Requires verified-SFT checkpoint')
    return {'global_step':version['global_step'],'trainable_numel':numel,'trainable_tensors':len(manifest['trainable_parameters']),'runtime_mem_enabled':False}

def ensure_no_privileged(observation):
    bad=FORBIDDEN_POLICY_KEYS.intersection(observation)
    if bad:raise ValueError(f'Privileged diagnostics found in policy observation: {sorted(bad)}')

def subtask_side(text):
    found=re.findall(r'initially on the (left|right)',text.lower())
    return found[0] if len(set(found))==1 and found else None

class LoggedActor:
    def __init__(self,actor,path,case,oracle=False,enforce_no_grad=False):
        self.actor,self.path,self.case,self.oracle=actor,Path(path),case,oracle
        self.version=actor.version;self.versions=getattr(actor,'versions',{})
        self.generation_calls=0;self.action_calls=0;self.enforce_no_grad=enforce_no_grad
    def inference_receipt(self):
        if not self.enforce_no_grad:return {'guard_enabled':False}
        import torch
        enabled=torch.is_grad_enabled()
        training=bool(self.actor.model.training)
        if enabled or training:raise RuntimeError('Inference requires no_grad and model.eval')
        return {'guard_enabled':True,'grad_enabled':enabled,'model_training':training}
    def generate(self,observation,mode,context):
        guard=self.inference_receipt()
        ensure_no_privileged(observation);start=time.monotonic();result=self.actor.generate(observation,mode,context)
        event={'kind':'generate','call':self.generation_calls,'mode':mode,'context':context,'text':result.text,'token_ids':result.token_ids,'ended_with_eos':result.ended_with_eos,'elapsed_s':time.monotonic()-start,'policy_observation_keys':sorted(observation),'inference_guard':guard}
        self.generation_calls+=1
        if mode=='SUBTASK':
            event['expected_first_side']=self.case['first_side'];event['generated_side']=subtask_side(result.text)
            event['first_role_matches']=event['generated_side']==self.case['first_side']
            # Preserve the raw output even in an explicitly labeled oracle motor control.
            # Missing EOS remains a protocol failure; an oracle never repairs output.
            if self.oracle and result.ended_with_eos and result.text.strip():
                text=f"Pick up and hold the sample initially on the {self.case['first_side']}."
                from rlinf.workers.rollout.hf.hierarchical_recap_worker import GeneratedText
                ids=self.actor.tokenizer.encode_target(text).tolist();result=GeneratedText(text,ids,True)
                event['oracle_subtask_override']={'text':text,'token_ids':ids,'diagnostic_only':True}
        append_json(self.path,event)
        return result
    def actions(self,observation,context):
        guard=self.inference_receipt()
        inputs=policy_observation_receipt(observation)
        motor_seed=seed_motor_query(self.action_calls)
        start=time.monotonic();actions=np.asarray(self.actor.actions(observation,context),dtype=np.float32)
        if actions.shape!=(16,24) or not np.isfinite(actions).all():raise ValueError(f'Expected H16 SDK24 finite actions, got{actions.shape}')
        append_json(self.path,{'kind':'actions','call':self.action_calls,'context':context,'elapsed_s':time.monotonic()-start,'shape':actions.shape,'proposed_H16_SDK24':actions,'sha256':hashlib.sha256(actions.tobytes()).hexdigest(),'executed_prefix':1,'policy_observation_keys':sorted(observation),'inference_guard':guard,'motor_noise_seed':motor_seed,'motor_query_index':self.action_calls,'actual_motor_input':inputs,'noise_scope':'Paired seed per query; actual observations can differ and are hashed.'})
        self.action_calls+=1;return actions

class LoggedEnv:
    def __init__(self,env,path):self.env,self.path,self.controls=env,Path(path),0;self.last_info=None
    def reset(self,*args,**kwargs):return self.env.reset(*args,**kwargs)
    def step(self,actions):
        value=np.asarray(actions)
        if value.shape!=(1,1,24):raise ValueError('Only one SDK24 control may execute per inference')
        observations,rewards,terminated,truncated,infos=self.env.step(actions);self.last_info=infos[0]
        self.controls+=1
        append_json(self.path,{'control':self.controls,'issued_K1_SDK24':value,'rewards':rewards,'terminated':terminated,'truncated':truncated,'info':infos[0]})
        return observations,rewards,terminated,truncated,infos
    def close(self):return self.env.close()

def classify(episode_result,policy_events,last_info):
    sub=[x for x in policy_events if x.get('kind')=='generate' and x.get('mode')=='SUBTASK']
    info=last_info or {};metrics=info.get('episode_metrics',{})
    metadata=episode_result.get('episode_metadata',{})
    evidence=episode_result.get('episode_evidence',{})
    last=evidence.get('last_step_terminal_fields') or {}
    errors={'python_exception':episode_result.get('error'),'episode_metadata_errors':{key:metadata[key] for key in ('protocol_error','collection_error','failure_reason','termination_reason','error') if key in metadata},'environment_failure_reason':info.get('failure_reason'),'last_step_termination_reason':last.get('termination_reason')}
    primary=episode_result.get('error') or metadata.get('protocol_error') or metadata.get('collection_error') or metadata.get('failure_reason') or info.get('failure_reason')
    return {'protocol_or_execution_error':primary,'all_failure_evidence':errors,'subtask_decisions':len(sub),
        'subtask_first_role_matches':sum(x.get('first_role_matches',False) for x in sub),
        'autonomous_subtask':not any('oracle_subtask_override' in x for x in sub),
        'task_SR':bool(info.get('episode_success',False)),'task_SR_interpretation':'Diagnostic only:128-controls are not complete ordered verify-return training evidence',
        'verified_count':metrics.get('verified_count'),'returned_count':metrics.get('returned_count'),
        'task_events':metrics.get('task_events',[]),'failure_reason':info.get('failure_reason'),
        'sim_time_s':info.get('sim_time_s'),'physics_step_count':info.get('physics_step_count'),
        'motor_steps_recorded':info.get('step_count'),
        'role_check_limit':'Matches static first role only; this dataset did not train phase-dependent goal progress.'}

def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment-root',type=Path,default=DEFAULT_ROOT)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--config',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=128);p.add_argument('--cases',type=int,choices=[2,4],default=2)
    p.add_argument('--oracle-subtask',action='store_true');p.add_argument('--execute',action='store_true')
    a=p.parse_args(argv)
    if not 1<=a.steps<=512:p.error('Use an explicit small closed-loop cap1..512')
    a.checkpoint=a.checkpoint or a.experiment_root/'reports/continuation_v1/language_warmup_96_2200/checkpoint'
    a.config=a.config or a.checkpoint.parent/'effective_config.yaml';return a

def main(argv=None):
    a=parse_args(argv);audit=check_checkpoint(a.checkpoint);case_list=cases(a.cases)
    report={'status':'planned','checkpoint':str(a.checkpoint),'checkpoint_audit':audit,'cases':case_list,'steps_per_case':a.steps,
        'physical_time_cap_s':a.steps*.06,'mode':'oracle_subtask_motor_diagnostic' if a.oracle_subtask else 'autonomous_high_level',
        'optimizer_constructed':False,'optimizer_updates':0,'formal_3000_training':False,'task_SR_claim':False,
        'contract':{'SDK_dim':24,'cameras':3,'period_s':.06,'H':16,'K_exec':1},'original_entry_modified':False,'runner_version':'v3_frozen','script_sha256':sha(__file__),
        'motor_noise_protocol':{'query_seed':'7+action_calls','reset_each_case':True,'torch_CPU_and_CUDA_seeded':True,'same_observation_claim':False,'source_path':'NativePi05Actor.actions->_predict_eval(noise=None,rng=None)->Pi0.sample_actions torch.randn(generator=None); highlevel greedy argmax'}}
    if not a.execute:print(json.dumps(report,ensure_ascii=False,indent=2));return report
    a.output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(a.experiment_root/'code/RLinf'));sys.path.insert(0,str(a.experiment_root/'scripts'));sys.path.insert(0,str(a.experiment_root/'code/RoboTwin'))
    configure_process_environment(a.experiment_root)
    import yaml,torch
    torch.set_num_threads(4)
    from rlinf.runners.hierarchical_recap_runner import NativeHierarchicalBackend
    from rlinf.workers.rollout.hf.hierarchical_recap_worker import HierarchicalReCapRolloutWorker
    from integrations.realman_rm65b_gen4.rl_env import RealManVectorEnv
    from protocol_pilot import parameter_digest
    cfg=yaml.safe_load(a.config.read_text());cfg['model']['model_path']=str(a.checkpoint.resolve())
    report['process_environment']=configure_process_environment(a.experiment_root,Path(cfg['tokenizer']['text_tokenizer_path']))
    sources={'selected_weights':a.checkpoint/'full_weights.pt','checkpoint_version':a.checkpoint/'version.json','tokenizer':Path(cfg['tokenizer']['text_tokenizer_path']),'norm_stats':Path(cfg['model']['openpi_data']['norm_stats_path'])}
    report['source_files_before']=source_files_receipt(sources)
    actor=None;before=None;report['status']='running';report['results']=[];started=time.monotonic();write_json(a.output/'report.json',report)
    try:
        actor=NativeHierarchicalBackend(cfg)._actor();actor.model.eval()
        report['load_receipt']=check_loaded_adapters(actor.model,a.checkpoint)
        report['motor_sampling_implementation']={'sample_actions':actor.model.sample_actions.__func__.__qualname__,'predict_eval':actor.model._predict_eval.__func__.__qualname__,'generate_text_tokens':actor.model.generate_text_tokens.__func__.__qualname__}
        if report['motor_sampling_implementation']['sample_actions']!='Pi0.sample_actions' or report['motor_sampling_implementation']['predict_eval']!='EnvIO._predict_eval':raise ValueError('Unreviewed motor RNG implementation; seed pairing cannot be asserted')
        if actor.model.mem_enabled or actor.model.recap_training_stage!='language_warmup':raise ValueError('Loaded model scope differs fromC2200')
        if sum(p.numel() for p in actor.model.parameters() if p.requires_grad)!=3686400:raise ValueError('Expected all-language attentionLoRA')
        before={str(trainable):parameter_digest(actor.model,trainable=trainable) for trainable in [False,True]};report['parameters_before']=before
        versions={name:p._version for name,p in actor.model.named_parameters()}
        report['parameter_version_counters_before']=versions
        report['gradients_before']=sum(p.grad is not None for p in actor.model.parameters())
        if report['gradients_before']:raise ValueError('Inference actor unexpectedly has preexisting gradients')
        for case in case_list:
            out=a.output/case['case_id'];out.mkdir();effective=config_for_case(cfg,a.experiment_root,a.checkpoint,case,out,a.steps)
            (out/'effective_config.yaml').write_text(yaml.safe_dump(effective,sort_keys=False));entry={'case':case,'status':'running'};report['results'].append(entry);write_json(a.output/'report.json',report)
            env=None;logged=None;start=time.monotonic()
            try:
                env=RealManVectorEnv(effective['environment']['task_config'],n_envs=1,env_seeds=[case['seed']])
                logged=LoggedEnv(env,out/'environment_info.jsonl');policy=LoggedActor(actor,out/'policy_trace.jsonl',case,a.oracle_subtask,enforce_no_grad=True)
                controller=HierarchicalReCapRolloutWorker(policy,logged,effective['memory']['catalogue'],high_level_window=32,max_steps=a.steps,history_num_frames=1,strict_memory_protocol=True)
                with torch.no_grad():
                    episode=controller.collect_episode(out/'episode',episode_id=case['case_id'],goal=case['goal'],seed=case['seed'],evaluate=True)
                entry.update(status='completed',episode_path=str(out/'episode/episode.json'),episode_metadata=deepcopy(episode.metadata),success=bool(episode.metadata.get('success',False)))
            except Exception:
                entry.update(status='failed',error=traceback.format_exc())
            finally:
                if env is not None:
                    env.close(clear_cache=True)
                    entry['workers_closed']=all(not process.is_alive() for process in env._processes)
                    entry['worker_exit_receipts']=[{'pid':process.pid,'exitcode':process.exitcode,'alive':process.is_alive()} for process in env._processes]
                    if not entry['workers_closed']:entry.update(status='failed',error='Simulator worker remains alive after close')
                entry['elapsed_s']=time.monotonic()-start
                entry['episode_evidence']=episode_evidence(out/'episode')
                entry['episode_metadata']=entry['episode_evidence']['metadata']
                events=[json.loads(x) for x in (out/'policy_trace.jsonl').read_text().splitlines()] if (out/'policy_trace.jsonl').exists() else []
                action_events=[event for event in events if event.get('kind')=='actions']
                entry['motor_noise_seeds']=[event['motor_noise_seed'] for event in action_events]
                entry['first_motor_input']=action_events[0]['actual_motor_input'] if action_events else None
                entry['diagnosis']=classify(entry,events,logged.last_info if logged else None)
                entry['executed_controls']=logged.controls if logged else 0
                write_json(out/'case_report.json',entry);write_json(a.output/'report.json',report)
        if len(report['results'])>=2:
            left,right=report['results'][:2]
            report['first_case_observation_comparison']={'both_first_motor_inputs_recorded':left.get('first_motor_input') is not None and right.get('first_motor_input') is not None,'actual_first_motor_inputs_exact':left.get('first_motor_input')==right.get('first_motor_input') if left.get('first_motor_input') is not None and right.get('first_motor_input') is not None else None,'claim':'Goal order differs intentionally; seeds pair motor noise per query, not subsequent observations.'}
        report['status']='completed'
    except Exception:
        report['status']='failed';report['error']=traceback.format_exc();raise
    finally:
        if actor is not None and before is not None:
            after={str(trainable):parameter_digest(actor.model,trainable=trainable) for trainable in [False,True]}
            report['parameters_after']=after;report['all_parameters_unchanged']=after==before
            after_versions={name:p._version for name,p in actor.model.named_parameters()}
            report['parameter_version_counters_after']=after_versions
            report['all_parameter_versions_unchanged']=after_versions==versions
            report['gradients_after']=sum(p.grad is not None for p in actor.model.parameters())
            if after!=before or after_versions!=versions or report['gradients_after']:
                report['status']='failed';report['parameter_protection_error']='Inference unexpectedly changed parameters/version or created gradients'
        report['source_files_after']=source_files_receipt(sources)
        report['all_source_files_unchanged']=report['source_files_after']==report['source_files_before']
        if not report['all_source_files_unchanged']:
            report['status']='failed';report['source_protection_error']='Selected weights/version/tokenizer/norm changed during inference'
        report['elapsed_s']=time.monotonic()-started;write_json(a.output/'report.json',report)
    return report
if __name__=='__main__':main()
