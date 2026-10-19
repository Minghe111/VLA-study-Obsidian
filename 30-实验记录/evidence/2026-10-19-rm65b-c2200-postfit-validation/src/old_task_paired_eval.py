"""Old-task native-policy evaluation, baseline or C2200, no training.

Default is plan-only. Each --execute invocation loads one model condition in
one parent and one spawned SAPIEN worker, using paired reset seeds/noise.
Original entry points, weights, normalization and assets are read-only.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

DEFAULT_ROOT=Path('/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/recap_memory_20261017/experiments/protocol_repair_20261019_v2')
ORIGINAL=Path('/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/rlinf_local_5090_minppo_20261013/models/pi05_80000_openpi')
INSTRUCTION='Pick up the bottle and adjust it to an upright position.'
CAMERAS=('full_image','left_wrist_image','right_wrist_image')


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(2**20),b''):digest.update(chunk)
    return digest.hexdigest()


def jsonable(value):
    if isinstance(value,dict):return {str(k):jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [jsonable(v) for v in value]
    if hasattr(value,'tolist'):return jsonable(value.tolist())
    if hasattr(value,'item'):return value.item()
    return value


def write(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.staging')
    temp.write_text(json.dumps(jsonable(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(path)


def noise_seed(master,index):
    if index<0:raise ValueError('Query index must be nonnegative')
    return (master+index)%2**63


def parser_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--variant',choices=['baseline','c2200'],required=True)
    parser.add_argument('--candidate-checkpoint',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seeds',nargs='+',type=int,default=[2010,2013])
    parser.add_argument('--budget',type=int,default=400)
    parser.add_argument('--noise-master-seed',type=int,default=7)
    parser.add_argument('--frames-every',type=int,default=20)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args(argv)
    if not 1<=args.budget<=400:parser.error('Old-task local protocol budget must be1..400 control targets')
    if len(args.seeds)!=2 or len(set(args.seeds))!=2 or any(s not in range(2010,2018) for s in args.seeds):parser.error('Choose exactly2 distinct fixed eval seeds from2010..2017')
    if args.frames_every<1 or not 0<=args.noise_master_seed<2**63:parser.error('Invalid frames/noise seed')
    args.root=args.root.resolve();args.output=args.output.resolve()
    args.candidate_checkpoint=(args.candidate_checkpoint or args.root/'reports/continuation_v1/language_warmup_96_2200/checkpoint').resolve()
    return args


def validate_chunk(value):
    import numpy as np
    chunk=np.asarray(value,dtype=np.float32)
    if chunk.shape!=(1,16,24) or not np.isfinite(chunk).all():raise ValueError('Expected finite decoded SDK24 H16[1,16,24]')
    return chunk[:,:1].copy()


def observation_record(obs):
    import numpy as np
    state=np.asarray(obs['state'],dtype=np.float32)
    if state.shape!=(24,) or not np.isfinite(state).all():raise ValueError('Expected finite measured SDK24')
    hashes={}
    for key in CAMERAS:
        rgb=np.asarray(obs[key])
        if rgb.shape!=(480,640,3) or rgb.dtype!=np.uint8 or not np.any(rgb):raise ValueError('Expected actual three-camera RGB8[480,640,3]')
        hashes[key]=hashlib.sha256(rgb.tobytes()).hexdigest()
    if obs['instruction']!=INSTRUCTION:raise ValueError('Old task instruction changed')
    return {'measured_state24':state.tolist(),'rgb_sha256':hashes,'instruction':INSTRUCTION}


def save_frame(obs,path,label):
    import numpy as np
    from PIL import Image,ImageDraw
    frame=Image.fromarray(np.concatenate([obs[k] for k in CAMERAS],axis=1))
    draw=ImageDraw.Draw(frame);draw.rectangle((0,0,frame.width,24),fill=(0,0,0));draw.text((8,5),label,fill=(255,255,255));frame.save(path)


def validate_step(info,*,seed,previous_ticks):
    if info['executed_action_count']!=1 or list(info['valid_action_mask'])!=[True]:raise ValueError('Expected exactly one actually executed control target')
    if info['episode_contract']['seed']!=seed:raise ValueError('Unexpected automatic reset/seed change')
    ticks=info['physics_step_count']-previous_ticks
    if not 0<=ticks<=15 or (ticks==0 and not info['failure_reason']):raise ValueError('Physics clock failed60ms contract')
    if not math.isclose(info['sim_time_s'],info['physics_step_count']*.004,abs_tol=1e-8):raise ValueError('Physics clock time differs')
    if not info['episode_terminated'] and not info['episode_truncated'] and ticks!=15:raise ValueError('Nonterminal control target did not execute15ticks')
    return ticks


def check_loaded_adapters(model,variant,checkpoint):
    """Check actual C2200 adapter values, not merely a path/config claim."""
    import torch
    from rlinf.models.embodiment.openpi.recap_lora import AttentionLoRALinear
    language={n:m for n,m in model.named_modules() if isinstance(m,AttentionLoRALinear) and n.startswith('llm.layers.') and n.endswith('.0')}
    if len(language)!=72:raise ValueError('Expected18 language layers times4 attention projections')
    trainable={n:p for n,p in model.named_parameters() if p.requires_grad}
    if len(trainable)!=144 or sum(p.numel() for p in trainable.values())!=3686400:raise ValueError('Language parameter manifest differs')
    receipt={'constructed_language_attention_adapters':len(language),'trainable_manifest_tensors':len(trainable),'trainable_manifest_numel':sum(p.numel() for p in trainable.values()),'training_performed':False,'model_factory_task':'recap_hier','strict_recap_full_load':variant=='c2200'}
    if variant=='c2200':
        version=json.loads((checkpoint/'version.json').read_text())
        if version['global_step']!=2200 or version['training_manifest']['training_stage']!='language_warmup':raise ValueError('Expected actual C2200 language_warmup checkpoint')
        source=torch.load(checkpoint/'full_weights.pt',map_location='cpu',weights_only=True,mmap=True)
        expected={key for key in model.state_dict()}
        if set(source)!=expected:raise ValueError('Full checkpoint keys differ; no partial/ignored LoRA loading allowed')
        for name,param in trainable.items():
            if not torch.equal(param.detach().cpu(),source[name]):raise ValueError('Loaded adapter does not equal C2200: '+name)
        nonzero=sum(bool(module.lora_B.detach().abs().sum()) for module in language.values())
        if nonzero!=72:raise ValueError('C2200 expected nonzero learned B in all72 language projections')
        receipt.update(source_global_step=2200,full_state_key_count=len(expected),all144_loaded_adapters_exact=True,nonzero_language_B_matrices=nonzero,version_sha256=sha(checkpoint/'version.json'))
        del source
    else:
        if any(bool(module.lora_B.detach().abs().sum()) for module in language.values()):raise ValueError('Original SFT baseline unexpectedly has learned LoRA')
        receipt.update(source_original_sft=str(ORIGINAL),all_language_B_zero=True)
    return receipt


def main(argv=None):
    args=parser_args(argv)
    plan={'execute':args.execute,'variant':args.variant,'training':False,'optimizer_updates':0,'root':str(args.root),'output':str(args.output),'instruction':INSTRUCTION,'seeds':args.seeds,'max_targets_per_episode':args.budget,'nominal_max_sim_seconds_per_episode':args.budget*.06,'success_mode':'native','H':16,'K_exec':1,'same_query_noise':'CUDA generator seed=(master+query_index)%2**63, reset query index each episode','noise_master_seed':args.noise_master_seed,'camera_count':3,'physical_action_dim':24,'MEM':False,'high_level_generation':False,'human_or_expert_intervention':False,'note':'native success is position/height only, not upright-and-hold success'}
    if not args.execute:
        print(json.dumps(plan,ensure_ascii=False,indent=2));return plan
    args.output.mkdir(parents=True,exist_ok=False)
    report={**plan,'status':'running','episodes':[],'script_sha256':sha(__file__),'wall_started_unix':time.time()}
    write(args.output/'report.json',report);model=None;env=None;monitor=None;started=time.monotonic()
    try:
        os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
        os.environ['JAX_PLATFORMS']='cpu';os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false';os.environ['PYTHONDONTWRITEBYTECODE']='1'
        os.environ['PYTHONPATH']=str(args.root/'code/RLinf')+os.pathsep+str(args.root/'code/RoboTwin')
        sys.path.insert(0,str(args.root/'code/RLinf'));sys.path.insert(0,str(args.root/'code/RoboTwin'));sys.path.insert(0,str(args.root/'scripts'))
        import numpy as np
        import torch
        import fit_comparison as fit
        fit.ROOT=args.root
        from protocol_pilot import parameter_digest
        from rlinf.runners.hierarchical_recap_runner import NativeHierarchicalBackend
        from rlinf.models.embodiment.openpi.recap_lora import legacy_prefix_context,AttentionLoRALinear
        from integrations.realman_rm65b_gen4.rl_contract import EnvConfig,sample_scene
        from integrations.realman_rm65b_gen4.rl_env import RealManVectorEnv
        torch.set_num_threads(4);random.seed(20261019);np.random.seed(20261019);torch.manual_seed(20261019)
        torch.cuda.set_device(args.device);torch.cuda.reset_peak_memory_stats()
        config=fit.build_config('language_warmup',96)
        config['runtime']['device']=args.device
        config['model']['model_path']=str(ORIGINAL if args.variant=='baseline' else args.candidate_checkpoint)
        # Model shape/config is identical for both conditions, only checkpoint and adapter flags differ.
        tokenizer_file=Path(config['tokenizer']['text_tokenizer_path'])
        if not tokenizer_file.is_file():raise FileNotFoundError('Expected existing local PaliGemma tokenizer')
        os.environ['OPENPI_DATA_HOME']=str(tokenizer_file.parent.parent)
        norm=Path(config['model']['openpi_data']['norm_stats_path'])
        if norm!=ORIGINAL/'realman_rgb50_full50/norm_stats.json':raise ValueError('Unexpected normalization path')
        report['norm_stats']={'path':str(norm),'sha256':sha(norm)}
        if report['norm_stats']['sha256']!='34cbdd09c831e04549d6c4a4e2eb321fb975bdea13bb816ec8aa3a2332d1ccc5':raise ValueError('Normalization differs from SFT source')
        report['model_path']=config['model']['model_path']
        report['source_weights']={'path':str(ORIGINAL/'model.safetensors' if args.variant=='baseline' else args.candidate_checkpoint/'full_weights.pt')}
        # Hash actual on-disk selected weights once; no second model is loaded.
        report['source_weights']['sha256']=sha(report['source_weights']['path'])
        monitor=fit.CardMonitor();monitor.__enter__()
        actor=NativeHierarchicalBackend(config)._actor();model=actor.model.eval()
        report['load_receipt']=check_loaded_adapters(model,args.variant,args.candidate_checkpoint)
        report['model_audit']=fit.audit_model(model,'language_warmup')
        report['parameter_hash_before']={str(t):parameter_digest(model,trainable=t) for t in (True,False)}
        versions={n:p._version for n,p in model.named_parameters()}
        task={'task_name':'adjust_bottle','profile':'full50_sdk24','asset_root':'/home/admin123/liminghe/vla-benchmark/RoboTwin/assets','runtime_dir':str(args.output/'runtime'),'max_episode_steps':args.budget,'instruction':INSTRUCTION,'success_mode':'native','success_reward':1.,'failure_reward':-1.,'hold_seconds':1.,'min_lift_m':.01,'max_upright_angle_deg':15.,'rpc_timeout_s':150.,'renderer_device':args.device,'renderer_denoiser':'optix'}
        report['environment_config']=task
        report['planned_scenes']={str(seed):sample_scene(EnvConfig.from_mapping(task),seed,0) for seed in args.seeds}
        with legacy_prefix_context(model,disable_adapters=args.variant=='baseline'):
            if model.mem_enabled:raise ValueError('MEM must be bypassed for old task')
            enabled=[n for n,m in model.named_modules() if isinstance(m,AttentionLoRALinear) and n.startswith('llm.layers.') and n.endswith('.0') and m.enabled]
            if len(enabled)!=(0 if args.variant=='baseline' else 72):raise ValueError('Actual inference adapter flags differ')
            report['language_adapters_enabled_during_rollout']=len(enabled)
            write(args.output/'report.json',report)
            for seed in args.seeds:
                episode_dir=args.output/f'seed_{seed}';episode_dir.mkdir();frames=episode_dir/'frames';frames.mkdir()
                env=RealManVectorEnv(task,n_envs=1,env_seeds=[seed]);obs=env.get_obs()[0]
                episode={'seed':seed,'status':'running','initial_observation':observation_record(obs),'frames_every_control_steps':args.frames_every,'frames':[],'queries':0,'maximum_bottom_gap_m':None,'minimum_upright_angle_deg':None,'maximum_lift_hold_seconds':0.,'native_success':False,'upright_hold_success_verified':False}
                report['episodes'].append(episode);write(args.output/'report.json',report)
                save_frame(obs,frames/'step_000000.png',f'{args.variant} seed={seed} t=0.00s');episode['frames'].append('frames/step_000000.png')
                query_times=[];step_times=[];previous_ticks=0;final_info=None
                with (episode_dir/'transitions.jsonl').open('x') as journal:
                    for index in range(args.budget):
                        initial=observation_record(obs);generator_seed=noise_seed(args.noise_master_seed,index)
                        rng=torch.Generator(device=args.device).manual_seed(generator_seed)
                        noise=torch.randn((1,16,32),generator=rng,device=args.device,dtype=torch.float32)
                        torch.cuda.synchronize();start=time.monotonic()
                        with torch.inference_mode():
                            action,prediction=model.predict_action_batch(actor.env_io(obs,INSTRUCTION),mode='eval',noise=noise)
                        torch.cuda.synchronize();query_times.append(time.monotonic()-start)
                        proposal=action.detach().float().cpu().numpy();target=validate_chunk(proposal)
                        internal=prediction['model_actions'].detach().float().cpu().numpy()
                        if internal.shape!=(1,16,32) or not np.isfinite(internal).all():raise ValueError('Internal H16/32 proposal differs')
                        start=time.monotonic();observations,rewards,terms,truncs,infos=env.step(target);step_times.append(time.monotonic()-start)
                        obs=observations[0];info=infos[0];validate_step(info,seed=seed,previous_ticks=previous_ticks);previous_ticks=info['physics_step_count'];final_info=info
                        metrics=info['episode_metrics'];episode['queries']=index+1
                        if 'bottom_gap_m' in metrics:
                            value=metrics['bottom_gap_m'];prior=episode['maximum_bottom_gap_m']
                            episode['maximum_bottom_gap_m']=value if prior is None else max(prior,value)
                        if 'upright_angle_deg' in metrics:
                            value=metrics['upright_angle_deg'];prior=episode['minimum_upright_angle_deg']
                            episode['minimum_upright_angle_deg']=value if prior is None else min(prior,value)
                        episode['maximum_lift_hold_seconds']=max(episode['maximum_lift_hold_seconds'],metrics.get('hold_seconds',0.))
                        # Native hold counter tracks lift/contact only; NEVER relabel it as upright-hold.
                        episode['native_success']|=bool(metrics.get('native_success',False))
                        record={'query_index':index,'noise_seed':generator_seed,'noise_sha256':hashlib.sha256(noise.detach().cpu().numpy().tobytes()).hexdigest(),'input':initial,'proposal_sdk24_H16':proposal[0].tolist(),'executed_sdk24':target[0,0].tolist(),'next_measured_state24':np.asarray(obs['state']).tolist(),'reward':rewards.tolist(),'terminated':terms.tolist(),'truncated':truncs.tolist(),'info':info,'query_wall_s':query_times[-1],'environment_step_wall_s':step_times[-1]}
                        journal.write(json.dumps(jsonable(record),allow_nan=False)+'\n');journal.flush()
                        if (index+1)%args.frames_every==0 or bool(terms.any()) or bool(truncs.any()):
                            name=f'frames/step_{index+1:06}.png';save_frame(obs,episode_dir/name,f'{args.variant} seed={seed} t={info["sim_time_s"]:.2f}s');episode['frames'].append(name)
                            write(args.output/'report.json',report)
                        if bool(terms.any()) or bool(truncs.any()):break
                if final_info is None:raise RuntimeError('No physical transition completed')
                final_name='frames/final.png';save_frame(obs,episode_dir/final_name,f'{args.variant} seed={seed} FINAL');episode['frames'].append(final_name)
                episode.update(status='completed',final_observation=observation_record(obs),final_info=final_info,actual_control_steps=final_info['step_count'],actual_physics_ticks=final_info['physics_step_count'],actual_sim_time_s=final_info['sim_time_s'],success=bool(final_info['episode_success']),termination_reason=final_info['failure_reason'] or ('native_success' if final_info['episode_success'] else 'time_limit' if final_info['episode_truncated'] else 'other'),query_wall_s={'mean':sum(query_times)/len(query_times),'max':max(query_times)},environment_step_wall_s={'mean':sum(step_times)/len(step_times),'max':max(step_times)},transitions_sha256=sha(episode_dir/'transitions.jsonl'))
                env.close(clear_cache=True);episode['workers_closed']=all(not p.is_alive() for p in env._processes);env=None
                write(episode_dir/'summary.json',episode);write(args.output/'report.json',report)
        report['parameter_hash_after']={str(t):parameter_digest(model,trainable=t) for t in (True,False)}
        report['parameters_unchanged']=report['parameter_hash_before']==report['parameter_hash_after']
        report['parameter_versions_unchanged']=all(p._version==versions[n] for n,p in model.named_parameters())
        report['no_parameter_gradients']=all(p.grad is None for p in model.parameters())
        if not report['parameters_unchanged'] or not report['parameter_versions_unchanged'] or not report['no_parameter_gradients']:raise RuntimeError('Inference changed parameters or produced gradients')
        report['peak_allocated_mib']=torch.cuda.max_memory_allocated()/2**20;report['peak_reserved_mib']=torch.cuda.max_memory_reserved()/2**20
        report['status']='completed'
    except Exception:
        report['status']='failed';report['error']=traceback.format_exc();raise
    finally:
        if env is not None:env.close(clear_cache=True)
        if monitor is not None:
            monitor.__exit__(None,None,None);report['sampled_whole_card_peak_mib']=max(monitor.samples) if monitor.samples else None
        report['elapsed_s']=time.monotonic()-started;write(args.output/'report.json',report)
    print(json.dumps({'status':report['status'],'variant':args.variant,'episodes':[{'seed':e['seed'],'success':e.get('success'),'steps':e.get('actual_control_steps')} for e in report['episodes']],'parameters_unchanged':report['parameters_unchanged']}))
    return report


if __name__=='__main__':main()
