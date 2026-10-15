"""Additive LOCAL JAX runtime, preserving the prior official inference chain."""
from pathlib import Path
import hashlib,json,os,sys,time,importlib.metadata as metadata

GEN4=Path('/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4')
TASK=GEN4/'pi05_rgb50_full50_100k_b8_4gpu_161_20261007_v1'
CHECKPOINT=TASK/'checkpoints/pi05_realman_rgb50_all50_full_v1/full50_100k_b64_fallback32_save20k_20261008_v4_bs64_formal100k/80000'
ROOT=Path('/tmp/realman-eval80k-20261012/policy_runtime')
COMMIT='215abfb217dbac7d5f1273282331b9b1866c0479'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def write_new(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False)

class Runtime:
    def __init__(self,proof_dir,openpi_root,source_identity):
        openpi_root=Path(openpi_root).resolve()
        source_identity=Path(source_identity).resolve()
        source=json.loads(source_identity.read_text())
        assert source['official_commit']==COMMIT and source['official_worktree_clean'] is True
        rows=source['source_files']
        assert isinstance(rows,list) and rows
        for row in rows:
            p=(openpi_root/row['relative']).resolve()
            assert p.is_relative_to(openpi_root) and p.is_file()
            assert sha(p)==row['sha256']
        sys.path[:0]=[str(openpi_root/'src'),str(openpi_root/'packages/openpi-client/src'),str(TASK/'training_v4_batch_save20k')]
        # Import Torch first so JAX's older CUDA loader cannot override Torch's newer
        # runtime library. Torch is imported by official code but never runs this model.
        import torch
        import numpy as np,jax
        from openpi.models.pi0_config import Pi0Config
        from openpi.training import config
        from openpi.policies import policy_config
        import flax.nnx as nnx
        import flax.traverse_util as traverse
        import full50_pi05_config_161_v3_batch as own
        self.np=np
        assert os.environ['CUDA_VISIBLE_DEVICES']=='0'
        assert len(jax.devices())==1 and jax.devices()[0].platform=='gpu'
        assert Path(policy_config.__file__).resolve().is_relative_to(openpi_root)
        own.bind_environment()
        os.environ['OPENPI_DATA_HOME']=str(ROOT/'cache')
        tokenizer=ROOT/'cache/big_vision/paligemma_tokenizer.model'
        assert tokenizer.is_file(), 'Tokenizer must already be present in isolated cache'
        assert sha(tokenizer)==source['tokenizer']['sha256']
        saved=CHECKPOINT/'assets'/own.ASSET_ID/'norm_stats.json'
        assert sha(saved)==own.NORM_SHA256
        assert (CHECKPOINT/'_CHECKPOINT_METADATA').is_file() and (CHECKPOINT/'params/_METADATA').is_file()
        DataConfig,_,_=own.runtime_classes()
        model=Pi0Config(pi05=True,dtype='bfloat16',action_dim=32,action_horizon=16,paligemma_variant='gemma_2b',action_expert_variant='gemma_300m')
        # Local mirror omits derived data50_v1/assets; the checkpoint carries the
        # same hash-pinned training normalization. Bind via the public factory API.
        data=DataConfig(assets=config.AssetsConfig(assets_dir=str(CHECKPOINT/'assets'),asset_id=own.ASSET_ID))
        cfg=config.TrainConfig(name=own.NAME,exp_name='checkpoint80000_local_inference_only',model=model,data=data,assets_base_dir=str(TASK/'training_assets'),policy_metadata={'physical_action_dimension':24,'model_action_dimension':32,'action_horizon':16,'hand_absolute':True,'sdk_scale':1000.,'all50_train':True})
        inventory=[{'relative':str(p.relative_to(CHECKPOINT)),'bytes':p.stat().st_size} for p in sorted((CHECKPOINT/'params').rglob('*')) if p.is_file()]
        print('RESTORING_LOCAL_CHECKPOINT_80000',flush=True)
        start=time.monotonic()
        self.policy=policy_config.create_trained_policy(cfg,CHECKPOINT,sample_kwargs={'num_steps':10})
        params=traverse.flatten_dict(nnx.state(self.policy._model,nnx.Param).to_pure_dict(),sep='/')
        param_rows=[{'name':name,'shape':list(value.shape),'dtype':str(value.dtype),'elements':int(value.size)} for name,value in sorted(params.items())]
        assert len(param_rows)==51 and sum(r['elements'] for r in param_rows)==3353433872
        assert all(r['dtype']=='bfloat16' for r in param_rows)
        self.identity={'checkpoint':str(CHECKPOINT),'checkpoint_step':80000,'full50_train':True,'sample_num_steps':10,'action_horizon':16,'payload_action_dim':24,'nominal_action_period_s':0.06,'official_commit':COMMIT,'official_worktree_clean':True,'official_source_path':str(openpi_root),'official_source_identity_path':str(source_identity),'official_source_identity_sha256':sha(source_identity),'official_create_trained_policy_used':True,'base_fallback':False,'optimizer_restored':False,'norm_stats_sha256':sha(saved),'norm_equal_to_training':True,'tokenizer_sha256':sha(tokenizer),'tokenizer_path':str(tokenizer),'training_config_path':str(Path(own.__file__)),'training_config_sha256':sha(own.__file__),'data_adapter_sha256':sha(own.ADAPTER_PATH),'checkpoint_metadata_sha256':sha(CHECKPOINT/'_CHECKPOINT_METADATA'),'params_metadata_sha256':sha(CHECKPOINT/'params/_METADATA'),'params_file_inventory':inventory,'restored_parameter_leaves':len(param_rows),'restored_parameter_elements':sum(r['elements'] for r in param_rows),'restored_parameters':param_rows,'physical_gpu_index':0,'jax_devices':[str(d) for d in jax.devices()],'load_wall_s':time.monotonic()-start,'expert_actions_used':False,'hardware_IO':False,'model_outputs_unclipped':True,'official_policy_rng_initial_seed':0,'local_policy_inference':True,'system_conda_mutated':False,'torch_used_for_inference':False,'runtime_python':sys.executable,'runtime_versions':{n:metadata.version(n) for n in ('jax','jaxlib','flax','orbax-checkpoint','ml-dtypes','tensorstore','numpy','optax','chex','torch','transformers')},'xla_memory_fraction':os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION'),'xla_preallocate':os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE')}
        write_new(Path(proof_dir)/'model_identity.json',self.identity)
        self.identity_sha256=sha(Path(proof_dir)/'model_identity.json')

    def infer(self,obs):
        start=time.monotonic();r=self.policy.infer(obs);v=self.np.asarray(r['actions'])
        assert v.shape==(16,24) and self.np.isfinite(v).all()
        return v,{'full_infer_wall_s':time.monotonic()-start,'official_policy_timing':r.get('policy_timing',{})}
