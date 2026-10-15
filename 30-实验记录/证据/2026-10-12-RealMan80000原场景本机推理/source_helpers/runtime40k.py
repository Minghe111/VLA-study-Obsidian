from pathlib import Path
import hashlib,json,os,sys,subprocess,time
GEN4=Path('/bigdata/liminghe/VLA-benchmark/deploy/realman_gen4')
TASK=GEN4/'pi05_rgb50_full50_100k_b8_4gpu_161_20261007_v1'
BASE=GEN4/'pi05_rgb20_20261004'
CHECKPOINT=TASK/'checkpoints/pi05_realman_rgb50_all50_full_v1/full50_100k_b64_fallback32_save20k_20261008_v4_bs64_formal100k/60000'
COMMIT='215abfb217dbac7d5f1273282331b9b1866c0479'
sys.path[:0]=[str(BASE/'openpi/src'),str(TASK/'training_v4_batch_save20k')]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def write_new(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
class Runtime:
    def __init__(self,proof_dir):
        import numpy as np,jax
        from openpi.models.pi0_config import Pi0Config
        from openpi.training import config,checkpoints
        from openpi.policies import policy_config
        import full50_pi05_config_161_v3_batch as own
        self.np=np
        assert os.environ['CUDA_VISIBLE_DEVICES']=='2'
        head=subprocess.check_output(['git','-C',str(BASE/'openpi'),'rev-parse','HEAD'],text=True).strip()
        assert head==COMMIT
        assert not subprocess.check_output(['git','-C',str(BASE/'openpi'),'status','--porcelain'],text=True).strip()
        assert len(jax.devices())==1 and jax.devices()[0].platform=='gpu'
        own.bind_environment()
        saved=CHECKPOINT/'assets'/own.ASSET_ID/'norm_stats.json'
        assert sha(saved)==sha(own.NORM_PATH)==own.NORM_SHA256
        assert (CHECKPOINT/'_CHECKPOINT_METADATA').is_file() and (CHECKPOINT/'params/_METADATA').is_file()
        DataConfig,_,_=own.runtime_classes()
        model=Pi0Config(pi05=True,dtype='bfloat16',action_dim=32,action_horizon=16,paligemma_variant='gemma_2b',action_expert_variant='gemma_300m')
        cfg=config.TrainConfig(name=own.NAME,exp_name='checkpoint60000_inference_only',model=model,data=DataConfig(),assets_base_dir=str(TASK/'training_assets'),policy_metadata={'physical_action_dimension':24,'model_action_dimension':32,'action_horizon':16,'hand_absolute':True,'sdk_scale':1000.,'all50_train':True})
        inventory=[{'relative':str(p.relative_to(CHECKPOINT)),'bytes':p.stat().st_size} for p in sorted((CHECKPOINT/'params').rglob('*')) if p.is_file()]
        print('RESTORING_CHECKPOINT_60000',flush=True)
        start=time.monotonic()
        self.policy=policy_config.create_trained_policy(cfg,CHECKPOINT,sample_kwargs={'num_steps':10})
        self.identity={'checkpoint':str(CHECKPOINT),'checkpoint_step':60000,'full50_train':True,'sample_num_steps':10,'action_horizon':16,'payload_action_dim':24,'nominal_action_period_s':0.06,'official_commit':head,'official_worktree_clean':True,'official_create_trained_policy_used':True,'base_fallback':False,'optimizer_restored':False,'norm_stats_sha256':sha(saved),'norm_equal_to_training':True,'training_config_path':str(Path(own.__file__)),'training_config_sha256':sha(own.__file__),'data_adapter_sha256':sha(own.ADAPTER_PATH),'checkpoint_metadata_sha256':sha(CHECKPOINT/'_CHECKPOINT_METADATA'),'params_metadata_sha256':sha(CHECKPOINT/'params/_METADATA'),'params_file_inventory':inventory,'physical_gpu_index':2,'jax_devices':[str(d) for d in jax.devices()],'load_wall_s':time.monotonic()-start,'expert_actions_used':False,'hardware_IO':False,'model_outputs_unclipped':True,'official_policy_rng_initial_seed':0}
        write_new(Path(proof_dir)/'model_identity.json',self.identity)
        self.identity_sha256=sha(Path(proof_dir)/'model_identity.json')
    def infer(self,obs):
        start=time.monotonic();r=self.policy.infer(obs);v=self.np.asarray(r['actions'])
        assert v.shape==(16,24) and self.np.isfinite(v).all()
        return v,{'full_infer_wall_s':time.monotonic()-start,'official_policy_timing':r.get('policy_timing',{})}
