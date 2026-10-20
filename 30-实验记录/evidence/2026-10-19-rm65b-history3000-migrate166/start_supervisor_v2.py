from pathlib import Path
import json,os,subprocess,time,socket,sys
ROOT=Path('/bigdata2/liminghe/rlinf_history3000_20261019_v1');NEW=ROOT/'source_tree/deploy/realman_gen4/recap_memory_20261017/experiments/high_sft_history3000_20261019_v1';PY=ROOT/'venv/bin/python'
assert ROOT.resolve()==ROOT and socket.gethostname()=='user-4U-GPU-Server'and os.environ.get('USER')=='liminghe'
assert json.loads((ROOT/'reports/migration_preparation.json').read_text())['status']=='prepared_CPU_verified_pending_actual_policy_resume'
assert json.loads((ROOT/'reports/weight_repair_v1.json').read_text())['status']=='passed_weight_repaired_byte_exact_and_CPU_deserialized'
variables=json.loads((ROOT/'runtime_env.json').read_text());env={**os.environ,**variables}
for k,v in variables.items():os.environ[k]=v
sys.path[:0]=[str(NEW/'scripts'),str(NEW/'code/RLinf'),str(ROOT/'vendor')]
from train_history3000 import validate_compact_metadata
from evaluate_history3000_checkpoint import read_baseline,read_suite
checkpoint=NEW/'reports/migrated_s002000/checkpoint';version=validate_compact_metadata(checkpoint);assert version['global_step']==2000
baseline=read_baseline(NEW/'reports/c2200_baseline_v2/retention/baseline_report.json',version)
plan=json.loads((NEW/'data/evaluation_v1/evaluation_plan.json').read_text())
for name in ('val180','train_quick96','full_val600','revision_only40','history_ablation16','legacy_regression96'):read_suite(plan,name)
uuid=variables['CUDA_VISIBLE_DEVICES'];gpu=subprocess.check_output(['nvidia-smi','--id='+uuid,'--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True).strip().split(',');assert gpu[0].strip()==uuid and float(gpu[1])<512,gpu
claim=ROOT/'reports/remote_supervisor_launch_v2.json';assert not claim.exists(),'Do not launch duplicate supervisors'
output=NEW/'reports/supervisor_166_v2';assert not output.exists()
argv=[str(PY),str(NEW/'scripts/run_history3000_sft_supervisor.py'),'--root',str(NEW),'--start-checkpoint',str(checkpoint),'--output',str(output),'--max-updates','24000','--execute']
stdout=ROOT/'reports/supervisor_166_v2_stdout.log'
with stdout.open('x')as stream:p=subprocess.Popen(argv,cwd=NEW,env=env,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
report={'status':'launched_pending_actual_eval_and_policy_updates','started_unix':time.time(),'host':socket.gethostname(),'user':os.environ['USER'],'pid':p.pid,'owned_session':p.pid,'argv':argv,'stdout_log':str(stdout),'summary':str(output/'summary.json'),'source_branch_step':2000,'source_microbatch_cursor':8000,'physical_gpu_index':1,'gpu_uuid':uuid,'CUDA_VISIBLE_DEVICES':uuid,'stdin':'DEVNULL','detached_from_SSH':True,'mathematical_training_contract_unchanged':True,'full_test_not_started':True}
claim.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
