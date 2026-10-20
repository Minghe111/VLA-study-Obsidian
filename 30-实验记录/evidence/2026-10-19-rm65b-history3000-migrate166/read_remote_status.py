import subprocess
REMOTE=r'''
from pathlib import Path
import json,time,subprocess,os,socket
root=Path('/bigdata2/liminghe/rlinf_history3000_20261019_v1')
assert root.resolve()==root and socket.gethostname()=='user-4U-GPU-Server' and os.getuid()==1014
new=root/'source_tree/deploy/realman_gen4/recap_memory_20261017/experiments/high_sft_history3000_20261019_v1'
job=new/'reports/supervisor_166_v2'
def read(p):return json.loads(p.read_text()) if p.exists() else {}
s=read(job/'summary.json');phase=(s.get('phases')or[{}])[-1];e=read(job/'eval_s002000/summary.json');t=read(job/'train_s002500/report.json');p=read(job/'train_s002500/progress.json');g=read(job/'train_s002500/first_gradient_receipt.json')
report={'time_unix':time.time(),'supervisor_status':s.get('status'),'phase':{k:phase.get(k) for k in ('phase','status','pid','returncode','elapsed_s')},'last_saved_branch_step':s.get('last_saved_branch_step'),'evaluation':{'status':e.get('status'),'model_load_count':e.get('model_load_count'),'all144_reloaded':e.get('actual_saved_adapter_reload144',{}).get('all144_exact'),'readonly':e.get('readonly_model_proof'),'suites':{k:{'status':v.get('status'),'strict':(v.get('summary')or{}).get('overall',{}).get('strict_passed')}for k,v in e.get('suites',{}).items()},'retention':{k:v.get('passed')for k,v in e.get('retention',{}).items()},'memory':e.get('memory'),'sampled_card_peak_mib':e.get('sampled_card_peak_mib')},'training':{'status':t.get('status'),'training_started':t.get('training_started'),'resume':t.get('resume'),'compact_overlay':{k:v for k,v in t.get('compact_overlay',{}).items()if k not in ('tensors','checks')},'progress':p,'gradient_receipt':{k:v for k,v in g.items()if k!='tensors'}},'gpu':subprocess.check_output(['nvidia-smi','--id=GPU-b8006bad-7fb4-41be-9016-7822db997854','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],universal_newlines=True).strip()}
for name,value in [('supervisor',s),('evaluation',e),('training',t)]:
 if value.get('error'):report[name+'_error']=value['error'][-3500:]
if e.get('status')=='failed' or t.get('status')=='failed' or s.get('status')=='failed':
 log=Path(phase.get('log','/nonexistent'))
 if log.is_file():report['error_log_tail']=log.read_text(errors='replace')[-4500:]
print(json.dumps(report,ensure_ascii=False))
'''
cmd=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15','-o','ProxyCommand=/usr/bin/python3 /tmp/ssh_166_via_enp5s0.py %h %p','liminghe@172.17.27.166','python3','-']
raise SystemExit(subprocess.run(cmd,input=REMOTE,text=True).returncode)
