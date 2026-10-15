"""Create fresh bounded feedback-only retries from unchanged slot25 sample."""
from pathlib import Path
import copy,json
work=Path(__file__).resolve().parent
source=json.loads((work/'smoke_worker_right_v8_jobs.json').read_text())[0]
jobs=[]
for gain in (.6,.9):
    job=copy.deepcopy(source)
    job['name']='right_coke_upright_slot25_gain'+str(int(gain*10))+'_v9'
    job['spec']['grasp_setting']={'feedback_force_gain':gain}
    jobs.append(job)
target=work/'smoke_worker_right_feedback_v9_jobs.json'
with target.open('x',encoding='utf-8') as stream:stream.write(json.dumps(jobs,indent=2)+'\n')
print(str(target))
