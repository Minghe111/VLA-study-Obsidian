"""Geometry-only upright retries; default feedback gain and timing retained."""
from pathlib import Path
import copy,json
work=Path(__file__).resolve().parent
source=json.loads((work/'smoke_worker_right_v8_jobs.json').read_text())[0]
jobs=[]
for shift in (.02,.01):
    job=copy.deepcopy(source)
    job['name']='right_coke_upright_slot25_axial'+str(int(shift*1000))+'mm_defaultgain_v9'
    job['spec']['grasp_setting']={'hand_x':shift}
    jobs.append(job)
target=work/'smoke_worker_right_upright_axial_v9_jobs.json'
with target.open('x',encoding='utf-8') as stream:stream.write(json.dumps(jobs,indent=2)+'\n')
print(str(target))
