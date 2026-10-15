"""Fresh real-contact candidates; JSON alone is not geometric evidence."""
from pathlib import Path
import copy,json
work=Path(__file__).resolve().parent
queue=work/'candidate_queue_v9_gpu3_7_20261007'
found={}
for path in queue.glob('*_jobs.json'):
    for job in json.loads(path.read_text()):
        if job['spec']['index'] in (6,11):found[job['spec']['index']]=job
assert set(found)=={6,11}
jobs=[]
for index,x,roll in ((11,.005,None),(6,.02,5.),(6,.02,-5.),(11,.002,None)):
    source=found[index];spec=copy.deepcopy(source['spec'])
    spec['grasp_setting']['hand_x']=x
    if roll is not None:spec['grasp_setting']['palm_roll_deg']=roll
    spec['sampling_revision']='Fresh axial/roll grasp geometry retry; original nearest-arm position/seed, unchanged force LP/contact guards; v2 initializer axial_roll must match actual scene matrix'
    name=f"slot{index:07d}_axial{x*1000:.0f}mm_roll{roll if roll is not None else 'keep'}_v9"
    jobs.append({'name':name,'spec':spec})
target=work/'smoke_worker_axial_geometry_v9_jobs.json'
with target.open('x',encoding='utf-8') as stream:stream.write(json.dumps(jobs,indent=2)+'\n')
print(str(target))
