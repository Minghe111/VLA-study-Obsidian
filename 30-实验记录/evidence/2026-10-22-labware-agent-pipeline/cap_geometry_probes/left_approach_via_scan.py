"""Bounded wrist/arm ordering scan, geometry only; native auditor still required."""
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np

OUT=Path(__file__).resolve().parent
PIPE=OUT.parent
sys.path[:0]=[str(PIPE/'robot_cap_task')]
from cap_grasp_probe import Geometry,sha
from left_cap_held_probe import set_hand

snapshot=PIPE/'robot_cap_task/static_hold_probes/held115_static_reference_v1/actual_terminal_snapshot.json'
rebind=OUT/'left115_static_rebind400/rebind.json'
folder=OUT/'left115_static_via_ordering';folder.mkdir(exist_ok=True)
g=Geometry(folder,snapshot=snapshot,geometry_spec_path=PIPE/'robot_cap_task/profiles/bottle115/physics_spec115.json')
d=json.loads(rebind.read_text());start=set_hand(g,g.q0,d['open_sdk6']);goal=np.asarray(d['open_q36']);arm=[g.index[f'l_joint{i}']for i in range(1,7)]
patterns=[[3,4,5],[2,3,4,5],[2,3],[3],[4,5],[2],[1],[1,3,4,5],[1,2,3]]

def check(waypoints,step):
    records=[]
    for segment,(a,b)in enumerate(zip(waypoints[:-1],waypoints[1:])):
        count=max(2,int(np.ceil(np.max(np.abs(b[arm]-a[arm]))/step)))
        for fraction in np.linspace(0,1,count+1):
            q=a+(b-a)*fraction;audit=g.screen(q);contacts=g.cap_contact_depths()
            clear=audit['collision_free_except_intended_cap_contact']and not any(contacts['groups'].values())
            records.append({'segment':segment,'fraction':float(fraction),'clear':clear,'collisions':audit['collisions'],'cap_contacts':bool(any(contacts['groups'].values()))})
            if not clear:return False,records
    return True,records

t0=time.time();results=[];chosen=None
for pattern in patterns:
    via=start.copy();via[[arm[i]for i in pattern]]=goal[[arm[i]for i in pattern]]
    waypoints=[start,via,goal];ok,coarse=check(waypoints,.04)
    row={'first_adjust_joint_numbers':[i+1 for i in pattern],'coarse_clear':ok,'coarse_samples':coarse}
    if ok:
        refined,details=check(waypoints,.008);row['refined_discrete_clear']=refined;row['refined_samples']=details
        if refined:
            row['arm6_waypoints']=[q[arm].tolist()for q in waypoints];chosen=row['arm6_waypoints'];results.append(row);break
    results.append(row);print(json.dumps({'tested':len(results),'elapsed_s':time.time()-t0,'found':chosen is not None}),flush=True)
summary={'scope':'CPU finite ordering/via line scan, no physical rollout','reference_snapshot_sha256':sha(snapshot),'rebind_sha256':sha(rebind),'source_sha256':sha(__file__),
    'selected_arm6_waypoints':chosen,'left_sdk6_fixed':d['open_sdk6'],'tested_count':len(results),'elapsed_s':time.time()-t0,'results':results,
    'continuous_swept_certified':False,'right_body_reference_is_not_permanent_attachment':True,
    'execution_requirement':'Use native full-mesh continuous curve audit and 60ms velocity/acceleration retiming before actual execution; preserve actual right drive/force feedback.'}
(folder/'via_scan.json').write_text(json.dumps(summary,indent=2));print(json.dumps({k:summary[k]for k in ['selected_arm6_waypoints','tested_count','elapsed_s']},indent=2))
