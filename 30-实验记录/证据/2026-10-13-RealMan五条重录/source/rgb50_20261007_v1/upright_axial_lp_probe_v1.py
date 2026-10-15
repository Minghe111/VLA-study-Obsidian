"""Frozen-contact axial sensitivity only, not a new physical grasp."""
from pathlib import Path
import copy,json,sys
import numpy as np
WORK=Path(__file__).resolve().parent
LEGACY=WORK.parent/'adjust_bottle_20261002'
sys.path.insert(0,str(LEGACY))
from purecontact_lp_v26 import allocate_contact_forces
root=WORK/'smoke_worker_right_v8/right_coke_upright_slot25_v8'
r=json.loads((root/'expert_report.json').read_text())
t=json.loads((root/'physics_trace.json').read_text())
first_close=next(row['step'] for row in t if row['phase']['name']=='close')
entry=next(e for e in r['adaptive_force_allocations'] if e['allocation']['feasible'] and e['wrench_mode']=='static_mg')
row=t[first_close+entry['step']-2]
assert row['phase']['name']=='close'
pose=np.asarray(row['bottle']['pose']);axis=pose[:3,1];axis/=np.linalg.norm(axis)
properties=r['actual_bottle_mass_properties'];local=np.asarray(properties['cmass_local_matrix'])
com=(pose@local)[:3,3];contacts=entry['contacts']
rows=[]
for shift in (-.02,-.01,-.005,0.,.005,.01,.015,.02,.025,.03):
    candidate=copy.deepcopy(contacts)
    for point in candidate:point['point_world_m']=(np.asarray(point['point_world_m'])-shift*axis).tolist()
    result=allocate_contact_forces(candidate,com_world_m=com,mass_kg=properties['mass_kg'],
        min_main_force_N=entry['lp_options']['min_main_force_N'])
    rows.append({'hand_x_shift_m':shift,'feasible':result['feasible'],'reason':result['reason'],
        'target_normal_forces_N':result['target_normal_forces_N'],
        'force_residual_inf_N':result['force_residual_inf_N'],'torque_residual_inf_Nm':result['torque_residual_inf_Nm']})
result={'scope':'Pure CPU frozen observed contact points translated along recorded bottle axis; normals fixed; not contact reachability, support or episode success',
        'source_report':str(root/'expert_report.json'),'source_trace':str(root/'physics_trace.json'),
        'selected_LP_control_step':entry['step'],'matched_physics_step':row['step'],
        'actual_mass_kg':properties['mass_kg'],'actual_COM_world_m':com.tolist(),
        'contact_axial_offsets_from_COM_m':[float((np.asarray(p['point_world_m'])-com)@axis) for p in contacts],
        'rows':rows}
output=WORK/'upright_axial_lp_probe_v1_results.json'
with output.open('x',encoding='utf-8') as stream:stream.write(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
