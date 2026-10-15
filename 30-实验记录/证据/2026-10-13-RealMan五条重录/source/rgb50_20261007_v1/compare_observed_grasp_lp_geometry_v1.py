"""Read actual LP contact geometry at the pre-step pose, not a hypothetical grasp."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('episodes', nargs='+', type=Path)
ap.add_argument('--output', required=True, type=Path)
opt = ap.parse_args()
rows=[]
for root in opt.episodes:
    root=root.resolve()
    report_path=root/'expert_report.json';trace_path=root/'physics_trace.json'
    report=json.loads(report_path.read_text(encoding='utf-8'))
    trace=json.loads(trace_path.read_text(encoding='utf-8'))
    history=report.get('adaptive_force_allocations',[])
    first_close=next((r['step'] for r in trace if r['phase']['name']=='close'),None)
    usable=[e for e in history if e['wrench_mode']=='static_mg' and e['phase']['name'] in ('close','close_hold')]
    if not usable or first_close is None:
        rows.append({'episode':str(root),'scope':'No executed close LP geometry available'})
        continue
    entry=next((e for e in usable if e['allocation']['feasible']),usable[-1])
    issued_before_step=first_close+entry['step']-1
    before=next((r for r in trace if r['step']==issued_before_step-1),None)
    if before is None:
        raise RuntimeError('No actual previous physics row for selected LP inputs')
    pose=np.asarray(before['bottle']['pose'])
    mass=report['actual_bottle_mass_properties']
    com=(pose@np.asarray(mass['cmass_local_matrix']))[:3,3]
    axis=pose[:3,1]/np.linalg.norm(pose[:3,1])
    contacts=[]
    for item in entry['contacts']:
        normal=np.asarray(item['force_on_bottle_normal_world'])
        point=np.asarray(item['point_world_m'])
        contacts.append({'channel':item['channel'],'point_world_m':point.tolist(),
            'axial_offset_from_pre_step_COM_m':float((point-com)@axis),
            'normal_world':normal.tolist(),
            'moment_per_unit_normal_Nm_per_N':np.cross(point-com,normal).tolist()})
    dots=[]
    for i,a in enumerate(contacts):
        for b in contacts[i+1:]:
            if a['channel']!=b['channel']:
                dots.append({'channels':[a['channel'],b['channel']],
                    'normal_dot':float(np.dot(a['normal_world'],b['normal_world']))})
    rows.append({'episode':str(root),'accepted':report.get('accepted'),
        'error':report.get('error'),'physical_steps':len(trace),
        'selected_LP_control_step':entry['step'],'selected_LP_phase':entry['phase'],
        'LP_issued_before_physics_step':issued_before_step,
        'actual_pose_row_before_LP_step':before['step'],
        'pose_comes_from_previous_actual_physics_not_post_LP_step':True,
        'actual_mass_kg':mass['mass_kg'],'pre_step_COM_world_m':com.tolist(),
        'pre_step_bottle_axis_world':axis.tolist(),'contacts':contacts,
        'different_channel_normal_dots':dots,
        'allocation':entry['allocation'],
        'inputs_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (report_path,trace_path)}})
result={'scope':'Actual frozen recorded geometry only; not a reachability prediction or new physical success',
    'rows':rows}
with opt.output.open('x',encoding='utf-8') as stream:
    stream.write(json.dumps(result,indent=2)+'\n')
for row in rows:
    if 'contacts' in row:
        row={**row,'allocation':{k:row['allocation'].get(k) for k in
            ('feasible','reason','target_normal_forces_N','force_residual_inf_N','torque_residual_inf_Nm')}}
    print(json.dumps(row),flush=True)
