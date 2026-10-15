"""Read-only bounded summary of actual failed expert contact evidence."""
from pathlib import Path
import json
import sys
import numpy as np

for root in map(Path, sys.argv[1:]):
    for path in sorted(root.glob('*/expert_report.json')):
        report = json.loads(path.read_text())
        trace_path = path.parent/'physics_trace.json'
        if not trace_path.exists():
            continue
        trace = json.loads(trace_path.read_text())
        if not trace:
            continue
        last = trace[-1]
        contacts = last['grip']['contacts']
        row = {key: report.get(key) for key in
               ('accepted', 'error', 'physical_steps', 'canonical_sources_unchanged')}
        row.update(trial=path.parent.name,
            final_relative_translation_drift_m=last.get('relative_translation_drift_m'),
            final_relative_rotation_drift_deg=last.get('relative_rotation_drift_deg'),
            final_bottle_bottom_gap_m=last.get('bottle_bottom_gap_m'),
            closure_max_translation_m=max((r['fp_delta_m'] for r in trace if r['phase']['name']=='close'), default=0.),
            closure_max_rotation_deg=max((r['rotation_delta_deg'] for r in trace if r['phase']['name']=='close'), default=0.),
            final_phase=last['phase'],
            final_contact_forces_N={token:sum(c['normal_force_N'] for c in contacts if token in c['link'])
                                    for token in ('little','ring','middle','index','thumb')},
            final_opposing_dot=last['grip']['opposing_normals_dot'],
            final_latched=last['capture_latched'],
            final_pair_streak=last['capture_actual_pair_streak'],
            final_bottle_pose=last['bottle']['pose'],
            final_contacts=contacts)
        if report.get('adaptive_force_allocations'):
            entry=report['adaptive_force_allocations'][-1]
            row['final_LP']=entry
        if report.get('pre_retreat_grip') and report.get('actual_bottle_mass_properties'):
            reference=np.asarray(report['pre_retreat_grip']['actual']['relative_bottle_palm'])
            current=np.asarray(last['grip']['relative_bottle_palm'])
            local=np.asarray(report['actual_bottle_mass_properties']['cmass_local_matrix'])[:3,3]
            initial_pose=np.asarray(report['before']['pose'])
            point=np.asarray(report['before']['functional_point0'])[:3]
            local_fp=(np.linalg.inv(initial_pose)@np.r_[point,1.])[:3]
            row['frozen_relative_motion_decomposition']={
                'origin_drift_m':float(np.linalg.norm(current[:3,3]-reference[:3,3])),
                'COM_drift_m':float(np.linalg.norm((current[:3,:3]@local+current[:3,3])-(reference[:3,:3]@local+reference[:3,3]))),
                'functional_point0_drift_m':float(np.linalg.norm((current[:3,:3]@local_fp+current[:3,3])-(reference[:3,:3]@local_fp+reference[:3,3]))),
                'scope':'read-only point-motion decomposition; origin 2mm guard stays unchanged'}
        print(json.dumps(row,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else str(x)),flush=True)
