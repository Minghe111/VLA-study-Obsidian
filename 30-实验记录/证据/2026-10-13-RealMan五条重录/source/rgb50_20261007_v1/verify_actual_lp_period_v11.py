"""Read actual completed LP histories; no controller or acceptance changes."""
from pathlib import Path
import argparse, hashlib, json

ap = argparse.ArgumentParser()
ap.add_argument('episodes', nargs='+', type=Path)
ap.add_argument('--output', type=Path, required=True)
opt = ap.parse_args()
rows = []
for episode in opt.episodes:
    episode = episode.resolve()
    report_file = episode/'expert_report.json'
    trace_file = episode/'physics_trace.json'
    report = json.loads(report_file.read_text(encoding='utf-8'))
    trace = json.loads(trace_file.read_text(encoding='utf-8'))
    history = report.get('adaptive_force_allocations', [])
    period = report.get('LP_refresh_experiment', {}).get('selected_period_physics_steps', 50)
    first_close = next((r['step'] for r in trace if r['phase']['name']=='close'), None)
    interval_rows, first_rows = [], []
    by_group = {}
    for entry in history:
        phase = entry['phase']
        key = (phase['name'], phase['dense_call'])
        if key not in by_group:
            physical = next((r['step'] for r in trace if
                r['phase']['name']==phase['name'] and r['phase']['dense_call']==phase['dense_call']), None)
            inferred = first_close+entry['step']-1 if first_close is not None else None
            first_rows.append({'phase':phase,'first_LP_control_step':entry['step'],
                'first_actual_physics_step':physical,'inferred_LP_physics_step':inferred,
                'same_first_step':physical==inferred if physical is not None else None})
        previous = by_group.get(key)
        if previous is not None:
            elapsed = entry['step']-previous['step']
            mode_changed = entry['wrench_mode']!=previous['wrench_mode']
            interval_rows.append({'phase':phase,'previous_control_step':previous['step'],
                'control_step':entry['step'],'elapsed_physics_steps':elapsed,
                'mode_changed':mode_changed,'on_period_boundary':entry['step']%period==0,
                'expected_refresh':entry['step']%period==0 or mode_changed,
                'within_selected_period':0<elapsed<=period})
        by_group[key] = entry
    intervals_proven = bool(interval_rows) and all(r['expected_refresh'] and
        r['within_selected_period'] for r in interval_rows)
    carry_first = [r for r in first_rows if r['phase']['name']=='after_grasp']
    rows.append({'episode':str(episode),'accepted':report.get('accepted'),
        'error':report.get('error'),'expert_version':report.get('expert_version'),
        'physical_steps':len(trace),'LP_period_physics_steps':period,
        'LP_updates':len(history),'LP_within_segment_intervals':interval_rows,
        'within_segment_period_physically_proven':intervals_proven,
        'first_update_by_phase':first_rows,
        'every_executed_carry_segment_refreshes_first_step':
            bool(carry_first) and all(r['same_first_step'] for r in carry_first),
        'LP_refresh_experiment':report.get('LP_refresh_experiment'),
        'LP_refresh_final_clock':report.get('LP_refresh_final_clock'),
        'last_actual_physics':{k:trace[-1].get(k) for k in
            ('step','sim_time_s','phase','bottle_bottom_gap_m',
             'relative_translation_drift_m','relative_rotation_drift_deg')} if trace else None,
        'inputs_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (report_file,trace_file)}})
result = {'scope':'Read-only actual physics/LP evidence; no source, controller, scene or acceptance edits',
    'static_checks_not_used_as_physical_success':True,'rows':rows}
with opt.output.open('x',encoding='utf-8') as stream:
    stream.write(json.dumps(result,indent=2)+'\n')
for row in rows:
    print(json.dumps({k:v for k,v in row.items() if k!='LP_within_segment_intervals'}))
