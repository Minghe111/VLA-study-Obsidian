"""Read recorded closure contacts and applied SDK targets without simulation."""
from pathlib import Path
import hashlib
import json
import sys

for argument in sys.argv[1:]:
    root = Path(argument)
    paths = [root / 'expert_report.json'] if (root / 'expert_report.json').is_file() else sorted(root.glob('*/expert_report.json'))
    for path in paths:
        report = json.loads(path.read_bytes())
        trace_path = path.parent / 'physics_trace.json'
        if not trace_path.is_file():
            continue
        trace = json.loads(trace_path.read_bytes())
        closure = [row for row in trace if row['phase']['name'] in ('close', 'close_hold')]
        if not closure:
            continue
        arm = report['sample_spec']['arm']
        offset = 6 if arm == 'left' else 18
        tokens = ('little', 'ring', 'middle', 'index', 'thumb')
        channels = {}
        for index, token in enumerate(tokens):
            force_rows = [(row['step'], sum(contact['normal_force_N'] for contact in row['grip']['contacts'] if token in contact['link'])) for row in closure]
            first_latch = next((row['step'] for row in closure if row['capture_latched'][index]), None)
            peak_step, peak = max(force_rows, key=lambda item: item[1])
            channels[token] = {
                'first_recorded_latch_step': first_latch,
                'peak_force_N': peak, 'peak_force_step': peak_step,
                'rows_above_0p025N': sum(value > .025 for _, value in force_rows),
                'rows_above_0p05N': sum(value > .05 for _, value in force_rows),
                'last_recorded_force_N': force_rows[-1][1],
            }
        print(json.dumps({'scope': 'Read-only recorded physics; no new simulation or success claim',
                          'trial': str(path.parent.resolve()), 'accepted': report.get('accepted'),
                          'error': report.get('error'), 'closure_rows': len(closure),
                          'closure_first_step': closure[0]['step'], 'closure_last_step': closure[-1]['step'],
                          'first_applied_hand_sdk': closure[0]['command24_actual'][offset:offset + 6],
                          'last_applied_hand_sdk': closure[-1]['command24_actual'][offset:offset + 6],
                          'max_three_contact_streak': max(row['capture_actual_pair_streak'] for row in closure),
                          'LP_history_count': len(report.get('adaptive_force_allocations', [])),
                          'channels': channels,
                          'immutable_inputs_sha256': {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in (path, trace_path)}}), flush=True)
