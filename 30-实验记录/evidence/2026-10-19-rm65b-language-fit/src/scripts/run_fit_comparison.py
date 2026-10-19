"""Serial small-fit sequence with explicit per-run bounds and preserved receipts."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from protocol_pilot import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--enable-training', action='store_true')
    p.add_argument('--max9', type=int, choices=[100, 200, 300], default=200)
    p.add_argument('--max96', type=int, choices=[100, 200, 300], default=300)
    a = p.parse_args()
    if not a.execute or not a.enable_training:
        p.error('Explicit execute and training switches required')
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / 'logs').mkdir()
    state = {'status': 'running', 'scope': 'bounded 9-to-96 fitting comparison only',
        'formal_3000_training': False, 'automatic_promotion': False, 'independent_from_same_base': True,
        'max_updates_per_9_run': a.max9, 'max_updates_per_96_run': a.max96,
        'steps': [], 'started_unix': time.time(), 'final_checkpoints': {}}
    def save(): write_json(a.output / 'summary.json', state)
    def invoke(name, command, stage='protocol', size=9, total=100, resume=None, checkpoint=None, full=False):
        out = a.output / name
        argv = [sys.executable, str(ROOT / 'scripts/fit_comparison.py'), command,
            '--stage', stage, '--size', str(size), '--output', str(out), '--execute']
        if command == 'fit': argv += ['--enable-training', '--total-updates', str(total)]
        if resume: argv += ['--resume', str(resume)]
        if checkpoint: argv += ['--checkpoint', str(checkpoint)]
        if command != 'baseline': argv += ['--baseline', str(a.output / '00_baseline/retention/snapshot.pt')]
        if full: argv += ['--eval-all']
        log = a.output / 'logs' / f'{name}.log'
        entry = {'name': name, 'argv': argv, 'status': 'running', 'log': str(log), 'started_unix': time.time()}
        state['steps'].append(entry); save()
        start = time.monotonic(); process = None
        try:
            with log.open('x') as f:
                process = subprocess.Popen(argv, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                    start_new_session=True, env={**os.environ, 'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1'})
                code = process.wait()
            entry['exit_code'] = code
            if code: raise RuntimeError(f'{name} exited {code}; {log}')
            report = json.loads((out / 'report.json').read_text())
            if report['status'] != 'completed': raise RuntimeError(f'{name}: incomplete report')
            entry['status'] = 'completed'; entry['evaluation'] = report['evaluation']
            entry['memory'] = report['memory']; entry['sampled_card_peak_mib'] = report['sampled_card_peak_mib']
            if 'training' in report: entry['training'] = report['training']
            if 'retention' in report: entry['retention_summary'] = report['retention'].get('summary')
            return report
        except BaseException:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired: os.killpg(process.pid, signal.SIGKILL); process.wait()
            entry['status'] = 'failed'; entry['error'] = traceback.format_exc(); raise
        finally:
            entry['elapsed_s'] = time.monotonic()-start; save()
    try:
        invoke('00_baseline', 'baseline')
        # One update is part of the bounded 9-row run, then exact process resume.
        for stage in ('protocol', 'language_warmup'):
            r = invoke(f'{stage}_9_001', 'fit', stage, 9, 1)
            state['final_checkpoints'][f'{stage}_9'] = r['checkpoint']; save()
        for total in range(100, a.max9 + 1, 100):
            for stage in ('protocol', 'language_warmup'):
                key = f'{stage}_9'
                r = invoke(f'{stage}_9_{total:03}', 'fit', stage, 9, total,
                    resume=Path(state['final_checkpoints'][key]))
                if r['training']['global_step'] != total or not r['resume']['exact_resume']:
                    raise RuntimeError('9-row strict resume/update budget failed')
                state['final_checkpoints'][key] = r['checkpoint']; save()
        # 96-row arms reset to the same original SFT, not the 9-row checkpoint.
        for total in range(100, a.max96 + 1, 100):
            for stage in ('protocol', 'language_warmup'):
                key = f'{stage}_96'
                r = invoke(f'{stage}_96_{total:03}', 'fit', stage, 96, total,
                    resume=Path(state['final_checkpoints'][key]) if key in state['final_checkpoints'] else None,
                    full=total == a.max96)
                if r['training']['global_step'] != total:
                    raise RuntimeError('96-row update budget failed')
                state['final_checkpoints'][key] = r['checkpoint']; save()
        state['status'] = 'completed'
        state['total_optimizer_updates'] = 2*a.max9 + 2*a.max96
        state['observed_metrics_only'] = True
    except BaseException:
        state['status'] = 'failed'; state['error'] = traceback.format_exc(); raise
    finally:
        state['finished_unix'] = time.time(); save()


if __name__ == '__main__': main()
