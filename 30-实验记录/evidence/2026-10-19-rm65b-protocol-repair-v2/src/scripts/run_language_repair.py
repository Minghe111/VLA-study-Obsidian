"""Serial 96-row C300 continuation. Plans by default; never starts formal training."""
from __future__ import annotations
import argparse
import json
import math
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


def planned_targets(start, cap, first=600, interval=200):
    if not 0 <= start < first <= cap <= 2400 or interval < 1:
        raise ValueError('Expected 0 <= source step < first <= explicit cap <= 2400')
    targets = list(range(first, cap + 1, interval))
    if targets[-1] != cap:
        targets.append(cap)
    return targets


def retention_signals(report):
    """Return trustworthy local gates, or None when receipts are incomplete."""
    passed, kv, action = [], [], []
    for key in ('retention', 'retention_holdout'):
        receipt = report.get(key)
        if not isinstance(receipt, dict):
            return None
        summary = receipt.get('summary')
        if not isinstance(summary, dict) or summary.get('count') != 8:
            return None
        count = summary.get('passed')
        if type(count) is not int or not 0 <= count <= 8:
            return None
        # A boolean alone cannot establish acceptance in an incomplete receipt.
        if receipt.get('passed') is not (count == 8):
            return None
        measurements = []
        for field in ('max_case_masked_all_layer_kv_relative_rms', 'max_case_normalized_physical24_rms'):
            value = summary.get(field)
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value) or value < 0):
                return None
            measurements.append(float(value))
        passed.append(count)
        kv.append(measurements[0])
        action.append(measurements[1])
    return {'passed': sum(passed), 'kv': kv, 'action': action}


def both_gates_pass(report):
    evaluation = report['evaluation']
    retention = retention_signals(report)
    return (evaluation.get('count') == 96 and evaluation.get('all_fit_passed') is True
            and retention is not None and retention['passed'] == 16)


def decide_stop(reports, total, cap):
    if len(reports) >= 2 and all(both_gates_pass(x) for x in reports[-2:]):
        return 'converged_all96_and_local_retention_twice'
    # Do not mistake language saturation for overall stagnation while the local
    # KV/action retention gates are still improving. Missing receipts are not a
    # plateau signal, and are never an acceptance signal.
    if total >= 1000 and len(reports) >= 3:
        window = reports[-3:]
        ce = [x['evaluation']['teacher_forced_ce'] for x in window]
        gain = (ce[0] - min(ce[1:])) / max(abs(ce[0]), 1e-12)
        fit = [x['evaluation']['fit_passed']['passed'] for x in window]
        signals = [retention_signals(x) for x in window]
        retention_stagnant = False
        if all(x is not None for x in signals):
            no_pass_gain = max(x['passed'] for x in signals[1:]) <= signals[0]['passed']
            no_metric_gain = True
            # Check both pools individually so progress in holdout observations
            # cannot be hidden behind a larger stationary anchor-pool error.
            for metric in ('kv', 'action'):
                for pool in (0, 1):
                    first = signals[0][metric][pool]
                    best_later = min(x[metric][pool] for x in signals[1:])
                    relative_gain = (first - best_later) / max(abs(first), 1e-12)
                    no_metric_gain = no_metric_gain and relative_gain < .05
            retention_stagnant = no_pass_gain and no_metric_gain
        if gain < .05 and max(fit[1:]) <= fit[0] and retention_stagnant:
            return 'plateau_ready_for_controlled_lr_kv_comparison'
    if total >= cap:
        return 'budget_exhausted_not_converged'
    return None


def run_subprocess(argv, cwd, log):
    process = None
    try:
        with log.open('x') as stream:
            process = subprocess.Popen(argv, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT,
                start_new_session=True, env={**os.environ, 'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1'})
            code = process.wait()
        if code:
            raise RuntimeError(f'Fit subprocess exited {code}; inspect {log}')
    except BaseException:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        raise


def parse_arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resume', type=Path, required=True, help='C300 exact continuation checkpoint')
    p.add_argument('--baseline', type=Path, required=True, help='Original all-adapters-off retention snapshot')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-total-updates', type=int, default=2000)
    p.add_argument('--first-eval', type=int, default=600)
    p.add_argument('--eval-interval', type=int, default=200)
    p.add_argument('--learning-rate', type=float, default=5e-5)
    p.add_argument('--kv-weight', type=float, default=1.0)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--enable-training', action='store_true')
    a = p.parse_args(argv)
    if a.execute and not a.enable_training:
        p.error('Optimizer updates require --enable-training')
    if not 600 <= a.first_eval <= a.max_total_updates <= 2400 or a.eval_interval < 1:
        p.error('96-row diagnostic requires first-eval >=600 and cap<=2400')
    return a


def main(argv=None):
    a = parse_arguments(argv)
    source = a.resume.resolve()
    version = json.loads((source / 'version.json').read_text())
    previous = int(version['global_step'])
    if version.get('training_manifest', {}).get('training_stage') != 'language_warmup':
        raise ValueError('Continuation expects the full-language-attention stage')
    if version.get('metadata', {}).get('fit_dataset_size') != 96:
        raise ValueError('Continuation requires the same 96-row checkpoint')
    from fit_comparison import select_training_labels
    labels, label_provenance = select_training_labels({}, checkpoint=source)
    targets = planned_targets(previous, a.max_total_updates, a.first_eval, a.eval_interval)
    state = {'status': 'planned', 'scope': '96-row full-language-attention step continuation',
        'formal_3000_training': False, 'automatic_promotion': False, 'task_SR_measured': False,
        'source_checkpoint': str(source), 'source_checkpoint_global_step': previous,
        'branch_global_step_start': previous, 'max_total_updates': a.max_total_updates,
        'additional_optimizer_update_cap': a.max_total_updates - previous,
        'eval_targets': targets, 'evaluation_count': 96, 'steps': [], 'started_unix': time.time(),
        'stop_rule': 'two consecutive strict all96 AND anchor8 AND holdout8 passes; plateau earliest1000 requires language and both KV/action retention pools stagnant; hard cap'}
    if not a.execute:
        print(json.dumps(state, indent=2), flush=True)
        return state
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / 'logs').mkdir()
    state['status'] = 'running'
    def save():
        write_json(a.output / 'summary.json', state)
    save()
    reports = []
    try:
        if not a.baseline.is_file():
            raise ValueError('Original retention baseline snapshot is required')
        for total in targets:
            name = f'language_warmup_96_{total:04d}'
            out = a.output / name
            args = [sys.executable, str(ROOT / 'scripts/fit_comparison.py'), 'fit',
                '--stage', 'language_warmup', '--size', '96', '--output', str(out),
                '--resume', str(source), '--training-labels', str(labels), '--baseline', str(a.baseline.resolve()),
                '--total-updates', str(total), '--max-fit-updates', str(a.max_total_updates),
                '--learning-rate', str(a.learning_rate), '--kv-weight', str(a.kv_weight),
                '--eval-all', '--execute', '--enable-training']
            log = a.output / 'logs' / f'{name}.log'
            entry = {'name': name, 'argv': args, 'status': 'running', 'log': str(log),
                'start_global_step': previous, 'target_global_step': total,
                'additional_optimizer_updates': total - previous, 'started_unix': time.time()}
            state['steps'].append(entry)
            save()
            start = time.monotonic()
            run_subprocess(args, ROOT, log)
            report = json.loads((out / 'report.json').read_text())
            if (report.get('status') != 'completed' or report['training']['global_step'] != total
                or not report['resume']['exact_resume'] or report['evaluation']['count'] != 96):
                raise RuntimeError('Exact resume/update/full96 evaluation acceptance failed')
            if not report['frozen_parameters_exact'] or report['gradient_audit']['frozen_with_grad']:
                raise RuntimeError('Frozen parameters/gradient protection failed')
            reports.append(report)
            entry.update(status='completed', elapsed_s=time.monotonic()-start,
                evaluation=report['evaluation'], memory=report['memory'],
                sampled_card_peak_mib=report.get('sampled_card_peak_mib'),
                retention_summary=report.get('retention', {}).get('summary'),
                retention_holdout_summary=report.get('retention_holdout', {}).get('summary'),
                both_gates_pass=both_gates_pass(report))
            source = out / 'checkpoint'
            previous = total
            state['final_checkpoint'] = str(source)
            state['total_additional_optimizer_updates'] = total - state['source_checkpoint_global_step']
            reason = decide_stop(reports, total, a.max_total_updates)
            if reason is not None:
                state['stop_reason'] = reason
                break
            save()
        state['status'] = 'completed'
    except BaseException:
        state['status'] = 'failed'
        state['error'] = traceback.format_exc()
        if state['steps'] and state['steps'][-1]['status'] == 'running':
            state['steps'][-1]['status'] = 'failed'
        raise
    finally:
        state['finished_unix'] = time.time()
        save()
    return state


if __name__ == '__main__':
    main()
