"""Aggregate bounded fit receipts without loading a model or changing a run."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    reports = a.root / 'reports/comparison_v1'
    sequence = json.loads((reports / 'summary.json').read_text())
    result = {'sequence_status': sequence['status'], 'formal_3000_training': False,
              'task_SR_measured': False, 'runs': [], 'final_arms': {}}
    for entry in sequence['steps']:
        file = reports / entry['name'] / 'report.json'
        if not file.exists():
            continue
        d = json.loads(file.read_text())
        if d['status'] != 'completed':
            continue
        row = {'name': entry['name'], 'stage': d['stage'], 'size': d['size'],
               'source_report': str(file), 'source_sha256': sha(file),
               'evaluation': d['evaluation'], 'memory': d['memory'],
               'sampled_card_peak_mib': d['sampled_card_peak_mib'],
               'elapsed_s': d['elapsed_s']}
        for key in ('training', 'resume', 'gradient_audit', 'frozen_parameters_exact',
                    'changed_trainable_tensors', 'max_trainable_delta'):
            if key in d:
                row[key] = d[key]
        if 'model_audit' in d:
            row['trainable_numel'] = d['model_audit']['trainable_numel']
            row['trainable_tensors'] = len(d['model_audit']['trainable_parameters'])
        for key in ('retention', 'retention_holdout'):
            if key in d and 'summary' in d[key]:
                r = d[key]
                row[key] = {'summary': r['summary'], 'passed': r['passed'],
                            'thresholds': r['thresholds'],
                            'physical_action_gate_passed': sum(c['action_metrics']['normalized_physical24']['rms'] <= r['thresholds']['normalized_physical24_rms_max'] for c in r['cases']),
                            'kv_gate_passed': sum(c['kv_metrics']['relative_rms'] <= r['thresholds']['masked_all_layer_kv_relative_rms_max'] for c in r['cases']),
                            'max_sdk_arm_rad': max(c['action_metrics']['sdk_arm_rad']['max_abs'] for c in r['cases']),
                            'max_sdk_hand_units': max(c['action_metrics']['sdk_hand_units']['max_abs'] for c in r['cases'])}
        if 'sampler_before' in d:
            row['sampler'] = {key: {field: d[key][field] for field in ('microbatch_cursor', 'data_fingerprint')}
                              for key in ('sampler_before', 'sampler_after')}
        result['runs'].append(row)
        if 'training' in row:
            result['final_arms'][f"{d['stage']}_{d['size']}"] = row
    for key, final in result['final_arms'].items():
        stage, size = final['stage'], final['size']
        runs = [r for r in result['runs'] if r['stage'] == stage and r['size'] == size and 'training' in r]
        trace = []
        for r in runs:
            trace.extend(json.loads((reports / r['name'] / 'sample_trace.json').read_text()))
        ids = Counter(r['sample_id'] for r in trace)
        final['total_sample_presentations'] = len(trace)
        final['per_sample_presentations_min_max'] = [min(ids.values()), max(ids.values())]
        final['observed_distinct_training_rows'] = len(ids)
        final['row_epoch_equivalent'] = len(trace) / size
        final['consecutive_microbatches'] = [r['microbatch'] for r in trace] == list(range(1, len(trace)+1))
        final['resume_boundary_sampler_exact'] = all(
            json.loads((reports / left['name'] / 'report.json').read_text())['sampler_after'] ==
            json.loads((reports / right['name'] / 'report.json').read_text())['sampler_before']
            for left, right in zip(runs, runs[1:]))
    result['total_optimizer_updates'] = sum(r.get('training', {}).get('updates', 0) for r in result['runs'])
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    for key, d in result['final_arms'].items():
        v = d['evaluation']
        print(key, 'updates', d['training']['global_step'], 'CE', round(v['teacher_forced_ce'], 4),
              'JSON', v['json'], 'EOS', v['eos'], 'fit', v['fit_passed'],
              'retention', d.get('retention', {}).get('summary'),
              'holdout', d.get('retention_holdout', {}).get('summary'))


if __name__ == '__main__':
    main()
