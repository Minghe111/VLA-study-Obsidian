"""Bounded 9/96-row fit diagnostics. No formal-data or autonomous robot run."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/recap_memory_20261017/experiments/protocol_sft_20261018_v1')
sys.path.insert(0, str(ROOT / 'code/RLinf'))
sys.path.insert(0, str(ROOT / 'scripts'))
from protocol_pilot import audit_records, check_output, parameter_digest, read_records, sha256, write_json


class CardMonitor:
    def __init__(self):
        self.stop = threading.Event()
        self.samples = []
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stop.is_set():
            try:
                value = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True, timeout=5)
                self.samples.append(float(value.strip().splitlines()[0]))
            except Exception:
                pass
            self.stop.wait(2)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=6)


def audit_model(model, stage):
    from rlinf.models.embodiment.openpi.recap_lora import recap_training_manifest
    manifest = recap_training_manifest(model)
    expected = 131072 if stage == 'protocol' else 3686400
    names = [name for name, value in model.named_parameters() if value.requires_grad]
    count = sum(value.numel() for value in model.parameters() if value.requires_grad)
    if count != expected or model.mem_enabled or manifest['training_stage'] != stage:
        raise ValueError(f'Unexpected training scope {stage}: {count}, MEM={model.mem_enabled}')
    if any(not name.startswith('llm.layers.') or '.0.lora_' not in name for name in names):
        raise ValueError('Only explicit language attention adapters may train')
    return {**manifest, 'trainable_numel': count, 'runtime_mem_enabled': model.mem_enabled}


def validate_fit_records(records, size):
    if len(records) != size or Counter(r['mode'] for r in records) != {k: size//3 for k in ('READ', 'WRITE', 'SUBTASK')}:
        raise ValueError('Fit rows/mode balance differs from explicit diagnostic size')
    digests = [r.get('actor_input_digest') for r in records]
    if any(not isinstance(x, str) or len(x) != 64 for x in digests) or len(set(digests)) != size:
        raise ValueError('Fit samples must be audited distinct actual actor inputs')


def check_resume_lr(state, requested):
    if any(group['lr'] != requested or group.get('weight_decay', 0) != 0
           for group in state['optimizer']['param_groups']):
        raise ValueError('Resume optimizer LR/weight decay differs from requested config')


def build_config(stage, size, learning_rate=5e-5, kv_weight=1.0):
    import yaml
    cfg = yaml.safe_load((ROOT / 'configs/rm65b_protocol_sft_pilot.yaml').read_text())
    cfg['model']['openpi']['recap_training_stage'] = stage
    cfg['model']['version'] = f'fit_{stage}_{size}_initial'
    cfg['model']['openpi']['recap_lora_rank'] = 16
    cfg['model']['openpi']['recap_lora_alpha'] = 16
    cfg['actor_training'].update(learning_rate=learning_rate, max_updates=300, micro_batch_size=1,
        gradient_accumulation=4, strict_resume=True, sampler={'shuffle': True, 'seed': 20261019},
        sequential_objectives=True, loss_weights={'high_ce': 1.0, 'fast_ce': 0.0, 'fm': 0.0,
        'kv_anchor': kv_weight if stage == 'language_warmup' else 0.0})
    cfg['data']['verified_high_sft_paths'] = [str(ROOT / f'data/fit_subsets_v1/fit{size}.train.jsonl')]
    cfg['environment']['robotwin_root'] = str(ROOT / 'code/RoboTwin')
    cfg['environment']['task_config']['runtime_dir'] = str(ROOT / 'runtime_fit')
    cfg['experiment'] = {'scope': '9/96-row diagnostic only; no 3000-row training',
        'same_initial_checkpoint': cfg['model']['model_path'], 'max_updates_per_run': 300,
        'retention_scope': 'Eight original-task states, not skill SR',
        'kv_weight': cfg['actor_training']['loss_weights']['kv_anchor']}
    return cfg


def evaluate(actor, labels_path, destination):
    import torch
    from rlinf.data.datasets.recap.hierarchical_collator import NativeJointCollator
    from rlinf.data.datasets.recap.hierarchical_dataset import load_observation
    from rlinf.models.embodiment.openpi.recap_tokenizer import canonical_json
    records = read_records([labels_path])
    actor.model.eval()
    collator = NativeJointCollator(actor.model, actor.tokenizer, loss_weights={'high_ce': 1.0}, training_mode='verified_sft')
    report = {'scope': 'training-set fitting only, no heldout success claim', 'cases': [],
              'decoding': 'unrestricted greedy, max96, actual EOS; no JSON repair'}
    with torch.no_grad():
        for record in records:
            batch = collator([record], [])
            targets = batch['high_target_ids'][batch['high_loss_mask']]
            decoded = []
            original_decode = actor.model.llm.embedder.decode
            def capture(hidden):
                logits = original_decode(hidden)
                decoded.append(logits.float().detach())
                return logits
            actor.model.llm.embedder.decode = capture
            try:
                ce = float(actor.model(batch=batch)['high_ce'])
            finally:
                actor.model.llm.embedder.decode = original_decode
            gold = torch.cat(decoded, dim=0)
            context = record['context']
            output = actor.generate(load_observation(record['observation']), record['mode'], {
                'goal': context['goal'], 'memory_index': context['catalogue'],
                'history': context['visible_history'] if isinstance(context['visible_history'], str) else canonical_json(context['visible_history']),
                'visible_memory': context.get('actor_memory', context.get('selected_memory', {}))})
            checks = check_output(record, output.text, output.ended_with_eos)
            exact = output.text.strip() == record['target_text'].strip()
            checks['fit_passed'] = checks['passed'] and (record['mode'] != 'SUBTASK' or exact)
            checks['text_exact'] = exact
            checks['first_token_match'] = bool(output.token_ids) and output.token_ids[0] == int(targets[0])
            report['cases'].append({'sample_id': record.get('sample_id'), 'mode': record['mode'],
                'target_text': record['target_text'], 'text': output.text, 'token_ids': output.token_ids,
                'ce': ce, 'gold_correct': int((gold.argmax(-1) == targets).sum()), 'gold_tokens': len(targets),
                'gold_eos_probability': float(gold[-1].softmax(-1)[actor.tokenizer.eos_token_id]),
                'gold_first_probability': float(gold[0].softmax(-1)[targets[0]]), 'checks': checks})
            write_json(destination, report)
            del gold, decoded, batch
    cases = report['cases']
    summary = {'count': len(cases), 'modes': dict(Counter(c['mode'] for c in cases)),
        'teacher_forced_ce': sum(c['ce'] for c in cases) / len(cases),
        'gold_tokens': sum(c['gold_tokens'] for c in cases), 'gold_correct': sum(c['gold_correct'] for c in cases)}
    for field in ('eos', 'nonempty', 'first_token_match', 'json', 'schema_legal', 'target_match', 'required_terms', 'side_binding_lexical', 'text_exact', 'passed', 'fit_passed'):
        values = [c['checks'][field] for c in cases if field in c['checks']]
        summary[field] = {'passed': sum(values), 'eligible': len(values)}
    summary['all_fit_passed'] = all(c['checks']['fit_passed'] for c in cases)
    report['summary'] = summary
    write_json(destination, report)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['baseline', 'fit', 'eval'])
    parser.add_argument('--stage', choices=['protocol', 'language_warmup'], default='protocol')
    parser.add_argument('--size', type=int, choices=[9, 96], default=9)
    parser.add_argument('--total-updates', type=int, default=100)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--eval-all', action='store_true')
    parser.add_argument('--learning-rate', type=float, default=5e-5)
    parser.add_argument('--kv-weight', type=float, default=1.0)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--enable-training', action='store_true')
    args = parser.parse_args(argv)
    if not args.execute or (args.command == 'fit' and not args.enable_training):
        parser.error('Execution and optimizer updates require explicit switches')
    if args.command == 'fit' and not 1 <= args.total_updates <= 300:
        parser.error('Diagnostic limit is 300 updates per independent run')
    if args.resume and args.checkpoint:
        parser.error('Choose exact resume or evaluation checkpoint')
    if args.command == 'fit' and args.checkpoint:
        parser.error('Fit arms start independently from original SFT or exact same-dataset resume')
    args.output.mkdir(parents=True, exist_ok=False)
    report_path = args.output / 'report.json'
    report = {'status': 'running', 'command': args.command, 'stage': args.stage, 'size': args.size,
        'started_unix': time.time(), 'formal_3000_training': False, 'task_SR_measured': False,
        'contract': {'action_dim': 24, 'cameras': 3, 'period_s': 0.06, 'H': 16, 'execute': 1},
        'script_sha256': sha256(Path(__file__)), 'output': str(args.output)}
    started = time.monotonic()
    write_json(report_path, report)
    monitor = CardMonitor()
    try:
        import numpy as np
        import torch
        import yaml
        from fit_retention import evaluate_retention
        from rlinf.data.datasets.recap.hierarchical_collator import NativeJointCollator
        from rlinf.data.datasets.recap.hierarchical_dataset import load_observation
        from rlinf.runners.hierarchical_recap_runner import ActorMicrobatchSampler, NativeHierarchicalBackend, asset_versions
        from rlinf.workers.sft.fsdp_hierarchical_recap_worker import FSDPHierarchicalReCapWorker
        random.seed(20261019); np.random.seed(20261019); torch.manual_seed(20261019)
        torch.set_num_threads(4)
        cfg = build_config(args.stage, args.size, args.learning_rate, args.kv_weight)
        manifest_path = SOURCE / 'data/retention_anchor/manifest.json'
        labels = Path(cfg['data']['verified_high_sft_paths'][0])
        records, data_audit = audit_records([labels], cfg, training=True)
        validate_fit_records(records, args.size)
        subset_audit = json.loads((ROOT / 'reports/fit_subsets_audit.json').read_text())
        if sha256(labels) != subset_audit['subsets'][labels.stem]['sha256']:
            raise ValueError('Fit labels differ from the CPU-audited subset')
        report['data_audit'] = {k: v for k, v in data_audit.items() if k != 'rows'}
        report['initial_model_path'] = cfg['model']['model_path']
        previous = 0
        if args.resume:
            previous = json.loads((args.resume / 'version.json').read_text())['global_step']
            trainer_state = torch.load(args.resume / 'trainer_state.pt', map_location='cpu', weights_only=True)
            check_resume_lr(trainer_state, args.learning_rate)
            del trainer_state
            cfg['model']['model_path'] = str(args.resume.resolve())
        elif args.checkpoint:
            cfg['model']['model_path'] = str(args.checkpoint.resolve())
        report['loaded_model_path'] = cfg['model']['model_path']
        (args.output / 'effective_config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
        with monitor:
            actor = NativeHierarchicalBackend(cfg)._actor()
            model = actor.model
            report['model_audit'] = audit_model(model, args.stage)
            report['frozen_sha_before'] = parameter_digest(model, trainable=False)
            before = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
            if args.command == 'fit':
                if previous >= args.total_updates:
                    raise ValueError('Requested target must exceed restored global_step')
                anchor_manifest = json.loads(manifest_path.read_text())
                # Include every anchor observation/instruction in the strict data fingerprint.
                retention_refs = [{'observation': {'npz_path': str(manifest_path.parent / x['observation']['npz_path'])},
                    'instruction': x['instruction']} for x in anchor_manifest['cases']]
                if cfg['actor_training']['loss_weights']['kv_anchor']:
                    for record in records:
                        record['_fit_retention_contract'] = retention_refs
                sampler = ActorMicrobatchSampler(records, [], [], [], batch_size=1, gradient_accumulation=4,
                    anchor_interval=4, training_mode='verified_sft', loss_weights=cfg['actor_training']['loss_weights'],
                    shuffle=True, seed=20261019)
                optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=0.0)
                worker = FSDPHierarchicalReCapWorker(model, optimizer, gradient_accumulation=4,
                    max_grad_norm=1.0, enable_training=True, sequential_objectives=True, sampler=sampler, strict_resume=True)
                if args.resume:
                    report['resume'] = worker.restore_trainer_state(args.resume)
                    if not report['resume']['exact_resume']:
                        raise RuntimeError('Expected exact sampler/optimizer/RNG restoration')
                else:
                    report['resume'] = {'restored': False, 'exact_resume': False, 'reason': 'independent original-SFT initialization'}
                start_cursor = sampler.microbatch_cursor
                report['sampler_before'] = sampler.state_dict()
                model.gradient_checkpointing_enable()
                collator = NativeJointCollator(model, actor.tokenizer,
                    loss_weights={'high_ce': 1.0, 'fast_ce': 0.0, 'fm': 0.0, 'kv_anchor': 0.0}, training_mode='verified_sft')
                updates = args.total_updates - previous
                traces, losses = [], []
                def batches():
                    for _ in range(updates * 4):
                        h, a, anchor = sampler.next_batch()
                        assert not a and not anchor and len(h) == 1
                        batch = collator(h, [])
                        if cfg['actor_training']['loss_weights']['kv_anchor']:
                            ref = retention_refs[(sampler.microbatch_cursor - 1) % len(retention_refs)]
                            native = load_observation(ref['observation'])
                            batch['anchor_observation'] = model.env_obs_to_observation(actor.env_io(native, ref['instruction']))
                            batch['loss_weights']['kv_anchor'] = args.kv_weight
                        traces.append({'microbatch': sampler.microbatch_cursor, 'sample_id': h[0].get('sample_id')})
                        batch['supervision_source'] = 'verified_high_sft_fit'
                        yield batch
                def loss_hook(module, inputs, output):
                    del module, inputs
                    losses.append({k: float(v.detach()) for k, v in output.items() if k in ('loss', 'high_ce', 'kv_anchor') and v.ndim == 0})
                    if len(losses) % 40 == 0:
                        write_json(args.output / 'progress.json', {'global_update_forward_groups': previous + len(traces) // 4,
                            'microbatch_cursor': sampler.microbatch_cursor, 'latest_branch': losses[-1],
                            'elapsed_s': time.monotonic() - started})
                hook = model.register_forward_hook(loss_hook)
                report['training_started'] = True
                write_json(report_path, report)
                metrics = worker.train(batches(), max_updates=updates)
                hook.remove()
                assert worker.global_step == args.total_updates
                assert sampler.microbatch_cursor == start_cursor + updates * 4
                report['training'] = metrics
                report['sampler_after'] = sampler.state_dict()
                report['training_sample_presentations_this_process'] = len(traces)
                report['sample_trace_sha256'] = hashlib.sha256(json.dumps(traces, sort_keys=True).encode()).hexdigest()
                write_json(args.output / 'sample_trace.json', traces)
                write_json(args.output / 'loss_trace.json', losses)
                report['gradient_audit'] = {'trainable_with_grad': sum(p.grad is not None for p in model.parameters() if p.requires_grad),
                    'trainable_finite_grad': sum(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.requires_grad),
                    'frozen_with_grad': sum(p.grad is not None for p in model.parameters() if not p.requires_grad)}
                if report['gradient_audit']['frozen_with_grad']:
                    raise RuntimeError('Frozen parameter received gradient')
                worker.save_checkpoint(args.output / 'checkpoint', version=args.output.name + '/actor',
                    metadata={'training_mode': 'verified_sft', 'asset_versions': asset_versions(cfg),
                        'fit_dataset_size': args.size, 'same_source_initial_model': report['initial_model_path'],
                        'training_label_sha256': sha256(labels)})
                report['checkpoint'] = str(args.output / 'checkpoint')
                # Save RNG/state before evaluation; next process restores the training boundary.
                del worker, optimizer
            report['frozen_sha_after'] = parameter_digest(model, trainable=False)
            report['frozen_parameters_exact'] = report['frozen_sha_before'] == report['frozen_sha_after']
            if not report['frozen_parameters_exact']:
                raise RuntimeError('Frozen parameters changed')
            changes = {name: float((p.detach().cpu() - before[name]).abs().max())
                for name, p in model.named_parameters() if p.requires_grad}
            report['changed_trainable_tensors'] = sum(x > 0 for x in changes.values())
            report['max_trainable_delta'] = max(changes.values(), default=0)
            if args.command == 'fit' and not report['changed_trainable_tensors']:
                raise RuntimeError('Optimizer did not change an adapter')
            del before
            model.eval()
            evaluation_labels = labels if args.eval_all or args.size == 9 else ROOT / 'data/fit_subsets_v1/fit96.quick12.train.jsonl'
            evaluation_records = read_records([evaluation_labels])
            validate_fit_records(evaluation_records, args.size if args.eval_all or args.size == 9 else 12)
            if sha256(evaluation_labels) != subset_audit['subsets'][evaluation_labels.stem]['sha256']:
                raise ValueError('Evaluation labels differ from audited training-only subset')
            report['evaluation'] = evaluate(actor, evaluation_labels, args.output / 'evaluation.json')
            if args.command == 'baseline':
                report['retention'] = evaluate_retention(actor, manifest_path, args.output / 'retention')
            elif args.baseline:
                report['retention'] = evaluate_retention(actor, manifest_path, args.output / 'retention', args.baseline)
            holdout_manifest = ROOT / 'data/retention_holdout/manifest.json'
            if holdout_manifest.is_file() and args.command in ('fit', 'eval') and args.total_updates >= 100:
                holdout_teacher = ROOT / 'reports/retention_holdout_teacher/snapshot.pt'
                if not holdout_teacher.is_file():
                    # The shared base is frozen and hashed; disabling adapters recovers the original teacher.
                    evaluate_retention(actor, holdout_manifest, holdout_teacher.parent)
                report['retention_holdout'] = evaluate_retention(actor, holdout_manifest,
                    args.output / 'retention_holdout', holdout_teacher)
            report['memory'] = {'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
                'peak_reserved_mib': torch.cuda.max_memory_reserved()/2**20,
                'host_peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
        report['card_memory_samples'] = len(monitor.samples)
        report['sampled_card_peak_mib'] = max(monitor.samples) if monitor.samples else None
        report['status'] = 'completed'
    except Exception:
        report['status'] = 'failed'
        report['error'] = traceback.format_exc()
        raise
    finally:
        report['elapsed_s'] = time.monotonic() - started
        write_json(report_path, report)
    print(json.dumps({k: v for k, v in report.items() if k in ('status', 'stage', 'size', 'training', 'evaluation', 'memory', 'sampled_card_peak_mib', 'elapsed_s')}), flush=True)
    return report


if __name__ == '__main__':
    main()
