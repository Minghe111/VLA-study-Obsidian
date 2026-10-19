"""Read-only C300 language diagnostics. Does not construct or step an optimizer.

Run only after other GPU jobs exit. Uses the isolated experiment's model loader,
text loss and native frozen-teacher KV objective without editing those sources.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import time
import traceback


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.staging')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def logits_stats(logits, labels):
    import torch
    values = logits.float()
    gold = values.gather(1, labels[:, None]).squeeze(1)
    top_score, top_id = values.max(-1)
    normalizer = values.logsumexp(-1)
    return {'top1': top_id.cpu().tolist(), 'gold_logit': gold.cpu().tolist(),
            'gold_probability': (gold-normalizer).exp().cpu().tolist(),
            'gold_ce': (normalizer-gold).cpu().tolist(),
            'gold_rank': (1+(values > gold[:, None]).sum(-1)).cpu().tolist(),
            'top1_logit': top_score.cpu().tolist()}


def fp32_projection_stats(hidden, weight, labels, *, chunk_size=4096):
    """Exact FP32 tied readout in vocab chunks; no full FP32 weight copy.

    Weight values are the loaded BF16 values cast to FP32, so this isolates
    arithmetic precision, not recovery of absent original FP32 weight bits.
    """
    import torch
    import torch.nn.functional as F
    if chunk_size < 1 or hidden.ndim != 2 or labels.shape != hidden.shape[:1]:
        raise ValueError('Invalid readout projection dimensions/chunk')
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        x = hidden.float()
        # Retaining [target_tokens,vocab] is modest here; avoid a full FP32
        # [vocab,hidden] weight copy. Gold ranks use the exact same matmul
        # values as the denominator, avoiding dot-vs-GEMM self-rank drift.
        projected = [F.linear(x, weight[begin:begin+chunk_size].float())
                     for begin in range(0, len(weight), chunk_size)]
        return logits_stats(torch.cat(projected, dim=-1), labels)



def token_regions(text, token_ids, tokenizer):
    """Token offsets from SentencePiece proto, with a marked decode fallback."""
    non_eos = token_ids[:-1]
    processor = tokenizer.text_tokenizer
    offsets, origin = [], 'sentencepiece_immutable_proto'
    try:
        proto = processor.encode_as_immutable_proto(text)
        if [piece.id for piece in proto.pieces] != non_eos:
            raise ValueError('Offset tokenizer IDs differ from target IDs')
        offsets = [(piece.begin, piece.end) for piece in proto.pieces]
    except (AttributeError, ValueError):
        origin = 'cumulative_decode_approximate'
        previous = 0
        for index in range(len(non_eos)):
            end = len(tokenizer.decode_text(non_eos[:index+1]))
            offsets.append((previous, max(previous, end)))
            previous = max(previous, end)
    keys = [(m.start(), m.end()-1) for m in re.finditer(r'"(?:[^"\\]|\\.)*"\s*:', text)]
    result = []
    for index, (begin, end) in enumerate(offsets):
        result.append({'index': index, 'first': index == 0, 'eos': False,
            'region': 'field_key' if any(begin < finish and end > start for start, finish in keys) else 'body_or_syntax',
            'char_begin': begin, 'char_end': end,
            'piece': processor.id_to_piece(non_eos[index]) if hasattr(processor, 'id_to_piece') else None})
    result.append({'index': len(non_eos), 'first': not non_eos, 'eos': True,
        'region': 'eos', 'char_begin': None, 'char_end': None, 'piece': '<eos>'})
    return result, origin


def summarize_readout(cases):
    """Aggregate same-position diagnostic accuracy by mode and token region."""
    groups = defaultdict(lambda: {'tokens': 0, 'native_correct': 0, 'fp32_correct': 0,
        'cache_correct': 0, 'native_ce_sum': 0.0, 'fp32_ce_sum': 0.0,
        'native_fp32_top1_agreement': 0, 'native_cache_top1_agreement': 0})
    for case in cases:
        for index, gold in enumerate(case['target_ids']):
            region = case['token_regions'][index]
            tags = {'all', region['region']}
            if region['first']:
                tags.add('first')
            for tag in tags:
                group = groups[f"{case['mode']}/{tag}"]
                group['tokens'] += 1
                group['native_correct'] += case['native_bf16']['top1'][index] == gold
                group['fp32_correct'] += case['same_hidden_fp32']['top1'][index] == gold
                group['cache_correct'] += case['cache_native_bf16']['top1'][index] == gold
                group['native_ce_sum'] += case['native_bf16']['gold_ce'][index]
                group['fp32_ce_sum'] += case['same_hidden_fp32']['gold_ce'][index]
                group['native_fp32_top1_agreement'] += case['native_bf16']['top1'][index] == case['same_hidden_fp32']['top1'][index]
                group['native_cache_top1_agreement'] += case['native_bf16']['top1'][index] == case['cache_native_bf16']['top1'][index]
    return {name: {**values, 'native_ce_mean': values['native_ce_sum']/values['tokens'],
        'fp32_ce_mean': values['fp32_ce_sum']/values['tokens']} for name, values in sorted(groups.items())}


def gradient_summary(names, ce, kv, kv_weight=1.0):
    """Gradient arrays are objective means on a fixed set, before clipping."""
    import torch
    totals = defaultdict(lambda: {'ce2': 0.0, 'kv2': 0.0, 'dot': 0.0, 'numel': 0})
    for name, left, right in zip(names, ce, kv, strict=True):
        layer = re.match(r'llm\.layers\.\d+', name)
        scopes = ('all', layer.group() if layer else name.rsplit('.', 1)[0])
        a, b = left.double(), right.double()
        for scope in scopes:
            totals[scope]['ce2'] += float(a.square().sum())
            totals[scope]['kv2'] += float(b.square().sum())
            totals[scope]['dot'] += float((a*b).sum())
            totals[scope]['numel'] += a.numel()
    report = {}
    for scope, values in totals.items():
        ce_norm, kv_norm = values['ce2']**.5, values['kv2']**.5
        weighted = kv_weight*kv_norm
        report[scope] = {'ce_grad_norm': ce_norm, 'kv_grad_norm': kv_norm,
            'weighted_kv_grad_norm': weighted,
            'weighted_kv_over_ce_norm': weighted/ce_norm if ce_norm else None,
            'cosine': values['dot']/(ce_norm*kv_norm) if ce_norm and kv_norm else None,
            'dot': values['dot'], 'numel': values['numel'],
            'weighted_combined_grad_norm': max(0.0, values['ce2'] + kv_weight**2*values['kv2'] + 2*kv_weight*values['dot'])**.5}
    return report


def cache_predictors(model, observation, target_ids):
    """Teacher-forced diagnostic cache path; no generated answer or training."""
    import torch
    from rlinf.models.embodiment.openpi.modules import model as model_lib
    observation = model_lib.preprocess_observation(observation, train=False)
    observation = model_lib._observation_to_dtype(observation, model.embed_dtype)
    hidden, valid, cache = model.build_prefix_cache(observation)
    index = torch.arange(valid.shape[1], device=valid.device)[None].expand_as(valid)
    final = index.masked_fill(~valid, -1).max(-1).values
    last = hidden[torch.arange(len(hidden), device=hidden.device), final]
    results = [last]
    for token in target_ids[:-1]:
        positions = valid.sum(-1, keepdim=True)
        valid = torch.cat([valid, torch.ones_like(valid[:, :1])], dim=1)
        output, cache = model.llm([model.llm.embed(token.reshape(1, 1)), None],
            positions=positions, mask=valid[:, None, :], kv_cache=cache)
        last = output[0][:, -1]
        results.append(last)
    return torch.cat(results, dim=0)


class LoRARoundingProbe:
    """All adapter rows on three mode representatives, sampled over token rows.

    Base/delta are recomputed using each module's native input dtype on at most
    64 deterministic evenly spaced rows. Native output is recorded for the same
    rows. Row-count-dependent matmul kernels may cause tiny recompute differences.
    """
    def __init__(self, model, max_rows=64):
        from rlinf.models.embodiment.openpi.recap_lora import AttentionLoRALinear
        self.rows = max_rows
        self.values = defaultdict(lambda: defaultdict(float))
        self.hooks = [module.register_forward_hook(self.make_hook(name))
            for name, module in model.named_modules()
            if isinstance(module, AttentionLoRALinear) and name.startswith('llm.layers.') and module.enabled]

    def make_hook(self, name):
        def hook(module, inputs, output):
            import torch
            import torch.nn.functional as F
            with torch.no_grad():
                x = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
                rows = torch.linspace(0, len(x)-1, min(len(x), self.rows), device=x.device).long()
                x = x[rows]
                base = F.linear(x, module.weight, module.bias)
                delta = F.linear(F.linear(x, module.lora_A.to(x.dtype)), module.lora_B.to(x.dtype))
                delta = delta * module.lora_scaling.to(x.dtype)
                added = base + delta
                actual = output.detach().reshape(-1, output.shape[-1])[rows]
                values = self.values[name]
                values['elements'] += added.numel()
                values['calls'] += 1
                values['base_square_sum'] += float(base.float().square().sum())
                values['delta_square_sum'] += float(delta.float().square().sum())
                values['delta_nonzero'] += int((delta != 0).sum())
                values['unchanged_after_add'] += int((added == base).sum())
                values['nonzero_delta_lost_to_add'] += int(((delta != 0) & (added == base)).sum())
                values['recomputed_output_max_abs'] = max(values['recomputed_output_max_abs'], float((added.float()-actual.float()).abs().max()))
        return hook

    def finish(self):
        for hook in self.hooks:
            hook.remove()
        report = {}
        for name, values in self.values.items():
            n = values['elements']
            report[name] = {**dict(values), 'base_rms': (values['base_square_sum']/n)**.5,
                'delta_rms': (values['delta_square_sum']/n)**.5,
                'delta_over_base_rms': (values['delta_square_sum']/values['base_square_sum'])**.5 if values['base_square_sum'] else None,
                'unchanged_after_add_fraction': values['unchanged_after_add']/n,
                'nonzero_delta_lost_fraction': values['nonzero_delta_lost_to_add']/values['delta_nonzero'] if values['delta_nonzero'] else None}
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--anchor-manifest', type=Path)
    parser.add_argument('--labels', type=Path)
    parser.add_argument('--kv-weight', type=float, default=1.0)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if not args.execute:
        parser.error('GPU diagnostics require explicit --execute; no optimizer updates occur')
    if args.kv_weight < 0:
        parser.error('KV weight must be nonnegative')
    args.root = args.root.resolve()
    sys.path.insert(0, str(args.root/'code/RLinf'))
    sys.path.insert(0, str(args.root/'scripts'))
    import fit_comparison as fit
    fit.ROOT = args.root
    from protocol_pilot import parameter_digest, read_records
    args.labels = args.labels or args.root/'data/fit_subsets_v1/fit96.quick12.train.jsonl'
    args.anchor_manifest = args.anchor_manifest or fit.SOURCE/'data/retention_anchor/manifest.json'
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'optimizer_constructed': False, 'optimizer_updates': 0,
        'checkpoint': str(args.checkpoint.resolve()), 'labels': str(args.labels),
        'anchor_manifest': str(args.anchor_manifest), 'kv_weight': args.kv_weight,
        'scope': 'fixed training diagnostic; no task SR or semantic generalization claim',
        'tf32_enabled': False,
        'same_hidden_fp32_note': 'loaded frozen tied embedding cast to FP32 in chunks; hidden/base computation unchanged',
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'contract': {'action_dim': 24, 'cameras': 3, 'period_s': .06, 'H': 16, 'execute': 1}}
    path = args.output/'report.json'
    write_json(path, report)
    start = time.monotonic()
    model = None
    try:
        import numpy as np
        import torch
        from rlinf.data.datasets.recap.hierarchical_collator import NativeJointCollator
        from rlinf.data.datasets.recap.hierarchical_dataset import load_observation
        from rlinf.models.embodiment.openpi.recap_tokenizer import canonical_json
        from rlinf.runners.hierarchical_recap_runner import NativeHierarchicalBackend
        random.seed(20261019); np.random.seed(20261019); torch.manual_seed(20261019)
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        cfg = fit.build_config('language_warmup', 96, kv_weight=args.kv_weight)
        cfg['model']['model_path'] = str(args.checkpoint.resolve())
        actor = NativeHierarchicalBackend(cfg)._actor()
        model = actor.model
        report['model_audit'] = fit.audit_model(model, 'language_warmup')
        report['parameters_before'] = {str(t): parameter_digest(model, trainable=t) for t in (True, False)}
        records = read_records([args.labels])
        fit.validate_fit_records(records, len(records))
        collator = NativeJointCollator(model, actor.tokenizer, loss_weights={'high_ce': 1.0}, training_mode='verified_sft')
        model.eval()
        report['readout'] = []
        rounding = LoRARoundingProbe(model)
        seen_modes = set()
        with torch.no_grad():
            for record in records:
                if record['mode'] in seen_modes:
                    for hook in rounding.hooks:
                        hook.remove()
                    rounding.hooks = []
                elif not rounding.hooks:
                    # Each mode's first representative needs its own hook lifetime.
                    extra = LoRARoundingProbe(model)
                    extra.values = rounding.values
                    rounding.hooks = extra.hooks
                seen_modes.add(record['mode'])
                batch = collator([record], [])
                labels = batch['high_target_ids'][batch['high_loss_mask']]
                captured_h, captured_logits = [], []
                decode = model.llm.embedder.decode
                def capture(hidden):
                    logits = decode(hidden)
                    captured_h.append(hidden.detach())
                    captured_logits.append(logits.detach().float())
                    return logits
                model.llm.embedder.decode = capture
                try:
                    ce = float(model(batch=batch)['high_ce'])
                finally:
                    model.llm.embedder.decode = decode
                # Stop rounding hooks before cache comparisons and free generation.
                for hook in rounding.hooks:
                    hook.remove()
                rounding.hooks = []
                hidden = torch.cat(captured_h, dim=0)
                native = logits_stats(torch.cat(captured_logits, dim=0), labels)
                fp32 = fp32_projection_stats(hidden, model.llm.embedder.embedding.weight, labels)
                cache_hidden = cache_predictors(model, batch['high_observation'], labels)
                cache_native = logits_stats(decode(cache_hidden).float(), labels)
                context = record['context']
                output = actor.generate(load_observation(record['observation']), record['mode'], {
                    'goal': context['goal'], 'memory_index': context['catalogue'],
                    'history': context['visible_history'] if isinstance(context['visible_history'], str) else canonical_json(context['visible_history']),
                    'visible_memory': context.get('actor_memory', context.get('selected_memory', {}))})
                regions, offset_origin = token_regions(record['target_text'], labels.cpu().tolist(), actor.tokenizer)
                first_wrong = next((i for i, (a,b) in enumerate(zip(native['top1'], labels.cpu().tolist(), strict=True)) if a != b), None)
                report['readout'].append({'sample_id': record['sample_id'], 'mode': record['mode'], 'teacher_forced_ce': ce,
                    'target_ids': labels.cpu().tolist(), 'target_text': record['target_text'], 'token_regions': regions,
                    'offset_origin': offset_origin, 'native_bf16': native, 'same_hidden_fp32': fp32,
                    'cache_native_bf16': cache_native, 'first_teacher_forced_error_index': first_wrong,
                    'same_hidden_fp32_top1_agreement': sum(a==b for a,b in zip(native['top1'], fp32['top1'], strict=True)),
                    'full_cache_top1_agreement': sum(a==b for a,b in zip(native['top1'], cache_native['top1'], strict=True)),
                    'full_cache_hidden_max_abs': float((hidden.float()-cache_hidden.float()).abs().max()),
                    'generated_text': output.text, 'generated_ids': output.token_ids,
                    'generated_eos': bool(output.ended_with_eos),
                    'generated_text_exact': output.text.strip() == record['target_text'].strip(),
                    'checks': fit.check_output(record, output.text, output.ended_with_eos)})
                write_json(path, report)
                del batch, captured_h, captured_logits, hidden, cache_hidden, native, fp32, cache_native
        report['lora_rounding'] = rounding.finish()
        report['readout_summary'] = summarize_readout(report['readout'])
        write_json(path, report)
        names_and_params = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
        names, params = zip(*names_and_params, strict=True)
        ce_grad = [torch.zeros_like(p, device='cpu', dtype=torch.float32) for p in params]
        kv_grad = [torch.zeros_like(p, device='cpu', dtype=torch.float32) for p in params]
        model.gradient_checkpointing_enable()
        model.train()
        losses = {'high_ce': [], 'kv_anchor': []}
        unused = {'high_ce': [], 'kv_anchor': []}
        def collect(loss, accumulator, count, objective):
            gradients = torch.autograd.grad(loss, params, allow_unused=True)
            absent = []
            for name, store, gradient in zip(names, accumulator, gradients, strict=True):
                if gradient is None:
                    absent.append(name)
                else:
                    if not bool(torch.isfinite(gradient).all()):
                        raise RuntimeError('Non-finite diagnostic gradient: '+name)
                    store.add_(gradient.detach().float().cpu(), alpha=1/count)
            unused[objective].append(absent)
        for record in records:
            batch = collator([record], [])
            loss = model(batch=batch)['high_ce']
            losses['high_ce'].append(float(loss.detach()))
            collect(loss, ce_grad, len(records), 'high_ce')
            del batch, loss
        manifest = json.loads(args.anchor_manifest.read_text())
        if len(manifest['cases']) != 8:
            raise ValueError('Expected all 8 native retention anchors')
        for case in manifest['cases']:
            ref = dict(case['observation'])
            if not Path(ref['npz_path']).is_absolute():
                ref['npz_path'] = str(args.anchor_manifest.parent/ref['npz_path'])
            observation = model.env_obs_to_observation(actor.env_io(load_observation(ref), case['instruction']))
            loss = model.kv_anchor_loss(observation).mean()
            losses['kv_anchor'].append(float(loss.detach()))
            collect(loss, kv_grad, len(manifest['cases']), 'kv_anchor')
            del loss, observation
        report['gradients'] = {'ce_samples': len(records), 'kv_samples': len(manifest['cases']),
            'losses': losses, 'unused_by_sample': unused, 'scope': 'per-objective mean gradients before clipping; diagnostic only',
            'metrics': gradient_summary(names, ce_grad, kv_grad, args.kv_weight)}
        model.zero_grad(set_to_none=True)
        report['parameters_after'] = {str(t): parameter_digest(model, trainable=t) for t in (True, False)}
        report['parameters_unchanged'] = report['parameters_before'] == report['parameters_after']
        if not report['parameters_unchanged']:
            raise RuntimeError('Read-only diagnostic changed parameters')
        report['peak_allocated_mib'] = torch.cuda.max_memory_allocated()/2**20
        report['peak_reserved_mib'] = torch.cuda.max_memory_reserved()/2**20
        report['status'] = 'completed'
    except Exception:
        report['status'] = 'failed'
        report['error'] = traceback.format_exc()
        if model is not None and 'parameters_before' in report:
            report['parameters_after_failure'] = {str(t): parameter_digest(model, trainable=t) for t in (True, False)}
            report['parameters_unchanged'] = report['parameters_before'] == report['parameters_after_failure']
        raise
    finally:
        report['elapsed_s'] = time.monotonic()-start
        write_json(path, report)
    print(json.dumps({k: report[k] for k in ('status', 'optimizer_updates', 'parameters_unchanged', 'elapsed_s')}))
    return report


if __name__ == '__main__':
    main()
