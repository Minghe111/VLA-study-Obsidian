"""Read-only full-gold CE/cached-decoder audit; no optimizer or backward."""
import contextlib
import hashlib
import json
import pathlib
import sys
import time

ROOT = pathlib.Path('/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/recap_memory_20261017/experiments/protocol_sft_20261018_v1')
OUT = pathlib.Path('/tmp/rm65b_protocol_audit_20261019')
sys.path.insert(0, str(ROOT / 'code/RLinf'))
sys.path.insert(0, str(ROOT / 'scripts'))

import torch
import torch.nn.functional as F
import yaml
from verify_protocol_logit_alignment import select_record
from rlinf.data.datasets.recap.hierarchical_collator import NativeJointCollator
from rlinf.models.embodiment.openpi.modules import model as model_lib
from rlinf.models.embodiment.openpi.pi0 import make_attn_mask
from rlinf.models.embodiment.openpi.recap_lora import legacy_prefix_context
from rlinf.runners.hierarchical_recap_runner import NativeHierarchicalBackend


def dump():
    OUT.mkdir(exist_ok=True)
    (OUT / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')


def digest_adapters(model):
    d = hashlib.sha256()
    for name, value in model.named_parameters():
        if value.requires_grad:
            d.update(name.encode())
            d.update(value.detach().cpu().contiguous().numpy().tobytes())
    return d.hexdigest()


def full_gold_probe(model, tokenizer, collator, record, precision, adapters=True):
    batch = collator([record], [])
    obs, targets = batch['high_observation'], batch['high_target_ids']
    loss_mask = batch['high_loss_mask'].bool()
    captured = []
    decode = model.llm.embedder.decode

    def capture(hidden):
        logits = decode(hidden)
        captured.append(logits.float().detach())
        return logits

    model.llm.embedder.decode = capture
    try:
        # These logits come directly from the production text_loss, not a substitute.
        actual_ce = model.text_loss(obs, targets, loss_mask, target_mask=batch['high_target_mask'])
    finally:
        model.llm.embedder.decode = decode
    gold_logits = torch.cat(captured, dim=0)
    manual_ce = F.cross_entropy(gold_logits, targets[loss_mask], reduction='mean')
    prepared = model_lib.preprocess_observation(obs, train=False)
    prepared = model_lib._observation_to_dtype(prepared, model.embed_dtype)
    hidden, valid, cache = model.build_prefix_cache(prepared)
    index = torch.arange(valid.shape[1], device=model.device)[None]
    final = index.masked_fill(~valid, -1).max(-1).values
    last = hidden[torch.arange(1, device=model.device), final]
    case = {'mode': record['mode'], 'sample_id': record.get('sample_id'), 'precision': precision,
            'adapters_enabled': adapters, 'target_text': record['target_text'],
            'target_ids': targets[0].tolist(), 'loss_mask': loss_mask[0].tolist(),
            'production_ce': float(actual_ce[0]), 'reconstructed_ce': float(manual_ce),
            'ce_abs_difference': float((actual_ce[0] - manual_ce).abs()), 'steps': []}
    assert targets[0, -1] == tokenizer.eos_token_id
    assert loss_mask.all()
    assert gold_logits.shape[0] == targets.shape[1]
    for i, target in enumerate(targets[0]):
        cached = decode(last).float()[0]
        full = gold_logits[i]
        diff = (cached - full).abs()
        target_id = int(target)
        row = {'step': i, 'target': target_id, 'is_eos': target_id == tokenizer.eos_token_id,
               'full_top1': int(full.argmax()), 'cached_top1': int(cached.argmax()),
               'same_top1': int(full.argmax()) == int(cached.argmax()),
               'full_correct': int(full.argmax()) == target_id,
               'cached_correct': int(cached.argmax()) == target_id,
               'full_target_probability': float(full.softmax(-1)[target]),
               'cached_target_probability': float(cached.softmax(-1)[target]),
               'full_target_rank': int((full > full[target]).sum()) + 1,
               'cached_target_rank': int((cached > cached[target]).sum()) + 1,
               'max_abs_logit_difference': float(diff.max()), 'mean_abs_logit_difference': float(diff.mean()),
               'cached_nll': float(F.cross_entropy(cached[None], target.reshape(1)))}
        case['steps'].append(row)
        if i < targets.shape[1] - 1:
            pos = valid.sum(-1, keepdim=True)
            valid = torch.cat([valid, torch.ones_like(target.reshape(1, 1), dtype=torch.bool)], dim=1)
            result, cache = model.llm([model.llm.embed(target.reshape(1, 1)), None],
                                      positions=pos, mask=valid[:, None, :], kv_cache=cache)
            last = result[0][:, -1]
    case['cached_gold_ce'] = sum(x['cached_nll'] for x in case['steps']) / len(case['steps'])
    case['full_token_accuracy'] = sum(x['full_correct'] for x in case['steps']) / len(case['steps'])
    case['cache_top1_agreement'] = sum(x['same_top1'] for x in case['steps']) / len(case['steps'])
    case['max_abs_logit_difference'] = max(x['max_abs_logit_difference'] for x in case['steps'])
    # Diagnostic 16-token free generation, never counted as the original validation.
    generated = model.generate_text_tokens(obs, max_new_tokens=16, eos_token_id=tokenizer.eos_token_id)
    case['free16_ids'] = generated[0].tolist()
    case['free16_text'] = tokenizer.decode_text(case['free16_ids'])
    case['free16_eos'] = tokenizer.eos_token_id in case['free16_ids']
    return case


start = time.monotonic()
report = {'status': 'running', 'training': False, 'backward_calls': 0, 'optimizer_steps': 0,
          'production_code_modified': False, 'checkpoint_modified': False,
          'scope': 'gold-history diagnostic only, not robot success or official validation',
          'fp32_compute_source': 'saved checkpoint base rounded to BF16; FP32 arithmetic control, not original FP32 source',
          'fp32_max_abs_tolerance': 0.001, 'ce_reconstruction_tolerance': 0.00001, 'cases': []}
dump()
try:
    torch.set_num_threads(4)
    torch.manual_seed(20261019)
    config = yaml.safe_load((ROOT / 'configs/rm65b_protocol_sft_pilot.yaml').read_text())
    config['model']['model_path'] = str(ROOT / 'reports/protocol_sft_sequence_v1/05_fit128/checkpoint')
    actor = NativeHierarchicalBackend(config)._actor()
    model = actor.model.eval()
    assert not model.mem_enabled and not actor.advantage_condition
    before = digest_adapters(model)
    report['adapter_sha_before'] = before
    report['tokenizer'] = {'vocab': actor.tokenizer.vocab_size, 'eos': actor.tokenizer.eos_token_id}
    records = [select_record(ROOT / 'data/protocol_labels_reviewed_v1/train.shuffled.reviewed.jsonl', 'READ', {'train'}),
               select_record(ROOT / 'data/protocol_labels_reviewed_v1/val.reviewed.jsonl', 'WRITE', {'val'}),
               select_record(ROOT / 'data/protocol_labels_reviewed_v1/val.reviewed.jsonl', 'SUBTASK', {'val'})]
    collator = NativeJointCollator(model, actor.tokenizer, loss_weights={'high_ce': 1.0}, training_mode='verified_sft')
    original_tf32 = torch.backends.cuda.matmul.allow_tf32
    original_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    report['default_tf32'] = {'matmul': original_tf32, 'cudnn': original_cudnn_tf32}
    with torch.inference_mode():
        for precision in ('bf16_default', 'bf16_tf32_off', 'fp32_tf32_off'):
            if precision != 'bf16_default':
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
            if precision == 'fp32_tf32_off':
                model.float()
                model.embed_dtype = torch.float32
                model.llm.embed_dtype = torch.float32
                model.img.dtype_mm = torch.float32
            for record in records:
                case = full_gold_probe(model, actor.tokenizer, collator, record, precision)
                report['cases'].append(case)
                dump()
                print(json.dumps({k: case[k] for k in ('mode', 'precision', 'production_ce', 'ce_abs_difference', 'cache_top1_agreement', 'max_abs_logit_difference', 'full_token_accuracy', 'free16_eos')}), flush=True)
            if precision == 'bf16_default':
                with legacy_prefix_context(model, disable_adapters=True):
                    for record in records:
                        case = full_gold_probe(model, actor.tokenizer, collator, record, 'bf16_base_adapter_off', adapters=False)
                        report['cases'].append(case)
                        dump()
    report['adapter_sha_after'] = digest_adapters(model)
    report['trainable_parameters_unchanged'] = report['adapter_sha_after'] == before
    report['all_gradients_none'] = all(p.grad is None for p in model.parameters())
    report['ce_reconstruction_passed'] = all(c['ce_abs_difference'] <= report['ce_reconstruction_tolerance'] for c in report['cases'])
    fp32 = [c for c in report['cases'] if c['precision'] == 'fp32_tf32_off']
    report['fp32_cache_full_passed'] = all(c['max_abs_logit_difference'] <= report['fp32_max_abs_tolerance'] and c['cache_top1_agreement'] == 1.0 for c in fp32)
    report['peak_allocated_mib'] = torch.cuda.max_memory_allocated() / 2**20
    report['status'] = 'completed'
except Exception:
    import traceback
    report['status'] = 'failed'
    report['error'] = traceback.format_exc()
    raise
finally:
    report['elapsed_s'] = time.monotonic() - start
    dump()
