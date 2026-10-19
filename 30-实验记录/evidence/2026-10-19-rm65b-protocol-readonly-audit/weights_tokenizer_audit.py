"""Read-only, CPU-only weight and tokenizer audit; writes receipts under /tmp."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

# No CUDA initialization or model forward is needed for this audit.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
import safetensors
import sentencepiece as sp

OUT = Path(__file__).resolve().parent
ROOT = Path('/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/recap_memory_20261017/experiments/protocol_sft_20261018_v1')
BASE = Path('/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/rlinf_local_5090_minppo_20261013/models/pi05_80000_openpi/model.safetensors')
TOKENIZER = Path('/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/recap_memory_20261017/cache/openpi/big_vision/paligemma_tokenizer.model')
CK32 = ROOT/'reports/protocol_sft_sequence_v1/03_fit32/checkpoint/full_weights.pt'
CK128 = ROOT/'reports/protocol_sft_sequence_v1/05_fit128/checkpoint/full_weights.pt'


def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def main():
    started = time.time()
    torch.set_num_threads(2)
    final = torch.load(CK128, map_location='cpu', weights_only=True, mmap=True)
    stage32 = torch.load(CK32, map_location='cpu', weights_only=True, mmap=True)
    tokenizer = sp.SentencePieceProcessor(model_file=str(TOKENIZER))
    report = {
        'schema': 'rm65b_weights_tokenizer_readonly_cpu_audit_v1',
        'started_unix': started,
        'runtime': {'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'], 'device': 'cpu', 'torch_num_threads': torch.get_num_threads(), 'no_optimizer_or_forward': True},
        'paths': {'base': str(BASE), 'checkpoint32': str(CK32), 'checkpoint128': str(CK128), 'tokenizer': str(TOKENIZER)},
        'sha256': {name: digest(path) for name, path in [('script', Path(__file__)), ('base', BASE), ('checkpoint32', CK32), ('checkpoint128', CK128), ('tokenizer', TOKENIZER)]},
        'tokenizer': {'vocab_size': tokenizer.vocab_size(), 'bos': tokenizer.bos_id(), 'eos': tokenizer.eos_id(), 'pad': tokenizer.pad_id(), 'unk': tokenizer.unk_id()},
    }
    roundtrips = []
    for text in ['{"read_ids":["sample_identity"]}', '{"operations":[]}', 'Pick up and hold the sample initially on the left.']:
        ids = tokenizer.encode(text, add_eos=True)
        decoded = tokenizer.decode(ids)
        roundtrips.append({'text': text, 'target_ids': ids, 'decoded': decoded, 'exact_roundtrip': text == decoded, 'eos_last': ids[-1] == tokenizer.eos_id()})
    report['target_roundtrips'] = roundtrips
    with safetensors.safe_open(BASE, framework='pt', device='cpu') as base:
        base_keys = set(base.keys())
        counts = {'direct_cast_equal': 0, 'bf16_roundtrip_fp32_island_equal': 0}
        mismatches, rows = [], []
        for key in sorted(base_keys):
            old, new = base.get_tensor(key), final[key]
            direct = torch.equal(old.to(new.dtype), new)
            roundtrip = (not direct and new.dtype == torch.float32 and torch.equal(old.bfloat16().float(), new))
            kind = 'direct_cast_equal' if direct else 'bf16_roundtrip_fp32_island_equal' if roundtrip else 'mismatch'
            if kind in counts:
                counts[kind] += 1
            else:
                mismatches.append(key)
            rows.append({'key': key, 'shape': list(new.shape), 'base_dtype': str(old.dtype), 'saved_dtype': str(new.dtype), 'comparison': kind})
        embedding = 'llm.embedder.embedding.weight'
        report['base_to_128'] = {'base_key_count': len(base_keys), 'checkpoint128_key_count': len(final), 'missing_base_keys': sorted(base_keys-set(final)), 'extra_keys_count': len(set(final)-base_keys), 'counts': counts, 'unexplained_mismatches': mismatches, 'per_key': rows, 'embedding_shape': list(final[embedding].shape), 'embedding_equal_after_bf16_cast': next(row['comparison'] for row in rows if row['key'] == embedding) == 'direct_cast_equal'}
    changed = []
    for key in final:
        if not torch.equal(final[key], stage32[key]):
            changed.append({'key': key, 'shape': list(final[key].shape), 'max_abs_delta': float((final[key]-stage32[key]).abs().max())})
    report['checkpoint32_to_128'] = {'compared_keys': len(final), 'same_key_set': set(stage32) == set(final), 'changed_keys': changed, 'unchanged_key_count': len(final)-len(changed)}
    for name, section in [('02_eval_baseline','baseline_evaluation'), ('06_eval128','evaluation128')]:
        path = ROOT/'reports/protocol_sft_sequence_v1'/name/'report.json'
        receipt = json.loads(path.read_text())
        report[section] = {'path': str(path), 'sha256': digest(path), 'model_checkpoint': receipt['model_checkpoint'], 'summary': receipt['summary'], 'first_case': {key:receipt['cases'][0][key] for key in ['mode','target_text','text','token_ids']}}
    manifest = ROOT/'reports/protocol_sft_sequence_v1/05_fit128/checkpoint/version.json'
    report['checkpoint_manifest'] = {'path': str(manifest), 'sha256': digest(manifest), 'contents': json.loads(manifest.read_text())}
    report['elapsed_s'] = time.time() - started
    report['status'] = 'completed'
    (OUT/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'output': str(OUT/'report.json'), 'elapsed_s': report['elapsed_s'], 'counts': report['base_to_128']['counts'], 'unexplained_mismatches': report['base_to_128']['unexplained_mismatches'], 'changed_keys': [item['key'] for item in changed], 'tokenizer': report['tokenizer']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
