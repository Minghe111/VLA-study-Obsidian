import json
import sentencepiece as spm
from pathlib import Path
from collections import Counter

root=Path('/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/recap_memory_20261017/experiments/protocol_fit_20261019_v1')
sp=spm.SentencePieceProcessor(model_file='/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/recap_memory_20261017/cache/openpi/big_vision/paligemma_tokenizer.model')
sequence=json.loads((root/'reports/comparison_v1/summary.json').read_text())
result={'sequence_status':sequence['status'],'target_eos_id':sp.eos_id(),'arms':{}}
for size in (9,96):
    rows=[json.loads(line) for line in (root/f'data/fit_subsets_v1/fit{size}.train.jsonl').read_text().splitlines()]
    lengths={r['sample_id']:len(sp.encode(r['target_text'],add_eos=True)) for r in rows}
    for stage in ('protocol','language_warmup'):
        trace=[]
        for entry in sequence['steps']:
            if entry['status']=='completed' and entry['name'].startswith(f'{stage}_{size}_'):
                trace+=json.loads((root/'reports/comparison_v1'/entry['name']/'sample_trace.json').read_text())
        counts=Counter(row['sample_id'] for row in trace)
        result['arms'][f'{stage}_{size}']={'optimizer_updates':len(trace)//4,
            'sample_presentations':len(trace),'supervised_tokens_including_eos':sum(lengths[row['sample_id']] for row in trace),
            'eos_supervision_positions':len(trace),'distinct_training_rows_seen':len(counts),
            'per_sample_presentations_min_max':[min(counts.values()),max(counts.values())] if counts else None,
            'row_epoch_equivalent':len(trace)/size}
Path('/tmp/fit_consumed_token_counts.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
