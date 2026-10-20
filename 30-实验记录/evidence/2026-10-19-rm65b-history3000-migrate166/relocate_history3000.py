"""CPU-only, task-local structured-asset and compact-state relocation.

Never edits a source checkpoint. Does not train or initialize any neural model.
Use only against independently copied files inside an authorized task directory.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n')


def replace_tree(value, path_map, digest_map=None):
    digest_map=digest_map or {}
    if isinstance(value,dict): return {k:replace_tree(v,path_map,digest_map) for k,v in value.items()}
    if isinstance(value,list): return [replace_tree(v,path_map,digest_map) for v in value]
    if isinstance(value,tuple): return tuple(replace_tree(v,path_map,digest_map) for v in value)
    if isinstance(value,str):
        if value in digest_map:return digest_map[value]
        for source,target in sorted(path_map.items(),key=lambda kv:-len(kv[0])):
            if value==source or value.startswith(source+'/'):return target+value[len(source):]
    return value


def load_structured(path):
    path=Path(path)
    if path.suffix=='.jsonl':return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if path.suffix=='.json':return json.loads(path.read_text())
    if path.suffix in ('.yaml','.yml'):
        import yaml
        return yaml.safe_load(path.read_text())
    raise ValueError(f'Unsupported structured suffix: {path}')


def encode_structured(path,value):
    suffix=Path(path).suffix
    if suffix=='.jsonl':return ''.join(json.dumps(row,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n' for row in value).encode()
    if suffix=='.json':return (json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    import yaml
    return yaml.safe_dump(value,sort_keys=False,allow_unicode=True).encode()


def structured(args):
    """Map contains path_map and files[{source,target,sha256}], source needn't exist.

    Files must have been copied byte-exact to target. No file may contain its
    own hash; cycles intentionally refuse to converge. Preserve all original
    structured bytes in output/original_structured/ to make inversion auditable.
    """
    spec=json.loads(args.manifest.read_text()); paths=spec['path_map']; rows=spec['files']
    out=args.output.resolve()
    if out.exists():raise FileExistsError(out)
    for row in rows:
        p=Path(row['target']).resolve()
        if not any(p==Path(v) or Path(v) in p.parents for v in paths.values()):
            raise ValueError(f'Target outside declared task roots: {p}')
        if sha(p)!=row['sha256']:raise ValueError(f'Copied source SHA differs: {p}')
    originals={row['target']:load_structured(row['target']) for row in rows}
    if any(isinstance(value,dict) and value.get('format')=='rm65b_k1_retention_anchor_v1' for value in originals.values()):
        raise ValueError('Retention manifests are sealed by reference snapshot SHA: preserve original bytes and exclude from structured rewriting')
    digests={};encoded={};rounds=0
    for rounds in range(1,len(rows)+3):
        encoded={row['target']:encode_structured(row['target'],replace_tree(originals[row['target']],paths,digests)) for row in rows}
        updated={row['sha256']:hashlib.sha256(encoded[row['target']]).hexdigest() for row in rows}
        if updated==digests:break
        digests=updated
    else:raise ValueError('SHA dependency cycle did not converge; no target files changed')
    out.mkdir(parents=True)
    receipt=[]
    for index,row in enumerate(rows):
        p=Path(row['target']); backup=out/'original_structured'/f'{index:06d}{p.suffix}'
        backup.parent.mkdir(exist_ok=True);shutil.copy2(p,backup)
        p.write_bytes(encoded[row['target']])
        receipt.append({**row,'target_sha256':sha(p),'original_backup':str(backup)})
    result={'status':'relocated_structured_copies_only','path_map':paths,'digest_map':digests,'files':receipt,'dependency_rounds':rounds}
    write_json(out/'receipt.json',result)
    print(json.dumps({'status':result['status'],'files':len(rows),'rounds':rounds}))
    return result


def same_tree(left,right,path='root'):
    """Return exact value differences, including every optimizer/RNG tensor."""
    import torch
    if isinstance(left,torch.Tensor) or isinstance(right,torch.Tensor):
        return [] if isinstance(left,torch.Tensor) and isinstance(right,torch.Tensor) and left.dtype==right.dtype and left.shape==right.shape and torch.equal(left,right) else [path]
    if type(left)!=type(right):return [path]
    if isinstance(left,dict):
        if set(left)!=set(right):return [path+'.keys']
        return [d for k in left for d in same_tree(left[k],right[k],path+'.'+str(k))]
    if isinstance(left,(list,tuple)):
        if len(left)!=len(right):return [path+'.length']
        return [d for i,(a,b) in enumerate(zip(left,right)) for d in same_tree(a,b,path+f'[{i}]')]
    return [] if left==right else [path]


def fingerprint_with_resolver(streams,path_map):
    """Verbatim fingerprint algorithm, only NPZ byte reads resolve old paths."""
    import numpy as np
    digest=hashlib.sha256();reference_hashes={}
    def update(value,key=None):
        if isinstance(value,dict):
            digest.update(b'dict{')
            for name in sorted(value):update(str(name));update(value[name],name)
            digest.update(b'}')
        elif isinstance(value,(list,tuple)):
            digest.update(b'list[')
            for item in value:update(item)
            digest.update(b']')
        elif isinstance(value,np.ndarray):
            digest.update(b'array'+str(value.dtype).encode());digest.update(str(value.shape).encode());digest.update(np.ascontiguousarray(value).tobytes())
        elif hasattr(value,'detach') and hasattr(value,'cpu'):update(value.detach().cpu().numpy())
        elif isinstance(value,Path):update(str(value),key)
        elif isinstance(value,(str,int,float,bool,type(None))):
            digest.update(json.dumps(value,ensure_ascii=False,allow_nan=False).encode())
            if key=='npz_path':
                if str(value) not in reference_hashes:reference_hashes[str(value)]=sha(replace_tree(value,path_map))
                digest.update(reference_hashes[str(value)].encode())
        elif isinstance(value,np.generic):update(value.item(),key)
        else:raise TypeError(type(value).__name__)
    update(streams);return digest.hexdigest()


def compact(args):
    import torch
    from types import SimpleNamespace
    relocation=json.loads(args.relocation_receipt.read_text());paths=relocation['path_map'];digests=relocation['digest_map']
    inverse_paths={v:k for k,v in paths.items()};inverse_digests={v:k for k,v in digests.items()}
    src=args.source_checkpoint.resolve();dest=args.output.resolve()
    if dest.exists():raise FileExistsError(dest)
    old_version=json.loads((src/'version.json').read_text())
    for name in ('adapter_weights.pt','trainer_state.pt'):
        if sha(src/name)!=old_version['files'][name]['sha256']:raise ValueError('Source compact seal differs: '+name)
    trainpath=args.train_script.resolve()
    sys.path.insert(0,str(trainpath.parent));sys.path.insert(0,str(trainpath.parents[1]/'code/RLinf'))
    spec=importlib.util.spec_from_file_location('history3000_relocated_train',trainpath)
    train=importlib.util.module_from_spec(spec);spec.loader.exec_module(train)
    source_contract=old_version['run_contract']
    base=Path(replace_tree(old_version['base_checkpoint']['path'],paths))
    data_root=Path(replace_tree(source_contract['dataset']['acceptance_path'],paths)).parents[1]
    anchor=Path(replace_tree(source_contract['anchor']['manifest_path'],paths))
    prepared=train.prepare(SimpleNamespace(base_checkpoint=base,data_root=data_root,anchor_manifest=anchor,device='cuda:0',
        learning_rate=source_contract['learning_rate'],kv_weight=source_contract['kv_weight']))
    cfg,records,data,identity,refs,new_contract=prepared
    # Inverting paths and closed dependency digests must recover the exact old
    # source contract, not merely a numerically similar optimizer config.
    restored_contract=replace_tree(new_contract,inverse_paths,inverse_digests)
    contract_diffs=same_tree(source_contract,restored_contract)
    if contract_diffs:raise ValueError('Non-relocation contract changes: '+repr(contract_diffs))
    source_records=replace_tree(records,inverse_paths,inverse_digests)
    streams={'high':records,'actions':[],'anchor_high':[],'anchor_actions':[]}
    source_streams={'high':source_records,'actions':[],'anchor_high':[],'anchor_actions':[]}
    from rlinf.runners.hierarchical_recap_runner import ActorMicrobatchSampler,actor_data_fingerprint
    new_fingerprint=actor_data_fingerprint(streams)
    canonical_fingerprint=fingerprint_with_resolver(source_streams,paths)
    original=torch.load(src/'trainer_state.pt',map_location='cpu',weights_only=True)
    if canonical_fingerprint!=original['sampler_state']['data_fingerprint']:
        raise ValueError('Source canonical ordered-content fingerprint was not reproduced')
    changed=deepcopy(original);changed['sampler_state']['data_fingerprint']=new_fingerprint
    differences=same_tree(original,changed)
    if differences!=['root.sampler_state.data_fingerprint']:
        raise ValueError('Only sampler physical-path fingerprint may change: '+repr(differences))
    sampler=ActorMicrobatchSampler(records,[],[],[],batch_size=1,gradient_accumulation=4,anchor_interval=4,
        training_mode='verified_sft',loss_weights=cfg['actor_training']['loss_weights'],shuffle=True,seed=20261019)
    sampler.load_state_dict(changed['sampler_state'])
    if sampler.microbatch_cursor!=4*original['global_step']:raise ValueError('Saved update/data clock differs')
    # NPZ bytes and ordered samples have already been hashed; optimizer, RNG,
    # cursor/order/stream-RNG are recursively exact and never reconstructed.
    dest.mkdir(parents=True)
    shutil.copy2(src/'adapter_weights.pt',dest/'adapter_weights.pt')
    torch.save(changed,dest/'trainer_state.pt')
    readback=torch.load(dest/'trainer_state.pt',map_location='cpu',weights_only=True)
    if same_tree(changed,readback):raise ValueError('Serialized trainer state readback differs')
    new_version=deepcopy(old_version);new_version['base_checkpoint']=identity;new_version['run_contract']=new_contract
    for name in ('adapter_weights.pt','trainer_state.pt'):
        new_version['files'][name]={'bytes':(dest/name).stat().st_size,'sha256':sha(dest/name)}
    new_version.setdefault('metadata',{})['relocation']={'source_version_sha256':sha(src/'version.json'),
        'source_adapter_sha256':sha(src/'adapter_weights.pt'),'source_trainer_sha256':sha(src/'trainer_state.pt'),
        'receipt_sha256':sha(args.relocation_receipt),'source_checkpoint':str(src),'mathematical_update_contract_unchanged':True}
    write_json(dest/'version.json',new_version)
    train.validate_compact_metadata(dest,base_checkpoint=base)
    receipt={'status':'cpu_relocation_verified_no_model_or_optimizer_updates','source_checkpoint':str(src),'target_checkpoint':str(dest),
        'branch_global_step':original['global_step'],'microbatch_cursor':sampler.microbatch_cursor,'trainer_changed_fields':differences,
        'optimizer_all_tensors_and_steps_exact':True,'all_global_and_stream_rng_exact':True,'sampler_order_cursor_exact':True,
        'source_canonical_fingerprint_exact':True,'source_fingerprint':canonical_fingerprint,'target_fingerprint':new_fingerprint,
        'adapter_file_byte_exact':sha(src/'adapter_weights.pt')==sha(dest/'adapter_weights.pt'),
        'source_version_sha256':sha(src/'version.json'),'target_version_sha256':sha(dest/'version.json'),
        'relocation_receipt_sha256':sha(args.relocation_receipt),'train_script_sha256':sha(trainpath),
        'target_base_checkpoint':str(base),'target_data_root':str(data_root),'target_anchor_manifest':str(anchor),
        'cross_gpu_bitwise_reproducibility_claimed':False}
    write_json(dest/'relocation_receipt.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False))
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('structured');p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('compact');p.add_argument('--relocation-receipt',type=Path,required=True);p.add_argument('--source-checkpoint',type=Path,required=True);p.add_argument('--train-script',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();return structured(args) if args.command=='structured' else compact(args)


if __name__=='__main__':main()
