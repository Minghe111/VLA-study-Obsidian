from pathlib import Path
import json,hashlib,datetime,os,time
ROOT=Path(__file__).resolve().parent
paths=json.loads((ROOT/'local_paths.json').read_text());stage=Path(paths['staging']);final=Path(paths['final'])
assert stage.resolve()==stage and not stage.is_symlink() and not final.exists()
inventory=json.loads((ROOT/'source_inventory.json').read_text());source=json.loads((ROOT/'source_sha256.json').read_text())
expected={f['path']:f for f in source['files']};first={f['path']:f for f in inventory['files']}
assert expected.keys()==first.keys() and source['total_bytes']==inventory['total_bytes']
for rel,f in expected.items():
    assert '..' not in Path(rel).parts and not Path(rel).is_absolute()
    assert f['bytes']==first[rel]['bytes'] and f['mtime_ns']==first[rel]['mtime_ns']
    if 'sha256' in first[rel]:assert f['sha256']==first[rel]['sha256']
actual={p.relative_to(stage).as_posix() for p in stage.rglob('*') if p.is_file()}
assert actual==expected.keys(),{'missing':sorted(expected.keys()-actual),'extra':sorted(actual-expected.keys())}
started=datetime.datetime.now(datetime.timezone.utc);checked=[]
for index,(rel,f) in enumerate(expected.items(),1):
    p=stage/rel;assert not p.is_symlink() and p.stat().st_size==f['bytes']
    h=hashlib.sha256()
    with p.open('rb') as fp:
        while True:
            block=fp.read(8*1024*1024)
            if not block:break
            h.update(block)
    digest=h.hexdigest();assert digest==f['sha256'],rel
    checked.append({'path':rel,'bytes':f['bytes'],'sha256':digest,'source_sha256_matched':True})
    if index%10==0:print('FILES_VERIFIED',index,len(expected),flush=True)
for p in sorted(stage.rglob('.rsync-partial'),key=lambda p:len(p.parts),reverse=True):
    assert not any(p.iterdir());p.rmdir()
actual_dirs={p.relative_to(stage).as_posix() for p in stage.rglob('*') if p.is_dir()}
assert actual_dirs==set(inventory['directories'])
meta=json.loads((stage/'_CHECKPOINT_METADATA').read_text());assert meta==inventory['checkpoint_metadata'] and meta['commit_timestamp_nsecs']
assert all((stage/p).exists() for p in ['assets','params','train_state'])
assert not final.exists();stage.rename(final)
completed=datetime.datetime.now(datetime.timezone.utc)
receipt={'status':'complete','checkpoint_step':80000,'complete_checkpoint_including_optimizer_state':True,'local_path':str(final),'source_root':source['source_root'],'source_host':source['host'],'source_user':source['user'],'file_count':len(checked),'total_bytes':sum(f['bytes'] for f in checked),'subtree_bytes':inventory['subtree_bytes'],'all_files_source_sha256_equal':True,'verification_started_utc':started.isoformat(),'completed_utc':completed.isoformat(),'completed_beijing':completed.astimezone(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'source_inventory_sha256':hashlib.sha256((ROOT/'source_inventory.json').read_bytes()).hexdigest(),'source_sha256_manifest_sha256':hashlib.sha256((ROOT/'source_sha256.json').read_bytes()).hexdigest(),'checkpoint_metadata_sha256':hashlib.sha256((final/'_CHECKPOINT_METADATA').read_bytes()).hexdigest(),'checkpoint_commit_timestamp_nsecs':meta['commit_timestamp_nsecs'],'source_files_changed_during_hashing':source['source_files_changed_during_hashing'],'originals_overwritten':False,'server_source_modified':False,'model_loaded_or_inference_run':False,'files':checked}
target=final.parent/'80000.download_receipt_20261012.json';assert not target.exists();target.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
(ROOT/'complete_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:receipt[k] for k in ['status','local_path','file_count','total_bytes','all_files_source_sha256_equal','completed_beijing']},ensure_ascii=False,indent=2),flush=True)
