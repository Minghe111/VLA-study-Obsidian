from pathlib import Path
import json,socket,subprocess,os,datetime,hashlib,stat
ROOT=Path('/bigdata/liminghe/VLA-benchmark/deploy/realman_gen4/pi05_rgb50_full50_100k_b8_4gpu_161_20261007_v1')
CHECKPOINT=ROOT/'checkpoints/pi05_realman_rgb50_all50_full_v1/full50_100k_b64_fallback32_save20k_20261008_v4_bs64_formal100k/80000'
assert socket.gethostname()=='user-NF5468-M7-A0-R0-00'
assert subprocess.check_output(['id','-un'],text=True).strip()=='liminghe'
assert CHECKPOINT.resolve()==CHECKPOINT and not CHECKPOINT.is_symlink()
meta=json.loads((CHECKPOINT/'_CHECKPOINT_METADATA').read_text());assert meta['commit_timestamp_nsecs']
files=[];symlinks=[];dirs=[]
for p in sorted(CHECKPOINT.rglob('*')):
    rel=p.relative_to(CHECKPOINT).as_posix()
    if p.is_symlink():symlinks.append(rel);continue
    if p.is_dir():dirs.append(rel);continue
    st=p.stat();assert stat.S_ISREG(st.st_mode)
    row={'path':rel,'bytes':st.st_size,'mtime_ns':st.st_mtime_ns}
    if st.st_size<=1048576:row['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
    files.append(row)
totals={}
for f in files:
    k=f['path'].split('/')[0];totals[k]=totals.get(k,0)+f['bytes']
assert not symlinks,symlinks
print(json.dumps({'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':socket.gethostname(),'user':'liminghe','source_root':str(CHECKPOINT),'source_root_realpath':str(CHECKPOINT.resolve()),'source_root_is_symlink':False,'checkpoint_metadata':meta,'checkpoint_metadata_sha256':hashlib.sha256((CHECKPOINT/'_CHECKPOINT_METADATA').read_bytes()).hexdigest(),'file_count':len(files),'total_bytes':sum(f['bytes'] for f in files),'subtree_bytes':totals,'directories':dirs,'symlinks':symlinks,'files':files,'rsync_path':subprocess.check_output(['which','rsync'],text=True).strip(),'scope':'Committed complete checkpoint inventory; metadata-only inspection; source untouched'},ensure_ascii=False))
