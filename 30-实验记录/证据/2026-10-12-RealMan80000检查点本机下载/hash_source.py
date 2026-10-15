from pathlib import Path
import json,socket,subprocess,datetime,hashlib,os
ROOT=Path('/bigdata/liminghe/VLA-benchmark/deploy/realman_gen4/pi05_rgb50_full50_100k_b8_4gpu_161_20261007_v1/checkpoints/pi05_realman_rgb50_all50_full_v1/full50_100k_b64_fallback32_save20k_20261008_v4_bs64_formal100k/80000')
assert ROOT.resolve()==ROOT and socket.gethostname()=='user-NF5468-M7-A0-R0-00'
assert subprocess.check_output(['id','-un'],text=True).strip()=='liminghe'
os.nice(10)
start=datetime.datetime.now(datetime.timezone.utc);files=[]
for p in sorted(ROOT.rglob('*')):
    assert not p.is_symlink()
    if not p.is_file():continue
    before=p.stat();h=hashlib.sha256()
    with p.open('rb') as f:
        while True:
            block=f.read(8*1024*1024)
            if not block:break
            h.update(block)
    after=p.stat();assert before.st_size==after.st_size and before.st_mtime_ns==after.st_mtime_ns
    files.append({'path':p.relative_to(ROOT).as_posix(),'bytes':after.st_size,'mtime_ns':after.st_mtime_ns,'sha256':h.hexdigest()})
print(json.dumps({'source_root':str(ROOT),'host':socket.gethostname(),'user':'liminghe','started_utc':start.isoformat(),'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'file_count':len(files),'total_bytes':sum(f['bytes'] for f in files),'files':files,'source_files_changed_during_hashing':False,'algorithm':'SHA256','hardware_io':False,'source_modified':False},ensure_ascii=False))
