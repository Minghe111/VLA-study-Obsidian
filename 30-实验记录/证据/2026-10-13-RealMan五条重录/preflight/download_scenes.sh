set -o pipefail
ssh -S /tmp/realman-rerecord-recovery-161.sock liminghe@2001:da8:9000:a837:ce29:cc6:56d1:1c3e 'python3 -c '"'"'from pathlib import Path
import tarfile,sys,io,json,hashlib
root=Path('"'"'"'"'"'"'"'"'/bigdata/liminghe/VLA-benchmark/deploy/realman_gen4/rgb50_final_20261007/dataset/episodes'"'"'"'"'"'"'"'"')
files={}
for i in [17,18,20,29,34]:
 p=root/f'"'"'"'"'"'"'"'"'episode_{i:07d}'"'"'"'"'"'"'"'"'
 for q in p.iterdir():
  if q.is_file() and q.suffix in ('"'"'"'"'"'"'"'"'.json'"'"'"'"'"'"'"'"','"'"'"'"'"'"'"'"'.yml'"'"'"'"'"'"'"'"','"'"'"'"'"'"'"'"'.yaml'"'"'"'"'"'"'"'"'):
   q.resolve().relative_to(root);files[f'"'"'"'"'"'"'"'"'episode_{i:07d}/{q.name}'"'"'"'"'"'"'"'"']=q
receipt={'"'"'"'"'"'"'"'"'root'"'"'"'"'"'"'"'"':str(root),'"'"'"'"'"'"'"'"'read_only'"'"'"'"'"'"'"'"':True,'"'"'"'"'"'"'"'"'files'"'"'"'"'"'"'"'"':{}}
with tarfile.open(fileobj=sys.stdout.buffer,mode='"'"'"'"'"'"'"'"'w|gz'"'"'"'"'"'"'"'"') as out:
 for n,p in sorted(files.items()):
  b=p.read_bytes();receipt['"'"'"'"'"'"'"'"'files'"'"'"'"'"'"'"'"'][n]={'"'"'"'"'"'"'"'"'source'"'"'"'"'"'"'"'"':str(p),'"'"'"'"'"'"'"'"'sha256'"'"'"'"'"'"'"'"':hashlib.sha256(b).hexdigest(),'"'"'"'"'"'"'"'"'bytes'"'"'"'"'"'"'"'"':len(b)}
  x=tarfile.TarInfo(n);x.size=len(b);out.addfile(x,io.BytesIO(b))
 b=json.dumps(receipt,indent=2).encode();x=tarfile.TarInfo('"'"'"'"'"'"'"'"'download_receipt.json'"'"'"'"'"'"'"'"');x.size=len(b);out.addfile(x,io.BytesIO(b))
'"'"'' > '/home/admin123/liminghe/VLA-WAM-RL/30-实验记录/证据/2026-10-13-RealMan五条重录/preflight/download_scenes.tar.gz'
