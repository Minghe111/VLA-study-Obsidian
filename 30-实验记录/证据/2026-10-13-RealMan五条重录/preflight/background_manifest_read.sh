ssh -S /tmp/realman-rerecord-recovery-161.sock liminghe@2001:da8:9000:a837:ce29:cc6:56d1:1c3e 'python3 -c '"'"'import pathlib,hashlib,json,socket,getpass
r=pathlib.Path('"'"'"'"'"'"'"'"'/bigdata/liminghe/VLA-benchmark/RoboTwin/assets/background_texture/seen'"'"'"'"'"'"'"'"');assert getpass.getuser()=='"'"'"'"'"'"'"'"'liminghe'"'"'"'"'"'"'"'"' and socket.gethostname()=='"'"'"'"'"'"'"'"'user-NF5468-M7-A0-R0-00'"'"'"'"'"'"'"'"';r.resolve().relative_to(pathlib.Path('"'"'"'"'"'"'"'"'/bigdata/liminghe'"'"'"'"'"'"'"'"'))
files={}
for p in sorted(r.iterdir()):
 if p.is_file():files[p.name]={'"'"'"'"'"'"'"'"'sha256'"'"'"'"'"'"'"'"':hashlib.sha256(p.read_bytes()).hexdigest(),'"'"'"'"'"'"'"'"'bytes'"'"'"'"'"'"'"'"':p.stat().st_size}
assert len(files)==10000
print(json.dumps({'"'"'"'"'"'"'"'"'server'"'"'"'"'"'"'"'"':socket.gethostname(),'"'"'"'"'"'"'"'"'user'"'"'"'"'"'"'"'"':getpass.getuser(),'"'"'"'"'"'"'"'"'source'"'"'"'"'"'"'"'"':str(r),'"'"'"'"'"'"'"'"'files'"'"'"'"'"'"'"'"':files,'"'"'"'"'"'"'"'"'read_only'"'"'"'"'"'"'"'"':True}))
'"'"'' > '/home/admin123/liminghe/VLA-WAM-RL/30-实验记录/证据/2026-10-13-RealMan五条重录/preflight/background_source_sha_manifest_20261014.json'
