import pathlib,tarfile,sys,io,json,hashlib,socket,getpass
root=pathlib.Path('/bigdata/liminghe/VLA-benchmark/deploy/realman_gen4');assert root.resolve()==root
assert getpass.getuser()=='liminghe' and socket.gethostname()=='user-NF5468-M7-A0-R0-00'
files={};missing=[]
for d in ['rgb50_20261007_v1','adjust_bottle_20261002']:
 for p in (root/d).iterdir():
  if p.is_file() and p.suffix in ('.py','.json','.yml','.yaml') and p.stat().st_size<2_000_000:
   files[str(p.relative_to(root))]=p
man=root/'rgb50_final_20261007/dataset/manifest.json'
if not man.exists():
 candidates=list((root/'pi05_rgb50_full50_100k_b8_4gpu_161_20261007_v1/source_training_files').glob('*manifest*.json'))
 man=next((p for p in candidates if 'source_manifest' in p.name),None)
if man is None or not man.exists():
 manifest=json.load(sys.stdin)
else:manifest=json.loads(man.read_text())
for i in [17,18,20,29,34]:
 ep=manifest['episodes'][i];p=pathlib.Path(ep['source_trial_dir'].replace('/bigdata2/liminghe','/bigdata/liminghe',1))
 for n in ['sample_spec.json','scene_sample.json','recording_report.json','expert_generation_report.json','expert_report.json','run_config.json','.cache/episode0/0.pkl']:
  q=p/n
  if q.is_file():files[f'selected_scenes/episode_{i:07d}/{n}']=q
  else:missing.append(str(q))
 # Capture metadata only, no old future-action labels or trajectories.
 if p.is_dir():
  for q in p.iterdir():
   if q.is_file() and q.suffix=='.json' and q.stat().st_size<3_000_000:files[f'selected_scenes/episode_{i:07d}/{q.name}']=q
receipt={'host':socket.gethostname(),'user':getpass.getuser(),'root':str(root),'read_only':True,'files':{},'missing_requested_metadata':missing}
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as out:
 for rel,p in sorted(files.items()):
  p.resolve().relative_to(root)
  b=p.read_bytes(); receipt['files'][rel]={'source':str(p),'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
  info=tarfile.TarInfo(rel);info.size=len(b);out.addfile(info,io.BytesIO(b))
 b=json.dumps(receipt,indent=2).encode();info=tarfile.TarInfo('download_receipt.json');info.size=len(b);out.addfile(info,io.BytesIO(b))
