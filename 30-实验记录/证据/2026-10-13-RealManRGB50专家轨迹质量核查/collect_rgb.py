import datetime,getpass,hashlib,io,json,socket,sys,zipfile
from pathlib import Path
import h5py,numpy as np
indices=[int(s) for s in sys.argv[1:]]
assert indices and all(0<=i<50 for i in indices)
root=Path('/bigdata/liminghe/VLA-benchmark/deploy/realman_gen4/pi05_rgb50_full50_100k_b8_4gpu_161_20261007_v1/source_training_files')
assert socket.gethostname()=='user-NF5468-M7-A0-R0-00' and getpass.getuser()=='liminghe' and not root.is_symlink()
manifest=json.loads((root/'source_manifest.json').read_text());receipt={'scope':'Exact original HDF stored three-camera JPEG bytes, no rerender or model inference','checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'episodes':[],'server_modified':False}
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',compression=zipfile.ZIP_STORED) as z:
 for i in indices:
  e=next(e for e in manifest['episodes'] if e['index']==i);p=root/f'data/episode_{i:07d}.hdf5';h=hashlib.sha256()
  st=p.stat()
  with p.open('rb') as f:
   for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
  expected=e['published_files'][f'data/episode_{i:07d}.hdf5']['sha256'];assert h.hexdigest()==expected
  files=[]
  with h5py.File(p,'r') as f:
   times=np.asarray(f['realman_dexterous/sample_time_s'][...]).tolist()
   for camera in ['cam_head','cam_left_wrist','cam_right_wrist']:
    for n,data in enumerate(f[f'vision/{camera}/colors'][...]):
     data=bytes(data) if not isinstance(data,np.ndarray) else data.tobytes()
     assert data.startswith(b'\xff\xd8') and data.endswith(b'\xff\xd9')
     name=f'rgb/episode_{i:07d}/{camera}/{n:04d}.jpg';z.writestr(name,data);files.append({'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
  after=p.stat();assert (st.st_size,st.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
  receipt['episodes'].append({'index':i,'source_hdf':str(p),'source_sha256':expected,'source_unchanged':True,'frames':len(times),'times':times,'files':files})
 z.writestr('rgb_receipt.json',json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
sys.stdout.buffer.write(buf.getvalue())
