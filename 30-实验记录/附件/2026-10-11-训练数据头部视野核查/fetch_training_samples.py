import subprocess
from pathlib import Path

ROOT = Path('/tmp/realman-training-camera-audit-20261011')
SCRIPT = r'''
import io,json,pathlib,h5py,numpy as np,hashlib,tarfile,sys,socket,getpass,datetime
p=pathlib.Path('/bigdata/liminghe/VLA-benchmark/deploy/realman_gen4/pi05_rgb50_full50_100k_b8_4gpu_161_20261007_v1')
s=p/'source_training_files'
raw=(s/'source_manifest.json').read_bytes();m=json.loads(raw)
idx_raw=(p/'data50_v1/native_index.json').read_bytes();idx=json.loads(idx_raw)
assert hashlib.sha256(raw).hexdigest()==idx['source_manifest_sha256']
assert len(m['episodes'])==50 and len(idx['episodes'])==50
reps={}
for e in m['episodes']:
 reps.setdefault((e['pose_class'],e['arm']),e['index'])
selected=set(reps.values())
meta={'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':socket.gethostname(),'user':getpass.getuser(),'source_root':str(s),'source_manifest_sha256':hashlib.sha256(raw).hexdigest(),'native_index_sha256':hashlib.sha256(idx_raw).hexdigest(),'actual_counts':m['actual_counts'],'representatives':{'_'.join(k):v for k,v in reps.items()},'episodes':[],'images':[],'head_extrinsics_semantics':'OpenGL camera model matrix, forward=-column2; differs from evaluation NPZ entity pose'}
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as tar:
 def add(name,blob):
  t=tarfile.TarInfo(name);t.size=len(blob);tar.addfile(t,io.BytesIO(blob))
 add('source_manifest.json',raw)
 add('native_index.json',idx_raw)
 for e in m['episodes']:
  i=e['index'];f=s/'data'/f'episode_{i:07d}.hdf5'
  with h5py.File(f,'r') as h:
   mats=h['vision/cam_head/extrinsics_matrix'][:]
   ks=h['vision/cam_head/intrinsic_matrix'][:]
   n=len(mats);forward=-mats[:,:3,2]
   pitch=np.degrees(np.arctan2(-forward[:,2],np.linalg.norm(forward[:,:2],axis=1)))
   row={k:e[k] for k in ['index','model_id','brand','arm','pose_class','seed','state_action_pairs']}
   row.update({'hdf_relative':str(f.relative_to(s)),'hdf_bytes':f.stat().st_size,'expected_hdf_sha256':idx['episodes'][i]['hdf5_sha256'],'head_frame_count':n,'pitch_deg_min':float(pitch.min()),'pitch_deg_max':float(pitch.max()),'extrinsics_first':mats[0].tolist(),'extrinsics_max_abs_change':float(np.max(np.abs(mats-mats[0]))),'intrinsics_first':ks[0].tolist(),'intrinsics_max_abs_change':float(np.max(np.abs(ks-ks[0]))),'image_shape':h['vision/cam_head/shape'][:].tolist(),'source_trial_dir':e['source_trial_dir']})
   meta['episodes'].append(row)
   requests=[('cam_head',0)]
   if i in selected:
    requests.extend((cam,j) for cam in ['cam_head','cam_left_wrist','cam_right_wrist'] for j in sorted(set([0,n//3,2*n//3,n-1])) if (cam,j)!=('cam_head',0))
   for cam,j in requests:
    blob=bytes(h['vision/'+cam+'/colors'][j]);assert blob[:2]==b'\xff\xd8'
    name=f'images/ep{i:02d}_{cam}_{j:04d}.jpg'
    add(name,blob)
    meta['images'].append({'name':name,'episode_index':i,'camera':cam,'frame_index':j,'sample_time_s':float(h['realman_dexterous/sample_time_s'][j]),'bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest()})
 # This is an inventory read, not a copy or write on the server.
 evalroot=p/'evaluation/ckpt40000_20261010_v1'
 meta['evaluation_comparison_candidates']=[str(f.relative_to(evalroot)) for f in evalroot.rglob('*.json') if any(k in f.name for k in ['initial','train','compare'])][:30] if evalroot.exists() else []
 add('training_camera_readback.json',json.dumps(meta,ensure_ascii=False,indent=2).encode())
'''

cmd=['ssh','-6','-S',str(ROOT/'ssh.sock'),'-o','ControlMaster=no','-o','HostKeyAlias=172.17.27.161','liminghe@2001:da8:9000:a837:ce29:cc6:56d1:1c3e','/bigdata/liminghe/envs/robotwin/bin/python','-B','-']
with (ROOT/'training_samples.tar').open('wb') as output:
    r=subprocess.run(cmd,input=SCRIPT.encode(),stdout=output,stderr=subprocess.PIPE,timeout=55)
if r.returncode:
    raise RuntimeError(r.stderr.decode())
print('sample archive bytes:',(ROOT/'training_samples.tar').stat().st_size)
