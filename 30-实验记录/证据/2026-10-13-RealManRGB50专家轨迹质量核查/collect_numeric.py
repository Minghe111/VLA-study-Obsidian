import datetime,getpass,hashlib,io,json,socket,sys,zipfile
from pathlib import Path
import h5py,numpy as np

root=Path('/bigdata/liminghe/VLA-benchmark/deploy/realman_gen4/pi05_rgb50_full50_100k_b8_4gpu_161_20261007_v1/source_training_files')
assert socket.gethostname()=='user-NF5468-M7-A0-R0-00' and getpass.getuser()=='liminghe'
assert root.is_dir() and not root.is_symlink() and root.resolve()==root
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

mf=root/'source_manifest.json';manifest=json.loads(mf.read_text())
assert manifest['complete'] and manifest['count']==50
fields=[
 'realman_dexterous/measured_state24','realman_dexterous/next_command24',
 'realman_dexterous/sample_time_s','realman_dexterous/next_sample_time_s',
 'realman_dexterous_physics/applied_command24',
 'realman_dexterous_physics/full36_qpos_before_rad','realman_dexterous_physics/full36_qpos_after_rad',
 'realman_dexterous_physics/measured_state24_before','realman_dexterous_physics/measured_state24_after',
 'realman_dexterous_physics/physics_step','realman_dexterous_physics/sim_time_before_s','realman_dexterous_physics/sim_time_after_s',
 'state/left_arm_joint_states','state/right_arm_joint_states','action/left_arm_joint_states','action/right_arm_joint_states',
 'state/left_ee_poses','state/right_ee_poses','action/left_ee_poses','action/right_ee_poses',
 'vision/cam_head/extrinsics_matrix','vision/cam_left_wrist/extrinsics_matrix','vision/cam_right_wrist/extrinsics_matrix',
]
receipt={'scope':'All50 formal training original HDF exact numeric arrays only; no image bytes, no simulator or model execution',
 'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':socket.gethostname(),'user':getpass.getuser(),
 'source_root':str(root),'source_manifest_sha256':sha(mf),'source_manifest_complete':manifest['complete'],
 'field_names':fields,'joint_ordering':manifest['joint_ordering'],'physics_dt_s':manifest['physics_dt_s'],
 'observation_dt_s':manifest['observation_dt_s'],'episodes':[],'server_files_modified':False}
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',compression=zipfile.ZIP_STORED) as z:
 for e in sorted(manifest['episodes'],key=lambda e:e['index']):
  i=e['index'];assert i in range(50)
  rel=f'data/episode_{i:07d}.hdf5';p=root/rel;expected=e['published_files'][rel]
  assert p.is_file() and not p.is_symlink() and p.resolve().parent==(root/'data').resolve()
  before=p.stat();actualsha=sha(p);assert actualsha==expected['sha256'] and before.st_size==expected['bytes']
  with h5py.File(p,'r') as f:
   arrays={key:np.asarray(f[key][...]) for key in fields}
   shape_rows={key:{'shape':list(a.shape),'dtype':str(a.dtype),'array_sha256':hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()} for key,a in arrays.items()}
   assert all(np.all(np.isfinite(a)) for a in arrays.values())
  after=p.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
  out=io.BytesIO();np.savez_compressed(out,**arrays);payload=out.getvalue();name=f'numeric/episode_{i:07d}.npz';z.writestr(name,payload)
  row={k:e[k] for k in ['index','seed','arm','brand','model_id','pose_class','state_action_pairs','physical_steps','physical_success','boundary_used']}
  row.update({'source_hdf':str(p),'source_hdf_sha256':actualsha,'source_bytes':before.st_size,'source_unchanged_during_read':True,
   'numeric_file':name,'numeric_file_sha256':hashlib.sha256(payload).hexdigest(),'numeric_bytes':len(payload),'fields':shape_rows})
  receipt['episodes'].append(row)
 receipt['total_observations']=sum(e['state_action_pairs'] for e in receipt['episodes'])
 receipt['total_physics_steps']=sum(e['physical_steps'] for e in receipt['episodes'])
 receipt['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 z.writestr('numeric_receipt.json',json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
sys.stdout.buffer.write(buf.getvalue())
