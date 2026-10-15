"""Read back one accepted native RoboTwin episode, including the hand extension."""
from pathlib import Path
import sys,json,pickle,hashlib
ROOT=Path('/bigdata2/liminghe/VLA-benchmark/RoboTwin');sys.path.insert(0,str(ROOT))
import numpy as np
import h5py
from PIL import Image
from data.decode_image_bit import decode_image_bit
out=Path(sys.argv[1]);report=json.loads((out/'attempt_report.json').read_text())
assert report['result']=='one_episode_collected' and report['replay']['task_success']
files=list((out/'data').glob('*.hdf5'));assert len(files)==1 and files[0].name=='episode_0000000.hdf5'
pkls=sorted((out/'.cache/episode0').glob('*.pkl'),key=lambda p:int(p.stem))
samples=[pickle.loads(p.read_bytes()) for p in pkls];n=len(samples)-1
checks={'source_hashes_stable':bool(report['canonical_sources_unchanged']), 'native_plan_and_limit_checks':all(a['plan_success'] and a['task_success'] and a['planned_joints_legal'] for a in report['attempts'])};result={'episode_count':1,'pairs':n,'checks':checks}
with h5py.File(files[0],'r') as f:
 checks['state_action_pair_count']=n>1 and all(f[g+'/'+k].shape[0]==n for g in ['state','action'] for k in f[g])
 checks['head_only_camera']=list(f['vision'])==['cam_head']
 checks['rgb_frame_count']=f['vision/cam_head/colors'].shape[0]==n
 shapes=[];bad=[]
 for i,buf in enumerate(f['vision/cam_head/colors']):
  rgb=decode_image_bit(buf)
  if rgb.shape!=(240,320,3) or rgb.dtype!=np.uint8:bad.append(i)
  if i in [0,n//2,n-1]:Image.fromarray(rgb).save(out/f'dataset_frame_{i:05d}.png')
 checks['all_rgb_decode']=not bad
 checks['depth_finite']=np.isfinite(f['vision/cam_head/depths'][:]).all().item()
 checks['measured_state24_preserved']=bool(np.allclose(f['realman_dexterous/measured_state24'][:],np.asarray([s['dexterous_state']['state24'] for s in samples])[:-1]))
 checks['next_command24_preserved']=bool(np.allclose(f['realman_dexterous/next_command24'][:],np.asarray([s['dexterous_state']['command24'] for s in samples])[1:]))
 for side in ['left','right']:
  expected=np.asarray([s['joint_action'][side+'_arm'] for s in samples])
  checks[side+'_native_pair_alignment']=bool(np.allclose(f['state/'+side+'_arm_joint_states'][:],expected[:-1]) and np.allclose(f['action/'+side+'_arm_joint_states'][:],expected[1:]))
 times=f['realman_dexterous/sample_time_s'][:];next_times=f['realman_dexterous/next_sample_time_s'][:]
 checks['recorded_time_monotonic']=bool(np.all(np.diff(times)>=0) and np.all(next_times>=times))
 result['recorded_duration_s']=float(next_times[-1]-times[0]);result['zero_duration_pairs']=int(np.sum(np.isclose(next_times-times,0)))
 result['unique_sample_intervals_s']=np.unique(np.round(next_times-times,6)).tolist()
 instructions=json.loads(f['instructions'][()].decode('utf-8'));checks['language_nonempty']=bool(instructions and all(s.strip() and '{' not in s for s in instructions))
 result['instructions']=instructions;result['groups']=list(f)
 checks['numeric_states_finite']=all(np.isfinite(f[g+'/'+k][:]).all().item() for g in ['state','action'] for k in f[g])
 checks['extended_states_finite']=bool(np.isfinite(f['realman_dexterous/measured_state24'][:]).all())
scene=json.loads((out/'scene_info.json').read_text())['episode_0'];result['task_metrics']=scene['realman_bottle_audit'];checks['official_bottle_success_retained']=bool(result['task_metrics']['accepted'] and result['task_metrics']['immediate']['official_success'] and result['task_metrics']['after_hold']['official_success']);checks['selected_hand_load_bearing_contact']=result['task_metrics']['after_hold'].get('selected_hand_active_points',0)>0;checks['upright_and_stable_hold']=result['task_metrics']['after_hold'].get('upright_angle_deg',180)<15 and result['task_metrics'].get('hold_drift_m',1)<.02;result['passed']=all(checks.values());result['hdf5_sha256']=hashlib.sha256(files[0].read_bytes()).hexdigest()
(out/'dataset_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
assert result['passed']