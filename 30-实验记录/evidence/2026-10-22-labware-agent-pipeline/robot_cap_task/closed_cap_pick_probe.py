from pathlib import Path
import argparse
import gzip
import hashlib
import json
import sys
import traceback
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT.parent))
from labware_cap_task import labware_closed_cap_pick
from camera_framing import apply_camera_framing
from PIL import Image
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--seed',type=int,default=0)
p.add_argument('--preclose-replay',type=Path,help='Replay existing actual targets, stop before close and record full measured articulation; no new grasp attempt')
a=p.parse_args();out=ROOT/'closed_cap_pick_probes'/a.name
if out.exists():raise FileExistsError(out)
task=labware_closed_cap_pick();error=None
try:
 task.setup_demo(seed=a.seed,runtime_path=out,capture_rgb=False,center_xy=[.10,-.11])
 apply_camera_framing(task)
 initial=task.observe_rgb()
 for name in ('full_image','left_wrist_image','right_wrist_image'):Image.fromarray(initial[name]).save(out/('initial_'+name+'.png'))
 if a.preclose_replay:
  source=a.preclose_replay.resolve()/'transitions.json'
  records=json.loads(source.read_text())
  replayed=0
  for record in records:
   if record['phase']=='close':break
   task.execute_command24(record['action'],record['phase']);replayed+=1
  articulation=task.robot.left_entity
  snapshot={'scope':'Actual deterministic pre-close replay, not successful grasp',
      'source_transitions':str(source),'source_transitions_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
      'replayed_targets':replayed,'physics_tick':task.physics_step_count,
      'joint_names':[joint.name for joint in articulation.get_active_joints()],
      'q36':np.asarray(articulation.get_qpos()).tolist(),
      'qlimits':np.asarray(articulation.get_qlimits()).tolist(),
      'qvel36':np.asarray(articulation.get_qvel()).tolist(),
      'root_matrix':np.asarray(articulation.root_pose.to_transformation_matrix()).tolist(),
      'right_palm_matrix':np.asarray(articulation.find_link_by_name('r_inspire_base_link').entity.pose.to_transformation_matrix()).tolist(),
      'state24':task.get_state24().tolist(),
      'body_matrix':np.asarray(task.assembly.body.pose.to_transformation_matrix()).tolist(),
      'cap_matrix':np.asarray(task.assembly.cap.pose.to_transformation_matrix()).tolist(),
      'body_pose':task.assembly.state()['body_pose'],'cap_pose':task.assembly.state()['cap_pose'],
      'metrics':task.metrics,'robot_urdf':str(out/'robot_head_down.urdf')}
  (out/'measured_preclose_q36.json').write_text(json.dumps(snapshot,indent=2)+'\n')
 else:task.play_once()
except Exception as exc:
 error={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()};traceback.print_exc()
finally:
 out.mkdir(parents=True,exist_ok=True)
 if getattr(task,'assembly',None) is not None:
  contract=task.assembly.contract()
  with gzip.open(out/'assembly_trace.json.gz','wt') as f:json.dump(task.assembly.trace,f)
  final=task.observe_rgb()
  for name in ('full_image','left_wrist_image','right_wrist_image'):Image.fromarray(final[name]).save(out/('final_'+name+'.png'))
 else:contract=None
 result={'success':error is None and task.check_success(),'error':error,'metrics':task.metrics,
     'targets':getattr(task,'take_action_cnt',0),'ticks':task.physics_step_count,
     'scope':('Pre-close diagnostic replay, not grasp success' if a.preclose_replay else 'Actual robot-drive CLOSED filled-bottle hold only; no unscrew, pour or dataset label'),
     'assembly_contract':contract,'source_sha256':{str(path):hashlib.sha256(path.read_bytes()).hexdigest()
       for path in [ROOT/'labware_cap_task.py',Path(__file__),ROOT.parent/'robot_cap_assembly.py']}}
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
 (out/'transitions.json').write_text(json.dumps(task.transition_log))
 (out/'body_physics_trace.json').write_text(json.dumps(task.physics_trace))
 (out/'assembly_observations.json').write_text(json.dumps(task.assembly_observation_trace))
 task.close()
if error:raise SystemExit(1)
print(json.dumps({'success':result['success'],'targets':result['targets'],'ticks':result['ticks']}),flush=True)
