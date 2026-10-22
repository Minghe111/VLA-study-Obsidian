"""Replay exact269 targets, stop raising, then25 bounded force-feedback targets."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import copy
import gzip
import hashlib
import json
import shutil
import sys
import traceback
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT.parent))
from labware_cap_workflow import labware_cap_workflow
from labware_cap115_task import PROFILE
from robot_expert import adapted_native
from camera_framing import apply_camera_framing
from capsule_geometry import read_capsule_entities
from PIL import Image

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def snapshot(task):
 art=task.robot.left_entity
 return {'captured_at_utc':datetime.now(timezone.utc).isoformat(),'scope':'Actual finite post-lift reference; no cap/pour success',
    'physics_tick':task.physics_step_count,'assembly_physics_step':task.assembly.physics_steps,'sim_time_s':task.assembly.physics_steps*task.assembly.dt,
    'joint_names':[joint.name for joint in art.get_active_joints()],'q36':np.asarray(art.get_qpos()).tolist(),
    'qvel36':np.asarray(art.get_qvel()).tolist(),'qlimits':np.asarray(art.get_qlimits()).tolist(),
    'state24':task.get_state24().tolist(),'actual_drive_target24':adapted_native().command24(task.robot).tolist(),
    'root_matrix':np.asarray(art.root_pose.to_transformation_matrix()).tolist(),
    'right_palm_matrix':np.asarray(art.find_link_by_name('r_inspire_base_link').entity.pose.to_transformation_matrix()).tolist(),
    'left_palm_matrix':np.asarray(art.find_link_by_name('l_inspire_base_link').entity.pose.to_transformation_matrix()).tolist(),
    'body_matrix':np.asarray(task.assembly.body.pose.to_transformation_matrix()).tolist(),
    'cap_matrix':np.asarray(task.assembly.cap.pose.to_transformation_matrix()).tolist(),
    'tablet_world_pose_pq':[np.r_[e.pose.p,e.pose.q].tolist() for e in task.assembly.tablets],
    'metrics':copy.deepcopy(task.metrics)}

p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--source',type=Path,required=True)
a=p.parse_args();out=ROOT/'static_hold_probes'/a.name
out.mkdir(parents=True,exist_ok=False)
original=json.loads((a.source/'transitions.json').read_text())
if len(original)!=269:raise RuntimeError('This bounded continuation expects the exact reviewed269-target source')
sourcefiles=[Path(__file__),ROOT/'labware_cap_workflow.py',ROOT/'labware_cap115_task.py',ROOT/'labware_cap_task.py',
    ROOT/'capsule_geometry.py',ROOT.parent/'labware_task.py',ROOT.parent/'robot_expert.py',ROOT.parent/'robot_cap_assembly.py',
    *[x for x in PROFILE.iterdir() if x.is_file()]]
fingerprints={str(path):sha(path) for path in sourcefiles};dump(out/'source_manifest.json',fingerprints)
snapdir=out/'source_snapshot';snapdir.mkdir();snaps={}
for name,digest in fingerprints.items():
 dest=snapdir/(digest+'_'+Path(name).name);shutil.copyfile(name,dest);snaps[name]={'snapshot':str(dest),'sha256':digest}
dump(out/'snapshot_manifest.json',snaps)
task=labware_cap_workflow();error=None;static=[];prefix_checks=[];shadow=[];fixed=None
try:
 task.setup_demo(seed=0,runtime_path=out,capture_rgb=False,center_xy=[.10,-.11]);apply_camera_framing(task)
 native=adapted_native()
 setting={'open_sdk':[0.,0.,800.,800.,800.,0.],'close_sdk':[0.,0.,608.,610.,610.,0.],
    'pitch_deg':10.,'hand_x':-.015,'hand_y':.076,'hand_z':.177,'palm_roll_deg':5.,'approach_azimuth_deg':0.}
 feedback=native.BoundaryContactController(task,'right',setting)
 for index,row in enumerate(original):
  if not np.array_equal(task.get_state24(),np.asarray(row['state'],dtype=np.float32)):
   raise RuntimeError(f'Exact replay state mismatch at target{index}')
  if row['phase'] in ('close','lift_plan','lift','hold'):
   proposed=feedback.target(carry=row['phase'] in ('lift','hold'))
   difference=float(np.max(np.abs(proposed-np.asarray(row['action'][18:24]))))
   shadow.append({'source_action_index':index,'phase':row['phase'],'hand_sdk_max_abs_diff':difference})
   if difference>1e-8:raise RuntimeError(f'Historical feedback shadow differs at target{index}: {difference}')
  task.execute_command24(row['action'],row['phase'])
  if not np.array_equal(task.get_state24(),np.asarray(row['next_state'],dtype=np.float32)):
   raise RuntimeError(f'Exact replay next-state mismatch at target{index}')
  prefix_checks.append(True)
 dump(out/'moving_prefix_terminal_snapshot.json',snapshot(task))
 fixed=native.command24(task.robot)
 task.workflow_phase='right_static_hold'
 for index in range(25):
  target=fixed.copy();target[18:24]=feedback.target(carry=True)
  task.execute_command24(target,'right_fixed_arm_same_grip_feedback')
  snap=snapshot(task)
  snap['capsule_geometry']=read_capsule_entities(task.assembly.tablets,task.assembly.body.pose,
      half_length_m=.002,capsule_radius_m=.003,inner_radius_m=.021,floor_z_m=.003,rim_z_m=.115,tolerance_m=.0004)
  static.append(snap)
  print(json.dumps({'static_target':index+1,'speed_m_s':task.metrics['speed_m_s'],
      'cap_force_N':task.metrics['cap_robot_contact_N'],'cap_closed':task.metrics['cap_closed_observed'],
      'body_gap_m':task.metrics['bottom_gap_m'],'full_tablets_inside':snap['capsule_geometry']['full_geometry_inside_count']}),flush=True)
except Exception as exc:
 error={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()};traceback.print_exc()
finally:
 if getattr(task,'assembly',None) is not None:
  terminal=snapshot(task);terminal['source_sha256']=fingerprints;dump(out/'actual_terminal_snapshot.json',terminal)
  dump(out/'assembly_contract.json',task.assembly.contract())
  final=task.observe_rgb()
  for name in ('full_image','left_wrist_image','right_wrist_image'):Image.fromarray(final[name]).save(out/('final_'+name+'.png'))
  for name,value in [('assembly_trace.json.gz',task.assembly.trace),('world_kinematics_pre_post.json.gz',task.assembly.kinematics_trace)]:
   with gzip.open(out/name,'wt') as stream:json.dump(value,stream)
 if static:
  transforms=[np.linalg.inv(np.asarray(x['body_matrix']))@np.asarray(x['right_palm_matrix']) for x in static]
  tail=transforms[-10:];origin=tail[0]
  drift=max(float(np.linalg.norm(t[:3,3]-origin[:3,3])) for t in tail)
  rotation=max(float(np.degrees(np.arccos(np.clip((np.trace(origin[:3,:3].T@t[:3,:3])-1)/2,-1,1)))) for t in tail)
 else:drift=rotation=None
 result={'scope':'Finite1.5s static-reference continuation; full unscrew/pour goal remains incomplete','error':error,
    'full_task_success':False,'targets':len(task.transition_log),'static_targets':len(static),
    'replay_source':str(a.source.resolve()),'source_trajectory_sha256':sha(a.source/'transitions.json'),
    'prefix269_action_state_exact':len(prefix_checks)==269 and all(prefix_checks),
    'shadow_feedback_history_max_abs_difference':max((r['hand_sdk_max_abs_diff'] for r in shadow),default=None),
    'right_arm_drive_held_constant':fixed is not None and all(np.array_equal(np.asarray(row['action'][12:18]),fixed[12:18]) for row in task.transition_log[269:]),
    'last060s_body_palm_translation_range_m':drift,'last060s_body_palm_rotation_range_deg':rotation,
    'final_metrics':task.metrics,'sources_unchanged_during_probe':all(sha(path)==digest for path,digest in fingerprints.items())}
 dump(out/'result.json',result);dump(out/'static_reference_observations.json',static);dump(out/'feedback_shadow_checks.json',shadow)
 dump(out/'transitions.json',task.transition_log);dump(out/'measured_full_state_trace.json',task.full_state_trace)
 dump(out/'body_physics_trace.json',task.physics_trace);dump(out/'assembly_observations.json',task.assembly_observation_trace)
 task.close()
if error:raise SystemExit(1)
print(json.dumps({'static_targets':len(static),'full_task_success':False,'body_palm_last060s_m':drift}),flush=True)
