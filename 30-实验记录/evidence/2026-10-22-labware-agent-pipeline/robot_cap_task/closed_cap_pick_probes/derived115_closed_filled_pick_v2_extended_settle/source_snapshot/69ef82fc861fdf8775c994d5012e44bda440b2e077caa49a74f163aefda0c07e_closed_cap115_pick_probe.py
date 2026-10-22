"""One bounded actual local simulation for the independently reviewed115mm profile."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import shutil
import sys
import traceback
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT.parent))
from labware_cap115_task import labware_closed_cap115_pick,PROFILE
from camera_framing import apply_camera_framing
from PIL import Image

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--seed',type=int,default=0)
a=p.parse_args();out=ROOT/'closed_cap_pick_probes'/a.name
if out.exists():raise FileExistsError(out)
out.mkdir(parents=True)
sources=[Path(__file__),ROOT/'labware_cap115_task.py',ROOT/'labware_cap_task.py',ROOT.parent/'labware_task.py',
    ROOT.parent/'robot_expert.py',ROOT.parent/'robot_cap_assembly.py',*PROFILE.iterdir()]
fingerprints={str(path):sha(path) for path in sources if path.is_file()}
dump(out/'source_manifest.json',fingerprints)
snapshots=out/'source_snapshot';snapshots.mkdir()
snapshot_manifest={}
for name,digest in fingerprints.items():
 source=Path(name);target=snapshots/(digest+'_'+source.name);shutil.copyfile(source,target)
 snapshot_manifest[name]={'snapshot':str(target),'sha256':digest}
dump(out/'snapshot_manifest.json',snapshot_manifest)
task=labware_closed_cap115_pick();error=None
try:
 task.setup_demo(seed=a.seed,runtime_path=out,capture_rgb=False,center_xy=[.10,-.11])
 apply_camera_framing(task)
 initial=task.observe_rgb()
 for name in ('full_image','left_wrist_image','right_wrist_image'):Image.fromarray(initial[name]).save(out/('initial_'+name+'.png'))
 dump(out/'initial_assembly_contract.json',task.assembly.contract())
 task.play_once()
except Exception as exc:
 error={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()};traceback.print_exc()
finally:
 if getattr(task,'assembly',None) is not None:
  contract=task.assembly.contract()
  for name,value in [('assembly_trace.json.gz',task.assembly.trace),('world_kinematics_pre_post.json.gz',task.assembly.kinematics_trace)]:
   with gzip.open(out/name,'wt') as stream:json.dump(value,stream)
  final=task.observe_rgb()
  for name in ('full_image','left_wrist_image','right_wrist_image'):Image.fromarray(final[name]).save(out/('final_'+name+'.png'))
 else:contract=None
 result={'success':error is None and task.check_success(),'error':error,'metrics':task.metrics,
    'targets':getattr(task,'take_action_cnt',0),'ticks':task.physics_step_count,'early_stop':task.early_stop,
    'scope':'Independent115mm derived bottle actual body grasp/closed12tablet hold only; no unscrew/pour/dataset success',
    'assembly_contract':contract,'source_sha256':fingerprints,
    'sources_unchanged_during_probe':all(sha(path)==digest for path,digest in fingerprints.items())}
 dump(out/'result.json',result)
 for name,value in [('transitions.json',task.transition_log),('body_physics_trace.json',task.physics_trace),
       ('assembly_observations.json',task.assembly_observation_trace),('measured_full_state_trace.json',task.full_state_trace)]:dump(out/name,value)
 if getattr(task,'teacher',None) is not None:task.teacher.write_report()
 task.close()
if error:raise SystemExit(1)
print(json.dumps({'success':result['success'],'targets':result['targets'],'ticks':result['ticks']}),flush=True)
