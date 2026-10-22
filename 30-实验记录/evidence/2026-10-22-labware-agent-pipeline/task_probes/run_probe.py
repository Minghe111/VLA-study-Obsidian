from pathlib import Path
import argparse
import json
import sys
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from labware_task import labware_pick_place
p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=0);p.add_argument('--name',required=True);p.add_argument('--x',type=float,default=.10);p.add_argument('--y',type=float,default=-.11);p.add_argument('--kind',default='reagent_bottle_body')
a=p.parse_args()
out=ROOT/'task_probes'/a.name
if out.exists():raise RuntimeError('Fresh evidence directory required')
t=labware_pick_place(); error=None
try:
 t.setup_demo(seed=a.seed,runtime_path=out,center_xy=[a.x,a.y],asset_kind=a.kind)
 print('RESET COMPLETE',json.dumps(t.capture()['state'].tolist()),flush=True)
 t.play_once()
except Exception as e:
 error=repr(e);traceback.print_exc()
finally:
 out.mkdir(exist_ok=True,parents=True)
 (out/'result.json').write_text(json.dumps({'error':error,'success':t.check_success(),'metrics':t.metrics,'targets':getattr(t,'take_action_cnt',0),'ticks':t.physics_step_count,'hardware_io':False},indent=2))
 (out/'physics_trace.json').write_text(json.dumps(t.physics_trace))
 (out/'transitions.json').write_text(json.dumps(t.transition_log))
 t.close()
if error:raise SystemExit(1)
