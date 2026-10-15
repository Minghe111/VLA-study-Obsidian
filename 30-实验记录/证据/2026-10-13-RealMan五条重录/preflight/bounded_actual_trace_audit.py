"""Bounded raw-trace integrity benchmark; never fabricates native geometry.
Actual v13/v14 geometry runs in the LIVE collector via bounded_trace_overlay.
"""
import argparse,json,resource,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'planning'))
from bounded_trace_storage import LazyJSONList,stream_sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--trace-only',action='store_true',required=True);a=p.parse_args()
 if a.output.exists():raise RuntimeError('Refusing prior proof overwrite')
 tracepath=a.candidate/'physics_trace.json';before=stream_sha(tracepath);started=time.monotonic();trace=LazyJSONList(tracepath);last=0
 for row in trace:
  if type(row['step']) is not int or row['step']!=last+1:raise ValueError('Missing actual physical step')
  q=np.asarray(row['full_qpos_actual_rad'],float)
  if q.shape!=(36,) or not np.isfinite(q).all():raise ValueError('Invalid actual36 state')
  for side,idx in [('left',[1,3,5,7,9,11]),('right',[0,2,4,6,8,10])]:
   if not np.allclose(q[idx],row['arm_qpos'][side],atol=1e-6,rtol=0):raise ValueError('Recorded arm/full36 inconsistency')
  if last and abs(float(row['sim_time_s'])-lasttime-.004)>1e-8:raise ValueError('Actual 4ms clock gap')
  last=row['step'];lasttime=float(row['sim_time_s'])
 if before!=stream_sha(tracepath):raise ValueError('Raw trace bytes changed')
 result={'passed':True,'component_only':True,'accepted':False,'scope':'Every original actual36/arm/time row read bounded; no FCL geometry or episode acceptance claimed','rows':last,'trace_sha256':before,'trace_bytes':tracepath.stat().st_size,'elapsed_s':time.monotonic()-started,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
