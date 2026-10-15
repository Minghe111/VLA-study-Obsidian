from pathlib import Path
import json,sys,math,numpy as np,sapien.core as s,xml.etree.ElementTree as ET
R=Path('/bigdata2/liminghe/VLA-benchmark/RoboTwin');sys.path.insert(0,str(R))
from integrations.realman_rm65b_gen4.controller import HandMapping
from integrations.realman_rm65b_gen4.physics_setup import configure_collisions
M=R/'assets/embodiments/realman-rm65b-inspire-gen4';manifest=json.loads((M/'source_manifest.json').read_text());ready=json.loads((Path(__file__).parent/'ready_home_elevated.json').read_text())
x=ET.parse(M/'robot_sapien.urdf')
for mesh in x.findall('.//mesh'):mesh.set('filename',str((M/mesh.get('filename')).resolve()))
f=Path(__file__).parent/'coupling_cpu_model.urdf';x.write(f)
results=[]
for stiffness in [100,1000,10000,100000]:
 scene=s.Scene();scene.set_timestep(.004);loader=scene.create_urdf_loader();loader.fix_root_link=True;robot=loader.load(str(f));configure_collisions(robot,M)
 robot.set_solver_position_iterations(40);robot.set_solver_velocity_iterations(10)
 js={j.name:j for j in robot.get_active_joints()};ix={n:i for i,n in enumerate(js)};q=np.zeros(len(js));maps={side:HandMapping(manifest['hands'][side]) for side in ['left','right']}
 for side,prefix in [('left','l'),('right','r')]:
  for i in range(6):q[ix[f'{prefix}_joint{i+1}']]=ready[side+'_joint_rad'][i]
  for n,v in maps[side].from_sdk([1000]*5+[0],dict(zip(js,q))).items():q[ix[n]]=v
 robot.set_qpos(q)
 for n,j in js.items():j.set_drive_property(0,0,0)
 for side,prefix in [('left','l'),('right','r')]:
  meta=manifest['hands'][side]
  for n,m in meta['mimic'].items():
   c=js[n];a=float(m['multiplier']);robot.create_fixed_tendon([c.parent_link.parent,c.parent_link,c.child_link],[0,-a,1],[0,-1/a,1],stiffness=stiffness,damping=.01)
  for n in meta['channels']:js[n].set_drive_property(5,.1,10)
  for i in range(6):
   n=f'{prefix}_joint{i+1}';js[n].set_drive_property(1000,200,manifest['arm_limits'][n]['effort']);js[n].set_drive_target(q[ix[n]])
 max_error=0.;valid=True
 for step in range(1000):
  val=max(0,min(1,step/500));sdk=[int(round(1000-val*600))]*4+[int(round(1000-val*500)),0]
  for side in maps:
   for n,t in maps[side].from_sdk(sdk,dict(zip(js,robot.get_qpos()))).items():
    if n in manifest['hands'][side]['channels']:js[n].set_drive_target(t)
  robot.set_qf(robot.compute_passive_force(gravity=True,coriolis_and_centrifugal=True));scene.step();current=robot.get_qpos()
  if not np.isfinite(current).all():valid=False;break
  for meta in manifest['hands'].values():
   for n,m in meta['mimic'].items():max_error=max(max_error,abs(float(current[ix[n]]-float(m['multiplier'])*current[ix[m['joint']]])))
 e={'stiffness':stiffness,'valid':valid,'last_step':step,'max_actual_mimic_residual_rad':max_error,'sdk_measured':{z:maps[z].to_sdk(dict(zip(js,current))) for z in maps} if valid else None};results.append(e);print('CPU_COUPLING',json.dumps(e),flush=True)
 scene.clear();del robot,scene
(Path(__file__).parent/'coupling_cpu_audit_20261003.json').write_text(json.dumps(results,indent=2))