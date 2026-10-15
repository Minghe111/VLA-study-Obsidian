"""CPU auxiliary IK readback; does not execute trajectories or validate success."""
from pathlib import Path
import sys,os,json,time
import numpy as np
WORK=Path(__file__).resolve().parent;ROOT=Path('/bigdata2/liminghe/VLA-benchmark/RoboTwin');LEGACY=WORK.parent/'adjust_bottle_20261002'
sys.path[:0]=[str(WORK),str(ROOT),str(LEGACY)];os.chdir(ROOT)
from geometry_overlay import install_native_shape_count_overlay
install_native_shape_count_overlay(LEGACY)
from benchmark_geometry_cache import cpu_snapshot_scene
import full_mesh_path_audit as fma
rows=[]
for name in ['left_coke_attempt00/slot0000002_attempt00']:
 p=WORK/'batch_candidates'/name
 task,receipt=cpu_snapshot_scene(p)
 a=fma.FullMeshPathAuditor(task);names=a.sim_joints;side='left';idx=[names.index(n) for n in task.robot.left_arm_joints_name]
 with np.load(p/'physics_telemetry.npz',allow_pickle=False) as z: q=z['full36_qpos_after_rad'][-1].copy()
 trace=json.loads((p/'physics_trace.json').read_text())[-1];fp=np.asarray(trace['bottle']['functional_point0'][:3])
 target=np.array([-.25,-.12,.95]);root=task.robot.left_entity.root_pose.to_transformation_matrix();pk=a.pk
 palm=list(pk.get_link_names()).index('l_inspire_base_link');pk.compute_forward_kinematics(q)
 goal=pk.get_link_pose(palm).to_transformation_matrix();goal[:3,3]+=root[:3,:3].T@(target-fp)
 lim=np.concatenate([np.asarray(j.limits).reshape(-1,2) for j in a.art.active_joints]);mask=np.ones(36,dtype=bool);mask[idx]=False
 rng=np.random.default_rng(20261007);solutions=[];diag=[];t=time.time()
 for n in range(24):
  seed=q.copy()
  if n:seed[idx]=rng.uniform(lim[idx,0],lim[idx,1])
  out,ok,err=pk.compute_IK_CLIK(palm,a.Pose(goal),seed,mask.tolist(),eps=1e-7,max_iter=2000,dt=.1,damp=1e-8);out=np.asarray(out)
  valid=bool(ok) and np.all(out[idx]>=lim[idx,0]-1e-5) and np.all(out[idx]<=lim[idx,1]+1e-5) and np.max(abs(out[mask]-q[mask]))<1e-8
  diag.append({'attempt':n,'converged':bool(ok),'limits_and_frozen_pass':bool(valid)})
  if valid:solutions.append(out[idx].tolist())
 row={'source_trial':str(p),'source_step':trace['step'],'actual_fp':fp.tolist(),'official_target_fp':target.tolist(),'pure_translation_world_m':(target-fp).tolist(),'IK_solutions':solutions,'attempts':diag,'elapsed_s':time.time()-t,'scope':'Endpoint reachability only; no collision/path/physics success claim','scene_steps_during_probe':0,'sapien_state_writes_during_probe':0,'snapshot':receipt}
 rows.append(row);print(json.dumps(row),flush=True)
(WORK/'upright_carry_translation_ik_results.json').write_text(json.dumps({'rows':rows},indent=2))
