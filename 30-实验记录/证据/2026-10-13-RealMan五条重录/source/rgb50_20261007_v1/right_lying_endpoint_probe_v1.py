"""CPU-only free-hand endpoint grid; no physical success or path certificate."""
from pathlib import Path
from types import SimpleNamespace
import itertools,json,os,sys,copy
import numpy as np
import yaml
WORK=Path(__file__).resolve().parent
ROOT=Path('/bigdata2/liminghe/VLA-benchmark/RoboTwin')
LEGACY=WORK.parent/'adjust_bottle_20261002'
sys.path[:0]=[str(ROOT),str(LEGACY),str(WORK)];os.chdir(ROOT)
from geometry_overlay import install_native_shape_count_overlay
install_native_shape_count_overlay(LEGACY)
from benchmark_geometry_cache import cpu_snapshot_scene
from expert_generation_v2 import ExpertSession,DEFAULT_SETTING
from envs.robot.robot import Robot
import full_mesh_path_audit as fma
from mplib import PlanningWorld

trial=WORK/'smoke_worker_right_v6/right_coke_horizontal_mirrored_default_v6'
out=WORK/'right_lying_endpoint_probe_v1_results.json'
if out.exists():raise ValueError('Output already exists')
task,receipt=cpu_snapshot_scene(trial)
cfg=yaml.safe_load((trial/'run_config.yml').read_text())['right_embodiment_config']
task.robot.right_gripper_bias=float(cfg['gripper_bias'])
task.robot.right_inv_delta_matrix=np.linalg.inv(cfg['delta_matrix'])
pose=task.bottle.actor.pose
data=json.loads((ROOT/'assets/objects/001_bottle/model_data13.json').read_text())
contact=np.asarray(data['contact_points_pose'][0],float);contact[:3,3]*=data['scale']
contact_world=pose.to_transformation_matrix()@contact
actor=SimpleNamespace(get_pose=lambda:pose,get_contact_point=lambda *_:contact_world)
native_q=task.robot.left_entity.qpos.copy();native_v=task.robot.left_entity.qvel.copy()
audit=fma.FullMeshPathAuditor(task);model=audit.model;old_q=model.get_qpos().copy()
model.set_move_group('r_inspire_base_link');pk=audit.pk
palm_index=list(pk.get_link_names()).index('r_inspire_base_link')
world=PlanningWorld([model],list(audit.static.values()))
acm=world.get_allowed_collision_matrix()
for pair in audit.allowed:
    a,b=sorted(pair);acm.set_entry(a,b,True)
indices=[audit.sim_joints.index(n) for n in task.robot.right_arm_joints_name]
mask=np.ones(36,dtype=bool);mask[indices]=False
limits=np.concatenate([np.asarray(j.limits).reshape(-1,2) for j in audit.art.active_joints])
inverse=np.linalg.inv(task.robot.left_entity.root_pose.to_transformation_matrix())
rng=np.random.default_rng(100010);rows=[]
try:
    for pitch,hand_y,hand_z,hand_x in itertools.product((10.,20.,30.,40.),(.075,.085,.095),(.155,.165,.175),(-.015,0.,.015)):
        setting=dict(DEFAULT_SETTING,pitch_deg=pitch,hand_y=hand_y,hand_z=hand_z,hand_x=hand_x)
        session=SimpleNamespace(side='right',setting=setting,report={})
        pre,goal=ExpertSession.grasp_pose(session,actor,'right',pre_dis=.1)
        target=inverse@Robot._trans_from_gripper_to_endlink(task.robot,goal,arm_tag='right').to_transformation_matrix()
        solutions=[];diagnostics=[]
        for attempt in range(6):
            seed=native_q.copy()
            if attempt:seed[indices]=rng.uniform(limits[indices,0],limits[indices,1])
            q,ok,error=pk.compute_IK_CLIK(palm_index,audit.Pose(target),seed,mask.tolist(),eps=1e-7,max_iter=2000,dt=.1,damp=1e-8)
            q=np.asarray(q);valid=bool(bool(ok) and np.all(q[indices]>=limits[indices,0]-1e-5) and np.all(q[indices]<=limits[indices,1]+1e-5))
            collision=[]
            if valid:
                model.set_qpos(q,full=True);collision=[str(c) for c in world.check_collision()]
                if not collision:solutions.append(q.tolist())
            diagnostics.append({'IK_converged':bool(ok),'limits_pass':valid,'collisions':collision})
        row={'setting':{k:setting[k] for k in ('pitch_deg','hand_y','hand_z','hand_x')},
             'pre_pose':pre,'grasp_pose':goal,'collision_free_endpoint_solutions':solutions,'attempts':diagnostics}
        rows.append(row)
        if solutions:print('FREE_ENDPOINT',json.dumps(row['setting']),len(solutions),flush=True)
    assert np.array_equal(native_q,task.robot.left_entity.qpos) and np.array_equal(native_v,task.robot.left_entity.qvel)
    result={'scope':'CPU-only endpoint IK/native collision check; no path, contact, LP, physical motion or success validation',
            'passed_execution_without_physical_mutation':True,'CPU_snapshot_receipt':receipt,'candidate_count':len(rows),
            'collision_free_candidate_count':sum(bool(r['collision_free_endpoint_solutions']) for r in rows),'rows':rows}
    out.write_text(json.dumps(result,indent=2)+'\n')
    print('SUMMARY',json.dumps({k:v for k,v in result.items() if k not in ('rows','CPU_snapshot_receipt')}),flush=True)
finally:model.set_qpos(old_q,full=True)
