"""Actual CPU right-arm factory/FK/OMPL proof; no SAPIEN or physical success.

The fixture is the existing frozen full robot, original collision meshes and the
new slot17 actual initial measured full q. Right J6 is full-q index 10; left and
hand states remain frozen. No environment, helper or production file is edited.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib, json, math, time, xml.etree.ElementTree as ET
import numpy as np
import mplib
from mplib import PlanningWorld, Pose
from mplib.planning.ompl import OMPLPlanner, FixedJoint
from auxiliary_joint_window import create_model, contains_arm_path

ROOT=Path(__file__).resolve().parent.parent
URDF=ROOT/'preflight/robot_head_down_cpu_relative_meshes.urdf'
SRDF=Path('/home/admin123/liminghe/vla-benchmark/RoboTwin/assets/embodiments/realman-rm65b-inspire-gen4/robot.srdf')
ACTUAL=Path('/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/rgb50_motion_quality_20261013_v1/new_physical_episodes/slot_0000017_randomized_attempt_000')
INPUT=ACTUAL/'free_approach_0.json'; REPORT=ACTUAL/'expert_report.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
input_hashes={str(p):sha(p) for p in [URDF,SRDF,INPUT,REPORT,ROOT/'planning/auxiliary_joint_window.py',ROOT/'planning/auxiliary_window_overlay.py']}
meta=json.loads(INPUT.read_text())['planning_metadata']
names=json.loads(REPORT.read_text())['full_qpos_joint_names']
base=np.asarray(meta['initial_actual_full_qpos_rad'],dtype=float)
right_names=['r_joint'+str(i) for i in range(1,7)]
indices=[names.index(n) for n in right_names]
assert len(names)==36 and base.shape==(36,) and indices==[0,2,4,6,8,10]
assert abs(base[10] - (-.930358350276947))<1e-12
tree=ET.parse(URDF).getroot();links=[n.get('name') for n in tree.findall('link')]
assert len(links)==93
source=mplib.Planner(urdf=URDF,srdf=SRDF,move_group='r_inspire_base_link',
    user_joint_names=names,user_link_names=links,use_convex=False)
model=source.robot;model.set_base_pose(Pose());model.set_qpos(base,full=True)
assert list(model.get_move_group_joint_indices())==indices
original_limits=np.concatenate(model.get_pinocchio_model().get_joint_limits()).copy()
original_collision_poses={o.name:np.asarray(o.pose.to_transformation_matrix()).copy()
                          for o in model.get_fcl_model().get_collision_objects()}
for link in tree.findall('link'):
    for child in list(link):
        if child.tag in ('visual','collision'):link.remove(child)
bare=ET.tostring(tree,encoding='unicode')
allowed={frozenset([n.get('link1'),n.get('link2')]) for n in ET.parse(SRDF).getroot().findall('disable_collisions')}
auditor=SimpleNamespace(model=model,pk=model.get_pinocchio_model(),urdf=bare,urdf_tree=tree,
    initial_sim_q=base.copy(),sim_joints=names,links={n:None for n in links},allowed=allowed,
    task=SimpleNamespace(robot=SimpleNamespace(right_arm_joints_name=right_names)))
bounded,receipt=create_model(auditor,'right')
assert list(bounded.get_move_group_joint_indices())==indices
assert list(bounded.get_move_group_joint_names())==right_names
bounded_limits=np.concatenate(bounded.get_pinocchio_model().get_joint_limits())
frozen=np.ones(36,dtype=bool);frozen[10]=False
assert np.array_equal(bounded_limits[frozen],original_limits[frozen])
assert receipt['selected_joint']=='r_joint6' and receipt['selected_full_q_index']==10
assert receipt['native_FK_checked_link_count']==93 and receipt['auxiliary_width_rad']<2*math.pi
goal=base[indices].copy();goal[-1]+=0.10
world=PlanningWorld([bounded],[])
start_collisions=world.check_collision()
target=base.copy();target[indices]=goal
bounded.set_qpos(target,full=True);world.set_qpos_all(goal)
goal_collisions=world.check_collision()
bounded.set_qpos(base,full=True);world.set_qpos_all(base[indices])
assert not start_collisions and not goal_collisions
fixed={FixedJoint(0,i,float(base[index])) for i,index in enumerate(indices[:-1])}
stamp=time.monotonic()
status,raw=OMPLPlanner(world).plan(base[indices],[goal],time=1.,range=.05,
                                  fixed_joints=fixed,simplify=True,verbose=False)
elapsed=time.monotonic()-stamp;path=np.asarray(raw,dtype=float)
assert str(status).rstrip('.')=='Exact solution' and path.ndim==2 and path.shape[1]==6 and len(path)>=2
assert np.max(abs(path[0]-base[indices]))<=1e-7 and np.max(abs(path[-1]-goal))<=1e-6
assert np.max(abs(path[:,:-1]-base[indices][:-1]))<=1e-8
assert contains_arm_path(bounded,path,indices)
# Compare every original FK link and FCL collision-link pose at the new right goal.
source_pk=model.get_pinocchio_model();window_pk=bounded.get_pinocchio_model()
source_pk.compute_forward_kinematics(target);window_pk.compute_forward_kinematics(target)
fk_error=max(float(abs(source_pk.get_link_pose(i).to_transformation_matrix()-
                       window_pk.get_link_pose(i).to_transformation_matrix()).max()) for i in range(93))
assert fk_error<=1e-8
assert np.array_equal(model.get_qpos(),base)
assert np.array_equal(np.concatenate(model.get_pinocchio_model().get_joint_limits()),original_limits)
assert all(np.array_equal(o.pose.to_transformation_matrix(),original_collision_poses[o.name])
           for o in model.get_fcl_model().get_collision_objects())
assert all(sha(p)==expected for p,expected in input_hashes.items())
proof={'date':'2026-10-14','passed':True,
    'scope':'Real CPU MPLib full-model right factory/FK/OMPL on frozen URDF and original meshes, synthetic short right goal from actual initial measured q; no randomized world/physical success asserted',
    'full_DOF':36,'FK_links':93,'right_arm_full_indices':indices,'right_joint_names':right_names,
    'actual_right_J6_start_rad':float(base[10]),'short_right_J6_goal_rad':float(goal[-1]),
    'window_factory':receipt,'left_and_other_frozen_limits_bitwise_unchanged':True,
    'source_model_qpos_limits_and_collision_poses_bitwise_unchanged':True,
    'full_goal_FK_matrix_max_error':fk_error,
    'OMPL':{'status':str(status),'elapsed_s':elapsed,'path_rows':len(path),'path':path.tolist(),
        'raw_start_error_rad':float(abs(path[0]-base[indices]).max()),'raw_goal_error_rad':float(abs(path[-1]-goal).max()),
        'other_five_arm_joints_error_rad':float(abs(path[:,:-1]-base[indices][:-1]).max()),
        'J6_total_variation_rad':float(abs(np.diff(path[:,-1])).sum()),'all_path_rows_within_auxiliary_window':True,
        'start_collision_count':0,'goal_collision_count':0,'returned_path_modified':False},
    'original_inputs_and_frozen_helpers_unchanged':True,'original_input_sha256':input_hashes,
    'test_sha256':sha(__file__),'GPU_started':False,'SAPIEN_scene_created':False,'physics_steps':0,
    'actual_vendor_URDF_FMA_physical_limits_or_active_worker_modified':False}
(ROOT/'planning/auxiliary_window_right_CPU_receipt_20261014.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps({'passed':True,'full_DOF':36,'FK_links':93,'right_indices':indices,
    'actual_right_J6':float(base[10]),'goal_right_J6':float(goal[-1]),'full_goal_FK_error':fk_error,
    'window':receipt['auxiliary_window_rad'],'OMPL_elapsed_s':elapsed,
    'raw_goal_error_rad':proof['OMPL']['raw_goal_error_rad'],'source_models_and_helpers_unchanged':True},indent=2))
