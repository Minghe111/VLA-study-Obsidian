"""Read-only CPU native FK reconstruction from rejected V3 17 a4.
No scene, physics, render, hardware IO or source changes. All endpoints original.
"""
from pathlib import Path
import hashlib,json,math,sys,xml.etree.ElementTree as ET
import numpy as np
import mplib
from mplib import Pose
E=Path(__file__).resolve().parent.parent
V=Path('/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/rgb50_motion_quality_20261013_v3_acquisition/jobs/slot17_runtime')
p=V/'new_physical_episodes/slot_0000017_randomized_attempt_004'
meta=json.loads((p/'free_approach_0.json').read_text())['planning_metadata']
report=json.loads((p/'expert_report.json').read_text())
URDF=E/'preflight/robot_head_down_cpu_relative_meshes.urdf'
SRDF=Path('/home/admin123/liminghe/vla-benchmark/RoboTwin/assets/embodiments/realman-rm65b-inspire-gen4/robot.srdf')
names=report['full_qpos_joint_names'];tree=ET.parse(URDF).getroot();links=[n.get('name') for n in tree.findall('link')]
planner=mplib.Planner(urdf=str(URDF),srdf=str(SRDF),move_group='l_inspire_base_link',user_joint_names=names,user_link_names=links,use_convex=False)
pk=planner.robot.get_pinocchio_model();link_names=list(pk.get_link_names());palm_index=link_names.index('l_inspire_base_link')
base=np.asarray(meta['initial_model_base_pose_matrix']);goal_world=np.asarray(meta['goal_palm_world_matrix']);goal=np.linalg.inv(base)@goal_world
canonical=np.asarray(Pose(goal).to_transformation_matrix())
def raw_angle(a,b):return math.acos(float(np.clip((np.trace(a@b.T)-1)/2,-1,1)))
def rot_measure(r):
 u,s,vt=np.linalg.svd(r);d=np.linalg.det(u@vt); rr=u@np.diag([1.,1.,d])@vt
 return {'determinant':float(np.linalg.det(r)),'orthogonality_max_error':float(np.max(abs(r.T@r-np.eye(3)))),'singular_values':s.tolist(),'polar_rotation':rr.tolist(),'projection_max_error':float(np.max(abs(rr-r))),'self_raw_trace_angle_rad':raw_angle(r,r)}
rows=[]
for attempt in meta['RRT_attempts']:
 q=np.asarray(attempt['predeclared_native_free_goals_full_rad'][0]);pk.compute_forward_kinematics(q);actual=np.asarray(pk.get_link_pose(palm_index).to_transformation_matrix())
 ar=rot_measure(actual[:3,:3]);gr=rot_measure(goal[:3,:3]);proj_a=np.asarray(ar['polar_rotation']);proj_g=np.asarray(gr['polar_rotation'])
 delta=proj_g@proj_a.T
 # atan2 is stable near 0; both matrices have been independently checked SO3.
 angle=math.atan2(float(np.linalg.norm(np.array([delta[2,1]-delta[1,2],delta[0,2]-delta[2,0],delta[1,0]-delta[0,1]])))/2,float((np.trace(delta)-1)/2))
 rows.append({'branch':attempt['branch'],'actual_native_FK_rotation':ar,'recorded_bad_check':attempt['chosen_goal_target_FK_proof'],'reconstructed_raw_trace_angle_rad':raw_angle(goal[:3,:3],actual[:3,:3]),'same_native_Pose_target_trace_angle_rad':raw_angle(canonical[:3,:3],actual[:3,:3]),'polar_angle_rad':angle,'position_error_m':float(np.linalg.norm(actual[:3,3]-goal[:3,3])),'elementwise_matrix_max_error':float(np.max(abs(actual-goal)))})
result={'schema':'actual_rejected_goal_rotation_numeric_diagnosis_v1','scope':'CPU reconstructed native 93-link/36-DOF FK of all six recorded rejected goals; no new physical episode or collision acceptance','source_files_unchanged':True,'files_sha256':{str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in [Path(__file__),URDF,SRDF,p/'free_approach_0.json',p/'expert_report.json']},'goal_world_rotation':rot_measure(goal_world[:3,:3]),'base_rotation':rot_measure(base[:3,:3]),'raw_base_goal_rotation':rot_measure(goal[:3,:3]),'native_Pose_target_rotation':rot_measure(canonical[:3,:3]),'rows':rows}
(E/'preflight/slot17_a4_fk_rotation_numeric_diagnosis_20261014.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['goal_world_rotation','base_rotation','raw_base_goal_rotation','native_Pose_target_rotation']},indent=2))
print(json.dumps([{k:r[k] for k in ['branch','reconstructed_raw_trace_angle_rad','same_native_Pose_target_trace_angle_rad','polar_angle_rad','position_error_m','elementwise_matrix_max_error']} for r in rows],indent=2))
