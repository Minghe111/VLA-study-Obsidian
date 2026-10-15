"""Independent native CPU FK endpoint proof for rejected V3 slot17/20 goals."""
from pathlib import Path
import hashlib,json,sys,xml.etree.ElementTree as ET
import numpy as np
import mplib
E=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(E/'planning'))
import collector_quality_overlay as quality
import strict_goal_pose_roundtrip_overlay as corrected
URDF=E/'preflight/robot_head_down_cpu_relative_meshes.urdf'
SRDF=Path('/home/admin123/liminghe/vla-benchmark/RoboTwin/assets/embodiments/realman-rm65b-inspire-gen4/robot.srdf')
V=Path('/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/rgb50_motion_quality_20261013_v3_acquisition/jobs')
rows=[];files=[Path(__file__),Path(corrected.__file__),URDF,SRDF]
for index,attempt in [(17,4),(20,1)]:
 p=V/f'slot{index}_runtime/new_physical_episodes/slot_{index:07d}_randomized_attempt_{attempt:03d}'
 meta=json.loads((p/'free_approach_0.json').read_text())['planning_metadata'];report=json.loads((p/'expert_report.json').read_text())
 files.extend([p/'free_approach_0.json',p/'expert_report.json'])
 names=report['full_qpos_joint_names'];tree=ET.parse(URDF).getroot();links=[n.get('name') for n in tree.findall('link')]
 side=report['active_arm'];palm_name=('l_' if side=='left' else 'r_')+'inspire_base_link'
 planner=mplib.Planner(urdf=str(URDF),srdf=str(SRDF),move_group=palm_name,user_joint_names=names,user_link_names=links,use_convex=False)
 pk=planner.robot.get_pinocchio_model();native_links=list(pk.get_link_names());palm=native_links.index(palm_name)
 def native_fk(q):
  pk.compute_forward_kinematics(q)
  return np.asarray([pk.get_link_pose(i).to_transformation_matrix() for i in range(len(native_links))])
 base=np.asarray(meta['initial_actual_full_qpos_rad']);goal=np.linalg.inv(np.asarray(meta['initial_model_base_pose_matrix']))@np.asarray(meta['goal_palm_world_matrix'])
 for entry in meta['RRT_attempts']:
  q=np.asarray(entry['predeclared_native_free_goals_full_rad'][0])
  r=corrected.strict_native_goal_proof(quality._strict_palm_goal,q,base,meta['selected_arm_indices'],native_fk,goal,palm,len(native_links))
  assert r['new_quality_position_tolerance_m']==5e-5 and r['new_quality_rotation_tolerance_rad']==2e-4
  rows.append({'slot':index,'attempt':attempt,'arm':side,'palm_name':palm_name,'branch':entry['branch'],'original_bad_check':entry['chosen_goal_target_FK_proof'],'corrected_native_CPU_check':r,'no_physical_episode_or_collision_acceptance_asserted':True})
result={'schema':'actual_rejected_goals_roundtrip_CPU_v1','passed':all(r['corrected_native_CPU_check']['passed'] for r in rows),'native_links':93,'native_full_DOF':36,'endpoint_count':len(rows),'scope':'read-only CPU native full-model FK on original recorded endpoint qpos; no new physical episode, scene, collision or hardware acceptance','original_source_and_candidate_bytes_unmodified':True,'source_files_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'rows':rows}
(E/'preflight/actual_goals_roundtrip_CPU_receipt_20261014.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'passed':result['passed'],'endpoints':len(rows),'rows':[{'slot':r['slot'],'branch':r['branch'],'position_error_m':r['corrected_native_CPU_check']['task_palm_target_position_error_m'],'rotation_error_rad':r['corrected_native_CPU_check']['task_palm_target_rotation_error_rad']} for r in rows]},indent=2))
