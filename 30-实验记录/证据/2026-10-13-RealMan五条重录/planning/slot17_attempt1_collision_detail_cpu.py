"""Read-only artifact/FK analysis; no new planning or physical execution."""
from pathlib import Path
import json,hashlib,xml.etree.ElementTree as ET
import numpy as np
import mplib

ROOT=Path(__file__).resolve().parent.parent
ACTUAL=Path('/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/rgb50_motion_quality_20261013_v1/new_physical_episodes/slot_0000017_randomized_attempt_001')
INPUT=ACTUAL/'free_approach_0.json'
d=json.loads(INPUT.read_text());m=d['planning_metadata']
report=json.loads((ACTUAL/'expert_report.json').read_text())
names=report['full_qpos_joint_names'];base=np.asarray(m['initial_actual_full_qpos_rad'])
indices=m['selected_arm_indices'];world=np.asarray(m['initial_model_base_pose_matrix'])
URDF=ROOT/'preflight/robot_head_down_cpu_relative_meshes.urdf'
SRDF=Path('/home/admin123/liminghe/vla-benchmark/RoboTwin/assets/embodiments/realman-rm65b-inspire-gen4/robot.srdf')
links=[n.get('name') for n in ET.parse(URDF).getroot().findall('link')]
planner=mplib.Planner(urdf=URDF,srdf=SRDF,move_group='l_inspire_base_link',
    user_joint_names=names,user_link_names=links,use_convex=False)
pk=planner.robot.get_pinocchio_model();link_names=list(pk.get_link_names())
def fk(full,link):
    pk.compute_forward_kinematics(full)
    return world@pk.get_link_pose(link_names.index(link)).to_transformation_matrix()

rows=[];maximum_goal_position_error=0.;maximum_goal_matrix_error=0.
for r in m['RRT_attempts']:
    raw=np.asarray(r.get('raw_path_arm_rad',[]));vertices=np.asarray(r.get('actual_TOPP_audit_vertices_arm_rad',[]))
    row={k:r.get(k) for k in ['branch','IK_attempt','RRT_status','elapsed_s','raw_path_rows','retimed_rows','rejection_reason','audit_passed']}
    if vertices.size:
        row['raw_vs_actual_TOPP_vertices_max_error_rad']=float(abs(raw-vertices).max()) if raw.shape==vertices.shape else None
        # FMA explicitly prepends measured start to the actual TOPP vertices.
        full=np.tile(base,(len(vertices)+1,1));full[1:,indices]=vertices
        end=fk(full[-1],'l_inspire_base_link');target=np.asarray(m['goal_palm_world_matrix'])
        pe=float(np.linalg.norm(end[:3,3]-target[:3,3]));me=float(abs(end-target).max())
        maximum_goal_position_error=max(maximum_goal_position_error,pe)
        maximum_goal_matrix_error=max(maximum_goal_matrix_error,me)
        row['offline_imported_model_goal_palm_FK_position_error_m']=pe
        row['offline_imported_model_goal_palm_FK_matrix_error']=me
        collisions=[]
        for event in r.get('audit',{}).get('collisions_first_64',[]):
            index=event['path_index'];fraction=event['interval_fraction']
            q=full[index]+fraction*(full[index+1]-full[index])
            origin=fk(q,event['link'])[:3,3]
            collisions.append({**event,'actual_TOPP_vertex_edge_indices':[index-1,index],
                'edge_from_is_measured_start':index==0,'interpolated_arm_q_rad':q[indices].tolist(),
                'offline_FK_collision_link_origin_world_m':origin.tolist(),
                'FK_origin_is_not_surface_contact_point':True})
        row['first_reported_actual_mesh_collisions']=collisions
        row['audit']={k:r['audit'].get(k) for k in ['passed','audited_samples','elapsed_s','unique_collision_pairs','collision_pair_sample_occurrences']}
    rows.append(row)
result={'date':'2026-10-14','scope':'Read-only failed actual artifact inspection plus CPU FK on frozen full imported model; no new planner, GPU, physical step or contact-surface reconstruction',
    'episode_status':d['status'],'physical_steps':report['physical_steps'],
    'branch_rows':rows,'offline_goal_FK_max_position_error_m':maximum_goal_position_error,
    'offline_goal_FK_max_matrix_error':maximum_goal_matrix_error,
    'direct_actual_free_backend':m['backend'],
    'actual_waypoint_detour_used':not report['bounded_goal_adapter']['waypoint_detour_not_used'],
    'waypoint_source_exists_but_not_selected':str(ROOT/'source/rgb50_20261007_v1/full_native_approach_equivalent_dense_waypoint_v8.py'),
    'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),INPUT,ACTUAL/'expert_report.json',URDF,SRDF]},
    'frozen_or_production_code_modified':False}
(ROOT/'planning/slot17_attempt1_collision_detail_receipt_20261014.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':result['episode_status'],'physical_steps':result['physical_steps'],
    'offline_goal_FK_max_position_error_m':maximum_goal_position_error,'offline_goal_FK_max_matrix_error':maximum_goal_matrix_error,
    'collisions':[{'branch':r['branch'],'events':r.get('first_reported_actual_mesh_collisions',[])} for r in rows]},indent=2))
