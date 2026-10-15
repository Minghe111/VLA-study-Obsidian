"""Actual CPU MPLib full-model FK equivalence proof; no scene or robot I/O.

Test poses are synthetic bounded endpoints on the actual frozen robot model.
This does not claim their collision freedom or task/recording success.
"""
from pathlib import Path
import hashlib, json, sys, xml.etree.ElementTree as ET
import numpy as np
import mplib
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT/'planning'), str(ROOT/'source/rgb50_20261007_v1')]
import equivalent_arm_goals_v5 as equivalent
import collector_quality_overlay as quality
ARCHIVE = Path('/home/admin123/liminghe/VLA-WAM-RL/30-实验记录/证据/2026-10-11-RealMan专家回放')
URDF = ROOT/'preflight/robot_head_down_cpu_relative_meshes.urdf'
SRDF = Path('/home/admin123/liminghe/vla-benchmark/RoboTwin/assets/embodiments/realman-rm65b-inspire-gen4/robot.srdf')
initial = json.loads((ARCHIVE/'episode_0000017/initial_state.json').read_text())
names = initial['actual_physical_contract']['full_qpos_joint_names']
tree = ET.parse(URDF).getroot(); links = [node.get('name') for node in tree.findall('link')]
types_by_name = {node.get('name'):node.get('type') for node in tree.findall('joint')}
rows = {}
for side in ('left','right'):
    palm = ('l_' if side=='left' else 'r_') + 'inspire_base_link'
    planner = mplib.Planner(urdf=str(URDF), srdf=str(SRDF), move_group=palm,
                           user_joint_names=names, user_link_names=links, use_convex=False)
    ids = list(planner.move_group_joint_indices); pk = planner.robot.get_pinocchio_model()
    link_names = list(pk.get_link_names()); palm_index = link_names.index(palm)
    current = np.zeros(len(names)); current[ids[-1]] = .93
    raw = current.copy(); raw[ids[-1]] = -5.25
    def fk(q):
        pk.compute_forward_kinematics(q)
        return np.asarray([pk.get_link_pose(i).to_transformation_matrix() for i in range(len(link_names))])
    goals, declaration = equivalent.declare_bounded_equivalent_goals(raw, ids, planner.joint_limits,
        [types_by_name[n] for n in names], fk)
    selected, selection = quality.choose_nearest_family({'goals':goals}, current, ids)
    matrices_raw = fk(raw); matrices_selected = fk(selected[0])
    proof = quality._strict_palm_goal(selected[0], current, ids, fk, matrices_raw[palm_index], palm_index, len(link_names))
    error = float(abs(matrices_raw-matrices_selected).max())
    endpoint_match = equivalent.match_declared_numeric_endpoint(np.stack([current[ids],selected[0][ids]]),current[ids],[selected[0][ids]])
    far_match = equivalent.match_declared_numeric_endpoint(np.stack([current[ids],raw[ids]]),current[ids],[selected[0][ids]])
    rows[side] = {'actual_native_joint_names': names, 'actual_native_FK_link_names': link_names,
        'declared_goal_proof': declaration, 'selected_single_goal': selection,
        'full_native_FK_max_matrix_error': error, 'strict_task_target_FK_proof': proof,
        'chosen_numeric_endpoint_match': endpoint_match, 'far_equivalent_endpoint_rejected': not far_match['passed'],
        'passed': bool(error<=1e-8 and proof['passed'] and endpoint_match['passed'] and not far_match['passed'])}
receipt = {'scope':'Real CPU MPLib/Pinocchio FK on frozen full robot; synthetic finite endpoints; no collision/task/physics success asserted',
    'source_sha256': {str(p):quality._sha(p) for p in [Path(__file__),Path(quality.__file__),URDF,SRDF,
        ROOT/'source/rgb50_20261007_v1/equivalent_arm_goals_v5.py']},
    'arms': rows,'passed':all(row['passed'] for row in rows.values()),'scene_created':False,'GPU_started':False,
    'planned_or_recorded_path_modified':False}
(ROOT/'planning/native_fk_near_goal_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'passed':receipt['passed'],'arms':{s:{'FK_links':len(v['actual_native_FK_link_names']),
    'FK_matrix_error':v['full_native_FK_max_matrix_error'],'chosen_J6':v['selected_single_goal']['chosen_arm_qpos_rad'][-1],
    'far_endpoint_rejected':v['far_equivalent_endpoint_rejected']} for s,v in rows.items()}},indent=2))
raise SystemExit(0 if receipt['passed'] else 1)
