"""Real CPU OMPL counterexample on the full frozen robot and original meshes.

Uses the new slot17 initial measured full q and a short only-J6 change to its
branch0 nearest endpoint. Five selected-arm joints and all other DOFs stay
frozen. This does not recreate the randomized table/clutter/physical scene.
"""
from pathlib import Path
import hashlib, json, math, time, xml.etree.ElementTree as ET
import numpy as np
import mplib
from mplib import PlanningWorld, Pose
from mplib.planning.ompl import OMPLPlanner, FixedJoint
from auxiliary_joint_window import build_auxiliary_window_model

ROOT = Path(__file__).resolve().parent.parent
URDF = ROOT/'preflight/robot_head_down_cpu_relative_meshes.urdf'
SRDF = Path('/home/admin123/liminghe/vla-benchmark/RoboTwin/assets/embodiments/realman-rm65b-inspire-gen4/robot.srdf')
ACTUAL = Path('/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/rgb50_motion_quality_20261013_v1/new_physical_episodes/slot_0000017_randomized_attempt_000')
meta = json.loads((ACTUAL/'free_approach_0.json').read_text())['planning_metadata']
report = json.loads((ACTUAL/'expert_report.json').read_text())
names = report['full_qpos_joint_names']
base = np.asarray(meta['initial_actual_full_qpos_rad'], dtype=float)
side = 'left'; palm = 'l_inspire_base_link'
tree = ET.parse(URDF).getroot(); links = [link.get('name') for link in tree.findall('link')]
source = mplib.Planner(urdf=URDF, srdf=SRDF, move_group=palm,
                       user_joint_names=names, user_link_names=links, use_convex=False)
model = source.robot; model.set_base_pose(Pose()); model.set_qpos(base, full=True)
print('CPU_SOURCE_MODEL_CREATED', flush=True)
indices = list(model.get_move_group_joint_indices()); selected = indices[-1]
short_goal = base[indices].copy()
short_goal[-1] = meta['RRT_attempts'][0]['near_numeric_goal_selection']['chosen_arm_qpos_rad'][-1]
branch0_goal = np.asarray(meta['RRT_attempts'][0]['near_numeric_goal_selection']['chosen_arm_qpos_rad'])
for link in tree.findall('link'):
    for node in list(link):
        if node.tag in ('visual', 'collision'):
            link.remove(node)
bare = ET.tostring(tree, encoding='unicode')
bounded, window = build_auxiliary_window_model(model, bare, SRDF.read_text(),
    current_full_q=base, joint_name=names[selected], move_group=palm)
print('CPU_AUXILIARY_MODEL_CLONED', flush=True)
results = {}
for label, candidate, freeze, goal in [
    ('original_wide_vendor_model_fixed5', model, True, short_goal),
    ('auxiliary_narrow_numeric_window_fixed5', bounded, True, short_goal),
    ('original_wide_vendor_model_native6D', model, False, short_goal),
    ('auxiliary_narrow_numeric_window_native6D', bounded, False, short_goal),
    ('original_wide_vendor_model_actual_branch0', model, False, branch0_goal),
    ('auxiliary_narrow_numeric_window_actual_branch0', bounded, False, branch0_goal),
]:
    candidate.set_base_pose(Pose()); candidate.set_qpos(base, full=True)
    world = PlanningWorld([candidate], [])
    start_collisions = world.check_collision()
    target = base.copy(); target[indices] = goal
    candidate.set_qpos(target, full=True); world.set_qpos_all(goal)
    goal_collisions = world.check_collision()
    candidate.set_qpos(base, full=True); world.set_qpos_all(base[indices])
    if start_collisions or goal_collisions:
        results[label] = {'start_collision_count': len(start_collisions), 'goal_collision_count': len(goal_collisions),
                          'planning_not_attempted_because_collision': True}
        continue
    # OMPL FixedJoint indexes the current move-group state, not full interleaved q.
    fixed = ({FixedJoint(0, local_index, float(base[full_index]))
              for local_index, full_index in enumerate(indices[:-1])} if freeze else set())
    print('CPU_PLANNING_BEGIN', label, 'group_full_indices', indices, flush=True)
    stamp = time.monotonic()
    status, raw = OMPLPlanner(world).plan(base[indices], [goal], time=1., range=.05,
                                        fixed_joints=fixed, simplify=True, verbose=False)
    elapsed = time.monotonic()-stamp; path = np.asarray(raw, dtype=float)
    success = str(status).rstrip('.') == 'Exact solution' and path.ndim == 2 and path.shape[1] == 6 and len(path) >= 2
    row = {'status': str(status), 'elapsed_s': elapsed, 'path_rows': len(path),
           'start_collision_count': len(start_collisions), 'goal_collision_count': len(goal_collisions),
           'start_arm_q': base[indices].tolist(), 'specified_goal_arm_q': goal.tolist(), 'raw_path': path.tolist(),
           'raw_path_modified': False, 'fixed_joints_enabled': freeze,
           'unchanged_goal_selected_arm_joint_indices': indices[:-1]}
    if success:
        delta = path[-1]-goal
        row.update(raw_start_error_rad=float(abs(path[0]-base[indices]).max()),
                   raw_goal_error_rad=float(abs(delta).max()), raw_end_delta_rad=delta.tolist(),
                   fixed_other_joints_error_rad=float(abs(path[:,:-1]-base[indices][:-1]).max()),
                   J6_total_variation_rad=float(abs(np.diff(path[:,-1])).sum()),
                   J6_range_rad=float(np.ptp(path[:,-1])),
                   exact_near_endpoint_match=bool(abs(delta).max() <= 1e-6))
    results[label] = row
    candidate.set_qpos(base, full=True); world.set_qpos_all(base[indices])
receipt = {'date':'2026-10-14', 'scope':'Actual CPU OMPL on frozen full URDF/original collision meshes; only J6 moves; no table/clutter or physical success asserted',
    'actual_joint_names': names, 'FK_links': len(links), 'window': window, 'results': results,
    'passed': all(results[k].get('exact_near_endpoint_match')
                   and results[k].get('fixed_other_joints_error_rad',math.inf) <= 1e-8
                   and results[k].get('J6_total_variation_rad',math.inf) < math.pi
                   for k in ['auxiliary_narrow_numeric_window_fixed5','auxiliary_narrow_numeric_window_native6D'])
              and bool(results['auxiliary_narrow_numeric_window_actual_branch0'].get('exact_near_endpoint_match')),
    'original_native6D_far_alias_reproduced': bool(abs(results['original_wide_vendor_model_native6D'].get('raw_goal_error_rad',0.) - 2*math.pi) < 1e-6),
    'original_source_state_after_all_tests_matches_base': bool(np.array_equal(model.get_qpos(), base)),
    'GPU_started': False, 'SAPIEN_scene_created': False, 'active_worker_modified': False,
    'vendor_URDF_FMA_or_physical_limits_modified': False,
    'sources_sha256': {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
         [Path(__file__), ROOT/'planning/auxiliary_joint_window.py',URDF,SRDF,ACTUAL/'free_approach_0.json']}}
(ROOT/'planning/auxiliary_window_actual_ompl_receipt_20261014.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'passed':receipt['passed'],'window':window['auxiliary_window_rad'],
      'results':{k:{n:v for n,v in r.items() if n in ['status','elapsed_s','raw_goal_error_rad','J6_total_variation_rad','start_collision_count','goal_collision_count','exact_near_endpoint_match']} for k,r in results.items()}},indent=2))
raise SystemExit(0 if receipt['passed'] else 1)
