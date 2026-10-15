"""Installed full-native zero-step goal declaration probe, not a motion trial."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time
import numpy as np

opt = argparse.ArgumentParser()
opt.add_argument('--output', required=True)
args = opt.parse_args()
root = Path('/bigdata2/liminghe/VLA-benchmark')
work = root / 'deploy/realman_gen4/rgb50_20261007_v1'
legacy = work.parent / 'adjust_bottle_20261002'
sys.path[:0] = [str(work), str(legacy), str(root / 'RoboTwin')]
os.chdir(root / 'RoboTwin')
from geometry_overlay import install_native_shape_count_overlay
install_native_shape_count_overlay(legacy)
import closure_boundary_audit_v14 as boundary
import full_mesh_path_audit as fma
import equivalent_arm_goals_v5 as equivalent
from mplib import PlanningWorld
import mplib

source = work / 'independent_coke_right_diagonal_screen_20261007_v1/native_all93_best3_v3.json'
geometry = work / 'independent_coke_right_diagonal_screen_20261007_v1/geometry_screen.json'
trial = work / 'batch_candidates/v9_gpu3_attempt02/slot0000018_candidate_attempt02_v9'
source_bytes = source.read_bytes()
assert hashlib.sha256(source_bytes).hexdigest() == 'd897eb3ede833b2ecb9729f571e0c67afcc7bf6dd821c552420501818c70c728'
state = next(s for s in json.loads(source_bytes)['states'] if s['geometry_index'] == 9 and s['phase'] == 'pre')
geom = next(s for s in json.loads(geometry.read_bytes())['rows'] if s['index'] == 9)
q = np.asarray(state['native_full36_q_rad'], float)
task, constructor = boundary._event_snapshot_scene(trial, {'step': 0,
    'full_qpos_actual_rad': q.tolist(), 'bottle': {'pose': geom['bottle_pose_matrix']}})
physical = boundary._snapshot(task)
audit = fma.FullMeshPathAuditor(task)
model, pk = audit.model, audit.pk
old_pose, old_q = model.get_base_pose(), model.get_qpos().copy()
old_group = list(model.get_move_group_end_effectors())
old_indices = list(model.get_move_group_joint_indices())
names, links = list(audit.sim_joints), list(pk.get_link_names())
types_by_name = {j.get('name'): j.get('type') for j in audit.urdf_tree.findall('joint')}
types = [types_by_name[n] for n in names]
limits = np.concatenate([np.asarray(j.limits).reshape(-1, 2) for j in audit.art.active_joints])
source_paths = [Path(__file__), Path(equivalent.__file__), source, geometry,
                trial / 'run_config.yml', trial / 'scene_sample.json',
                Path(boundary.__file__), work / 'benchmark_geometry_cache.py',
                work / 'geometry_overlay.py', legacy / 'full_mesh_path_audit.py',
                Path(constructor['urdf'])]
binary = next(Path(mplib.__file__).parent.glob('pymp*.so'))
source_paths.append(binary)
hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
rows = []
started = time.monotonic()
try:
    inverse = np.linalg.inv(task.robot.left_entity.root_pose.to_transformation_matrix())
    model.set_base_pose(audit.Pose())
    obstacles = [audit.fcl.FCLObject('realman_' + name,
        audit.Pose(inverse @ source_obj.pose.to_transformation_matrix()),
        [audit.fcl.CollisionObject(s.get_collision_geometry()) for s in source_obj.shapes],
        [audit.Pose(p) for p in source_obj.shape_poses]) for name, source_obj in audit.static.items()]
    def native_fk(configuration):
        pk.compute_forward_kinematics(configuration)
        return np.asarray([fma._pose_matrix(pk.get_link_pose(i)) for i in range(len(links))])

    for side in ('left', 'right'):
        model.set_move_group(('l_' if side == 'left' else 'r_') + 'inspire_base_link')
        indices = [names.index(n) for n in getattr(task.robot, side + '_arm_joints_name')]
        assert list(model.get_move_group_joint_indices()) == indices
        world = PlanningWorld([model], obstacles)
        acm = world.get_allowed_collision_matrix()
        for pair in audit.allowed:
            a, b = sorted(pair)
            acm.set_entry(a, b, True)
        original = q.copy()
        # Explicit auxiliary IK-like reference within legal bounds; no native
        # articulation qpos/pose write occurs after scene initialization.
        original[indices[-1]] = .7
        declared, receipt = equivalent.declare_bounded_equivalent_goals(original, indices, limits, types, native_fk)
        assert len(declared) == 2
        checks = []
        for goal_index, candidate in enumerate(declared):
            model.set_qpos(candidate, full=True)
            world.set_qpos_all(candidate[indices])
            collisions = world.check_collision()
            checks.append({'declared_goal_index': goal_index, 'full_qpos_rad': candidate.tolist(),
                           'native_collision_count': len(collisions)})
        assert len({row['native_collision_count'] for row in checks}) == 1
        rows.append({'side': side, 'actual_selected_joint_names': [names[i] for i in indices],
                     'declaration': receipt, 'complete_native_world_checks': checks,
                     'all_link_FK_and_collision_count_equivalence_passed': True})
finally:
    model.set_move_group(old_group)
    model.set_base_pose(old_pose)
    model.set_qpos(old_q, full=True)
    assert list(model.get_move_group_end_effectors()) == old_group
    assert list(model.get_move_group_joint_indices()) == old_indices
    assert np.array_equal(model.get_qpos(), old_q)
    boundary._assert_snapshot(task, physical)
assert {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths} == hashes
output = {'passed': True, 'scope': 'Installed complete-native zero-step goal declaration only; no RRT/TOPP/physics/dataset success and no historical alias proof',
          'installed_mplib_version': mplib.__version__, 'source_hashes_before': hashes, 'source_hashes_after': hashes,
          'constructor_receipt': constructor, 'new_physics_steps': 0, 'render_systems_created': 0,
          'post_initialization_physical_state_writes': 0, 'physical_state_unchanged': True,
          'auxiliary_q_base_group_restored': True, 'full_q_dof': len(q),
          'native_FK_checked_link_count': len(links), 'native_collision_link_count': len(audit.objects),
          'native_collision_shape_count': sum(len(obj.shapes) for obj in audit.objects.values()),
          'native_bottle_shape_count': len(audit.static['bottle'].shapes),
          'native_table_shape_count': len(audit.static['table'].shapes),
          'original_allowed_self_pairs': len(audit.allowed), 'both_arm_results': rows,
          'raw_path_postprocessing_performed': False, 'elapsed_s': time.monotonic() - started}
with Path(args.output).open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(output, indent=2, allow_nan=False) + '\n')
print(json.dumps({key: output[key] for key in ('passed', 'scope', 'new_physics_steps', 'native_FK_checked_link_count', 'native_collision_link_count', 'physical_state_unchanged')}), flush=True)
