"""Zero-step finer audit of retained raw affine polylines, not old TOPP bytes."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--trial', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--branches', type=int, nargs='+', default=[0, 1, 5])
args = parser.parse_args()
root = Path('/bigdata2/liminghe/VLA-benchmark')
work = root / 'deploy/realman_gen4/rgb50_20261007_v1'
legacy = work.parent / 'adjust_bottle_20261002'
sys.path[:0] = [str(work), str(legacy), str(root / 'RoboTwin')]
os.chdir(root / 'RoboTwin')
from geometry_overlay import install_native_shape_count_overlay
install_native_shape_count_overlay(legacy)
from geometry_vector_overlay import install_vectorized_aabb_overlay
install_vectorized_aabb_overlay(legacy)
import closure_boundary_audit_v14 as boundary
import full_mesh_path_audit as fma

trial = Path(args.trial).resolve()
source = trial / 'free_approach_0.json'
original_bytes = source.read_bytes()
original = json.loads(original_bytes)
metadata = original['planning_metadata']
task, constructor = boundary._event_snapshot_scene(trial, {'step': 0,
    'full_qpos_actual_rad': metadata['initial_actual_full_qpos_rad'],
    'bottle': {'pose': metadata['initial_native_bottle_matrix']}})
snapshot = boundary._snapshot(task)
auditor = fma.FullMeshPathAuditor(task)
sources = [source, trial / 'run_config.yml', trial / 'scene_sample.json',
           trial / 'expert_report.json', Path(__file__), work / 'closure_boundary_audit_v14.py',
           work / 'benchmark_geometry_cache.py', work / 'geometry_overlay.py',
           work / 'geometry_vector_overlay.py', legacy / 'full_mesh_path_audit.py',
           Path(constructor['urdf'])]
hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
results = []
for branch in args.branches:
    row = next(r for r in metadata['RRT_attempts'] if r['branch'] == branch)
    old = row['audit']
    assert old['joint_limit_audit']['passed'] and old['unique_collision_pairs'] == 0 and not old['collisions_first_64']
    assert all(old['sampled_minimum_gap'][key]['distance_m'] >= value for key, value in old['thresholds_m'].items())
    for key in ('full_dof', 'collision_links', 'collision_shapes', 'bottle_shapes',
                'native_exemption_pairs', 'effective_exemption_pairs', 'checked_self_pairs',
                'urdf_kinematic_export_sha256'):
        assert auditor.metadata[key] == old['metadata'][key], (key, auditor.metadata[key], old['metadata'][key])
    path = np.asarray(row['raw_path_arm_rad'])
    saved_path = path.copy()
    started = time.monotonic()
    dense = auditor.audit(path, 'right', vertex_step_m=.0001,
        min_bottle_clearance_m=old['thresholds_m']['bottle'],
        min_self_clearance_m=old['thresholds_m']['self'],
        min_table_clearance_m=old['thresholds_m']['table'], max_samples=250000)
    assert np.array_equal(path, saved_path)
    boundary._assert_snapshot(task, snapshot)
    assert source.read_bytes() == original_bytes
    results.append({'branch': branch, 'original_raw_arm_path_rad': path.tolist(),
                    'raw_matrix_sha256_f64': hashlib.sha256(path.astype('<f8').tobytes()).hexdigest(),
                    'original_complete_audit': old, 'new_complete_dense_raw_audit': dense,
                    'original_rejection_kept_unchanged': True,
                    'original_TOPP_audit_vertices_not_retained_per_branch': True,
                    'new_audit_proves_raw_affine_polyline_only': True,
                    'vertex_step_m': .0001, 'max_samples_finite_capacity': 250000,
                    'no_sample_truncation': True, 'elapsed_s': time.monotonic() - started})
    print(json.dumps({'branch': branch, 'passed': dense['passed'], 'audited_samples': dense['audited_samples'],
                      'sampled_minimum_gap': dense['sampled_minimum_gap'],
                      'continuous_gap_lower_bound': dense['continuous_gap_lower_bound'],
                      'new_physics_steps': 0, 'elapsed_s': results[-1]['elapsed_s']}), flush=True)
assert {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources} == hashes
output = {'scope': 'Stricter zero-step native full-mesh certificate of original raw affine polylines only, not byte equality to unretained original TOPP vertices, not physical success or old trial acceptance',
          'source_hashes_before': hashes, 'source_hashes_after': hashes,
          'constructor_receipt': constructor, 'new_physics_steps': 0, 'render_systems_created': 0,
          'post_initialization_physical_state_writes': 0, 'physical_state_unchanged': True,
          'original_report_and_raw_source_unchanged': True, 'all_geometry_thresholds_unchanged': True,
          'native_metadata_exactly_matches_original_model': True, 'results': results}
with Path(args.output).open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(output, indent=2, allow_nan=False) + '\n')
print(json.dumps({'output': str(Path(args.output).resolve()), 'sha256': hashlib.sha256(Path(args.output).read_bytes()).hexdigest()}), flush=True)
