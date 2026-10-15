"""Zero-step mock integration: refinement gets actual retimed vertices, not raw."""
from pathlib import Path
import ast
import hashlib
import json
import numpy as np

work = Path(__file__).resolve().parent
fixture = work / 'equivalent_native_planner_contract_dense_v7.py'
source = fixture.read_text(encoding='utf-8').split('results = []\n', 1)[0]
tree = ast.parse(source)
auditor = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Auditor')
old = next(n for n in auditor.body if isinstance(n, ast.FunctionDef) and n.name == 'audit')
replacement = ast.parse('''def audit(self, vertices, side, **kwargs):
    calls['FMA'].append({'actual_input_vertices': np.asarray(vertices).copy(), **kwargs})
    refined = kwargs.get('vertex_step_m') == .0001
    return {'passed': refined, 'joint_limit_audit': {'passed': True},
        'unique_collision_pairs': 0, 'collisions_first_64': [],
        'metadata': {'native_collision_links': 93},
        'thresholds_m': {'bottle': .0005, 'table': 0., 'self': 0.},
        'sampled_minimum_gap': {'bottle': {'distance_m': .0008}, 'table': {'distance_m': .005}, 'self': {'distance_m': .002}},
        'continuous_gap_lower_bound': {'bottle': {'lower_bound_m': .0007 if refined else -.0001}, 'table': {'lower_bound_m': .004}, 'self': {'lower_bound_m': .001}}}
''').body[0]
auditor.body[auditor.body.index(old)] = replacement
topp = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'topp')
site = next(n for n in topp.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'position' for t in n.targets))
topp.body.insert(topp.body.index(site) + 1, ast.parse('position += 5e-11').body[0])
ast.fix_missing_locations(tree)
env = {'__file__': str(fixture), '__name__': '__private_zero_step_mock__'}
exec(compile(tree, str(fixture) + ':dense_integration_mock_only', 'exec'), env)
env['mode'] = 'valid_predeclared_alternate'
env['created'] = []
env['calls'] = {name: [] for name in ('RRT', 'TOPP', 'FMA', 'ACM', 'obstacles')}
env['calls']['IK_count'] = 0
result = env['helper'].plan_free_approach(env['task'], [0] * 7, 'right', ik_attempts=2)
assert result['status'] == 'Success' and len(env['calls']['FMA']) == 2
row = result['planning_metadata']['RRT_attempts'][0]
raw = np.asarray(row['raw_path_arm_rad'])
actual = np.asarray(row['actual_TOPP_audit_vertices_arm_rad'])
assert not np.array_equal(raw, actual)
assert np.array_equal(env['calls']['FMA'][0]['actual_input_vertices'], actual)
assert np.array_equal(env['calls']['FMA'][1]['actual_input_vertices'], actual)
assert row['strict_dense_certificate_refinement']['original_complete_default_grid_audit']['passed'] is False
assert row['strict_dense_certificate_refinement']['refined_complete_native_audit']['passed'] is True
assert result['planning_metadata']['physical_state_unchanged']
for model in env['created']:
    assert np.array_equal(model.q, env['original_aux_q'])
    assert np.array_equal(model.pose.m, env['original_base'])
    assert model.group == ['original_group']
output = {'passed': True, 'scope': 'Zero-step mocked call integration only, not real native TOPP/FCL or physical success',
          'new_physics_steps': 0, 'actual_retained_TOPP_vertices_used_twice_unchanged': True,
          'raw_path_was_not_substituted_for_actual_TOPP_vertices': True,
          'original_failed_full_audit_retained': True, 'all_auxiliary_state_restored': True,
          'source_sha256': {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (Path(__file__), fixture, Path(env['helper'].__file__), Path(env['helper'].dense_policy.__file__))}}
with (work / 'dense_planner_flow_contract_receipt_v7.json').open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(output, indent=2, allow_nan=False) + '\n')
print(json.dumps(output, indent=2), flush=True)
