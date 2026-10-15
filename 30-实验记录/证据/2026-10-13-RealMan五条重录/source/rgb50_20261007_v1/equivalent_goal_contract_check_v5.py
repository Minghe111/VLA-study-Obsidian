"""Zero-step fixtures for finite goal declaration; not native robot validation."""
from pathlib import Path
import ast
import hashlib
import json
import numpy as np
import equivalent_arm_goals_v5 as helper

work = Path(__file__).resolve().parent
q = np.linspace(-.2, .2, 36)
indices = list(range(6))
q[5] = 5.8
limits = np.tile([-3.11, 3.11], (36, 1))
limits[5] = [-6.28, 6.28]
types = ['revolute'] * 36


def periodic_fk(configuration):
    result = np.tile(np.eye(4), (93, 1, 1))
    for link in range(93):
        angle = float(configuration[5])
        result[link, :2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        result[link, :3, 3] = np.asarray(configuration[6:9]) + link * .001
    return result


def nonperiodic_fk(configuration):
    result = periodic_fk(configuration)
    result[23, 2, 3] += float(configuration[5]) * .01
    return result


cases = []
before = q.copy()
goals, receipt = helper.declare_bounded_equivalent_goals(q, indices, limits, types, periodic_fk)
assert len(goals) == 2 and np.array_equal(q, before)
assert all(np.array_equal(g[6:], q[6:]) for g in goals)
assert all(np.all(g[indices] >= limits[indices, 0]) and np.all(g[indices] <= limits[indices, 1]) for g in goals)
assert all(r['native_FK_checked_links'] == 93 and r['native_FK_equivalent'] for r in receipt['goal_attempts'])
cases.append({'case': 'two_explicit_bounded_native_FK_equivalent_goals', 'passed': True})
start = np.zeros(6)
alternate = next(g for g in goals if g[5] < 0.)
path = np.vstack([start, alternate[indices]])
saved_path = path.copy()
matched = helper.match_declared_numeric_endpoint(path, start, [g[indices] for g in goals])
assert matched['passed'] and np.array_equal(saved_path, path)
cases.append({'case': 'legal_predeclared_alternate_numeric_endpoint_without_path_modification', 'passed': True})
for name, mutation in (
    ('unmatched_2pi_endpoint', lambda p: p.__setitem__((-1, 0), p[-1, 0] + helper.TWO_PI)),
    ('finite_vendor_bound_exceeded_by_4pi_endpoint', lambda p: p.__setitem__((-1, 5), p[-1, 5] + 2 * helper.TWO_PI)),
    ('numerically_different_start_despite_periodicity', lambda p: p.__setitem__((0, 5), p[0, 5] + helper.TWO_PI)),
    ('nonfinite_raw_path', lambda p: p.__setitem__((-1, 2), np.nan)),
):
    negative = path.copy()
    mutation(negative)
    assert not helper.match_declared_numeric_endpoint(negative, start, [g[indices] for g in goals])['passed']
    cases.append({'case': name, 'passed': True})
inexact = path.copy()
inexact[-1, 3] += 1.0001e-6
assert not helper.match_declared_numeric_endpoint(inexact, start, [g[indices] for g in goals])['passed']
cases.append({'case': 'original_numeric_endpoint_tolerance_retained', 'passed': True})
wrong_fk_goals, wrong_fk_receipt = helper.declare_bounded_equivalent_goals(q, indices, limits, types, nonperiodic_fk)
assert len(wrong_fk_goals) == 1 and np.array_equal(wrong_fk_goals[0], q)
assert any(not row['native_FK_equivalent'] for row in wrong_fk_receipt['goal_attempts'])
assert not helper.match_declared_numeric_endpoint(path, start, [g[indices] for g in wrong_fk_goals])['passed']
cases.append({'case': 'one_nonpalm_link_FK_difference_rejects_alternate_goal', 'passed': True})
for name, changed_limits, changed_types in (
    ('continuous_joint_not_treated_as_bounded', limits.copy(), ['continuous'] + types[1:]),
    ('unbounded_limits_not_silently_used', np.vstack([[-np.inf, np.inf], limits[1:]]), types),
    ('reversed_limits_rejected', np.vstack([[3., -3.], limits[1:]]), types),
):
    try:
        helper.declare_bounded_equivalent_goals(q, indices, changed_limits, changed_types, periodic_fk)
    except ValueError:
        cases.append({'case': name, 'passed': True})
    else:
        raise AssertionError(name)
tree = ast.parse(Path(helper.__file__).read_text(encoding='utf-8'))
for node in ast.walk(tree):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        assert node.func.attr not in ('step', 'set_pose', 'set_qpos', 'set_qvel', 'set_drive_target', 'set_drive_velocity_target')
output = {'passed': True, 'scope': 'Pure finite-goal CPU fixtures only; no installed native robot, planning, render or physics success implied',
          'new_physics_steps': 0, 'physical_validation_completed': False,
          'input_q_and_returned_raw_path_unchanged': True, 'cases': cases,
          'source_sha256': {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (Path(helper.__file__), Path(__file__))}}
with (work / 'equivalent_goal_contract_receipt_v5.json').open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(output, indent=2, allow_nan=False) + '\n')
print(json.dumps(output, indent=2), flush=True)
