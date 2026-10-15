"""Create new v6 sources; preserve frozen v5 bytes and receipts."""
from pathlib import Path
import hashlib
import json

work = Path(__file__).resolve().parent
old = work / 'full_native_approach_equivalent_goals_v5.py'
source = old.read_text(encoding='utf-8')
needle = '''            if any(any(np.linalg.norm(c[indices] - old[indices]) < 1e-3
                       for c in legal for old in family['goals']) for family in families):
                row['duplicate_predeclared_numeric_goal_family'] = True
                continue'''
replacement = '''            covered = [any(np.linalg.norm(c[indices] - old[indices]) < 1e-3
                           for family in families for old in family['goals']) for c in legal]
            row['predeclared_numeric_goals_already_covered'] = covered
            if all(covered):
                row['all_predeclared_numeric_goals_already_covered'] = True
                continue'''
assert source.count(needle) == 1
source = source.replace(needle, replacement).replace('equivalent_goals_v5\'', 'equivalent_goals_v6\'')
new = work / 'full_native_approach_equivalent_goals_v6.py'
with new.open('x', encoding='utf-8', newline='\n') as stream: stream.write(source)
old_expert = work / 'expert_generation_v9_equivalent_goals_v5.py'
expert = old_expert.read_text(encoding='utf-8').replace('full_native_approach_equivalent_goals_v5', 'full_native_approach_equivalent_goals_v6')
expert = expert.replace("bounded_equivalent_goals_v5'", "bounded_equivalent_goals_v6'")
new_expert = work / 'expert_generation_v9_equivalent_goals_v6.py'
with new_expert.open('x', encoding='utf-8', newline='\n') as stream: stream.write(expert)
old_fixture = work / 'equivalent_native_planner_contract_v5.py'
fixture = old_fixture.read_text(encoding='utf-8').replace('full_native_approach_equivalent_goals_v5', 'full_native_approach_equivalent_goals_v6')
fixture = fixture.replace('expert_generation_v9_equivalent_goals_v5.py', 'expert_generation_v9_equivalent_goals_v6.py')
fixture = fixture.replace('equivalent_native_planner_contract_receipt_v5.json', 'equivalent_native_planner_contract_receipt_v6.json')
needle = "        q[5] = 5.8\n"
assert fixture.count(needle) == 1
fixture = fixture.replace(needle, needle + "        if mode == 'finite_goal_boundary': q[5] = .0030 if calls['IK_count'] == 0 else .0034\n        calls['IK_count'] += 1\n")
needle = "        endpoint = np.asarray(goals[0]).copy()\n"
assert fixture.count(needle) == 1
fixture = fixture.replace(needle, "        endpoint = np.asarray(goals[-1] if mode == 'finite_goal_boundary' else goals[0]).copy()\n")
fixture = fixture.replace("        if mode == 'approximate_status': return 'Approximate solution', path\n", "        if mode == 'approximate_status' or (mode == 'finite_goal_boundary' and len(goals) == 1): return 'Approximate solution', path\n")
fixture = fixture.replace("             'endpoint_collision', 'start_collision', 'frozen_IK_changed', 'FK_not_equivalent', 'world_exception'):", "             'endpoint_collision', 'start_collision', 'frozen_IK_changed', 'FK_not_equivalent', 'world_exception', 'finite_goal_boundary'):")
fixture = fixture.replace("    snapshot = physical.copy()\n", "    calls['IK_count'] = 0\n    snapshot = physical.copy()\n")
fixture = fixture.replace("        if mode == 'valid_predeclared_alternate':\n", "        if mode in ('valid_predeclared_alternate', 'finite_goal_boundary'):\n")
fixture = fixture.replace("            assert len(calls['RRT'][0]['goals']) == 2\n", "            assert len(calls['RRT'][-1]['goals']) == 2\n")
fixture = fixture.replace("            row = result['planning_metadata']['RRT_attempts'][0]\n", "            row = result['planning_metadata']['RRT_attempts'][-1]\n            if mode == 'finite_goal_boundary':\n                assert result['planning_metadata']['native_free_goal_family_count'] == 2\n                assert len(calls['RRT']) == 2\n                assert result['planning_metadata']['endpoint_attempts'][1]['predeclared_numeric_goals_already_covered'] == [True, False]\n                assert row['numeric_endpoint_match']['matched_declared_goal_index'] == 1\n")
new_fixture = work / 'equivalent_native_planner_contract_v6.py'
with new_fixture.open('x', encoding='utf-8', newline='\n') as stream: stream.write(fixture)
print(json.dumps({'scope': 'New source creation only; no physical trial', 'source_sha256':
                  {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (old, new, old_expert, new_expert, old_fixture, new_fixture, Path(__file__))}}, indent=2), flush=True)
