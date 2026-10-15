"""Create isolated dense-certificate v7; preserve all frozen v6 sources."""
from pathlib import Path
import hashlib
import json

work = Path(__file__).resolve().parent
old = work / 'full_native_approach_equivalent_goals_v6.py'
source = old.read_text(encoding='utf-8')
source = source.replace('import equivalent_arm_goals_v5 as equivalent\n',
    'import equivalent_arm_goals_v5 as equivalent\nimport dense_free_certificate_policy_v7 as dense_policy\n')
source = source.replace("equivalent_goals_v6'", "equivalent_goals_dense_v7'")
needle = "            row.update(retimed_joint_limit_audit=limit_check, retimed_rows=len(position))\n"
assert source.count(needle) == 1
source = source.replace(needle, needle + '''            row['complete_original_official_TOPP_retiming'] = retime
            row['actual_TOPP_audit_vertices_arm_rad'] = np.asarray(retime['audit_vertices']).tolist()
            row['retimed_commanded_position_arm_rad'] = np.asarray(position).tolist()
            row['retimed_commanded_velocity_arm_rad_s'] = np.asarray(velocity).tolist()
            row['retimed_official_native_times_s'] = np.asarray(times).tolist()
''')
needle = "            row.update(audit_passed=geometry['passed'], audit=geometry)\n"
assert source.count(needle) == 1
source = source.replace(needle, '''            geometry, refinement = dense_policy.refine_certificate(
                audit, retime['audit_vertices'], side, geometry)
            row['strict_dense_certificate_refinement'] = refinement
''' + needle)
new = work / 'full_native_approach_equivalent_goals_dense_v7.py'
with new.open('x', encoding='utf-8', newline='\n') as stream: stream.write(source)
old_expert = work / 'expert_generation_v9_equivalent_goals_v6.py'
expert = old_expert.read_text(encoding='utf-8').replace('full_native_approach_equivalent_goals_v6', 'full_native_approach_equivalent_goals_dense_v7')
expert = expert.replace("bounded_equivalent_goals_v6'", "bounded_equivalent_goals_dense_v7'")
needle = "                       Path(self.free_approach.equivalent.__file__).resolve(),\n"
assert expert.count(needle) == 1
expert = expert.replace(needle, needle + "                       Path(self.free_approach.dense_policy.__file__).resolve(),\n")
expert = expert.replace("'CPU_path_is_not_physical_success': True}",
    "'CPU_path_is_not_physical_success': True, 'dense_only_after_original_complete_insufficient_certificate': True, 'dense_vertex_step_m': .0001, 'dense_finite_sample_capacity': 250000, 'every_branch_actual_TOPP_vertices_and_retiming_retained': True}")
new_expert = work / 'expert_generation_v9_equivalent_goals_dense_v7.py'
with new_expert.open('x', encoding='utf-8', newline='\n') as stream: stream.write(expert)
fixture_source = (work / 'equivalent_native_planner_contract_v6.py').read_text(encoding='utf-8')
fixture_source = fixture_source.replace('full_native_approach_equivalent_goals_v6', 'full_native_approach_equivalent_goals_dense_v7')
fixture_source = fixture_source.replace('expert_generation_v9_equivalent_goals_v6.py', 'expert_generation_v9_equivalent_goals_dense_v7.py')
fixture_source = fixture_source.replace('equivalent_native_planner_contract_receipt_v6.json', 'equivalent_native_planner_contract_receipt_dense_v7.json')
new_fixture = work / 'equivalent_native_planner_contract_dense_v7.py'
with new_fixture.open('x', encoding='utf-8', newline='\n') as stream: stream.write(fixture_source)
print(json.dumps({'new_sources_only': True, 'source_sha256': {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in (old, new, old_expert, new_expert, new_fixture, work / 'dense_free_certificate_policy_v7.py', Path(__file__))}}, indent=2), flush=True)
