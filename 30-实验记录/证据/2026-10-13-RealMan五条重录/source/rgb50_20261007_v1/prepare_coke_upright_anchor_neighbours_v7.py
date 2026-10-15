"""Create fresh scene jobs around a formally read-back successful anchor.

This copies scene/profile specifications only. It does not copy a trajectory,
run physics, or declare the three new scenes successful.
"""
from pathlib import Path
import copy
import hashlib
import json

work = Path(__file__).resolve().parent
plan_path = work / 'sample_plan.json'
plan = json.loads(plan_path.read_bytes())
samples = plan['specs']
anchor_trial = work / 'batch_candidates/v9_equivalent_goals_dense_v7_attempt01/right_coke_upright_slot25_axial20mm_equivalent_goals_dense_v7'
formal_path = work / 'independent_v7_right_coke_slot25_full_readback_v3.json'
formal = json.loads(formal_path.read_bytes())
assert formal['passed'] is True and Path(formal['trial_dir']).resolve() == anchor_trial.resolve()
assert (formal['identity']['spec_index'], formal['identity']['seed']) == (25, 100025)
assert formal['synthetic_fixture_only'] is False and formal['readback_only'] is True
anchor_report = json.loads((anchor_trial / 'expert_report.json').read_bytes())
assert anchor_report['accepted'] is True and anchor_report['physical_steps'] == 2550
anchor = anchor_report['sample_spec']
assert anchor['index'] == 25 and anchor['seed'] == 100025
assert (anchor['model_id'], anchor['arm'], anchor['pose_class']) == (13, 'right', 'upright')
assert anchor['grasp_setting']['hand_x'] == .02
offsets = {12: (.0015, -.0010, .15), 31: (-.0015, .0015, -.15), 33: (.0010, .0020, .05)}
jobs = []
for index, (dx, dy, yaw) in offsets.items():
    original = next(s for s in samples if s['index'] == index)
    assert (original['model_id'], original['arm'], original['pose_class']) == (13, 'right', 'upright')
    spec = {key: copy.deepcopy(anchor[key]) for key in (
        'model_id', 'brand', 'arm', 'pose_class', 'yaw_semantics', 'position_reference',
        'acceptance_requires_settled_pose_check', 'grasp_setting')}
    spec.update(index=index, seed=2100000 + index,
                xy=[anchor['xy'][0] + dx, anchor['xy'][1] + dy],
                yaw_deg=anchor['yaw_deg'] + yaw,
                candidate_provenance={'scene_profile_anchor_trial': str(anchor_trial.resolve()),
                    'anchor_independent_formal_readback_receipt': str(formal_path.resolve()),
                    'anchor_independent_formal_readback_receipt_sha256': hashlib.sha256(formal_path.read_bytes()).hexdigest(),
                    'anchor_report_sha256': hashlib.sha256((anchor_trial / 'expert_report.json').read_bytes()).hexdigest(),
                    'original_sample_plan_sha256': hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                    'original_sampling_cell_spec': original,
                    'new_position_delta_m': [dx, dy], 'new_upright_label_yaw_delta_deg': yaw,
                    'fresh_scene_physics_and_native_planning_required': True,
                    'trajectory_or_dataset_not_copied': True,
                    'this_job_plan_is_not_an_accepted_episode': True})
    jobs.append({'name': f'slot{index:07d}_coke_right_upright_anchor_neighbour_v7', 'spec': spec})
output = work / 'coke_right_upright_anchor_neighbours_v7_jobs.json'
with output.open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(jobs, indent=2, allow_nan=False) + '\n')
print(json.dumps({'scope': 'Three fresh candidate scene specifications only; no new episode success implied',
                  'output': str(output), 'jobs': [{'name': j['name'], 'index': j['spec']['index'],
                      'seed': j['spec']['seed'], 'xy': j['spec']['xy'], 'yaw_deg': j['spec']['yaw_deg']} for j in jobs]}, indent=2), flush=True)
