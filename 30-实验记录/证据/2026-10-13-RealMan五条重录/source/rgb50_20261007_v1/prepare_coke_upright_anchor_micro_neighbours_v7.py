"""Prepare six fresh scene candidates, without copying any physical trajectory.

The scene/profile anchor must already have passed the independent v3 reader.
This file creates job specifications only; no candidate success is inferred.
"""
from pathlib import Path
import copy
import hashlib
import json

work = Path(__file__).resolve().parent
plan_path = work / 'sample_plan.json'
plan = json.loads(plan_path.read_bytes())
anchor_trial = work / 'batch_candidates/v9_equivalent_goals_dense_v7_attempt01/right_coke_upright_slot25_axial20mm_equivalent_goals_dense_v7'
formal_path = work / 'independent_v7_right_coke_slot25_full_readback_v3.json'
formal = json.loads(formal_path.read_bytes())
assert formal['passed'] is True
assert Path(formal['trial_dir']).resolve() == anchor_trial.resolve()
assert (formal['identity']['spec_index'], formal['identity']['seed']) == (25, 100025)
assert formal['synthetic_fixture_only'] is False and formal['readback_only'] is True
anchor_report_path = anchor_trial / 'expert_report.json'
anchor_report = json.loads(anchor_report_path.read_bytes())
assert anchor_report['accepted'] is True and anchor_report['physical_steps'] == 2550
anchor = anchor_report['sample_spec']
assert (anchor['index'], anchor['seed'], anchor['model_id'], anchor['arm'], anchor['pose_class']) == (25, 100025, 13, 'right', 'upright')
assert anchor['grasp_setting']['hand_x'] == .02

# Different positions, label yaws and seeds are initialized in new actual scenes.
# All entries retain the formal anchor's complete control profile unchanged.
offsets = [
    (12, 2200012, -.0005, -.0005, -.025),
    (31, 2200031, -.00025, .0005, -.05),
    (33, 2200033, .0005, .00025, .05),
    (12, 2210012, .0005, -.00025, .025),
    (31, 2210031, .00025, .0005, .025),
    (33, 2210033, -.0005, .00025, -.05),
]
assert len({entry[1] for entry in offsets}) == len(offsets)
assert all(entry[1] not in {s['seed'] for s in plan['specs']} for entry in offsets)
jobs = []
for index, seed, dx, dy, yaw in offsets:
    original = next(s for s in plan['specs'] if s['index'] == index)
    assert (original['model_id'], original['arm'], original['pose_class']) == (13, 'right', 'upright')
    spec = {key: copy.deepcopy(anchor[key]) for key in (
        'model_id', 'brand', 'arm', 'pose_class', 'yaw_semantics', 'position_reference',
        'acceptance_requires_settled_pose_check', 'grasp_setting')}
    spec.update(index=index, seed=seed,
                xy=[anchor['xy'][0] + dx, anchor['xy'][1] + dy],
                yaw_deg=anchor['yaw_deg'] + yaw,
                candidate_provenance={
                    'scene_profile_anchor_trial': str(anchor_trial.resolve()),
                    'anchor_independent_formal_readback_receipt': str(formal_path.resolve()),
                    'anchor_independent_formal_readback_receipt_sha256': hashlib.sha256(formal_path.read_bytes()).hexdigest(),
                    'anchor_report_sha256': hashlib.sha256(anchor_report_path.read_bytes()).hexdigest(),
                    'original_sample_plan_sha256': hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                    'original_sampling_cell_spec': original,
                    'new_position_delta_m': [dx, dy],
                    'new_upright_label_yaw_delta_deg': yaw,
                    'complete_grasp_setting_unchanged_from_formal_anchor': True,
                    'fresh_scene_physics_and_native_planning_required': True,
                    'trajectory_or_dataset_not_copied': True,
                    'this_job_plan_is_not_an_accepted_episode': True})
    assert spec['grasp_setting'] == anchor['grasp_setting']
    jobs.append({'name': f'slot{index:07d}_seed{seed}_coke_right_upright_anchor_micro_v7', 'spec': spec})
output = work / 'coke_right_upright_anchor_micro_neighbours_v7_jobs.json'
with output.open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(jobs, indent=2, allow_nan=False) + '\n')
print(json.dumps({'scope': 'Fresh candidate specifications only; no new physical episode success implied',
                  'output': str(output),
                  'jobs': [{'name': job['name'], 'index': job['spec']['index'],
                            'seed': job['spec']['seed'], 'xy': job['spec']['xy'],
                            'yaw_deg': job['spec']['yaw_deg']} for job in jobs]}, indent=2), flush=True)
