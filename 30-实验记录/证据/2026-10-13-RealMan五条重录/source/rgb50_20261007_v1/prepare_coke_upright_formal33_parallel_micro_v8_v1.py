"""Fresh micro-position candidate groups; no trajectories or successes copied."""
from pathlib import Path
import copy
import hashlib
import json


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


work = Path(__file__).resolve().parent
trial = (work / 'batch_candidates/v9_coke_right_upright_gain01_waypoint_v8_attempt05'
         / 'slot0000033_coke_right_upright_anchor_neighbour_v7_gain01_matched_parallel_waypoint_v8')
matched = None
for list_path in (work / 'accepted_list_v3_admission.json', work / 'accepted_list_v2.json'):
    list_bytes = list_path.read_bytes()
    current = json.loads(list_bytes)
    assert current['reader_version'] == 3
    rows = current.get('trials', current.get('accepted', []))
    candidates = [row for row in rows if row['identity']['spec_index'] == 33
                  and Path(row['trial_dir']).resolve() == trial.resolve()]
    if candidates:
        assert len(candidates) == 1
        matched = (list_path, list_bytes, current, candidates[0])
        break
assert matched is not None, 'Exact33 independent formal admission is required'
list_path, list_bytes, formal_list, row = matched
formal_path = Path(row['audit_receipt_path'])
assert digest(formal_path) == row['audit_receipt_sha256']
formal = json.loads(formal_path.read_bytes())
assert formal['passed'] is True and formal['readback_only'] is True
assert formal['synthetic_fixture_only'] is False and formal['new_physics_runs'] == 0
assert Path(formal['trial_dir']).resolve() == trial.resolve()
identity = formal['identity']
assert (identity['spec_index'], identity['seed'], identity['model_id'], identity['arm'],
        identity['pose_class']) == (33, 2100033, 13, 'right', 'upright')
assert digest(trial / 'expert_report.json') == formal['source_files']['expert_report.json']['sha256']
report = json.loads((trial / 'expert_report.json').read_bytes())
assert report['accepted'] is True and report['physical_steps'] == 3780
assert report['canonical_sources_unchanged'] is True
anchor = report['sample_spec']
assert anchor['grasp_setting']['feedback_force_gain'] == .1
assert anchor['grasp_setting']['hand_x'] == .02 and anchor['grasp_setting']['close_sdk'][4] == 887
control_sha = hashlib.sha256(json.dumps(anchor['grasp_setting'], sort_keys=True,
                                     allow_nan=False).encode()).hexdigest()
plan_path = work / 'sample_plan.json'
plan = json.loads(plan_path.read_bytes())
seen_seeds = {entry['identity']['seed'] for entry in formal_list.get('trials', formal_list.get('accepted', []))}
for spec_path in work.rglob('sample_spec.json'):
    try:
        seen_seeds.add(json.loads(spec_path.read_bytes())['seed'])
    except (json.JSONDecodeError, OSError):
        pass

groups = {1: ((12, 3300012, .0001, -.0001, .01), (12, 3310012, -.0002, .0002, -.05)),
          6: ((31, 3300031, -.0001, -.0001, -.01), (31, 3310031, .0002, .0002, .05))}
outputs = []
for gpu, cells in groups.items():
    jobs = []
    for index, seed, dx, dy, yaw in cells:
        assert seed not in seen_seeds, 'Only completely fresh actual scene seeds allowed'
        seen_seeds.add(seed)
        original = next(item for item in plan['specs'] if item['index'] == index)
        assert (original['model_id'], original['arm'], original['pose_class']) == (13, 'right', 'upright')
        spec = {key: copy.deepcopy(anchor[key]) for key in (
            'model_id', 'brand', 'arm', 'pose_class', 'yaw_semantics', 'position_reference',
            'acceptance_requires_settled_pose_check', 'grasp_setting')}
        spec.update(index=index, seed=seed, xy=[anchor['xy'][0] + dx, anchor['xy'][1] + dy],
                    yaw_deg=anchor['yaw_deg'] + yaw,
                    candidate_provenance={
                        'scene_profile_anchor_trial': str(trial.resolve()),
                        'anchor_independent_formal_readback_receipt': str(formal_path.resolve()),
                        'anchor_independent_formal_readback_receipt_sha256': digest(formal_path),
                        'anchor_report_sha256': digest(trial / 'expert_report.json'),
                        'anchor_full_grasp_setting_sha256': control_sha,
                        'formal_list_snapshot_sha256': hashlib.sha256(list_bytes).hexdigest(),
                        'original_sample_plan_sha256': digest(plan_path),
                        'original_sampling_cell_spec': original,
                        'new_position_delta_m': [dx, dy], 'new_upright_label_yaw_delta_deg': yaw,
                        'all_grasp_control_inputs_unchanged': True,
                        'expert_module': 'expert_generation_v9_equivalent_dense_waypoint_v8',
                        'all_direct_first_native_TOPP_FMA_and_actual_physics_guards_unchanged': True,
                        'nearest_arm_requires_actual_settled_scene_evidence': True,
                        'fresh_native_scene_paths_and_full_three_RGB_required': True,
                        'trajectory_and_dataset_not_copied': True,
                        'cancel_redundant_group_after_independent_formal_slot_qualifies': True,
                        'this_job_plan_is_not_an_accepted_episode': True})
        assert spec['grasp_setting'] == anchor['grasp_setting']
        jobs.append({'name': f'slot{index:07d}_seed{seed}_coke_right_upright_formal33_micro_v8', 'spec': spec})
    output = work / f'coke_right_upright_formal33_micro_parallel_v8_gpu{gpu}_jobs.json'
    with output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(jobs, indent=2, allow_nan=False) + '\n')
    outputs.append({'gpu': gpu, 'output': str(output.resolve()), 'output_sha256': digest(output),
                    'jobs': [{'name': job['name'], 'index': job['spec']['index'],
                              'seed': job['spec']['seed'], 'xy': job['spec']['xy'],
                              'yaw_deg': job['spec']['yaw_deg']} for job in jobs]})
print(json.dumps({'scope': 'Four fresh scene specifications; no new physics success inferred',
                  'anchor_formal_receipt': str(formal_path.resolve()),
                  'anchor_formal_receipt_sha256': digest(formal_path), 'outputs': outputs}, indent=2), flush=True)
