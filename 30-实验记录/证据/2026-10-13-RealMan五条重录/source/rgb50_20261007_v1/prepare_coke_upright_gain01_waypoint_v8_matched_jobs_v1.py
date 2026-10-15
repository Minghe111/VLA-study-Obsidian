"""Prepare new path-adapter candidates matched to completed gain-.1 failures.

All scene/control specification inputs are identical to the rejected baselines.
The selected expert module is changed separately by the worker CLI. Original
rejections remain immutable; fresh native paths and actual acceptance are needed.
"""
from pathlib import Path
import copy
import hashlib
import json

work = Path(__file__).resolve().parent
baseline_root = work / 'batch_candidates/v9_coke_right_upright_gain01_matched_parallel_v7_attempt03'
status_path = baseline_root / 'worker_status.json'
status = json.loads(status_path.read_bytes())
assert status.get('worker_ended') is True and status['jobs_total'] == status['finished'] == 3
assert len(status['receipts']) == 3 and all(receipt.get('accepted') is False for receipt in status['receipts'])
baseline_jobs_path = work / 'coke_right_upright_gain01_matched_parallel_v7_jobs.json'
baselines = json.loads(baseline_jobs_path.read_bytes())
assert {job['spec']['index'] for job in baselines} == {12, 31, 33}
jobs = []
for baseline_job in baselines:
    baseline = baseline_root / baseline_job['name']
    report_path, trace_path = baseline / 'expert_report.json', baseline / 'physics_trace.json'
    report = json.loads(report_path.read_bytes())
    assert report.get('accepted') is False and report.get('error')
    assert report.get('canonical_sources_unchanged') is True
    original_spec = report['sample_spec']
    assert (original_spec['model_id'], original_spec['arm'], original_spec['pose_class']) == (13, 'right', 'upright')
    assert original_spec['grasp_setting']['feedback_force_gain'] == .1
    assert original_spec['grasp_setting']['close_sdk'][4] == 887
    assert original_spec['grasp_setting']['hand_x'] == .02
    spec = copy.deepcopy(original_spec)
    for key in ('index', 'seed', 'xy', 'yaw_deg', 'grasp_setting'):
        assert spec[key] == original_spec[key]
    spec['candidate_provenance'] = {
        'matched_original_failed_v7_gain01_trial': str(baseline.resolve()),
        'original_failed_report_sha256': hashlib.sha256(report_path.read_bytes()).hexdigest(),
        'original_failed_trace_sha256': hashlib.sha256(trace_path.read_bytes()).hexdigest(),
        'completed_original_failed_worker_status_sha256': hashlib.sha256(status_path.read_bytes()).hexdigest(),
        'baseline_jobs_sha256': hashlib.sha256(baseline_jobs_path.read_bytes()).hexdigest(),
        'all_scene_and_control_specification_inputs_unchanged': True,
        'selected_expert_only_change': {'before': 'expert_generation_v9_equivalent_goals_dense_v7',
                                       'after': 'expert_generation_v9_equivalent_dense_waypoint_v8'},
        'original_frozen_v7_direct_attempt_retained_before_optional_waypoint': True,
        'waypoint_world_z_m': .08,
        'all_raw_numeric_limit_TOPP_FMA_and_actual_physics_guards_retained': True,
        'new_native_planning_can_change_actual_paths': True,
        'direct_success_cannot_be_attributed_to_waypoint_fallback': True,
        'fresh_full_three_RGB_physics_and_independent_formal_admission_required': True,
        'original_failed_trial_not_modified_or_relabelled': True,
        'this_specification_is_not_a_physical_success_or_causal_improvement_claim': True,
        'previous_baseline_provenance': copy.deepcopy(original_spec.get('candidate_provenance')),
    }
    jobs.append({'name': baseline_job['name'] + '_waypoint_v8', 'spec': spec})
output = work / 'coke_right_upright_gain01_waypoint_v8_matched_jobs.json'
with output.open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(jobs, indent=2, allow_nan=False) + '\n')
print(json.dumps({'scope': 'Same scene/control specifications for a separately reviewed path candidate; no physical success',
                  'output': str(output), 'indices': [job['spec']['index'] for job in jobs],
                  'seeds_reused_from_failed_nonpublished_baselines': [job['spec']['seed'] for job in jobs]}, indent=2), flush=True)
