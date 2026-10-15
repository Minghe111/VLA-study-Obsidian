"""Prepare isolated thumb-close candidates only after the six baseline failures.

No physical replay is performed. Baseline failures remain immutable. Each new
candidate reuses a failed scene specification and changes only close_sdk[4].
Separate native replanning and full actual physics/RGB acceptance are required.
"""
from pathlib import Path
import copy
import hashlib
import json

work = Path(__file__).resolve().parent
micro_root = work / 'batch_candidates/v9_coke_right_upright_anchor_micro_neighbours_v7_attempt02'
status_path = micro_root / 'worker_status.json'
status = json.loads(status_path.read_bytes())
assert status.get('worker_ended') is True
assert status['jobs_total'] == 6 and status['finished'] == 6
assert len(status['receipts']) == 6 and all(receipt.get('accepted') is False for receipt in status['receipts'])
baseline_root = work / 'batch_candidates/v9_coke_right_upright_anchor_neighbours_v7_attempt01'
jobs_path = work / 'coke_right_upright_anchor_neighbours_v7_jobs.json'
baselines = json.loads(jobs_path.read_bytes())
assert len(baselines) == 3
jobs = []
for baseline_job in baselines:
    baseline = baseline_root / baseline_job['name']
    report_path = baseline / 'expert_report.json'
    trace_path = baseline / 'physics_trace.json'
    report = json.loads(report_path.read_bytes())
    assert report.get('accepted') is False and report.get('error')
    assert report.get('canonical_sources_unchanged') is True
    original_spec = report['sample_spec']
    assert original_spec['model_id'] == 13 and original_spec['arm'] == 'right' and original_spec['pose_class'] == 'upright'
    assert original_spec['grasp_setting']['close_sdk'][4] == 887
    assert original_spec['grasp_setting']['hand_x'] == .02
    spec = copy.deepcopy(original_spec)
    spec['grasp_setting']['close_sdk'][4] = 850
    for key in ('index', 'seed', 'xy', 'yaw_deg'):
        assert spec[key] == original_spec[key]
    expected_setting = copy.deepcopy(original_spec['grasp_setting'])
    expected_setting['close_sdk'][4] = 850
    assert spec['grasp_setting'] == expected_setting
    spec['candidate_provenance'] = {
        'matched_original_failed_trial': str(baseline.resolve()),
        'original_failed_report_sha256': hashlib.sha256(report_path.read_bytes()).hexdigest(),
        'original_failed_trace_sha256': hashlib.sha256(trace_path.read_bytes()).hexdigest(),
        'completed_six_baseline_worker_status_sha256': hashlib.sha256(status_path.read_bytes()).hexdigest(),
        'input_parameter_delta_only': {'grasp_setting.close_sdk[4]': {'before': 887, 'after': 850}},
        'same_scene_spec_and_seed_fresh_setup_required': True,
        'all_LP_force_rate_five_finger_timing_and_acceptance_guards_unchanged': True,
        'new_native_planning_can_change_actual_paths': True,
        'original_failed_trial_not_modified_or_relabelled': True,
        'this_specification_is_not_a_physical_success_or_causal_improvement_claim': True,
        'previous_baseline_provenance': copy.deepcopy(original_spec.get('candidate_provenance')),
    }
    jobs.append({'name': baseline_job['name'] + '_thumb850_matched', 'spec': spec})
output = work / 'coke_right_upright_thumb850_matched_candidates_v7_jobs.json'
with output.open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(jobs, indent=2, allow_nan=False) + '\n')
print(json.dumps({'scope': 'Matched candidate specifications only; all six baseline failures retained',
                  'output': str(output), 'indices': [job['spec']['index'] for job in jobs],
                  'seeds_reused_from_failed_nonpublished_baselines': [job['spec']['seed'] for job in jobs],
                  'only_control_input_change': 'thumb close SDK 887 -> 850'}, indent=2), flush=True)
