"""Prepare the separately authorized gain-.1 group alongside the micro queue.

Only complete, rejected original baselines are used. The separate active micro
queue is neither edited nor waited on. No success or trajectory is copied.
"""
from pathlib import Path
import copy
import hashlib
import json

work = Path(__file__).resolve().parent
baseline_root = work / 'batch_candidates/v9_coke_right_upright_anchor_neighbours_v7_attempt01'
status_path = baseline_root / 'worker_status.json'
status = json.loads(status_path.read_bytes())
assert status.get('worker_ended') is True
assert status['jobs_total'] == 3 and status['finished'] == 3
assert len(status['receipts']) == 3 and all(receipt.get('accepted') is False for receipt in status['receipts'])
jobs_path = work / 'coke_right_upright_anchor_neighbours_v7_jobs.json'
original_jobs = json.loads(jobs_path.read_bytes())
assert {job['spec']['index'] for job in original_jobs} == {12, 31, 33}
jobs = []
# The completed hold-slip baseline is first; this changes no control input.
for index in (33, 12, 31):
    baseline_job = next(job for job in original_jobs if job['spec']['index'] == index)
    baseline = baseline_root / baseline_job['name']
    report_path = baseline / 'expert_report.json'
    trace_path = baseline / 'physics_trace.json'
    report = json.loads(report_path.read_bytes())
    assert report.get('accepted') is False and report.get('error')
    assert report.get('canonical_sources_unchanged') is True
    original_spec = report['sample_spec']
    assert (original_spec['model_id'], original_spec['arm'], original_spec['pose_class']) == (13, 'right', 'upright')
    assert original_spec['grasp_setting'].get('feedback_force_gain', .3) == .3
    assert original_spec['grasp_setting']['close_sdk'][4] == 887
    spec = copy.deepcopy(original_spec)
    spec['grasp_setting']['feedback_force_gain'] = .1
    for key in ('index', 'seed', 'xy', 'yaw_deg'):
        assert spec[key] == original_spec[key]
    expected_setting = copy.deepcopy(original_spec['grasp_setting'])
    expected_setting['feedback_force_gain'] = .1
    assert spec['grasp_setting'] == expected_setting
    spec['candidate_provenance'] = {
        'matched_original_failed_trial': str(baseline.resolve()),
        'original_failed_report_sha256': hashlib.sha256(report_path.read_bytes()).hexdigest(),
        'original_failed_trace_sha256': hashlib.sha256(trace_path.read_bytes()).hexdigest(),
        'completed_original_failed_worker_status_sha256': hashlib.sha256(status_path.read_bytes()).hexdigest(),
        'baseline_jobs_sha256': hashlib.sha256(jobs_path.read_bytes()).hexdigest(),
        'input_parameter_delta_only': {'grasp_setting.feedback_force_gain': {'before_effective_default': .3, 'after': .1}},
        'same_scene_spec_and_seed_fresh_setup_required': True,
        'LP_period_force_caps_rate_cap_five_finger_timing_skin_and_acceptance_guards_unchanged': True,
        'new_native_planning_can_change_actual_paths': True,
        'original_failed_trial_not_modified_or_relabelled': True,
        'separate_active_micro_queue_not_modified': True,
        'this_specification_is_not_a_physical_success_or_causal_improvement_claim': True,
        'previous_baseline_provenance': copy.deepcopy(original_spec.get('candidate_provenance')),
    }
    jobs.append({'name': baseline_job['name'] + '_gain01_matched_parallel', 'spec': spec})
output = work / 'coke_right_upright_gain01_matched_parallel_v7_jobs.json'
with output.open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(jobs, indent=2, allow_nan=False) + '\n')
print(json.dumps({'scope': 'Isolated gain input specifications only; fresh actual validation remains required',
                  'output': str(output), 'indices': [job['spec']['index'] for job in jobs],
                  'seeds_reused_from_failed_nonpublished_baselines': [job['spec']['seed'] for job in jobs],
                  'only_control_input_change': 'feedback force gain effective default .3 -> .1'}, indent=2), flush=True)
