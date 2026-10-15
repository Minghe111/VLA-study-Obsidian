"""Threshold separation on actual saved original numbers; no new physics run."""
from pathlib import Path
import csv
import hashlib
import json

from motion_gate import load_numeric, numerical_motion, load_json, CONFIG

SOURCE = Path(__file__).resolve().parents[2] / '2026-10-13-RealManRGB50专家轨迹质量核查'
OUT = Path(__file__).resolve().parent
REPLACEMENTS = [17, 18, 20, 29, 34]
SHORT_UPRIGHT_REFERENCES = [0, 2, 13, 14, 25, 48]
ADDITIONAL_MATCHED_ABNORMAL = [3, 16]


def main():
    cfg = load_json(CONFIG)
    receipt = load_json(SOURCE / 'numeric_receipt.json')
    metadata = {row['index']: row for row in receipt['episodes']}
    results, flat = [], []
    for index in REPLACEMENTS + SHORT_UPRIGHT_REFERENCES + ADDITIONAL_MATCHED_ABNORMAL:
        row = metadata[index]
        file = SOURCE / row['numeric_file']
        assert hashlib.sha256(file.read_bytes()).hexdigest() == row['numeric_file_sha256']
        result = numerical_motion(load_numeric(file, row['arm']), row, cfg)
        window = result['metrics']['before_substantial_commanded_closure']
        group = ('original_five_to_rerecord' if index in REPLACEMENTS else
                 'short_trajectory_upright_reference' if index in SHORT_UPRIGHT_REFERENCES else
                 'same_brand_arm_upright_but_abnormal_reference')
        item = {'index': index, 'brand': row['brand'], 'arm': row['arm'], 'pose_class': row['pose_class'],
                'group': group, 'source_hdf_sha256': row['source_hdf_sha256'],
                'numeric_sha256': row['numeric_file_sha256'], 'motion_only': result,
                'new_full_publish_gate_evaluated': False, 'actual_new_planner_branch_receipt_exists': False,
                'original_manifest_physical_success': row['physical_success']}
        results.append(item)
        actual, command, cam = window['actual_joints'], window['command_joints'], window['wrist_center_world_model_matrix']
        flat.append({'index': index, 'brand': row['brand'], 'arm': row['arm'], 'pose_class': row['pose_class'],
                     'group': group, 'motion_only_passed': result['passed'],
                     'maximum_actual_joint_range_deg': actual['maximum_range_deg'],
                     'maximum_command_joint_range_deg': command['maximum_range_deg'],
                     'maximum_actual_joint_total_variation_deg': actual['maximum_total_variation_deg'],
                     'maximum_command_joint_total_variation_deg': command['maximum_total_variation_deg'],
                     'maximum_actual4ms_step_deg': actual['maximum_raw_step_deg'],
                     'maximum_command4ms_step_deg': command['maximum_raw_step_deg'],
                     'maximum_command60ms_step_deg': window['next_command_joints']['maximum_raw_step_deg'],
                     'wrist_center_path_m': cam['path_m'], 'wrist_straight_displacement_m': cam['straight_displacement_m'],
                     'wrist_path_to_straight_ratio': cam['path_to_straight_ratio'],
                     'ee_path_report_only_m': window['legacy_ee_position_report_only']['path_m'],
                     'first_sdk_closure_time_s_not_contact': result['first_substantial_commanded_closure']['time_s'],
                     'failures': ';'.join(result['failures'])})
    summary = {'scope': 'Actual CPU replay of independent numerical motion gate against saved original13 numeric cases, not re-simulation or new-data acceptance',
               'thresholds_confirmed': True, 'config_sha256': hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
               'old_five_motion_rejected': sum(not x['motion_only']['passed'] for x in results if x['index'] in REPLACEMENTS),
               'old_five_count': len(REPLACEMENTS),
               'short_upright_reference_motion_passed': sum(x['motion_only']['passed'] for x in results if x['index'] in SHORT_UPRIGHT_REFERENCES),
               'short_upright_reference_count': len(SHORT_UPRIGHT_REFERENCES),
               'other_abnormal_matched_rejected': sum(not x['motion_only']['passed'] for x in results if x['index'] in ADDITIONAL_MATCHED_ABNORMAL),
               'matching_limits': {
                   'exact_normal_CocaCola_left_upright_reference': [2],
                   'Fanta_left_other_upright_normal_reference_in_original50': [],
                   'Fanta_right_other_upright_reference': [3],
                   'Fanta_right_reference3_is_abnormal_not_normal': True,
                   'other_short_references_are_same_upright_pose_and_same_or_mirrored_arm_but_cross_brand': [0, 13, 14, 25, 48]},
               'sdk_boundary_is_not_actual_first_contact': True,
               'new_five_scene_profile_is_demo_randomized_not_old_initial_scene_restoration': True,
               'original_datasets_models_and_remote_not_modified': True,
               'new_planner_evidence_and_physical_clutter_success_still_required': True}
    result = {'summary': summary, 'episodes': results}
    (OUT / 'old5_and_reference_motion_comparison.json').write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
    with (OUT / 'old5_and_reference_motion_comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    assert summary['old_five_motion_rejected'] == 5
    assert summary['short_upright_reference_motion_passed'] == 6
    assert summary['other_abnormal_matched_rejected'] == 2
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
