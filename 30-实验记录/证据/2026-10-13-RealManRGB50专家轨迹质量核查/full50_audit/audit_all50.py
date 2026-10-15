"""All50 original expert motion audit. No simulation, model, or source mutation."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path('/tmp/realman-data-motion-audit-20261013')
OUT = ROOT / 'full50_audit'
DEG = 180. / np.pi
CLOSE_SDK = 950.
ARM_BASE = {'left': 0, 'right': 12}
FULL36_ARM = {'left': [1, 3, 5, 7, 9, 11], 'right': [0, 2, 4, 6, 8, 10]}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def scalar(value):
    return float(value)


def max_monotonic_sweep(values, reversal_tolerance=1e-5):
    """Signed-step runs; tiny reversal noise is ignored, with tolerance reported."""
    if len(values) < 2:
        return 0.
    direction, start, best = 0., float(values[0]), 0.
    for i, delta in enumerate(np.diff(values)):
        if abs(delta) <= reversal_tolerance:
            continue
        new = float(np.sign(delta))
        if direction and new != direction:
            best = max(best, abs(float(values[i]) - start))
            start = float(values[i])
        direction = new
    return max(best, abs(float(values[-1]) - start))


def joint_metrics(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return None
    unwrapped = np.unwrap(values)
    raw_delta, delta = np.diff(values), np.diff(unwrapped)
    monotonic = max_monotonic_sweep(unwrapped)
    return {
        'samples': len(values),
        'raw_min_rad': scalar(values.min()), 'raw_max_rad': scalar(values.max()),
        'raw_range_deg': scalar(np.ptp(values) * DEG),
        'unwrapped_min_rad': scalar(unwrapped.min()), 'unwrapped_max_rad': scalar(unwrapped.max()),
        'unwrapped_range_deg': scalar(np.ptp(unwrapped) * DEG),
        'raw_net_deg': scalar((values[-1] - values[0]) * DEG),
        'unwrapped_net_deg': scalar((unwrapped[-1] - unwrapped[0]) * DEG),
        'raw_total_variation_deg': scalar(np.abs(raw_delta).sum() * DEG),
        'unwrapped_total_variation_deg': scalar(np.abs(delta).sum() * DEG),
        'positive_travel_deg': scalar(delta[delta > 0].sum() * DEG),
        'negative_travel_deg': scalar(-delta[delta < 0].sum() * DEG),
        'maximum_raw_step_deg': scalar(np.max(np.abs(raw_delta)) * DEG) if len(delta) else 0.,
        'maximum_unwrapped_step_deg': scalar(np.max(np.abs(delta)) * DEG) if len(delta) else 0.,
        'raw_step_over_pi_count': int(np.sum(np.abs(raw_delta) > np.pi)),
        'near_2pi_jump_count': int(np.sum((np.abs(np.abs(raw_delta) - 2 * np.pi) < .1)
                                         & (np.abs(raw_delta) > np.pi))),
        'unwrapping_adjusted_steps': int(np.sum(np.abs(raw_delta - delta) > 1e-6)),
        'maximum_monotonic_sweep_deg': scalar(monotonic * DEG),
        'continuous_coordinate_range_ge360': bool(np.ptp(unwrapped) >= 2 * np.pi),
        'monotonic_sweep_ge360': bool(monotonic >= 2 * np.pi),
        'total_variation_ge360_does_not_by_itself_prove_full_turn': bool(np.abs(delta).sum() >= 2 * np.pi),
    }


def position_metrics(points):
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return None
    delta = np.diff(points, axis=0)
    length = float(np.linalg.norm(delta, axis=1).sum())
    straight = float(np.linalg.norm(points[-1] - points[0]))
    return {'samples': len(points), 'path_m': length, 'straight_displacement_m': straight,
            'path_to_straight_ratio': length / straight if straight > 1e-6 else None,
            'maximum_distance_from_start_m': float(np.linalg.norm(points - points[0], axis=1).max()),
            'maximum_sample_step_m': float(np.linalg.norm(delta, axis=1).max()) if len(delta) else 0.,
            'start_m': points[0].tolist(), 'end_m': points[-1].tolist(),
            'coordinate_span_m': np.ptp(points, axis=0).tolist()}


def so3_metrics(matrices):
    """Unsigned SO(3) path length; does not establish fixed-axis winding."""
    matrices = np.asarray(matrices, dtype=np.float64)
    if len(matrices) == 0:
        return None
    raw = matrices[:, :3, :3]
    ortho = np.matmul(np.swapaxes(raw, 1, 2), raw) - np.eye(3)[None]
    determinant = np.linalg.det(raw)
    u, _, vh = np.linalg.svd(raw)
    corrected = u @ vh
    wrong = np.linalg.det(corrected) < 0
    if wrong.any():
        u[wrong, :, -1] *= -1
        corrected = u @ vh
    relative = np.swapaxes(corrected[:-1], 1, 2) @ corrected[1:]
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1, 1)
    angles = np.arccos(cosine)
    net = corrected[0].T @ corrected[-1]
    net_angle = np.arccos(np.clip((np.trace(net) - 1) / 2, -1, 1))
    return {'samples': len(matrices), 'unsigned_path_length_deg': float(angles.sum() * DEG),
            'first_last_shortest_angle_deg': float(net_angle * DEG),
            'maximum_observation_increment_deg': float(angles.max() * DEG) if len(angles) else 0.,
            'path_ge360_does_not_prove_same_axis_turn': bool(angles.sum() >= 2 * np.pi),
            'same_fixed_world_axis_full_turn_claimed': False,
            'maximum_rotation_orthogonality_residual': float(np.abs(ortho).max()),
            'maximum_abs_determinant_minus_one': float(np.abs(determinant - 1).max()),
            'maximum_svd_projection_correction': float(np.abs(corrected - raw).max())}


def nearest_indices(query, times):
    right = np.clip(np.searchsorted(times, query), 0, len(times) - 1)
    left = np.clip(right - 1, 0, len(times) - 1)
    choose_left = np.abs(times[left] - query) < np.abs(times[right] - query)
    return np.where(choose_left, left, right)


def first_closure(applied, times, base):
    # First five flexion channels, excluding thumb opposition/rotation (channel six).
    mask = np.any(applied[:, base + 6:base + 11] < CLOSE_SDK, axis=1)
    hits = np.flatnonzero(mask)
    if not len(hits):
        return {'command_time_s': None, 'command_row': None, 'found': False,
                'already_closed_at_first_command': False}
    index = int(hits[0])
    hand = applied[index, base + 6:base + 11]
    return {'command_time_s': float(times[index]), 'command_row': index, 'found': True,
            'already_closed_at_first_command': index == 0,
            'flexion_channels_below_threshold_1based': (np.flatnonzero(hand < CLOSE_SDK) + 1).tolist(),
            'first_closing_command_flexion_sdk': hand.tolist(),
            'threshold_sdk': CLOSE_SDK, 'thumb_rotation_channel_excluded': True,
            'actual_contact_time_claimed': False}


def first_360_excursion(values, times):
    """Earliest interval whose endpoint changes by>=2pi; may include reversals."""
    values = np.asarray(values, dtype=np.float64)
    lo = hi = 0
    for i in range(1, len(values)):
        if values[i] - values[lo] >= 2 * np.pi:
            start, direction = lo, 'positive'
        elif values[hi] - values[i] >= 2 * np.pi:
            start, direction = hi, 'negative'
        else:
            if values[i] < values[lo]:
                lo = i
            if values[i] > values[hi]:
                hi = i
            continue
        delta = np.diff(values[start:i + 1])
        opposed = -delta[delta < 0].sum() if direction == 'positive' else delta[delta > 0].sum()
        return {'start_row': int(start), 'end_row': int(i),
                'start_time_s': float(times[start]), 'end_time_s': float(times[i]),
                'elapsed_time_s': float(times[i] - times[start]), 'direction': direction,
                'start_rad': float(values[start]), 'end_rad': float(values[i]),
                'net_change_deg': float((values[i] - values[start]) * DEG),
                'unsigned_joint_path_deg': float(np.abs(delta).sum() * DEG),
                'opposed_direction_travel_deg': float(abs(opposed) * DEG),
                'maximum4ms_step_deg': float(np.abs(delta).max() * DEG),
                'strictly_monotonic_claimed': False}
    return None


def csv_write(path, rows):
    if not rows:
        return
    keys = sorted(set().union(*(row.keys() for row in rows)))
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def flatten_metrics(prefix, metrics):
    return {prefix + key: value for key, value in (metrics or {}).items()
            if isinstance(value, (float, int, bool)) or value is None}


def main():
    OUT.mkdir(exist_ok=True)
    receipt_path = ROOT / 'numeric_receipt.json'
    receipt = json.loads(receipt_path.read_text())
    assert sorted(e['index'] for e in receipt['episodes']) == list(range(50))
    joint_rows, arm_rows, episodes, integrity, large_q6 = [], [], [], [], []
    for source in receipt['episodes']:
        index = source['index']
        path = ROOT / source['numeric_file']
        before_stat = path.stat()
        assert sha(path) == source['numeric_file_sha256']
        with np.load(path, allow_pickle=False) as packed:
            arrays = {key: packed[key] for key in packed.files}
        for key, value in arrays.items():
            actual_array_sha = hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
            expected = source['fields'][key]
            assert actual_array_sha == expected['array_sha256']
            assert list(value.shape) == expected['shape'] and str(value.dtype) == expected['dtype']
            assert np.isfinite(value).all()
        measured_before = arrays['realman_dexterous_physics/measured_state24_before'].astype(np.float64)
        measured_after = arrays['realman_dexterous_physics/measured_state24_after'].astype(np.float64)
        dense = np.vstack((measured_before[:1], measured_after))
        tb = arrays['realman_dexterous_physics/sim_time_before_s'].astype(np.float64)
        ta = arrays['realman_dexterous_physics/sim_time_after_s'].astype(np.float64)
        td = np.r_[tb[:1], ta]
        applied = arrays['realman_dexterous_physics/applied_command24'].astype(np.float64)
        low_state = arrays['realman_dexterous/measured_state24'].astype(np.float64)
        next_command = arrays['realman_dexterous/next_command24'].astype(np.float64)
        tl = arrays['realman_dexterous/sample_time_s'].astype(np.float64)
        tn = arrays['realman_dexterous/next_sample_time_s'].astype(np.float64)
        low_to_dense = nearest_indices(tl, tb)
        physical_steps = arrays['realman_dexterous_physics/physics_step']
        assert np.all(np.diff(physical_steps) == 1)
        assert np.all(np.diff(tb) > 0) and np.all(np.diff(tl) > 0)
        assert len(applied) == source['physical_steps'] and len(low_state) == source['state_action_pairs']
        checks = {
            'index': index, 'numeric_sha_verified': True, 'all_array_sha_verified': True,
            'maximum_before_after_clock_delta_error_s': float(np.abs(ta - tb - receipt['physics_dt_s']).max()),
            'maximum_physics_clock_increment_error_s': float(np.abs(np.diff(tb) - receipt['physics_dt_s']).max()),
            'maximum_low_clock_increment_error_s': float(np.abs(np.diff(tl) - receipt['observation_dt_s']).max()),
            'maximum_next_time_delta_error_s': float(np.abs(tn - tl - receipt['observation_dt_s']).max()),
            'maximum_adjacent_measured_state24_chain_residual': float(np.abs(measured_before[1:] - measured_after[:-1]).max()),
            'maximum_low_dense_timestamp_residual_s': float(np.abs(tb[low_to_dense] - tl).max()),
        }
        episode = {key: source[key] for key in ['index', 'seed', 'arm', 'brand', 'model_id', 'pose_class',
                                                'state_action_pairs', 'physical_steps', 'physical_success', 'boundary_used']}
        episode.update({'source_hdf': source['source_hdf'], 'source_hdf_sha256': source['source_hdf_sha256'],
                        'numeric_sha256': source['numeric_file_sha256'], 'duration_s': float(ta[-1] - tb[0]),
                        'arms': {}, 'integrity': checks})
        for arm, base in ARM_BASE.items():
            active = arm == source['arm']
            closure = first_closure(applied, tb, base)
            ct = closure['command_time_s']
            masks = {
                'dense_measured': {'full': np.ones(len(td), bool), 'preclose': td <= ct + 1e-10 if ct is not None else np.ones(len(td), bool)},
                'dense_applied_command': {'full': np.ones(len(tb), bool), 'preclose': tb <= ct + 1e-10 if ct is not None else np.ones(len(tb), bool)},
                'low_measured': {'full': np.ones(len(tl), bool), 'preclose': tl <= ct + 1e-10 if ct is not None else np.ones(len(tl), bool)},
                'low_next_command': {'full': np.ones(len(tn), bool), 'preclose': tn <= ct + 1e-10 if ct is not None else np.ones(len(tn), bool)},
            }
            streams = {'dense_measured': dense[:, base:base + 6],
                       'dense_applied_command': applied[:, base:base + 6],
                       'low_measured': low_state[:, base:base + 6],
                       'low_next_command': next_command[:, base:base + 6]}
            times = {'dense_measured': td, 'dense_applied_command': tb, 'low_measured': tl, 'low_next_command': tn}
            arm_stats = {'active_task_arm': active, 'first_substantial_commanded_flexion': closure, 'joint_streams': {}}
            for stream, values in streams.items():
                arm_stats['joint_streams'][stream] = {}
                for stage in ['full', 'preclose']:
                    keep = masks[stream][stage]
                    per_joint = []
                    for j in range(6):
                        metric = joint_metrics(values[keep, j])
                        per_joint.append(metric)
                        if metric is None:
                            continue
                        joint_rows.append({'index': index, 'arm': arm, 'active_arm': active, 'joint_1based': j + 1,
                                           'stream': stream, 'stage': stage, 'first_closure_command_s': ct,
                                           'first_time_s': float(times[stream][keep][0]),
                                           'last_time_s': float(times[stream][keep][-1]), **metric})
                    arm_stats['joint_streams'][stream][stage] = per_joint
            q36before = arrays['realman_dexterous_physics/full36_qpos_before_rad'][:, FULL36_ARM[arm]]
            q36after = arrays['realman_dexterous_physics/full36_qpos_after_rad'][:, FULL36_ARM[arm]]
            matching_error = float(max(np.abs(q36before - measured_before[:, base:base + 6]).max(),
                                       np.abs(q36after - measured_after[:, base:base + 6]).max()))
            arm_stats['full36_arm_column_mapping_verified_numeric_residual_rad'] = matching_error
            checks[arm + '_full36_mapping_residual_rad'] = matching_error
            checks[arm + '_low_vs_dense_state_residual_rad'] = float(np.abs(low_state[:, base:base + 6] - measured_before[low_to_dense, base:base + 6]).max())
            checks[arm + '_low_state_vs_state_arm_alias_residual_rad'] = float(np.abs(low_state[:, base:base + 6] - arrays[f'state/{arm}_arm_joint_states']).max())
            checks[arm + '_next_command_vs_action_arm_alias_residual_rad'] = float(np.abs(next_command[:, base:base + 6] - arrays[f'action/{arm}_arm_joint_states']).max())
            checks[arm + '_first_applied_command_vs_initial_measured_arm_residual_rad'] = float(np.abs(applied[0, base:base + 6] - measured_before[0, base:base + 6]).max())
            wrist = arrays[f'vision/cam_{arm}_wrist/extrinsics_matrix']
            ee = arrays[f'state/{arm}_ee_poses'][:, :3]
            arm_stats['spatial_and_orientation'] = {}
            for stage in ['full', 'preclose']:
                keep = masks['low_measured'][stage]
                position = position_metrics(ee[keep])
                camera_position = position_metrics(wrist[keep, :3, 3])
                orientation = so3_metrics(wrist[keep])
                stage_metric = {'ee_position': position, 'wrist_camera_position': camera_position, 'wrist_camera_so3': orientation,
                                'first_observation_time_s': float(tl[keep][0]) if keep.any() else None,
                                'last_observation_time_s': float(tl[keep][-1]) if keep.any() else None,
                                'preclose_window_exists_before_threshold': bool(ct is not None and ct > td[0]) if stage == 'preclose' else True,
                                'preclose_window_last_observation_undershoot_s': float(ct - tl[keep][-1]) if stage == 'preclose' and ct is not None and keep.any() else None}
                arm_stats['spatial_and_orientation'][stage] = stage_metric
                dense_joint = arm_stats['joint_streams']['dense_measured'][stage]
                valid_joint = [j for j in dense_joint if j is not None]
                record = {'index': index, 'arm': arm, 'active_arm': active, 'brand': source['brand'],
                          'pose_class': source['pose_class'], 'stage': stage, 'first_closure_command_s': ct,
                          'maximum_single_joint_range_deg': max(j['unwrapped_range_deg'] for j in valid_joint),
                          'maximum_single_joint_total_variation_deg': max(j['unwrapped_total_variation_deg'] for j in valid_joint),
                          'maximum_single_joint_monotonic_sweep_deg': max(j['maximum_monotonic_sweep_deg'] for j in valid_joint),
                          **flatten_metrics('ee_', position), **flatten_metrics('camera_position_', camera_position),
                          **flatten_metrics('wrist_so3_', orientation)}
                arm_rows.append(record)
            episode['arms'][arm] = arm_stats
            full_q6 = arm_stats['joint_streams']['dense_measured']['full'][5]
            if active and full_q6['continuous_coordinate_range_ge360']:
                q = dense[:, base + 5]
                qu = np.unwrap(q)
                qcmd = applied[:, base + 5]
                qpre = arm_stats['joint_streams']['dense_measured']['preclose'][5]
                minimum, maximum = int(np.argmin(qu)), int(np.argmax(qu))
                command_metric = arm_stats['joint_streams']['dense_applied_command']['full'][5]
                crossing = first_360_excursion(qu, td)
                large_q6.append({
                    'index': index, 'seed': source['seed'], 'brand': source['brand'], 'arm': arm,
                    'pose_class': source['pose_class'], 'source_hdf_sha256': source['source_hdf_sha256'],
                    'first_substantial_commanded_closure_time_s': ct,
                    'closure_threshold_sdk': CLOSE_SDK,
                    'closure_threshold_is_not_physical_contact': True,
                    'physical_duration_s': float(td[-1] - td[0]),
                    'actual_initial_q6_rad': float(qu[0]), 'actual_final_q6_rad': float(qu[-1]),
                    'actual_min_q6_rad': float(qu[minimum]), 'actual_min_q6_time_s': float(td[minimum]),
                    'actual_max_q6_rad': float(qu[maximum]), 'actual_max_q6_time_s': float(td[maximum]),
                    'actual_full_range_deg': full_q6['unwrapped_range_deg'],
                    'actual_full_net_change_deg': full_q6['unwrapped_net_deg'],
                    'actual_full_total_variation_deg': full_q6['unwrapped_total_variation_deg'],
                    'actual_maximum_monotonic_sweep_deg': full_q6['maximum_monotonic_sweep_deg'],
                    'actual_preclose_range_deg': qpre['unwrapped_range_deg'],
                    'actual_preclose_net_change_deg': qpre['unwrapped_net_deg'],
                    'first360_excursion': crossing,
                    'first360_excursion_ends_before_commanded_closure': bool(crossing is not None and ct is not None and crossing['end_time_s'] <= ct),
                    'actual_maximum4ms_step_deg': full_q6['maximum_raw_step_deg'],
                    'actual_unwrapping_adjusted_steps': full_q6['unwrapping_adjusted_steps'],
                    'command_full_range_deg': command_metric['raw_range_deg'],
                    'command_maximum4ms_step_deg': command_metric['maximum_raw_step_deg'],
                    'command_near2pi_jump_count': command_metric['near_2pi_jump_count'],
                    'maximum_after_step_q6_command_tracking_residual_deg': float(np.abs(measured_after[:, base + 5] - qcmd).max() * DEG),
                    'same_fixed_world_wrist_axis_full_turn_claimed': False,
                })
        after_stat = path.stat()
        assert (before_stat.st_size, before_stat.st_mtime_ns) == (after_stat.st_size, after_stat.st_mtime_ns)
        checks['numeric_source_unchanged'] = True
        episodes.append(episode)
        integrity.append(checks)
    actual = [x for x in joint_rows if x['stream'] == 'dense_measured' and x['stage'] == 'full' and x['active_arm']]
    preactual = [x for x in joint_rows if x['stream'] == 'dense_measured' and x['stage'] == 'preclose' and x['active_arm']]
    commands = [x for x in joint_rows if x['stream'] == 'dense_applied_command' and x['stage'] == 'full' and x['active_arm']]
    lowcmd = [x for x in joint_rows if x['stream'] == 'low_next_command' and x['stage'] == 'full' and x['active_arm']]
    full_arm = [x for x in arm_rows if x['active_arm'] and x['stage'] == 'full']
    pre_arm = [x for x in arm_rows if x['active_arm'] and x['stage'] == 'preclose']
    threshold_cases = lambda rows, key, threshold: [x for x in rows if x[key] >= threshold]
    summary = {
        'source_scope': 'All50 formal original expert HDF numeric arrays, verified against original source manifest SHA',
        'source_receipt_sha256': sha(receipt_path), 'source_manifest_sha256': receipt['source_manifest_sha256'],
        'episode_count': len(episodes), 'actual_physics_steps': sum(e['physical_steps'] for e in episodes),
        'low_frequency_observations': sum(e['state_action_pairs'] for e in episodes),
        'actual_simulator_or_model_run': False, 'hardware_io': False, 'source_files_modified': False,
        'metric_contract': {
            'arm_columns_in_state24': {'left': [0, 1, 2, 3, 4, 5], 'right': [12, 13, 14, 15, 16, 17]},
            'full36_arm_columns_numerically_verified': FULL36_ARM,
            'joint_angle_unit': 'rad; reported derived angles in degrees',
            'dense_state_path': 'initial measured_before[0] concatenated with every measured_after',
            'preclose_definition': 'Before/on first applied SDK command <950 in any first-five flexion channels; excludes thumb rotation; not first physical contact',
            'preclose_threshold_sdk': CLOSE_SDK, 'sdk_closure_drop_relative_to_nominal_open': 50,
            'command_jump_near_2pi_tolerance_rad': .1,
            'unwrap_discontinuity_rad': np.pi,
            'command_unwrapping_is_descriptive_not_a_change_to_executed_targets': True,
            'monotonic_run_reversal_tolerance_rad': 1e-5,
            'same_fixed_world_axis_full_turn_inferred_from_so3_path': False,
            'so3_path': 'Sum of unsigned nearest SO(3) increments at actual60ms camera observations; distinct from same-axis winding',
            'so3_end_angle': 'First-last shortest SO(3) distance is bounded0..180deg; does not count whole turns',
            'ee_and_camera_position_paths': 'Actual saved60ms pose/matrix translation samples, no interpolation/FK or image-generated estimates',
            'ee_pose_semantics': 'RoboTwin corrected legacy world EE frame from collection adapter, including its fixed local+0.06m convention; not calibrated real-robot TCP',
            'matrix_translation_semantics': 'Camera-to-world model matrix as specified by collection provenance; matrix header extrinsics alone does not prove this convention',
            'legacy_state_arm_aliases_not_used_for_main_angle_results': True,
        },
        'counts': {
            'active_single_joint_coordinate_range_ge360': sum(x['continuous_coordinate_range_ge360'] for x in actual),
            'episodes_with_active_single_joint_coordinate_range_ge360': len({x['index'] for x in actual if x['continuous_coordinate_range_ge360']}),
            'active_single_joint_monotonic_sweep_ge360': sum(x['monotonic_sweep_ge360'] for x in actual),
            'preclose_single_joint_coordinate_range_ge360': sum(x['continuous_coordinate_range_ge360'] for x in preactual),
            'episodes_with_preclose_J1_or_J3_range_over180': len({x['index'] for x in preactual if x['joint_1based'] in [1, 3] and x['unwrapped_range_deg'] > 180}),
            'dense_applied_command_near2pi_jumps_all_arms': sum(x['near_2pi_jump_count'] for x in joint_rows if x['stream'] == 'dense_applied_command' and x['stage'] == 'full'),
            'low_next_command_near2pi_jumps_all_arms': sum(x['near_2pi_jump_count'] for x in joint_rows if x['stream'] == 'low_next_command' and x['stage'] == 'full'),
            'dense_measured_state_unwrapping_corrections_all_arms': sum(x['unwrapping_adjusted_steps'] for x in joint_rows if x['stream'] == 'dense_measured' and x['stage'] == 'full'),
            'active_preclose_ee_path_ge1m': sum(x['ee_path_m'] >= 1 for x in pre_arm),
            'active_preclose_ee_detour_ratio_ge5': sum(x['ee_path_to_straight_ratio'] is not None and x['ee_path_to_straight_ratio'] >= 5 for x in pre_arm),
            'active_preclose_wrist_so3_unsigned_path_ge360_not_fixed_axis_turn': sum(x['wrist_so3_unsigned_path_length_deg'] >= 360 for x in pre_arm),
        },
        'rankings': {
            'actual_single_joint_range': sorted(actual, key=lambda x: x['unwrapped_range_deg'], reverse=True)[:20],
            'actual_single_joint_total_variation': sorted(actual, key=lambda x: x['unwrapped_total_variation_deg'], reverse=True)[:20],
            'preclose_actual_single_joint_range': sorted(preactual, key=lambda x: x['unwrapped_range_deg'], reverse=True)[:20],
            'preclose_J1_or_J3_range_over180': sorted([x for x in preactual if x['joint_1based'] in [1, 3] and x['unwrapped_range_deg'] > 180], key=lambda x: x['unwrapped_range_deg'], reverse=True),
            'dense_command_maximum4ms_jump': sorted(commands, key=lambda x: x['maximum_raw_step_deg'], reverse=True)[:10],
            'low_command_maximum60ms_jump': sorted(lowcmd, key=lambda x: x['maximum_raw_step_deg'], reverse=True)[:10],
            'preclose_ee_path_length': sorted(pre_arm, key=lambda x: x['ee_path_m'], reverse=True)[:20],
            'preclose_ee_detour_ratio': sorted([x for x in pre_arm if x['ee_path_to_straight_ratio'] is not None], key=lambda x: x['ee_path_to_straight_ratio'], reverse=True)[:20],
            'preclose_wrist_so3_unsigned_path': sorted(pre_arm, key=lambda x: x['wrist_so3_unsigned_path_length_deg'], reverse=True)[:20],
            'full_wrist_so3_unsigned_path': sorted(full_arm, key=lambda x: x['wrist_so3_unsigned_path_length_deg'], reverse=True)[:20],
        },
        'checks': integrity,
    }
    summary['global_extrema'] = {
        'maximum_actual4ms_joint_step_deg_all_arms': max(x['maximum_raw_step_deg'] for x in joint_rows if x['stream'] == 'dense_measured' and x['stage'] == 'full'),
        'maximum_command4ms_joint_step_deg_all_arms': max(x['maximum_raw_step_deg'] for x in joint_rows if x['stream'] == 'dense_applied_command' and x['stage'] == 'full'),
        'maximum_command60ms_joint_step_deg_all_arms': max(x['maximum_raw_step_deg'] for x in joint_rows if x['stream'] == 'low_next_command' and x['stage'] == 'full'),
        'maximum_initial_applied_target_vs_measured_arm_difference_deg_all_arms': max(x[arm + '_first_applied_command_vs_initial_measured_arm_residual_rad'] * DEG for x in integrity for arm in ARM_BASE),
    }
    large_q6.sort(key=lambda x: x['actual_full_range_deg'], reverse=True)
    seven = {'scope': 'The seven active-arm J6 trajectories with actual continuous coordinate span>=360deg; no claim of fixed-world-axis wrist rotation',
             'closure_threshold_sdk': CLOSE_SDK, 'closure_definition_not_physical_contact': True,
             'count': len(large_q6), 'episodes': large_q6}
    (OUT / 'seven_actual_q6_ge360.json').write_text(json.dumps(seven, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
    seven_flat = []
    for row in large_q6:
        flat = {k: v for k, v in row.items() if k != 'first360_excursion'}
        flat.update({'first360_' + k: v for k, v in (row['first360_excursion'] or {}).items()})
        seven_flat.append(flat)
    csv_write(OUT / 'seven_actual_q6_ge360.csv', seven_flat)
    summary['seven_actual_q6_ge360_detail'] = str(OUT / 'seven_actual_q6_ge360.json')
    csv_write(OUT / 'joint_metrics.csv', joint_rows)
    csv_write(OUT / 'arm_path_orientation_metrics.csv', arm_rows)
    (OUT / 'all50_motion_audit.json').write_text(json.dumps({'summary': summary, 'episodes': episodes}, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
    print(json.dumps({'episodes': len(episodes), 'joint_rows': len(joint_rows), 'arm_rows': len(arm_rows), 'counts': summary['counts'],
                      'largest_joint_range': summary['rankings']['actual_single_joint_range'][0],
                      'largest_preclose_ee_path': summary['rankings']['preclose_ee_path_length'][0],
                      'largest_preclose_so3_path': summary['rankings']['preclose_wrist_so3_unsigned_path'][0]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
