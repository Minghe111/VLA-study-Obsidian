"""Seal completed physical recording without decoding images or producing a dataset.

The original physical success, airborne hold, and complete recorded-state
geometry audit are required here.  Independent motion/RGB gates remain pending. A seal is
never a published or independently accepted demonstration.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


SCHEMA = 'physical-collection-seal-v1'
INSTRUCTION = ('Pick up the soda bottle with the nearer arm, turn it upright, '
               'and hold it steadily in the air.')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # All fields supplied to this producer must already be JSON-compatible.
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False,
                  default=_json)
        stream.write('\n')


def _json(value):
    if callable(getattr(value, 'tolist', None)):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError('Unexpected non-JSON physical evidence value: ' + type(value).__name__)


def verify_pins(pins):
    if not isinstance(pins, dict) or not pins:
        raise RuntimeError('Complete unchanged physical producer/source pins required')
    for name, digest in pins.items():
        path = Path(name)
        if not path.is_file() or sha(path) != digest:
            raise RuntimeError('Physical source/pinned evidence changed: ' + str(path))
    return dict(pins)


def _file(path, root):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError('Expected actual regular recording file: ' + str(path))
    return {'path': str(path.resolve()), 'relative_path': str(path.relative_to(root)),
            'bytes': path.stat().st_size, 'sha256': sha(path)}


def recording_clock(samples, frames, report, actual_dt):
    """Validate native capture callback identities, not RGB contents or success."""
    if (not math.isfinite(actual_dt) or abs(actual_dt - .004) > 1e-9
            or not isinstance(samples, list) or len(samples) < 2
            or len(samples) != len(frames)):
        raise RuntimeError('Incomplete actual N+1 capture metadata or physical clock')
    if any(type(frame) is not int for frame in frames):
        raise RuntimeError('Actual capture callback steps must be exact integers')
    if type(report.get('physical_steps')) is not int or report['physical_steps'] <= 0:
        raise RuntimeError('Original completed physical step count missing')
    for index, (sample, frame) in enumerate(zip(samples, frames)):
        if sample.get('physics_step') != frame or frame != 15 * index:
            raise RuntimeError('Actual capture chronology differs from unchanged stride15')
        value = sample.get('sim_time_s')
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or abs(value - frame * actual_dt) > 1e-9):
            raise RuntimeError('Actual capture time differs from actual physical step')
        for field in ('measured_state24', 'command24'):
            vector = sample.get(field)
            if (not isinstance(vector, list) or len(vector) != 24
                    or any(isinstance(x, bool) or not isinstance(x, (int, float))
                           or not math.isfinite(x) for x in vector)):
                raise RuntimeError('Actual SDK24 capture metadata missing/nonfinite')
    if not 0 <= report['physical_steps'] - frames[-1] < 15:
        raise RuntimeError('Final physical endpoint differs from native capture stride')
    if report.get('recorded_frames') != len(samples):
        raise RuntimeError('Original expert recorded_frames differs from callback caches')
    return {'physics_dt_s_actual': actual_dt, 'physics_dt_s_nominal': .004,
            'observation_physics_step_stride': 15, 'sample_interval_s_nominal': .060,
            'physical_steps': report['physical_steps'], 'raw_observations': len(samples),
            'state_action_pairs': len(samples) - 1,
            'last_capture_physics_step': frames[-1], 'last_sample_time_s_actual': samples[-1]['sim_time_s'],
            'actual_physical_duration_s': report['physical_steps'] * actual_dt,
            'ordering': 'Larm6 rad,Lprimary SDK6,Rarm6 rad,Rprimary SDK6',
            'rgb_content_validation_pending': True}


def finalize_collected_phase(task, out, report, frames, *, finalize_environment,
                             read_hardware_contract, export_branch_receipt,
                             producer_pins):
    """Call while the actual scene is alive, then allow its worker to close.

    No image pickle is unpickled, no trajectory is replayed, and no physics step
    or pose is written.  The original task trajectory saver is retained.
    """
    out = Path(out).resolve()
    seal_path = out / 'physical_collection_seal.json'
    if seal_path.exists():
        raise RuntimeError('Refusing to overwrite completed collection seal')
    if not (report.get('accepted') is True and report.get('result') == 'physics_success'
            and report.get('canonical_sources_unchanged') is True
            and report.get('hold_passed') is True
            and float(report.get('hold_validation_s', 0)) >= 1 - 1e-7
            and float(report.get('minimum_hold_bottle_gap_m', 0)) >= .01):
        raise RuntimeError('Original physical task and actual airborne hold did not pass')
    geometry = report.get('actual36_geometry_audit')
    geometry_path = out / 'actual36_geometry_audit.json'
    original_geometry_path = out / 'actual36_geometry_audit_v13_original.json'
    if (not isinstance(geometry, dict) or geometry.get('passed') is not True
            or geometry.get('physical_scene_unchanged') is not True
            or not geometry_path.is_file() or not original_geometry_path.is_file()
            or report.get('actual36_geometry_audit_sha256') != sha(geometry_path)):
        raise RuntimeError('Complete actual36 geometry audit and source report SHA required')
    retained = out / 'expert_report_pre_actual_audit.json'
    retained_receipt = geometry.get('retained_pre_actual_audit_report', {})
    if (not retained.is_file() or retained_receipt.get('original_bytes_unmodified') is not True
            or retained_receipt.get('matches_original_report_sha256') is not True
            or retained_receipt.get('sha256') != sha(retained)):
        raise RuntimeError('Original v9 retained pre-actual-audit report proof missing')
    if getattr(task.robot, 'communication_flag', True):
        raise RuntimeError('Physical hardware IO forbidden during staged collection')
    required_pins = verify_pins(producer_pins)
    report_pins = verify_pins(report.get('source_hashes'))
    if any(report_pins.get(name) != digest for name, digest in required_pins.items()):
        raise RuntimeError('Original expert report did not register all staged producer pins')
    helper = str(Path(__file__).resolve())
    if report_pins.get(helper) != sha(helper):
        raise RuntimeError('Staged seal producer not registered before physical collection')
    environment = finalize_environment(task, out, report)
    if environment.get('passed') is not True:
        raise RuntimeError('Actual every-step environment execution did not pass')
    hardware = read_hardware_contract(task)
    if hardware.get('passed') is not True:
        raise RuntimeError('Actual post-collection hardware contract did not pass')
    hardware_path = out / 'actual_hardware_camera_clock_after_receipt.json'
    write_json(hardware_path, hardware)
    samples_path = out / 'recording_samples.json'
    samples = json.loads(samples_path.read_text(encoding='utf-8'))
    clock = recording_clock(samples, frames, report, float(task.scene.get_timestep()))
    cache = out / '.cache' / 'episode0'
    expected = [cache / (str(index) + '.pkl') for index in range(len(frames))]
    if set(cache.iterdir()) != set(expected):
        raise RuntimeError('Actual capture cache contains missing or extraneous frames')
    cached = [_file(path, out) for path in expected]
    names = [joint.get_name() for joint in task.robot.left_entity.get_active_joints()]
    hand_names = [name for side in ('left', 'right')
                  for name in task.robot._manifest['hands'][side]['limits']]
    if len(names) != 36 or len(set(names)) != 36 or len(hand_names) != 24 or len(set(hand_names)) != 24:
        raise RuntimeError('Actual native full36/hand24 joint inventory missing')
    if report.get('full_qpos_joint_names') != names:
        raise RuntimeError('Original actual trace and native joint order differ')
    (out / '_traj_data').mkdir(exist_ok=True)
    if Path(task.save_dir).resolve() != out:
        raise RuntimeError('Original task trajectory saver targets another candidate')
    task.save_traj_data(0)
    write_json(out / 'scene_info.json', {'episode_0': task.info})
    inventory_path = out / 'inventory_snapshot.json'
    inventory = {'schema': 'actual-post-collection-inventory-v1',
        'passed': True, 'physical_scene_alive_at_readback': True, 'hardware_io': False,
        'actual_native_active_joint_names': names, 'actual_joint_names': names,
        'full36_joint_names': names, 'hand24_joint_names': hand_names,
        'actual_joint_count': len(names), 'actual_link_count': hardware.get('actual_link_count'),
        'training_cameras': ['cam_head', 'cam_left_wrist', 'cam_right_wrist'],
        'clock': clock, 'actual_physics_timestep_s': hardware['actual_physics_timestep_s'],
        'observation_physics_step_stride': 15,
        'source_hashes': report_pins, 'sdk_mapping_calibrated': False,
        'hardware_readback_receipt_path': str(hardware_path),
        'hardware_readback_receipt_sha256': sha(hardware_path)}
    write_json(inventory_path, inventory)
    spec = json.loads((out / 'sample_spec.json').read_text(encoding='utf-8'))
    branch = Path(export_branch_receipt(out, spec))
    # The source physical report keeps its historical accepted=True meaning:
    # official task/hold/full actual geometry passed. Dataset acceptance differs.
    report.update(accepted_semantics='original physical task, airborne hold, and actual36 geometry only',
                  collected_physical=True, published_accepted=False,
                  independent_recorded_state_geometry_pending=False,
                  independent_motion_and_RGB_validation_pending=True)
    (out / 'expert_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2,
        allow_nan=False, default=_json) + '\n', encoding='utf-8')
    paths = ['expert_report.json', 'expert_report_pre_actual_audit.json', 'physics_trace.json',
             'physics_telemetry.npz', 'recording_samples.json', 'sample_spec.json',
             'run_config.yml', 'scene_sample.json', 'scene_info.json', '_traj_data/episode0.pkl',
             'randomized_scene_receipt.json', 'randomized_environment_execution.json',
             'actual_hardware_camera_clock_receipt.json',
             'actual_hardware_camera_clock_after_receipt.json',
             'pre_recording_clutter_settle_receipt.json',
             'nearest_periodic_branch_receipt.json', 'runtime_binding_receipt.json']
    paths += ['actual36_geometry_audit.json', 'actual36_geometry_audit_v13_original.json',
              'inventory_snapshot.json']
    if branch != out / 'nearest_periodic_branch_receipt.json':
        raise RuntimeError('Actual branch receipt path differs from source candidate')
    files = {name: _file(out / name, out) for name in paths}
    verified_after = verify_pins(report_pins)
    if verified_after != report_pins or verify_pins(required_pins) != required_pins:
        raise RuntimeError('Staged collection producer changed during sealing')
    seal = {'schema': SCHEMA, 'passed': True, 'collect_only': True,
            'collected_physical': True, 'data_accepted': False, 'accepted': False,
            'published_accepted': False, 'collection_phase_complete': True,
            'normal_worker_exit_verified': False,
            'normal_worker_exit_scope': 'Scene remains alive for this seal; parent exit receipt required.',
            'source_slot_index': int(spec['index']), 'actual_domain_seed': int(spec['seed']),
            'candidate_directory': str(out), 'hardware_io': False,
            'original_physical_success': True, 'original_airhold_passed': True,
            'actual_environment_execution_passed': True,
            'complete_actual36_geometry_audit_passed': True,
            'actual_hardware_before_after_readback_passed': True,
            'pending_release_gates': ['independent_motion_quality', 'complete_RGB_SDK_HDF5_validation',
                                     'native_video_full_decode', 'normal_collection_unit_exit'],
            'instruction': INSTRUCTION, 'clock': clock, 'full36_joint_names': names,
            'hand24_joint_names': hand_names,
            'training_cameras': ['cam_head', 'cam_left_wrist', 'cam_right_wrist'],
            'files': files, 'cache_frames': cached, 'source_hashes': report_pins,
            'cache_sha256': {Path(item['path']).name: item['sha256'] for item in cached},
            'inventory_snapshot_sha256': sha(inventory_path),
            'expert_report_sha256': files['expert_report.json']['sha256'],
            'recording_samples_sha256': files['recording_samples.json']['sha256'],
            'physics_telemetry_sha256': files['physics_telemetry.npz']['sha256'],
            'required_producer_hashes': required_pins,
            'source_hashes_verified_after_seal_preparation': True,
            'whole_trace_or_RGB_arrays_loaded_by_seal': False,
            'HDF5_or_video_export_started': False, 'physical_scene_close_not_yet_verified': True}
    write_json(seal_path, seal)
    write_json(out / 'recording_report.json', {'accepted': False, 'published_accepted': False,
        'collected_physical': True, 'stage': 'physical_collection_only',
        'physical_collection_seal': str(seal_path), 'physical_collection_seal_sha256': sha(seal_path),
        'native_state_action_pairs': clock['state_action_pairs'],
        'raw_observations': clock['raw_observations'], 'hdf5': None,
        'new_physics_episode': True, 'hardware_io': False, 'pending_release_gates': seal['pending_release_gates']})
    return str(seal_path)
