"""CPU-only bounded export of immutable RealMan N+1 capture caches.

No task/scene/robot is constructed. Native HDF5 values, scalar dimensions,
fixed-width marked JPEG bytes and state[k]/action[k+1] match pkl2hdf5.py.
Only the three native RGB buffers of ONE observation are resident. The first
pass discovers JPEG widths; the second writes rows. NPZ members are inflated
one at a time to disk and copied through 256-row mmap slices.

This producer never certifies an episode for training. Original physics,
actual geometry, independent motion/DR gates and normal worker exit remain
necessary. --benchmark-only labels interrupted/partial caches component_only.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import resource
import shutil
import subprocess
import tempfile
import time
import traceback
import zipfile

import h5py
import numpy as np

ROOT = Path('/home/admin123/liminghe/vla-benchmark/RoboTwin')
CODEC = ROOT / 'data/decode_image_bit.py'
CODEC_SHA = '90df99c52d3f127a09f5960680113f17e8b3e25b2e3cb9117abdeb249d7a337e'
NATIVE_EXPORTER = ROOT / 'envs/utils/pkl2hdf5.py'
NATIVE_SHA = 'c19a77f69d9b5c624c73f0162c2fee7b5f5979bdfddf9fdef3dda020912371bf'
INSTRUCTION = 'Pick up the soda bottle with the nearer arm, turn it upright, and hold it steadily in the air.'
CAMERA_MAP = {'head_camera': 'cam_head', 'left_camera': 'cam_left_wrist',
              'right_camera': 'cam_right_wrist', 'front_camera': 'cam_third_view'}
REQUIRED_CAMERAS = ('head_camera', 'left_camera', 'right_camera')
JOINT_FIELDS = {'left_arm': 'left_arm_joint_states', 'left_gripper': 'left_ee_joint_states',
                'right_arm': 'right_arm_joint_states', 'right_gripper': 'right_ee_joint_states'}
POSE_FIELDS = {'left_endpose': 'left_ee_poses', 'right_endpose': 'right_ee_poses'}
PHYSICS_FIELDS = ('physics_step', 'sim_time_before_s', 'sim_time_after_s',
    'measured_state24_before', 'applied_command24', 'measured_state24_after',
    'hand24_qpos_before_rad', 'hand24_qpos_after_rad', 'hand24_drive_target_rad',
    'hand24_actual_fd_rad_s', 'hand24_engine_qvel_rad_s',
    'full36_qpos_before_rad', 'full36_qpos_after_rad')
JSON_LIMIT = 64 * 1024**2
FRAME_LIMIT = 64 * 1024**2
MAX_FRAMES = 50000
NPY_MEMBER_LIMIT = 512 * 1024**2
NPY_TOTAL_LIMIT = 2 * 1024**3
CHUNK_ROWS = 256


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path, limit=JSON_LIMIT):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Missing or symlinked JSON: ' + str(path))
    require(path.stat().st_size <= limit, 'JSON exceeds bounded reader budget: ' + str(path))
    with path.open(encoding='utf-8') as f:
        return json.load(f)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def load_codec():
    require(sha(CODEC) == CODEC_SHA, 'Reviewed native image codec changed')
    require(sha(NATIVE_EXPORTER) == NATIVE_SHA, 'Reviewed native HDF5 semantics changed')
    spec = importlib.util.spec_from_file_location('_streaming_native_rgb_codec', CODEC)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Explicit CPU budget; the enclosing phase guard also limits the process tree.
    module.cv2.setNumThreads(2)
    return module.encode_image_bit


def cache_paths(candidate):
    candidate = Path(candidate)
    require(candidate.is_dir() and not candidate.is_symlink(), 'Candidate missing or symlinked')
    cache = candidate / '.cache/episode0'
    require(cache.is_dir() and not cache.is_symlink() and not cache.parent.is_symlink(), 'Raw observation cache missing or symlinked')
    files = []
    for p in cache.iterdir():
        require(not p.is_symlink(), 'Symlinked cache member refused')
        if p.suffix == '.pkl':
            require(p.stem.isdigit() and p.is_file(), 'Non-numeric or non-file pickle cache entry')
            files.append((int(p.stem), p))
    files.sort()
    require(2 <= len(files) <= MAX_FRAMES, 'Incomplete N+1 cache or excessive observation count')
    require([i for i, _ in files] == list(range(len(files))), 'Missing or duplicate observation index in N+1 cache')
    return [p for _, p in files]


def load_frame(path):
    require(path.stat().st_size <= FRAME_LIMIT, 'One cache frame exceeds bounded frame budget')
    with path.open('rb') as f:
        value = pickle.load(f)
        require(not f.read(1), 'Unexpected trailing data after cached observation')
    require(isinstance(value, dict), 'Cached observation is not a dictionary')
    return value


def vector(value, *, dtype=np.float32):
    array = np.asarray(value, dtype=dtype)
    if array.ndim == 0:
        array = array.reshape(1)
    require(array.ndim == 1 and len(array) > 0 and np.isfinite(array).all(), 'Invalid numeric state/pose row')
    return array


def extrinsic(value):
    value = np.asarray(value, dtype=np.float32)
    if value.shape == (3, 4):
        out = np.eye(4, dtype=np.float32)
        out[:3, :4] = value
        value = out
    require(value.shape == (4, 4) and np.isfinite(value).all(), 'Invalid native extrinsic matrix')
    return value


def normalized_frame(frame, *, expected_rgb_shape):
    joints = frame.get('joint_action', {})
    require(isinstance(joints, dict) and all(k in joints for k in JOINT_FIELDS), 'Missing native arm/scalar state field')
    numeric = {('joint', target): vector(joints[source]) for source, target in JOINT_FIELDS.items()}
    require(numeric['joint', 'left_arm_joint_states'].shape == (6,) and
        numeric['joint', 'right_arm_joint_states'].shape == (6,) and
        numeric['joint', 'left_ee_joint_states'].shape == (1,) and
        numeric['joint', 'right_ee_joint_states'].shape == (1,), 'RealMan native arm6/scalar shape differs')
    poses = frame.get('endpose', {})
    for source, target in POSE_FIELDS.items():
        if source in poses:
            numeric['joint', target] = vector(poses[source])
    observation = frame.get('observation', {})
    require(isinstance(observation, dict) and all(k in observation for k in REQUIRED_CAMERAS), 'Missing native three-camera observation')
    colors = {}
    for source, target in CAMERA_MAP.items():
        if source not in observation:
            continue
        camera = observation[source]
        require(isinstance(camera, dict) and 'rgb' in camera, 'Native camera RGB field missing')
        image = camera['rgb']
        require(isinstance(image, np.ndarray) and image.dtype == np.uint8 and image.ndim == 3 and
            image.shape[-1] == 3, 'RGB shape/dtype differs from original uint8 RGB contract')
        if expected_rgb_shape is not None:
            require(image.shape == tuple(expected_rgb_shape), 'Native RGB resolution differs from480x640x3')
        colors[target] = image
        if 'depth' in camera:
            value = np.asarray(camera['depth'])
            require(value.dtype.kind in 'fiu' and np.isfinite(value).all(), 'Invalid native depth row')
            numeric['camera', target + '/depths'] = value
        intrinsic = camera.get('intrinsic_cv', camera.get('intrinsic_matrix'))
        if intrinsic is not None:
            value = np.asarray(intrinsic)
            require(value.shape == (3, 3) and value.dtype.kind in 'fiu' and np.isfinite(value).all(), 'Invalid native intrinsic matrix')
            numeric['camera', target + '/intrinsic_matrix'] = value
        pose = camera.get('cam2world_gl')
        if pose is None:
            pose = camera.get('extrinsic_cv')
        if pose is None:
            pose = camera.get('extrinsics_matrix')
        if pose is not None:
            numeric['camera', target + '/extrinsics_matrix'] = extrinsic(pose)
    return numeric, colors


def schema_of(numeric, colors):
    return {'numeric': {kind + '/' + name: {'shape': list(v.shape), 'dtype': v.dtype.str}
        for (kind, name), v in numeric.items()},
        'colors': {name: {'shape': list(v.shape), 'dtype': v.dtype.str} for name, v in colors.items()}}


def original_physics_admission(candidate, report):
    require(report.get('accepted') is True and report.get('result') == 'physics_success' and
        report.get('canonical_sources_unchanged') is True and report.get('hardware_io') is False and
        report.get('official_task_methods_unmodified') is True, 'Original physical terminal report rejected or incomplete')
    require(report.get('hold_passed') is True and report.get('hold_validation_s', 0) >= 1 and
        report.get('minimum_hold_bottle_gap_m', 0) >= .01 and
        report.get('immediate', {}).get('official_success') is True and
        report.get('after_hold', {}).get('official_success') is True, 'Original success/airhold proof missing or rejected')
    audit_path = candidate / 'actual36_geometry_audit.json'
    audit = read_json(audit_path)
    require(audit.get('passed') is True and report.get('actual36_geometry_audit', {}).get('passed') is True,
        'Final actual36 physical geometry audit missing or rejected')
    require(report.get('actual36_geometry_audit_sha256') == sha(audit_path), 'Original final geometry audit SHA differs')
    sources = report.get('source_hashes', {})
    require(isinstance(sources, dict) and len(sources) >= 601, 'Protected original producer registry incomplete')
    for path, digest in sources.items():
        p = Path(path)
        require(p.is_file() and not p.is_symlink() and sha(p) == digest, 'Protected physical source changed: ' + path)
    seal_path = candidate / 'physical_collection_seal.json'
    seal = read_json(seal_path)
    require(seal.get('passed') is True and seal.get('expert_report_sha256') == sha(candidate / 'expert_report.json') and
        seal.get('inventory_snapshot_sha256') == sha(candidate / 'inventory_snapshot.json') and
        seal.get('recording_samples_sha256') == sha(candidate / 'recording_samples.json') and
        seal.get('physics_telemetry_sha256') == sha(candidate / 'physics_telemetry.npz'),
        'Immutable physical collection seal missing, failed or input SHA differs')
    terminal = read_json(candidate / 'raw_capture_terminal_receipt.json')
    require(terminal.get('collected_physical') is True and terminal.get('normal_worker_exit') is True and
        terminal.get('data_accepted') is False and terminal.get('exit_code') == 0 and
        terminal.get('collection_seal_sha256') == sha(seal_path),
        'Normal collect-only terminal receipt missing, failed or not bound to this report')
    return seal


def samples_and_inventory(candidate, report, benchmark_only):
    samples_path = candidate / 'recording_samples.json'
    samples = read_json(samples_path)
    require(isinstance(samples, list) and 2 <= len(samples) <= MAX_FRAMES, 'Invalid recorded SDK24 N+1 sample list')
    times = np.asarray([r['sim_time_s'] for r in samples], dtype=np.float64)
    steps = np.asarray([r['physics_step'] for r in samples], dtype=np.int64)
    states = np.asarray([r['measured_state24'] for r in samples], dtype=np.float64)
    commands = np.asarray([r['command24'] for r in samples], dtype=np.float64)
    require(states.shape == commands.shape == (len(samples), 24) and np.isfinite(states).all() and
        np.isfinite(commands).all(), 'SDK24 state/command shape or finite values differ')
    require(times[0] == 0 and steps[0] == 0 and np.isfinite(times).all() and
        np.all(np.diff(steps) == 15) and np.allclose(np.diff(times), .060, atol=1e-8, rtol=0), 'Original15x4ms capture clock differs')
    dt = report.get('action_clock', {}).get('physics_dt_s')
    require(isinstance(dt, (int, float)) and abs(dt - .004) <= 1e-9 and
        np.allclose(times, steps * dt, atol=1e-8, rtol=0), 'Recorded physical clock differs from actual original4ms')
    inventory_path = candidate / 'inventory_snapshot.json'
    inventory_source = 'raw_capture_inventory_snapshot'
    if inventory_path.exists():
        inventory = read_json(inventory_path)
        names = inventory['full36_joint_names']
        hands = inventory['hand24_joint_names']
    else:
        require(benchmark_only, 'Collect-only inventory_snapshot.json missing')
        # Historical raw34 has no detached snapshot. These are names actually
        # measured in its initialization receipt, not a reconstructed URDF.
        inventory_path = candidate / 'actual_hardware_camera_clock_receipt.json'
        actual = read_json(inventory_path)
        require(actual.get('passed') is True and actual.get('hardware_io') is False, 'Historical actual hardware receipt rejected')
        names = actual['actual_joint_names']
        hands = list(actual['actual_hand_drives'])
        inventory = {'full36_joint_names': names, 'hand24_joint_names': hands}
        inventory_source = 'historical_actual_hardware_readback_for_component_benchmark_only'
    require(isinstance(names, list) and isinstance(hands, list) and len(names) == len(set(names)) == 36 and
        len(hands) == len(set(hands)) == 24 and set(hands).issubset(names) and
        names == report.get('full_qpos_joint_names'), 'Actual full36/hand24 joint inventory differs')
    require(all(isinstance(n, str) for n in names + hands), 'Joint inventory must contain actual names')
    return samples, times, steps, states, commands, inventory, inventory_path, inventory_source


def inflate_npz(candidate, scratch):
    path = candidate / 'physics_telemetry.npz'
    require(path.is_file() and not path.is_symlink(), 'Physical telemetry archive missing or symlinked')
    checksum = sha(path)
    arrays = {}
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        require(len(infos) == len(PHYSICS_FIELDS) and {i.filename for i in infos} ==
            {name + '.npy' for name in PHYSICS_FIELDS}, 'Original physical telemetry schema missing, extra or duplicated')
        require(sum(i.file_size for i in infos) <= NPY_TOTAL_LIMIT and all(0 < i.file_size <= NPY_MEMBER_LIMIT for i in infos),
            'Physical telemetry exceeds bounded disk inflation budget')
        for name in PHYSICS_FIELDS:
            dest = scratch / (name + '.npy')
            with archive.open(name + '.npy') as source, dest.open('xb') as out:
                shutil.copyfileobj(source, out, length=1024**2)
            value = np.load(dest, mmap_mode='r', allow_pickle=False)
            require(value.ndim in (1, 2) and value.dtype.kind in 'fiu' and len(value) > 0, 'Invalid physical numeric member: ' + name)
            arrays[name] = value
    require(sha(path) == checksum, 'Physical telemetry input changed during bounded inflation')
    return arrays, checksum


def close_arrays(arrays):
    for value in arrays.values():
        mapping = getattr(value, '_mmap', None)
        if mapping is not None:
            mapping.close()


def validate_physics(arrays, report, steps):
    count = len(arrays['physics_step'])
    require(count == int(steps[-1]) == report.get('physical_steps'), 'Physical terminal steps differ from N+1 cache')
    dt = report['action_clock']['physics_dt_s']
    for name, value in arrays.items():
        width = 36 if name.startswith('full36_') else 24
        require(value.shape == ((count,) if name in ('physics_step', 'sim_time_before_s', 'sim_time_after_s') else (count, width)),
            'Original physical telemetry shape differs: ' + name)
    for start in range(0, count, CHUNK_ROWS):
        end = min(start + CHUNK_ROWS, count)
        seq = np.arange(start + 1, end + 1)
        require(np.array_equal(arrays['physics_step'][start:end], seq), 'Physical step gap/duplicate')
        require(np.allclose(arrays['sim_time_before_s'][start:end], (seq - 1) * dt, atol=1e-8, rtol=0) and
            np.allclose(arrays['sim_time_after_s'][start:end], seq * dt, atol=1e-8, rtol=0), 'Physical4ms timestamp differs')
        require(all(np.isfinite(v[start:end]).all() for v in arrays.values()), 'Non-finite physical telemetry')
    return count


def validate_frame_sample(frame, index, samples, arrays, inventory):
    sample = samples[index]
    clock = frame.get('recording_clock', {})
    require(clock.get('physics_step') == sample['physics_step'] and clock.get('sim_time_s') == sample['sim_time_s'],
        'Cached observation clock differs from its real recorded sample')
    dex = frame.get('dexterous_state', {})
    require(np.array_equal(np.asarray(dex.get('state24'), dtype=np.float64), sample['measured_state24']) and
        np.array_equal(np.asarray(dex.get('command24'), dtype=np.float64), sample['command24']), 'SDK24 cache/recorded sample differ')
    positions = dex.get('joint_position_rad', {})
    names, hands = inventory['full36_joint_names'], inventory['hand24_joint_names']
    require(set(positions) == set(names), 'Actual cache full36 names differ from captured inventory')
    row = sample['physics_step'] - 1
    suffix = 'after_rad' if row >= 0 else 'before_rad'
    row = max(row, 0)
    require(np.array_equal(np.asarray([positions[n] for n in names], dtype=np.float32), arrays['full36_qpos_' + suffix][row]),
        'Actual native full36 observation differs from captured physical column ordering')
    require(np.array_equal(np.asarray([positions[n] for n in hands], dtype=np.float32), arrays['hand24_qpos_' + suffix][row]),
        'Actual hand24 observation differs from captured physical column ordering')


def discover_layout(paths, samples, arrays, inventory, encoder, expected_rgb_shape):
    widths, schema, cache_pins = {}, None, []
    for i, path in enumerate(paths):
        digest = sha(path)
        frame = load_frame(path)
        numeric, colors = normalized_frame(frame, expected_rgb_shape=expected_rgb_shape)
        layout = schema_of(numeric, colors)
        if schema is None:
            schema = layout
        require(layout == schema, 'Observation schema/dtype/shape changed at frame ' + str(i))
        validate_frame_sample(frame, i, samples, arrays, inventory)
        if i < len(paths) - 1:
            for name, image in colors.items():
                encoded = encoder(image)
                require(isinstance(encoded, bytes) and encoded, 'Native RGB encoder returned no JPEG')
                widths[name] = max(widths.get(name, 0), len(encoded))
            del image, encoded
        require(sha(path) == digest, 'Raw observation changed during first export pass')
        cache_pins.append({'name': str(path.name), 'bytes': path.stat().st_size, 'sha256': digest})
        del frame, numeric, colors
        if i % 100 == 0:
            print(json.dumps({'phase': 'jpeg_width_and_real_capture_check', 'observation': i, 'total': len(paths)}, ensure_ascii=False), flush=True)
    return schema, widths, cache_pins


def write_native_hdf(paths, hdf, schema, widths, pins, encoder, instructions, expected_rgb_shape):
    count = len(paths) - 1
    with h5py.File(hdf, 'w', rdcc_nbytes=1024**2) as f:
        string_type = h5py.string_dtype(encoding='utf-8')
        f.attrs['source_format'] = 'RoboTwin'
        f.attrs['source_path'] = 'native_collection'
        f.create_dataset('data_format_version', data='v1.0', dtype=string_type)
        f.create_dataset('instructions', data=json.dumps(instructions, ensure_ascii=False), dtype=string_type)
        f.create_group('additional_info').create_dataset('frequency', data=np.asarray(15, dtype=np.int32))
        for group in ('state', 'action', 'vision'):
            f.create_group(group)
        for name, details in schema['numeric'].items():
            kind, target = name.split('/', 1)
            shape = (count,) + tuple(details['shape'])
            if kind == 'joint':
                for group in ('state', 'action'):
                    f[group].create_dataset(target, shape=shape, dtype=np.dtype(details['dtype']))
            else:
                camera, field = target.split('/', 1)
                f['vision'].require_group(camera).create_dataset(field, shape=shape, dtype=np.dtype(details['dtype']))
        for name, details in schema['colors'].items():
            camera = f['vision'].require_group(name)
            camera.create_dataset('colors', shape=(count,), dtype='S' + str(widths[name]))
            camera.create_dataset('shape', data=np.asarray(details['shape'], dtype=np.int32))
        for i, path in enumerate(paths):
            require(sha(path) == pins[i]['sha256'], 'Raw observation changed between export passes')
            frame = load_frame(path)
            numeric, colors = normalized_frame(frame, expected_rgb_shape=expected_rgb_shape)
            require(schema_of(numeric, colors) == schema, 'Raw schema changed between export passes')
            for (kind, name), value in numeric.items():
                if kind == 'joint':
                    if i < count:
                        f['state'][name][i] = value
                    if i > 0:
                        f['action'][name][i - 1] = value
                elif i < count:
                    f['vision'][name][i] = value
            if i < count:
                for name, image in colors.items():
                    encoded = encoder(image)
                    require(len(encoded) <= widths[name], 'Native JPEG width changed between passes')
                    f['vision'][name + '/colors'][i] = encoded
                del image, encoded
            require(sha(path) == pins[i]['sha256'], 'Raw observation changed during second export pass')
            del frame, numeric, colors
            if i % 100 == 0:
                print(json.dumps({'phase': 'write_native_hdf_rows', 'observation': i, 'total': len(paths)}, ensure_ascii=False), flush=True)


def append_dexterous(hdf, report_sha, times, states, commands, inventory, arrays):
    with h5py.File(hdf, 'a', rdcc_nbytes=1024**2) as f:
        group = f.create_group('realman_dexterous')
        group.attrs['sdk_mapping_calibrated'] = False
        group.attrs['ordering'] = 'Larm6 rad,Lprimary SDK6,Rarm6 rad,Rprimary SDK6'
        group.attrs['command_source'] = 'actual drive targets at next observation timestamps'
        for name, values in {'measured_state24': states[:-1], 'next_command24': commands[1:],
            'sample_time_s': times[:-1], 'next_sample_time_s': times[1:]}.items():
            group.create_dataset(name, data=values, compression='gzip')
        physics = f.create_group('realman_dexterous_physics')
        physics.attrs['sdk_mapping_calibrated'] = False
        physics.attrs['hand24_joint_names'] = json.dumps(inventory['hand24_joint_names'])
        physics.attrs['full36_joint_names'] = json.dumps(inventory['full36_joint_names'])
        physics.attrs['ordering'] = group.attrs['ordering']
        physics.attrs['command_source'] = 'actual drive targets before each physics step'
        for name in PHYSICS_FIELDS:
            values = arrays[name]
            target = physics.create_dataset(name, shape=values.shape, dtype=values.dtype, compression='gzip')
            for start in range(0, len(values), CHUNK_ROWS):
                target[start:start + CHUNK_ROWS] = values[start:start + CHUNK_ROWS]
        f.attrs['physical_success_report_sha256'] = report_sha
        f.attrs['physical_calibration_available'] = False
        f.attrs['new_physics_episode'] = True
        f.attrs['recording_clock'] = '60ms observations,4ms physical steps'


def encode_video(paths, pins, times, source_camera, dest, *, realtime, ffmpeg):
    count = int(np.ceil(times[-1] * 30)) if realtime else len(paths)
    process = subprocess.Popen([str(ffmpeg), '-nostdin', '-loglevel', 'error', '-f', 'rawvideo',
        '-pixel_format', 'rgb24', '-video_size', '640x480', '-framerate', '30', '-i', 'pipe:0',
        '-threads', '2', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18' if realtime else '23',
        '-movflags', '+faststart', str(dest)], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL)
    sample_index, image, max_age = -1, None, 0.
    try:
        for tick in range(count):
            t = min(tick / 30, times[-1])
            index = max(int(np.searchsorted(times, t, side='right') - 1), 0) if realtime else tick
            if index != sample_index:
                require(sha(paths[index]) == pins[index]['sha256'], 'Input changed during video export')
                image = None
                frame = load_frame(paths[index])
                image = np.asarray(frame['observation'][source_camera]['rgb'])
                require(image.shape == (480, 640, 3) and image.dtype == np.uint8, 'Native RGB video input differs')
                del frame
                sample_index = index
            if realtime:
                max_age = max(max_age, float(t - times[index]))
            process.stdin.write(image.tobytes())
    except BaseException:
        process.kill()
        process.wait()
        raise
    finally:
        if process.stdin is not None:
            process.stdin.close()
    require(process.wait() == 0, 'Bounded video encoder failed')
    require(not realtime or max_age <= .060 + 1e-7, 'ZOH video exceeded one real sample interval')
    # Decode every encoded frame through the same bounded process-tree budget.
    result = subprocess.run([str(ffmpeg), '-nostdin', '-v', 'error', '-threads', '2', '-i', str(dest),
        '-threads', '2', '-f', 'null', '-'], stdout=subprocess.DEVNULL, check=False)
    require(result.returncode == 0, 'Video full decode failed')
    import cv2
    cap = cv2.VideoCapture(str(dest))
    observed = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    shape = [int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))), int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))]
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    require(observed == count and shape == [480, 640] and abs(fps - 30) < 1e-6, 'Video frame count/dimensions/FPS differ')
    return {'frames': count, 'playback_fps': 30, 'duration_s': count / 30,
        'max_observation_age_s': max_age if realtime else None,
        'method': 'zero-order hold actual sample clock' if realtime else 'native N+1 preview at30fps',
        'full_ffmpeg_decode_passed': True, 'sha256': sha(dest), 'bytes': dest.stat().st_size}


def export_candidate(candidate, output, *, benchmark_only=False, hdf_only=False,
                     video_only=False, expected_rgb_shape=(480, 640, 3), ffmpeg=None):
    require(not (hdf_only and video_only), 'Select only one isolated export phase')
    candidate, output = Path(candidate), Path(output)
    require(not candidate.is_symlink(), 'Symlinked physical candidate refused')
    candidate = candidate.resolve()
    require(not output.exists() and not output.is_symlink(), 'Refusing to overwrite existing export output')
    require(output.parent.is_dir() and not output.parent.is_symlink(), 'Output parent must already exist and be a real directory')
    output = output.parent.resolve() / output.name
    require(not output.is_relative_to(candidate), 'Independent export output must be outside immutable physical candidate')
    start = time.monotonic()
    paths = cache_paths(candidate)
    report_path = candidate / 'expert_report.json'
    report_sha = sha(report_path)
    report = read_json(report_path)
    terminal = None if benchmark_only else original_physics_admission(candidate, report)
    samples, times, steps, states, commands, inventory, inv_path, inventory_source = samples_and_inventory(candidate, report, benchmark_only)
    require(len(samples) == len(paths) == report.get('recorded_frames'), 'Real sample/physical report/cache N+1 count differs')
    inputs = {str(p): sha(p) for p in (report_path, candidate / 'recording_samples.json',
        candidate / 'physics_telemetry.npz', inv_path)}
    if not benchmark_only:
        for p in (candidate / 'actual36_geometry_audit.json', candidate / 'raw_capture_terminal_receipt.json',
                  candidate / 'physical_collection_seal.json'):
            inputs[str(p)] = sha(p)
    staging = Path(tempfile.mkdtemp(prefix=output.name + '.partial.', dir=output.parent))
    scratch = staging / '_telemetry_mmap'
    scratch.mkdir()
    arrays = {}
    try:
        arrays, telemetry_sha = inflate_npz(candidate, scratch)
        physical_steps = validate_physics(arrays, report, steps)
        encoder = load_codec()
        schema, widths, pins = discover_layout(paths, samples, arrays, inventory, encoder, expected_rgb_shape)
        if terminal is not None:
            require(terminal.get('cache_sha256') == {row['name']: row['sha256'] for row in pins},
                'Terminal cache SHA inventory differs from actual consumed observations')
        outputs = {}
        if not video_only:
            (staging / 'data').mkdir()
            (staging / 'instruction').mkdir()
            hdf = staging / 'data/episode_0000000.hdf5'
            instructions = [INSTRUCTION]
            write_native_hdf(paths, hdf, schema, widths, pins, encoder, instructions, expected_rgb_shape)
            append_dexterous(hdf, report_sha, times, states, commands, inventory, arrays)
            outputs['hdf5'] = {'path': str(output / 'data/episode_0000000.hdf5'), 'sha256': sha(hdf), 'bytes': hdf.stat().st_size}
            instruction_path = staging / 'instruction/episode_0000000.json'
            write_json(instruction_path, {'instructions': instructions, 'seen': instructions, 'unseen': []})
            outputs['instruction'] = {'path': str(output / 'instruction/episode_0000000.json'), 'sha256': sha(instruction_path)}
        close_arrays(arrays)
        arrays = {}
        shutil.rmtree(scratch)
        if not hdf_only:
            ffmpeg = ffmpeg or shutil.which('ffmpeg')
            require(ffmpeg and Path(ffmpeg).is_file(), 'Video phase requires existing ffmpeg')
            (staging / 'video').mkdir()
            videos = [('head_camera', 'episode_0000000.mp4', False),
                ('head_camera', 'head_realtime.mp4', True), ('left_camera', 'left_wrist_realtime.mp4', True),
                ('right_camera', 'right_wrist_realtime.mp4', True)]
            for source, filename, realtime in videos:
                details = encode_video(paths, pins, times, source, staging / 'video' / filename, realtime=realtime, ffmpeg=ffmpeg)
                outputs[filename] = {'path': str(output / 'video' / filename), **details}
        for p, digest in inputs.items():
            require(sha(p) == digest, 'Protected raw input changed during export: ' + p)
        require(cache_paths(candidate) == paths, 'Cache inventory changed during export')
        require(all(sha(p) == row['sha256'] for p, row in zip(paths, pins)), 'Consumed raw cache changed before export finalization')
        receipt = {'version': 'realman_streaming_export_v1', 'export_complete': True,
            'accepted': False, 'component_only': bool(benchmark_only),
            'original_physical_terminal_and_actual36_admission_passed': not benchmark_only,
            'scope': 'Immutable CPU export; independent dataset/motion/DR acceptance still required',
            'candidate': str(candidate), 'output': str(output), 'hardware_io': False,
            'task_or_scene_constructed': False, 'raw_input_or_physical_source_writes': 0,
            'mode': 'hdf_only' if hdf_only else 'video_only' if video_only else 'sequential_hdf_then_video',
            'observations': len(paths), 'native_state_action_pairs': len(paths) - 1,
            'physical_steps': physical_steps, 'actual_duration_s': float(times[-1]),
            'native_frequency_field_unchanged': 15, 'SDK_action_dim': 24,
            'actual_full36_hand24_inventory_source': inventory_source,
            'source_sha256': {str(Path(__file__).resolve()): sha(__file__), str(CODEC): CODEC_SHA, str(NATIVE_EXPORTER): NATIVE_SHA},
            'protected_raw_inputs_sha256': inputs, 'consumed_cache_sha256': pins,
            'native_schema': schema, 'native_JPEG_fixed_width_bytes': widths,
            'bounded_strategy': {'resident_raw_observations_max': 1, 'JPEG_width_passes': 2,
                'physics_HDF_copy_rows_max': CHUNK_ROWS, 'physics_npz_full_array_loads': 0,
                'one_npz_member_inflated_to_disk_at_a_time': True, 'ffmpeg_workers_max': 1, 'ffmpeg_threads': 2,
                'small_JSON_file_limit_bytes': JSON_LIMIT},
            'outputs': outputs, 'elapsed_s': time.monotonic() - start,
            'python_peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
        write_json(staging / 'export_receipt.json', receipt)
        require(not output.exists(), 'Independent output appeared before atomic publication')
        staging.rename(output)
        return receipt
    except BaseException as error:
        close_arrays(arrays)
        write_json(staging / 'export_failure.json', {'export_complete': False, 'accepted': False,
            'component_only': bool(benchmark_only), 'candidate': str(candidate), 'requested_output': str(output),
            'partial_output_preserved': str(staging), 'error': str(error), 'traceback': traceback.format_exc(),
            'python_peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--benchmark-only', action='store_true')
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument('--hdf-only', action='store_true')
    phase.add_argument('--video-only', action='store_true')
    parser.add_argument('--ffmpeg', type=Path)
    args = parser.parse_args()
    receipt = export_candidate(args.candidate, args.output, benchmark_only=args.benchmark_only,
        hdf_only=args.hdf_only, video_only=args.video_only, ffmpeg=args.ffmpeg)
    print(json.dumps({key: receipt[key] for key in ('export_complete', 'accepted', 'component_only',
        'observations', 'native_state_action_pairs', 'physical_steps', 'python_peak_RSS_bytes', 'output')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
