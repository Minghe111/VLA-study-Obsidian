"""Release exactly five fully proven V5 captures as a separate immutable patch.

Bounded file/hash operations only: no simulation, no full trace/audit JSON load,
no original50 merge, no training and no Git. Default validates without writing.
staged_quality.py's SHA-pinned --publish-five dispatch supplies publish=True.
"""
from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import h5py

HERE = Path(__file__).resolve().parent
E = HERE.parent
V5 = Path('/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/rgb50_motion_quality_20261014_v5_staged')
ARCHIVE = Path('/home/admin123/liminghe/VLA-WAM-RL/30-实验记录/证据/2026-10-11-RealMan专家回放')
ORIGINAL50 = ARCHIVE / 'source_manifest.json'
ORIGINAL50_SHA = '2339d5749b650761011e3975cbeb8d42ceb31dbff4f1f2aba6d71cac9a3edf80'
SLOTS = (17, 18, 20, 29, 34)
BINDINGS = {17: ('Fanta', 'left'), 18: ('Fanta', 'left'), 20: ('Fanta', 'right'),
            29: ('Coca-Cola', 'left'), 34: ('Coca-Cola', 'left')}
GIB = 1024**3
JSON_MAX = 64 * 1024**2
STAGES = {'collect': (10, 12), 'export': (3, 4), 'quality': (2, 3), 'render': (2, 3)}
TOOLS = {'export': HERE / 'streaming_export_staged.py', 'quality': HERE / 'staged_quality.py',
         'render': HERE / 'render_staged_video.py'}
VIDEOS = ('episode_0000000.mp4', 'head_realtime.mp4',
          'left_wrist_realtime.mp4', 'right_wrist_realtime.mp4')
DR = {'random_background': True, 'cluttered_table': True, 'clean_background_rate': .02,
      'random_head_camera_dis': 0, 'random_table_height': .03, 'random_light': True,
      'crazy_random_light_rate': .02}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            value.update(chunk)
    return value.hexdigest()


def read_json(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= JSON_MAX,
            'Missing, symlinked or oversized bounded JSON: ' + str(path))
    with path.open(encoding='utf-8') as stream:
        return json.load(stream)


def real_directory(raw, *, allowed_roots):
    path = Path(raw)
    require(path.is_absolute() and path.is_dir() and not path.is_symlink() and path == path.resolve(),
            'Use exact absolute real stage directories')
    require(any(path.is_relative_to(root) for root in allowed_roots), 'Stage directory outside authorized task roots')
    return path


def require_disjoint_directories(candidate, outputs):
    """Every physical/stage directory is independent in BOTH directions."""
    paths = [Path(candidate), *map(Path, outputs.values())]
    for i, left in enumerate(paths):
        for right in paths[i + 1:]:
            require(not left.is_relative_to(right) and not right.is_relative_to(left),
                'Candidate/export/quality/render directories must be distinct and mutually non-containing')


def pinned_file(path, digest, *, size=None):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and isinstance(digest, str) and len(digest) == 64
        and sha(path) == digest and (size is None or path.stat().st_size == size),
        'Protected file missing or SHA/size changed: ' + str(path))
    return {'path': str(path.resolve()), 'sha256': digest, 'bytes': path.stat().st_size}


def read_bound_motion_result(quality, motion_path):
    motion_path = Path(motion_path)
    require(quality.get('independent_motion_quality_path') == str(motion_path),
        'Quality receipt does not identify this actual independent motion result')
    pinned_file(motion_path, quality.get('independent_motion_quality_sha256'))
    return read_json(motion_path)


def option(command, name, *, required=True):
    found = []
    for i, token in enumerate(command):
        if token == name:
            require(i + 1 < len(command), 'Missing command option value')
            found.append(command[i + 1])
        elif token.startswith(name + '='):
            found.append(token.split('=', 1)[1])
    require(len(found) == 1 if required else len(found) <= 1, 'Missing or duplicated phase option: ' + name)
    return found[0] if found else None


def resource_proof(path, stage, candidate, outputs, spec):
    path = Path(path)
    require(path.is_absolute() and path == path.resolve() and path.name == 'receipt.json'
        and path.parent.parent == HERE / 'resource_runs' and path.parent.name.startswith(stage + '_'),
        'Use actual task resource ledger receipt paths')
    receipt = read_json(path)
    require(receipt.get('schema') == 'realman_guarded_stage_v1' and receipt.get('stage') == stage
        and receipt.get('passed') is True and receipt.get('check_only') is False
        and receipt.get('launcher_returncode') == 0 and receipt.get('monitor_stop_reason') is None
        and receipt.get('cgroup_readback_verified') is True,
        'Resource phase incomplete, failed or not actually bounded: ' + stage)
    command = receipt.get('command')
    require(isinstance(command, list) and len(command) >= 2 and
        command[0] == '/home/admin123/liminghe/miniconda3/envs/robotwin/bin/python'
        and '--benchmark-only' not in command and '--publish-five' not in command,
        'Resource phase was a component benchmark or another command')
    entry = Path(command[1])
    require(entry.is_absolute() and entry.is_file() and not entry.is_symlink() and entry == entry.resolve(),
        'Resource phase entry is not the real producer')
    if stage == 'collect':
        runtime = V5 / 'jobs' / f"slot{spec['index']}_runtime"
        require(entry == runtime / 'preflight/run_staged_collector.py', 'Only new V5 collection is releasable')
        index, attempt = int(option(command, '--index')), int(option(command, '--attempt'))
        raw_output = option(command, '--output', required=False)
        if raw_output is None:
            tree = ast.parse(entry.read_text())
            defaults = [kw.value for node in ast.walk(tree) if isinstance(node, ast.Call) and
                isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument' and
                node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == '--output'
                for kw in node.keywords if kw.arg == 'default']
            require(len(defaults) == 1 and ast.unparse(defaults[0]) == "TASK_ROOT / 'new_physical_episodes'",
                'Collector --output parser default differs from exact reviewed V5 producer')
            raw_output = runtime / 'new_physical_episodes'
        expected = Path(raw_output).resolve() / f'slot_{index:07d}_randomized_attempt_{attempt:03d}'
        require(expected == candidate and index == spec['index'] and attempt == spec['rerecord_attempt']
            and command.count('--run') == 1 and '--prepare-runtime' not in command,
            'Resource collection identity differs from this actual candidate')
        domain = option(command, '--domain-seed', required=False)
        require((int(domain) if domain is not None else spec['source_seed'] + 10_000_000 * attempt) == spec['seed'],
            'Collected actual DR seed differs from source specification')
    else:
        require(entry == TOOLS[stage] and Path(option(command, '--candidate')).resolve() == candidate
            and Path(option(command, '--output')).resolve() == outputs[stage],
            'Resource phase belongs to a different candidate/output')
        if stage == 'quality':
            require(Path(option(command, '--export-output')).resolve() == outputs['export'],
                    'Quality process did not read this actual exporter result')
    binding = receipt.get('source_binding', {})
    require(binding.get('entry_sha256') == sha(entry) and
        binding.get('guard_sha256') == sha(HERE / 'run_guarded_stage.py'), 'Resource producer changed since actual execution')
    high, maximum = STAGES[stage]
    limits = receipt.get('limits', {})
    require(limits.get('MemoryHigh') == str(high * GIB) and limits.get('MemoryMax') == str(maximum * GIB)
        and limits.get('MemorySwapMax') == '0' and limits.get('CPUQuota') == '200%'
        and limits.get('KillMode') == 'control-group' and limits.get('OOMPolicy') == 'stop',
        'Resource phase does not preserve authorized desktop memory/CPU limits')
    worker = receipt.get('worker_terminal', {})
    require(read_json(path.parent / 'worker_terminal.json') == worker, 'Embedded worker terminal differs from actual guarded file')
    require(worker.get('stage_returncode') == 0 and worker.get('source_unchanged') is True,
        'Bounded worker did not terminate normally with frozen source')
    events = worker.get('resource', {}).get('memory.events', {})
    require(all(events.get(k) == 0 for k in ('max', 'oom', 'oom_kill', 'oom_group_kill')),
        'Phase exceeded hard memory cap or suffered OOM')
    unit = receipt.get('unit_terminal', {})
    require(unit.get('Result') == 'success' and unit.get('ExecMainStatus') == '0' and unit.get('ActiveState') == 'inactive',
        'Phase unit did not finish normal complete cleanup')
    readback = read_json(path.parent / 'cgroup_readback.json')
    actual = readback.get('actual', {})
    require(readback.get('limits_verified') is True and actual.get('memory.high') == str(high * GIB)
        and actual.get('memory.max') == str(maximum * GIB) and actual.get('memory.swap.max') == '0'
        and actual.get('cpu.max') == '200000 100000' and actual.get('pids.max') == '128',
        'Actual kernel cgroup limits differ from declared release phase budget')
    return {'receipt_path': str(path.resolve()), 'receipt_sha256': sha(path), 'passed': True,
        'producer_path': str(entry), 'producer_sha256': sha(entry), 'limits': limits,
        'peak': receipt.get('peak'), 'worker_memory_events': events, 'unit_terminal': unit}


def header_contract(hdf, pairs, physical_steps, report_sha):
    with h5py.File(hdf, 'r', rdcc_nbytes=1024**2) as f:
        require(f.attrs.get('source_format') == 'RoboTwin' and
            f.attrs.get('recording_clock') == '60ms observations,4ms physical steps' and
            f.attrs.get('physical_success_report_sha256') == report_sha, 'Native HDF provenance/clock differs')
        for name in ('cam_head', 'cam_left_wrist', 'cam_right_wrist'):
            colors = f['vision/' + name + '/colors']
            require(colors.shape == (pairs,) and colors.dtype.kind == 'S' and
                f['vision/' + name + '/shape'][()].tolist() == [480, 640, 3], 'Native three RGB resolutions/columns differ')
        for name in ('measured_state24', 'next_command24'):
            require(f['realman_dexterous/' + name].shape == (pairs, 24), 'Native SDK24 state/action dimension differs')
        require(f['realman_dexterous/sample_time_s'].shape == (pairs,) and
            f['realman_dexterous/next_sample_time_s'].shape == (pairs,) and
            f['realman_dexterous_physics/physics_step'].shape == (physical_steps,) and
            f['realman_dexterous_physics/full36_qpos_after_rad'].shape == (physical_steps, 36) and
            f['realman_dexterous_physics/hand24_qpos_after_rad'].shape == (physical_steps, 24),
            'Native physical DOF/sample column counts differ')
    return {'passed': True, 'numeric_or_RGB_episode_arrays_loaded': False,
            'native_state_action_pairs': pairs, 'physical_steps': physical_steps,
            'RGB_shape': [480, 640, 3], 'SDK_action_dim': 24, 'physics_dt_s': .004,
            'observation_dt_s': .060}


def source_proofs(candidate, runtime, report, seal):
    pins = report.get('source_hashes', {})
    require(isinstance(pins, dict) and pins == seal.get('source_hashes'), 'Physical seal and original source registry differ')
    source_receipt = runtime / 'source/download_receipt.json'
    bundle = read_json(source_receipt)['files']
    require(len(bundle) == 601, 'Original source bundle must retain601 protected files')
    for relative, row in bundle.items():
        name = Path(relative)
        require(not name.is_absolute() and '..' not in name.parts, 'Original source bundle path escaped')
        path = runtime / 'source' / name
        require(pins.get(str(path)) == row['sha256'], 'Original601 source not registered before physics')
    guard_path = ARCHIVE / 'hardware_guard.json'
    hardware = read_json(guard_path)['expected_sha256']
    require(len(hardware) == 116, 'Frozen hardware source guard must have116 entries')
    aliases = {
        '/tmp/realman-eval60k-20261011/local_eval/hardware/robot_head_down_fixed.urdf': ARCHIVE / 'shared_frozen_runtime/robot_head_down_fixed.urdf',
        '/tmp/realman-eval60k-20261011/local_eval/scripts/remote_base_task_snapshot.py': ARCHIVE / 'shared_frozen_runtime/remote_base_task_snapshot.py'}
    for old, digest in hardware.items():
        path = aliases.get(old, Path(old))
        require(pins.get(str(path)) == digest, 'Hardware116 source not registered before physics')
    for path, digest in pins.items():
        pinned_file(path, digest)
    require(pins.get(str(runtime / 'preflight/run_staged_collector.py')) == sha(runtime / 'preflight/run_staged_collector.py'),
            'Actual V5 staged producer missing from physical source registry')
    return {'source601_passed': True, 'hardware116_passed': True,
        'source_bundle_receipt': pinned_file(source_receipt, sha(source_receipt)),
        'hardware_guard_receipt': pinned_file(guard_path, sha(guard_path)),
        'original_registered_source_sha256': pins}


def verify_candidate(row, old):
    slot = row['slot']
    candidate = real_directory(row['candidate'], allowed_roots=(V5 / 'jobs',))
    runtime = V5 / 'jobs' / f'slot{slot}_runtime'
    require(candidate.is_relative_to(runtime / 'new_physical_episodes'), 'Candidate belongs to a different V5 slot/runtime')
    outputs = {stage: real_directory(row[stage + '_output'], allowed_roots=(E, V5))
        for stage in ('export', 'quality', 'render')}
    require_disjoint_directories(candidate, outputs)
    spec = read_json(candidate / 'sample_spec.json')
    brand, arm = BINDINGS[slot]
    require(spec.get('index') == slot and spec.get('source_slot_index') == slot and
        spec.get('pose_class') == 'upright' and spec.get('brand') == brand and spec.get('arm') == arm
        and spec.get('source_seed') == old['seed'] and spec.get('model_id') == old['model_id'],
        'Original five-slot brand/arm/upright/source identity differs')
    quality_path = outputs['quality'] / 'quality_receipt.json'
    quality = read_json(quality_path)
    require(quality.get('schema') == 'staged-independent-quality-v1' and quality.get('passed') is True
        and quality.get('gates_passed') is True and quality.get('component_only') is False
        and quality.get('accepted') is False and quality.get('data_accepted') is False
        and quality.get('native_dataset_readback_passed') is True
        and quality.get('independent_motion_quality_passed') is True
        and quality.get('candidate') == str(candidate) and quality.get('export_output') == str(outputs['export']),
        'Original physics/RGB/SDK/motion quality stage failed or is a component benchmark')
    require(quality.get('quality_stage_producer_sha256') == sha(HERE / 'staged_quality.py'), 'Independent quality producer changed')
    dataset_path = outputs['quality'] / 'dataset_validation.json'
    pinned_file(dataset_path, quality['dataset_validation_sha256'])
    require(quality['dataset_validation_path'] == str(dataset_path) and read_json(dataset_path).get('passed') is True,
        'Frozen original dataset validator did not pass this artifact')
    pinned_file(quality['native_validator_path'], quality['native_validator_sha256'])
    motion_path = outputs['quality'] / 'independent_motion_quality.json'
    motion = read_bound_motion_result(quality, motion_path)
    require(motion.get('overall_publish_gate_passed') is True and motion.get('numeric_source_unchanged') is True
        and motion.get('config_approved') is True and motion.get('within_confirmed_five_upright_scope') is True
        and motion.get('metadata') == {k: spec[k] for k in ('index', 'arm', 'pose_class', 'brand', 'seed')}
        and motion.get('config_sha256') == sha(E / 'quality/thresholds_confirmed.json')
        and all(motion.get(k, {}).get('passed') is True for k in ('motion', 'planner_branch', 'planner_source_binding', 'original_upstream_acceptance')),
        'Independent actual motion/IK/source/physical gate did not pass')
    require(sha(candidate / 'independent_motion_quality.json') == sha(motion_path), 'Motion gate copy differs from physical-owner proof')
    exported = read_json(outputs['export'] / 'export_receipt.json')
    require(exported.get('export_complete') is True and exported.get('accepted') is False
        and exported.get('component_only') is False and exported.get('mode') == 'hdf_only'
        and exported.get('candidate') == str(candidate) and exported.get('output') == str(outputs['export'])
        and exported.get('original_physical_terminal_and_actual36_admission_passed') is True,
        'Streaming native HDF export is incomplete, unbound or component-only')
    export_receipt_path = outputs['export'] / 'export_receipt.json'
    require(quality['exporter_receipt_path'] == str(export_receipt_path) and
        quality['exporter_receipt_sha256'] == sha(export_receipt_path), 'Quality belongs to another actual export')
    for phase in ('export', 'render'):
        binding_path = outputs[phase] / 'staged_export_binding.json'
        binding = read_json(binding_path)
        require(binding.get('schema') == 'bounded-full-audit-export-binding-v1' and
            binding.get('candidate') == str(candidate) and binding.get('accepted') is False and
            binding.get('export_receipt_sha256') == sha(outputs[phase] / 'export_receipt.json'),
            'Strict full-audit staged exporter source binding missing or different')
        for path, digest in binding.get('producer_sha256', {}).items():
            pinned_file(path, digest)
        require(binding.get('producer_sha256', {}).get(str(HERE / 'streaming_export_staged.py')) == sha(HERE / 'streaming_export_staged.py'),
            'Strict staged exporter producer not bound')
    for path, digest in exported['source_sha256'].items():
        pinned_file(path, digest)
    for path, digest in exported['protected_raw_inputs_sha256'].items():
        pinned_file(path, digest)
    hdf = outputs['export'] / 'data/episode_0000000.hdf5'
    hdf_sha = exported['outputs']['hdf5']['sha256']
    pinned_file(hdf, hdf_sha, size=exported['outputs']['hdf5']['bytes'])
    require(exported['outputs']['hdf5']['path'] == str(hdf) and quality['hdf5_sha256'] == hdf_sha and
        quality['hdf5'] == str(candidate / 'data/episode_0000000.hdf5') and
        sha(candidate / 'data/episode_0000000.hdf5') == hdf_sha and
        motion.get('numeric_source') == quality['hdf5'] and motion.get('numeric_source_sha256') == hdf_sha,
        'Physical-owner/native-copy HDF source binding differs')
    require(hdf_sha != old['published_files'][f'data/episode_{slot:07d}.hdf5']['sha256'],
        'New randomized recording reuses original50 HDF bytes')
    report = read_json(candidate / 'expert_report.json')
    report_sha = sha(candidate / 'expert_report.json')
    require(report.get('result') == 'physics_success' and report.get('accepted') is True and
        report.get('canonical_sources_unchanged') is True and report.get('hardware_io') is False and
        report.get('hold_passed') is True and report.get('hold_validation_s', 0) >= 1 and
        report.get('minimum_hold_bottle_gap_m', 0) >= .01 and
        report.get('immediate', {}).get('official_success') is True and
        report.get('after_hold', {}).get('official_success') is True and
        report.get('actual36_geometry_audit', {}).get('passed') is True,
        'Original physical success/airhold/actual36 proof incomplete')
    seal_path = candidate / 'physical_collection_seal.json'
    seal = read_json(seal_path)
    require(seal.get('passed') is True and seal.get('collect_only') is True and
        seal.get('collected_physical') is True and seal.get('candidate_directory') == str(candidate)
        and seal.get('data_accepted') is False and seal.get('expert_report_sha256') == report_sha,
        'Original physical collection seal incomplete or belongs elsewhere')
    for relative, info in seal['files'].items():
        rel = Path(relative)
        require(not rel.is_absolute() and '..' not in rel.parts, 'Physical sealed file path escaped')
        pinned_file(candidate / rel, info['sha256'], size=info['bytes'])
    terminal = read_json(candidate / 'raw_capture_terminal_receipt.json')
    require(terminal.get('passed') is True and terminal.get('collected_physical') is True and
        terminal.get('normal_worker_exit') is True and terminal.get('worker_ended') is True
        and terminal.get('exit_code') == 0 and terminal.get('original_scene_close_returned_normally') is True
        and terminal.get('data_accepted') is False and terminal.get('candidate_directory') == str(candidate)
        and terminal.get('collection_seal_sha256') == sha(seal_path), 'Actual collector did not return and close normally')
    cache_pins = {v['name']: v['sha256'] for v in exported['consumed_cache_sha256']}
    require(len(cache_pins) == exported['observations'] and cache_pins == seal['cache_sha256'] == terminal['cache_sha256'],
        'Consumed observations do not match complete original sealed N+1 cache')
    for name, digest in cache_pins.items():
        require(name.endswith('.pkl') and name[:-4].isdigit(), 'Unbounded raw cache name')
        pinned_file(candidate / '.cache/episode0' / name, digest)
    require(exported['native_state_action_pairs'] == exported['observations'] - 1 and
        report['recorded_frames'] == exported['observations'] and
        report['physical_steps'] == exported['physical_steps'] and
        abs(report['action_clock']['physics_dt_s'] - .004) <= 1e-9 and
        report['action_clock']['record_every_physics_steps'] == 15, 'Physical N+1/4ms/60ms contract differs')
    header = header_contract(hdf, exported['native_state_action_pairs'], exported['physical_steps'], report_sha)
    randomized = read_json(candidate / 'randomized_scene_receipt.json')
    require(randomized.get('passed') is True and randomized.get('official_domain_randomization') == DR
        and randomized.get('actual_domain_randomization') == DR and randomized.get('actual_seed') == spec['seed']
        and randomized.get('original_source_pose_restored') is False, 'Official fresh randomized scene proof missing')
    source = source_proofs(candidate, runtime, report, seal)
    render_path = outputs['render'] / 'render_stage_receipt.json'
    rendered = read_json(render_path)
    require(rendered.get('render_complete') is True and rendered.get('accepted') is False
        and rendered.get('component_only') is False and rendered.get('candidate') == str(candidate)
        and rendered.get('output') == str(outputs['render']) and rendered.get('source_clock_unchanged') is True,
        'Final native video stage incomplete, unbound or component-only')
    require(rendered.get('staged_export_binding_sha256') == sha(outputs['render'] / 'staged_export_binding.json'),
        'Rendered strict staged exporter binding changed')
    for path, digest in rendered['wrapper_and_export_producer_sha256'].items():
        pinned_file(path, digest)
    media_export = read_json(outputs['render'] / 'export_receipt.json')
    require(rendered['frozen_export_receipt_sha256'] == sha(outputs['render'] / 'export_receipt.json')
        and media_export.get('component_only') is False and media_export.get('mode') == 'video_only'
        and media_export.get('candidate') == str(candidate) and media_export.get('observations') == exported['observations']
        and media_export.get('actual_duration_s') == exported['actual_duration_s'], 'Video source clock differs from HDF capture')
    video_files = rendered['actual_three_camera_video_outputs']
    require(set(video_files) == set(VIDEOS) and video_files == media_export.get('outputs'), 'Required native4 video output set differs')
    for name, info in video_files.items():
        path = outputs['render'] / 'video' / name
        require(info.get('path') == str(path) and info.get('full_ffmpeg_decode_passed') is True
            and info.get('playback_fps') == 30, 'Native video full decode or30fps evidence missing')
        expected = exported['observations'] if name == VIDEOS[0] else math.ceil(exported['actual_duration_s'] * 30)
        require(info.get('frames') == expected and info.get('duration_s') == expected / 30,
            'Native/ZOH video clock and frame count differ')
        if name != VIDEOS[0]:
            require(0 <= info.get('max_observation_age_s', 1) <= .060 + 1e-7 and
                info.get('method') == 'zero-order hold actual sample clock', 'Video no longer follows actual observation clock')
        pinned_file(path, info['sha256'], size=info['bytes'])
    resources = row['resource_receipts']
    require(set(resources) == set(STAGES), 'All four separate bounded stage resource proofs required')
    resource_rows = {stage: resource_proof(resources[stage], stage, candidate, outputs, spec) for stage in STAGES}
    exit_proof = quality['collection_external_exit_proof']
    require(exit_proof['receipt_path'] == resource_rows['collect']['receipt_path'] and
        exit_proof['receipt_sha256'] == resource_rows['collect']['receipt_sha256'] and
        exit_proof.get('actual_external_process_exit_code') == 0, 'Quality and publication collection-exit proof differ')
    proofs = [candidate / name for name in ('sample_spec.json', 'expert_report.json', 'inventory_snapshot.json',
        'physical_collection_seal.json', 'raw_capture_terminal_receipt.json', 'recording_report.json',
        'scene_sample.json', 'scene_info.json', 'randomized_scene_receipt.json',
        'randomized_environment_execution.json', 'actual_hardware_camera_clock_receipt.json',
        'actual_hardware_camera_clock_after_receipt.json', 'nearest_periodic_branch_receipt.json',
        'independent_motion_quality.json', 'dataset_validation.json', 'upstream_acceptance.json')]
    for path in (candidate / 'actual36_geometry_audit_summary.json',
                 outputs['export'] / 'staged_export_binding.json', outputs['render'] / 'staged_export_binding.json'):
        if path.exists():
            proofs.append(path)
    proofs += [quality_path, export_receipt_path, render_path, outputs['render'] / 'export_receipt.json',
               motion_path, dataset_path, *(Path(v) for v in resources.values())]
    for path in proofs:
        require(path.is_file() and not path.is_symlink() and path.stat().st_size <= JSON_MAX,
                'Proof metadata missing or exceeds bounded copy budget')
    instruction = outputs['export'] / 'instruction/episode_0000000.json'
    pinned_file(instruction, exported['outputs']['instruction']['sha256'])
    require(exported['outputs']['instruction']['path'] == str(instruction), 'Instruction source belongs elsewhere')
    files = [(hdf, f'data/episode_{slot:07d}.hdf5'),
             (instruction, f'instruction/episode_{slot:07d}.json')]
    for name in VIDEOS:
        suffix = '' if name == VIDEOS[0] else '_' + name[:-4]
        files.append((outputs['render'] / 'video' / name, f'video/episode_{slot:07d}{suffix}.mp4'))
    trajectory = candidate / '_traj_data/episode0.pkl'
    require(trajectory.is_file() and sha(trajectory) == seal['files']['_traj_data/episode0.pkl']['sha256'], 'Original native trajectory missing or changed')
    files.append((trajectory, f'_traj_data/episode{slot}.pkl'))
    # File names are namespaced by phase to keep all copied source proof bytes.
    for i, path in enumerate(proofs):
        files.append((path, f'episodes/episode_{slot:07d}/proof/{i:02d}_{path.name}'))
    protected = set(report['source_hashes']) | {str(candidate / relative) for relative in seal['files']} | \
        {str(candidate / '.cache/episode0' / name) for name in cache_pins} | set(exported['protected_raw_inputs_sha256'])
    snapshots = {name: {'device': Path(name).stat().st_dev, 'inode': Path(name).stat().st_ino,
        'bytes': Path(name).stat().st_size, 'mtime_ns': Path(name).stat().st_mtime_ns} for name in sorted(protected)}
    return {'slot': slot, 'metadata': {k: spec[k] for k in ('index', 'arm', 'pose_class', 'brand', 'model_id', 'seed', 'source_seed')},
        'candidate_directory': str(candidate), 'export_output': str(outputs['export']),
        'quality_output': str(outputs['quality']), 'render_output': str(outputs['render']),
        'all_original_physical_RGB_SDK_motion_DR_video_resource_gates_passed': True,
        'header_contract': header, 'source_proofs': source, 'resources': resource_rows,
        'quality_receipt_sha256': sha(quality_path), 'render_receipt_sha256': sha(render_path),
        'physical_origin_retained_and_copy_does_not_rewrite_absolute_proof_paths': True,
        'large_raw_trace_and_full_actual_audit_not_copied_or_JSON_loaded': True,
        'verified_raw_and_source_file_stat_before_copy': snapshots,
        'large_original_evidence_references': {name: seal['files'][name] for name in
            ('physics_trace.json', 'actual36_geometry_audit.json', 'actual36_geometry_audit_v13_original.json')},
        'copy_files': [{'source': str(path), 'destination_relative': name,
            'sha256': sha(path), 'bytes': path.stat().st_size} for path, name in files]}


def request_rows(value):
    require(value.get('schema') == 'staged-five-release-request-v1' and isinstance(value.get('candidates'), list), 'Wrong release request schema')
    rows = value['candidates']
    require(len(rows) == 5 and all(type(row.get('slot')) is int for row in rows)
        and sorted(row['slot'] for row in rows) == list(SLOTS), 'Exactly one complete record for slots17,18,20,29,34 is required')
    require(len({row['candidate'] for row in rows}) == 5, 'Duplicate physical candidate reused across slots')
    return sorted(rows, key=lambda row: row['slot'])


def copy_exact(source, destination, digest, size):
    source, destination = Path(source), Path(destination)
    pinned_file(source, digest, size=size)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open('rb') as origin, destination.open('xb') as target:
        shutil.copyfileobj(origin, target, length=1024**2)
    pinned_file(destination, digest, size=size)
    pinned_file(source, digest, size=size)


def publish_request(request_json, output, *, publish=False):
    request_json, output = Path(request_json), Path(output)
    require(not output.exists() and not output.is_symlink() and output.parent.is_dir()
        and not output.parent.is_symlink(), 'Patch destination must be new with an existing real parent')
    output = output.parent.resolve() / output.name
    require(output.parent in (E, V5) or output.is_relative_to(E / 'releases') or output.is_relative_to(V5 / 'patches'),
        'Use a new dedicated patch destination at evidence/V5 root or releases/patches')
    value = read_json(request_json)
    rows = request_rows(value)
    pinned_file(ORIGINAL50, ORIGINAL50_SHA)
    old = read_json(ORIGINAL50)
    require(old.get('complete') is True and old.get('count') == 50 and
        {row['index'] for row in old['episodes']} == set(range(50)), 'Original50 reference is not the preserved complete manifest')
    originals = {row['index']: row for row in old['episodes']}
    producer = sha(__file__)
    admissions = [verify_candidate(row, originals[row['slot']]) for row in rows]
    for row in rows:
        require(not output.is_relative_to(Path(row['candidate'])) and
            all(not output.is_relative_to(Path(row[k])) for k in ('export_output', 'quality_output', 'render_output')),
            'Patch destination overlaps immutable inputs')
    manifest = {'schema': 'realman-staged-randomized-five-patch-v1', 'complete': bool(publish),
        'validated': True, 'count': 5, 'accepted': bool(publish), 'original_slots': list(SLOTS),
        'task': 'adjust_bottle', 'full50_randomized': False, 'original50_replaced_or_merged': False,
        'outside_scope_original45_not_claimed_motion_quality_passed': True,
        'hardware_io': False, 'new_physics_episodes': True, 'source_policy': 'V5 genuine randomized physical captures only',
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'original50_reference': {'path': str(ORIGINAL50), 'sha256': ORIGINAL50_SHA, 'copied_file': 'original50_reference_manifest.json'},
        'request_path': str(request_json.resolve()), 'request_sha256': sha(request_json),
        'publisher_path': str(Path(__file__).resolve()), 'publisher_sha256': producer,
        'training_contract': {'SDK_action_dim': 24, 'native_DOF': 36, 'RGB_views': 3,
            'RGB_shape': [480, 640, 3], 'physics_dt_s': .004, 'observation_dt_s': .060,
            'native_HDF_and_instruction_and_video_bytes_unchanged': True},
        'official_domain_randomization': DR, 'episodes': admissions}
    if not publish:
        return manifest
    stage = Path(tempfile.mkdtemp(prefix=output.name + '.partial.', dir=output.parent))
    try:
        copy_exact(ORIGINAL50, stage / 'original50_reference_manifest.json', ORIGINAL50_SHA, ORIGINAL50.stat().st_size)
        for episode in admissions:
            for row in episode['copy_files']:
                copy_exact(row['source'], stage / row['destination_relative'], row['sha256'], row['bytes'])
            for name, snapshot in episode['verified_raw_and_source_file_stat_before_copy'].items():
                stat = Path(name).stat()
                require(not Path(name).is_symlink() and snapshot == {'device': stat.st_dev, 'inode': stat.st_ino,
                    'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}, 'Protected raw/source changed while copying release')
        require(sha(ORIGINAL50) == ORIGINAL50_SHA and sha(__file__) == producer and
            sha(request_json) == manifest['request_sha256'], 'Publisher/request/original50 changed during patch copy')
        (stage / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        require(not output.exists(), 'Patch destination appeared during atomic preparation')
        stage.rename(output)
        return {'complete': True, 'accepted': True, 'count': 5, 'output': str(output),
                'manifest_sha256': sha(output / 'manifest.json'), 'full50_randomized': False,
                'original50_replaced_or_merged': False}
    except BaseException:
        (stage / 'publication_incomplete.json').write_text(json.dumps({'complete': False, 'accepted': False,
            'original50_unchanged_sha256': sha(ORIGINAL50), 'requested_output': str(output),
            'partial_copy_preserved': str(stage)}, ensure_ascii=False, indent=2), encoding='utf-8')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    result = publish_request(args.request, args.output, publish=args.publish)
    print(json.dumps({key: result[key] for key in ('complete', 'count', 'full50_randomized')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
