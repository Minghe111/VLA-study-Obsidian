"""Audit five genuine new physical candidates, then optionally publish a patch.

The default performs readback only. --publish requires exactly one independently
accepted candidate for each original slot. All acceptance runs at the original
physical recording locations; copied proofs keep their original bytes/paths.
Original 50 and existing training files are never merged or rewritten here.
"""
from __future__ import annotations

import argparse
import ast
import copy
import datetime
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import textwrap
import types
import uuid

import numpy as np
import yaml

TASK_ROOT = Path(__file__).resolve().parent.parent
SOURCE = TASK_ROOT / 'source'
PACKAGE = SOURCE / 'rgb50_20261007_v1'
LEGACY = SOURCE / 'adjust_bottle_20261002'
REPOSITORY = Path('/home/admin123/liminghe/vla-benchmark/RoboTwin')
ARCHIVE = Path('/home/admin123/liminghe/VLA-WAM-RL/30-实验记录/证据/2026-10-11-RealMan专家回放')
SOURCE50 = ARCHIVE / 'source_manifest.json'
SOURCE50_SHA = '2339d5749b650761011e3975cbeb8d42ceb31dbff4f1f2aba6d71cac9a3edf80'
SLOTS = (17, 18, 20, 29, 34)
BINDINGS = {17: ('Fanta', 'left'), 18: ('Fanta', 'left'), 20: ('Fanta', 'right'),
            29: ('Coca-Cola', 'left'), 34: ('Coca-Cola', 'left')}
DR = {'random_background': True, 'cluttered_table': True, 'clean_background_rate': .02,
      'random_head_camera_dis': 0, 'random_table_height': .03, 'random_light': True,
      'crazy_random_light_rate': .02}


def require(value, message):
    if not bool(value):
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def original50(path):
    require(sha(path) == SOURCE50_SHA, 'Original50 manifest differs from the verified reference')
    value = read(path)
    require(value.get('complete') is True and value.get('count') == 50, 'Original50 manifest is incomplete')
    rows = {row['index']: row for row in value['episodes']}
    require(set(rows) == set(range(50)), 'Original50 slot identities differ')
    return rows


def verify_source_bundle():
    receipt = read(SOURCE / 'download_receipt.json')
    files = receipt['files']
    require(len(files) == 601, 'Reviewed original collector bundle must contain exactly601 files')
    for name, proof in files.items():
        path = SOURCE / name
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe collector source relative path')
        require(path.is_file() and not path.is_symlink() and sha(path) == proof['sha256'], 'Original collector source changed: ' + name)
    return {'source_receipt': str(SOURCE / 'download_receipt.json'),
            'source_receipt_sha256': sha(SOURCE / 'download_receipt.json'),
            'source_file_count': len(files), 'all_original_source_sha256_matched': True}


class BindSourcePaths(ast.NodeTransformer):
    """Only relocates source root constants; never touches recorded proof paths."""
    def __init__(self):
        self.sites = []

    def visit_Constant(self, node):
        if not isinstance(node.value, str):
            return node
        local = node.value
        for remote, target in (
            ('/bigdata2/liminghe/VLA-benchmark/RoboTwin', REPOSITORY),
            ('/bigdata/liminghe/VLA-benchmark/RoboTwin', REPOSITORY),
            ('/bigdata2/liminghe/VLA-benchmark/deploy/realman_gen4', SOURCE),
            ('/bigdata/liminghe/VLA-benchmark/deploy/realman_gen4', SOURCE),
        ):
            if local == remote or local.startswith(remote + '/'):
                local = str(target) + local[len(remote):]
                break
        if local != node.value:
            self.sites.append({'line': node.lineno, 'original': node.value, 'local': local})
            return ast.copy_location(ast.Constant(value=local), node)
        return node


def install_original_reader(bindings):
    sys.path[:0] = [str(PACKAGE), str(LEGACY), str(TASK_ROOT / 'quality'), str(REPOSITORY)]
    audit = importlib.import_module('dataset50_audit_v3')
    reader = importlib.import_module('dataset50_reader_v3')
    cpu = importlib.import_module('dataset50_boundary_cpu_v2')

    def exact_recorded_path(raw, directory):
        path = Path(raw)
        path = (path if path.is_absolute() else REPOSITORY / path).resolve()
        require(path.is_relative_to(Path(directory).resolve()), 'Recorded geometry proof escapes original source candidate')
        return path

    reader.audit_geometry_chain.__globals__['resolve_recorded_path'] = exact_recorded_path
    bindings.append({'binding': 'recorded_relative_proof_producer_cwd', 'producer_cwd': str(REPOSITORY),
        'recorded_path_strings_modified': False, 'original_reader_source': str(reader.__file__),
        'original_reader_sha256': sha(reader.__file__), 'path_resolution_only': True})

    def bound_module(path, name):
        path = Path(path)
        before = sha(path)
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        transform = BindSourcePaths()
        tree = transform.visit(tree)
        ast.fix_missing_locations(tree)
        module = types.ModuleType(name)
        module.__file__, module.__package__ = str(path), ''
        sys.modules[name] = module
        # inspect.getsource in the frozen CPU verifier must still read the
        # original, SHA-pinned constructor file, not a synthetic filename.
        exec(compile(tree, str(path), 'exec'), module.__dict__)
        require(sha(path) == before, 'Bound CPU helper original bytes changed')
        if path.name == 'benchmark_geometry_cache.py':
            require(len(transform.sites) == 1 and transform.sites[0]['original'].endswith('/RoboTwin'),
                    'Exactly one frozen CPU constructor root binding required')
        bindings.append({'binding': 'CPU_helper_source_root_constants', 'source': str(path),
            'original_source_sha256': before, 'sites': transform.sites,
            'compiled_ast_sha256': hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest(),
            'source_bytes_modified': False})
        return module

    cpu.module_from = bound_module
    source, start = inspect.getsourcelines(cpu.verify_cpu_event)
    tree = ast.parse(textwrap.dedent(''.join(source)))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'robotwin':
            require(ast.unparse(node.value) == "work.parents[2] / 'RoboTwin'", 'Frozen CPU verifier root expression changed')
            found.append({'line': start + node.lineno - 1, 'original_expression': ast.unparse(node.value),
                          'local': str(REPOSITORY)})
            node.value = ast.Call(func=ast.Name(id='Path', ctx=ast.Load()), args=[ast.Constant(value=str(REPOSITORY))], keywords=[])
    require(len(found) == 1, 'Exactly one CPU verifier checkout-root binding required')
    ast.fix_missing_locations(tree)
    namespace = dict(cpu.verify_cpu_event.__globals__)
    exec(compile(tree, str(cpu.__file__) + ':checkout_root_binding_only', 'exec'), namespace)
    cpu.verify_cpu_event = namespace['verify_cpu_event']
    bindings.append({'binding': 'CPU_verifier_checkout_root', 'source': str(cpu.__file__),
        'original_source_sha256': sha(cpu.__file__), 'sites': found,
        'compiled_ast_sha256': hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest(),
        'source_bytes_modified': False, 'physics_thresholds_or_proofs_changed': False})
    return audit


def slot_gate(spec, baseline):
    index = spec.get('index')
    require(type(index) is int and index in SLOTS, 'Candidate is outside the authorized five slots')
    require((spec.get('brand'), spec.get('arm')) == BINDINGS[index] and spec.get('pose_class') == 'upright', 'Candidate slot taxonomy changed')
    require(spec.get('source_slot_index') == index and spec.get('source_seed') == baseline[index]['seed'], 'Original slot/seed provenance differs')
    require(type(spec.get('seed')) is int and spec.get('actual_domain_seed') == spec['seed'] and 0 <= spec['seed'] < 2**32, 'Actual domain seed missing/invalid')
    require(type(spec.get('rerecord_attempt')) is int and spec['rerecord_attempt'] >= 0, 'Rerecord attempt missing/invalid')
    policy = spec.get('domain_seed_policy')
    require(policy in ('source_seed + 10000000 * attempt', 'explicit --domain-seed'), 'Unreviewed domain seed policy')
    if policy != 'explicit --domain-seed':
        require(spec['seed'] == spec['source_seed'] + 10_000_000 * spec['rerecord_attempt'], 'Actual domain seed violates attempt policy')
    return index


def randomization_gate(directory, spec, original_audit):
    scene = read(directory / 'randomized_scene_receipt.json')
    execution = read(directory / 'randomized_environment_execution.json')
    coverage = read(directory / 'randomized_environment_coverage.json')
    config = yaml.safe_load((directory / 'run_config.yml').read_text())
    require(scene.get('passed') is True and scene.get('hardware_io') is False, 'Actual randomized scene receipt rejected')
    require(scene.get('official_domain_randomization') == scene.get('actual_domain_randomization') == DR == config.get('domain_randomization'), 'Actual official DR/config differs')
    require(scene.get('source_slot_index') == spec['index'] and scene.get('source_seed') == spec['source_seed'] and scene.get('actual_seed') == spec['seed'], 'Actual scene random seed provenance differs')
    require(scene.get('original_source_pose_restored') is False and scene.get('objects_settled_by_original_gravity_and_stability_chain') is True, 'Fresh randomized initial state evidence missing')
    bias = float(scene['table_z_bias_m'])
    base_bias = float(config.get('table_height_bias', 0))
    require(math.isfinite(bias) and base_bias - .03 - 1e-7 <= bias <= base_bias + 1e-7, 'Actual randomized table bias outside official range')
    count = scene.get('clutter_count')
    require(type(count) is int and count >= 0 and count == len(scene['official_clutter_records']), 'Actual clutter count/record mismatch')
    require(coverage.get('passed') is True and coverage.get('all_collidable_clutter_covered') is True and
            coverage.get('expected_clutter_count') == coverage.get('covered_clutter_count') == count, 'Incomplete generated clutter shape coverage')
    require(execution.get('passed') is True and execution.get('hardware_io') is False and
            execution.get('every_actual_physics_step_checked') is True and execution.get('complete_physics_telemetry_coverage_verified') is True,
            'Randomized execution lacks complete actual-step coverage')
    require(execution.get('steps_checked') == execution.get('original_report_physical_steps') == original_audit['physical_steps'], 'Clutter guard physical endpoint differs')
    require(execution.get('robot_clutter_active_contacts') == execution.get('bottle_clutter_active_contacts') == 0, 'Actual clutter collision occurred')
    trace = read(directory / 'physics_trace.json')
    distances = []
    for position, row in enumerate(trace, 1):
        item = row.get('randomized_environment', {})
        require(row['step'] == position and item.get('native_clutter_components_checked') == coverage['covered_collision_component_count'], 'Actual-step clutter geometry coverage differs')
        require(item.get('active_target_or_robot_clutter_contacts') == [], 'Actual trace contains clutter contact')
        distance = float(item['max_vertex_displacement_from_initial_m'])
        require(math.isfinite(distance) and 0 <= distance <= .001 + 1e-9, 'Actual clutter drift exceeds conservative obstacle budget')
        distances.append(distance)
    require(len(trace) == original_audit['physical_steps'] and math.isclose(max(distances), execution['max_clutter_vertex_displacement_m'], abs_tol=1e-12, rel_tol=0), 'Actual clutter drift summary does not reconstruct')
    return {'passed': True, 'actual_domain_randomization': DR, 'actual_seed': spec['seed'],
            'clutter_count': count, 'clean_sampling_outcome': scene['clutter_sampling_outcome'],
            'official_clean_2percent_draws_are_valid': True, 'physical_steps_checked': len(trace),
            'complete_clutter_coverage_and_zero_contacts_reconstructed': True, 'table_z_bias_m': bias,
            'original_initial_scene_match_is_not_required': True}


def hardware_gate(directory, original_audit):
    before = read(directory / 'actual_hardware_camera_clock_receipt.json')
    after = read(directory / 'actual_hardware_camera_clock_after_receipt.json')
    keys = ('frozen_sapien_urdf', 'frozen_sapien_urdf_sha256', 'actual_joint_names',
            'actual_arm_drives', 'actual_hand_drives', 'actual_tendon_installation_receipt',
            'current_native_collision_materials', 'current_native_collision_offsets_and_groups',
            'current_native_link_mass_and_inertia')
    for value in (before, after):
        require(value.get('passed') is True and value.get('hardware_io') is False and
                value.get('actual_joint_count') == 36 and value.get('actual_link_count') == 93 and
                value.get('observation_physics_step_stride') == 15 and value.get('vendor_inertia_preservation_enabled') is True,
                'Actual hardware/camera/clock contract rejected')
        require(abs(value['actual_physics_timestep_s'] - .004) < 1e-9, 'Actual4ms physics clock changed')
        require(sha(value['frozen_sapien_urdf']) == value['frozen_sapien_urdf_sha256'], 'Frozen SAPIEN URDF proof changed')
    require(all(before[key] == after[key] for key in keys), 'Static hardware mass/drive/collision contract changed during episode')
    require(original_audit['physical_steps'] == 15 * original_audit['state_action_pairs'], 'SDK24/60ms timeline mismatch')
    return {'passed': True, 'unchanged_static_readback_fields': list(keys), 'physics_dt_s': .004,
            'observation_dt_s': original_audit['observation_dt_s'], 'sdk24_joint_ordering': 'left arm6 rad,left hand6 SDK,right arm6 rad,right hand6 SDK',
            'hardware_io': False, 'real_hardware_calibration_claimed': False}


def inspect_video(path, ffmpeg):
    import cv2
    capture = cv2.VideoCapture(str(path))
    require(capture.isOpened(), 'Video cannot be opened: ' + str(path))
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    require(width > 0 and height > 0 and frames > 0 and math.isfinite(fps) and fps > 0, 'Invalid actual video metadata')
    command = [str(ffmpeg), '-nostdin', '-v', 'error', '-xerror', '-i', str(path), '-map', '0:v:0',
               '-progress', 'pipe:1', '-nostats', '-f', 'null', '-']
    result = subprocess.run(command, capture_output=True, text=True, timeout=600)
    require(result.returncode == 0 and not result.stderr.strip(), 'Full video decode failed: ' + result.stderr[-1500:])
    decoded = [int(line.split('=', 1)[1].strip()) for line in result.stdout.splitlines() if line.startswith('frame=')]
    require(decoded and decoded[-1] > 0 and abs(decoded[-1] - frames) <= 1 and 'progress=end' in result.stdout, 'Video full decode did not reach its container endpoint')
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size,
            'cv2_width': width, 'cv2_height': height, 'cv2_frame_count': frames, 'cv2_fps': fps,
            'actual_ffmpeg_full_decode_frame_count': decoded[-1], 'full_decode_passed': True,
            'duration_from_container_s': frames / fps, 'ffmpeg': str(ffmpeg), 'ffmpeg_sha256': sha(ffmpeg),
            'timeline_semantics': 'Video preview only; authoritative15x4ms timeline is exact raw/HDF/physicsNPZ, independently read back.'}


def audit_candidate(path, baseline, original_reader, ffmpeg, codec):
    path = Path(path)
    result = {'passed': False, 'validated_source_candidate': str(path.resolve()), 'hardware_io': False}
    try:
        require(not path.is_symlink(), 'Candidate itself is a symlink')
        path = path.resolve()
        spec = read(path / 'sample_spec.json')
        index = slot_gate(spec, baseline)
        result.update(original_slot_index=index, attempt=spec['rerecord_attempt'], actual_domain_seed=spec['seed'])
        recorded_quality = read(path / 'independent_motion_quality.json')
        require(recorded_quality.get('overall_publish_gate_passed') is True, 'Recorded independent motion quality rejected')
        require(recorded_quality.get('numeric_source') == str(path / 'data/episode_0000000.hdf5') and
                recorded_quality.get('numeric_source_sha256') == sha(path / 'data/episode_0000000.hdf5'), 'Recorded motion quality is not from this actual source candidate/HDF')
        recording = read(path / 'recording_report.json')
        require(recording.get('accepted') is True and recording.get('original_recording_export_accepted') is True and
                recording.get('additional_quality_report') == str(path / 'independent_motion_quality.json') and
                recording.get('additional_quality_report_sha256') == sha(path / 'independent_motion_quality.json'), 'Actual recording/quality SHA linkage differs')
        audit = original_reader.audit_trial(path, codec_path=codec, allow_synthetic=False)
        require(audit.get('passed') is True, 'Original v3 physics/RGB/SDK24/full36 readback failed: ' + str(audit.get('error')))
        require(audit.get('synthetic_fixture_only') is False and audit['identity']['spec_index'] == index, 'Original reader identity/synthetic status differs')
        from motion_gate import evaluate_candidate
        quality = evaluate_candidate(path / 'data/episode_0000000.hdf5',
            {key: spec[key] for key in ('index', 'arm', 'pose_class', 'brand', 'seed')},
            config_path=TASK_ROOT / 'quality/thresholds_confirmed.json',
            branch_receipt_path=path / 'nearest_periodic_branch_receipt.json', upstream_acceptance_path=path / 'upstream_acceptance.json')
        require(quality['overall_publish_gate_passed'] and quality['config_sha256'] == recorded_quality['config_sha256'], 'Independent fresh motion gate/config readback rejected')
        random = randomization_gate(path, spec, audit)
        hardware = hardware_gate(path, audit)
        videos = [inspect_video(path / name, ffmpeg) for name in audit['source_files'] if name.startswith('video/') and name.endswith('.mp4')]
        require(videos, 'No accepted physical candidate preview video')
        require(original_reader.inventory(path) == audit['source_files'], 'Source candidate changed during independent patch audit')
        old_hdf = baseline[index]['published_files'][f'data/episode_{index:07d}.hdf5']['sha256']
        require(audit['source_hdf5_sha256'] != old_hdf, 'Fresh randomized HDF duplicates the old50 HDF bytes')
        result.update(passed=True, source_candidate_retained=True, source_files=audit['source_files'],
            source_hdf5_sha256=audit['source_hdf5_sha256'], original50_hdf5_sha256=old_hdf,
            original_readback=audit, recomputed_independent_motion_quality=quality,
            actual_randomization=random, actual_hardware_clock=hardware, videos=videos,
            acceptance_performed_at_actual_recording_location=True, recorded_proof_paths_rewritten=False)
    except Exception as error:
        result.update(error=str(error), error_type=type(error).__name__)
    return result


def choose_five(rows):
    selected = []
    for index in SLOTS:
        candidates = [row for row in rows if row.get('passed') is True and row.get('original_slot_index') == index]
        if not candidates:
            return []
        selected.append(min(candidates, key=lambda row: (row['attempt'], row['validated_source_candidate'])))
    require(len({row['validated_source_candidate'] for row in selected}) == 5, 'Duplicate physical candidate selected')
    require(len({row['actual_domain_seed'] for row in selected}) == 5, 'Actual seeds are not unique across the five selected candidates')
    return selected


def transfer(path, target, expected):
    require(not path.is_symlink() and path.is_file() and sha(path) == expected['sha256'], 'Source changed before independent copy')
    target.parent.mkdir(parents=True, exist_ok=True)
    require(not target.exists(), 'Refusing to overwrite a patch file')
    shutil.copy2(path, target)
    require(target.stat().st_size == expected['bytes'] and sha(target) == expected['sha256'], 'Independent copied file differs')


def publish_patch(selected, output, audit_report, source50_path, original_reader):
    output = Path(output).resolve()
    require(len(selected) == 5 and {row['original_slot_index'] for row in selected} == set(SLOTS), 'Five accepted original slots required before publication')
    require(all(row.get('passed') is True and row.get('original_readback', {}).get('passed') is True and
                row.get('recomputed_independent_motion_quality', {}).get('overall_publish_gate_passed') is True
                for row in selected), 'Every selected candidate needs complete independent physics and motion acceptance')
    require(output.is_relative_to(TASK_ROOT / 'published') and not output.exists(), 'Publication requires a new directory under this task published/')
    for row in selected:
        candidate = Path(row['validated_source_candidate'])
        require(not output.is_relative_to(candidate) and not candidate.is_relative_to(output), 'Patch destination overlaps source candidate')
    require(not output.is_relative_to(source50_path.parent), 'Patch destination overlaps original50 source/archive')
    before50 = sha(source50_path)
    stage = output.parent / ('.' + output.name + '.staging-' + uuid.uuid4().hex)
    stage.mkdir(parents=True, exist_ok=False)
    manifest = {'schema': 'realman-randomized-rerecord5-patch-v1', 'complete': False,
        'status': 'staging_not_complete', 'task': 'adjust_bottle', 'count': 5, 'original_slot_indices': list(SLOTS),
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'full50_randomized': False, 'original45_randomized_or_quality_passed_is_not_claimed': True,
        'original50_manifest': str(source50_path), 'original50_manifest_sha256': SOURCE50_SHA,
        'original50_and_training_files_written': False, 'automatic45plus5_merge_performed': False,
        'hardware_io': False, 'sdk_mapping_calibrated': False, 'training_cameras': ['cam_head', 'cam_left_wrist', 'cam_right_wrist'],
        'physics_dt_s': .004, 'observation_dt_s': .060, 'rgb_only': True,
        'publisher_source_sha256': sha(__file__), 'transfer_mode': 'independent_source_copies_internal_patch_links_only',
        'source_candidate_policy': 'Retain all original accepted/failed candidates; acceptance runs at actual recording directories; copied original proof paths/bytes remain intact.',
        'episodes': []}
    try:
        transfer(source50_path, stage / 'original50_reference_manifest.json',
                 {'sha256': SOURCE50_SHA, 'bytes': source50_path.stat().st_size})
        manifest['original50_reference_manifest'] = 'original50_reference_manifest.json'
        manifest['original50_reference_manifest_sha256'] = sha(stage / 'original50_reference_manifest.json')
        write(stage / 'independent_patch_validation.json', audit_report)
        manifest['independent_patch_validation_sha256'] = sha(stage / 'independent_patch_validation.json')
        for row in selected:
            index = row['original_slot_index']
            origin = Path(row['validated_source_candidate'])
            require(original_reader.inventory(origin) == row['source_files'], 'Validated source candidate changed before publication')
            relative = Path('episodes') / f'episode_{index:07d}'
            for name, proof in row['source_files'].items():
                transfer(origin / name, stage / relative / name, proof)
            proof_path = relative / 'independent_patch_acceptance.json'
            require(not (stage / proof_path).exists(), 'Reserved patch receipt name appeared in source candidate')
            write(stage / proof_path, {key: value for key, value in row.items() if key != 'source_files'})
            mapping = {'data/episode_0000000.hdf5': f'data/episode_{index:07d}.hdf5',
                       'instruction/episode_0000000.json': f'instruction/episode_{index:07d}.json',
                       '_traj_data/episode0.pkl': f'_traj_data/episode{index}.pkl'}
            for name in row['source_files']:
                if name.startswith('video/') and name.endswith('.mp4'):
                    require(Path(name).parent == Path('video') and Path(name).name.startswith('episode_0000000'), 'Unreviewed nested/native video naming')
                    mapping[name] = 'video/' + Path(name).name.replace('episode_0000000', f'episode_{index:07d}', 1)
            require(len(set(mapping.values())) == len(mapping), 'Published top-level filename collision')
            top = {}
            for name, target in mapping.items():
                require(name in row['source_files'], 'Required native export missing from validated inventory')
                dest = stage / target
                dest.parent.mkdir(parents=True, exist_ok=True)
                os.link(stage / relative / name, dest)
                require(sha(dest) == row['source_files'][name]['sha256'], 'Internal published-file readback differs')
                top[target] = row['source_files'][name]
            manifest['episodes'].append({'index': index, 'original_slot_index': index,
                'validated_source_candidate': str(origin), 'source_candidate_retained': True,
                'physical_recording_location': str(origin), 'copied_proof_paths_rewritten': False,
                'accepted': True, 'physical_success': True, 'full_official_scene_randomization_enabled': True,
                'actual_domain_seed': row['actual_domain_seed'], 'rerecord_attempt': row['attempt'],
                'original50_hdf5_sha256': row['original50_hdf5_sha256'],
                'source_hdf5_sha256': row['source_hdf5_sha256'], 'copied_hdf5_sha256': top[f'data/episode_{index:07d}.hdf5']['sha256'],
                'copiedHDF_exactly_same_as_validated_source': True, 'state_action_pairs': row['original_readback']['state_action_pairs'],
                'raw_observations': row['original_readback']['raw_observations'], 'physical_steps': row['original_readback']['physical_steps'],
                'episode_proof_dir': str(relative), 'independent_patch_acceptance': str(proof_path),
                'independent_patch_acceptance_sha256': sha(stage / proof_path), 'source_files': row['source_files'], 'published_files': top,
                'preview_videos': row['videos']})
            require(original_reader.inventory(origin) == row['source_files'], 'Source changed during publication')
        require(sha(source50_path) == before50 == SOURCE50_SHA, 'Original50 manifest changed during publication')
        require(len(manifest['episodes']) == 5, 'Incomplete patch count')
        manifest.update(complete=True, status='complete_independently_validated_randomized5_patch_only')
        write(stage / 'manifest.json', manifest)
        require(not output.exists(), 'Patch target appeared before atomic rename')
        stage.rename(output)
        return {'complete': True, 'count': 5, 'output': str(output), 'manifest_sha256': sha(output / 'manifest.json'), 'full50_randomized': False}
    except Exception as error:
        manifest.update(complete=False, status='publication_failed_staging_preserved')
        write(stage / 'manifest.json', manifest)
        write(stage / 'publication_error.json', {'error': str(error), 'source_candidates_deleted': False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials-root', type=Path, default=TASK_ROOT / 'new_physical_episodes')
    parser.add_argument('--source50-manifest', type=Path, default=SOURCE50)
    parser.add_argument('--audit-output', type=Path, default=TASK_ROOT / 'preflight/randomized5_patch_validation.json')
    parser.add_argument('--output', type=Path, default=TASK_ROOT / 'published/realman_randomized_rerecord5_patch_v1/dataset')
    parser.add_argument('--codec', type=Path, default=REPOSITORY / 'data/decode_image_bit.py')
    parser.add_argument('--ffmpeg', type=Path, default=Path('/home/admin123/liminghe/miniconda3/envs/robotwin-official-data/bin/ffmpeg'))
    parser.add_argument('--publish', action='store_true', help='Atomic new patch publication only after all five independent audits pass')
    args = parser.parse_args()
    baseline = original50(args.source50_manifest)
    source_guard = verify_source_bundle()
    require(args.ffmpeg.is_file(), 'An actually installed ffmpeg executable is required')
    tools = subprocess.run([str(args.ffmpeg), '-version'], capture_output=True, text=True, timeout=15)
    require(tools.returncode == 0, 'Installed ffmpeg cannot execute')
    bindings = []
    reader = install_original_reader(bindings)
    root = args.trials_root.resolve()
    require(root.is_relative_to(TASK_ROOT / 'new_physical_episodes'), 'Candidate discovery must remain in the actual task collection area')
    paths = sorted(path for path in root.glob('slot_*_randomized_attempt_*') if path.is_dir())
    require(all(not path.is_symlink() for path in paths), 'Candidate discovery found a symlink')
    require(args.audit_output.resolve().is_relative_to(TASK_ROOT / 'preflight'), 'Audit receipt must remain in task preflight; original50/training/source outputs are forbidden')
    require(not args.audit_output.resolve().is_relative_to(root), 'Audit receipt must not alter candidate inventories')
    rows = [audit_candidate(path, baseline, reader, args.ffmpeg, args.codec) for path in paths]
    selected = choose_five(rows)
    audit = {'schema': 'randomized5-patch-independent-validation-v1', 'complete': len(selected) == 5,
        'scope': 'Five independent randomized physical replacements only; not a fully randomized50 dataset',
        'full50_randomized': False, 'original_slot_indices': list(SLOTS), 'candidate_count': len(rows),
        'accepted_candidate_count': sum(row.get('passed') is True for row in rows),
        'selected_source_candidates': [row['validated_source_candidate'] for row in selected], 'candidates': rows,
        'all_source_candidates_retained': True, 'original50_manifest': str(args.source50_manifest), 'original50_manifest_sha256': SOURCE50_SHA,
        'original_source_bundle': source_guard, 'offline_source_path_bindings': bindings,
        'validator_script_sha256': sha(__file__), 'new_physics_episodes': 0, 'model_inference_runs': 0, 'hardware_io': False,
        'independent_zero_step_CPU_boundary_reconstructions': sum(row.get('original_readback', {}).get('zero_step_cpu_event_reconstructions', 0) for row in rows),
        'ffmpeg_version': tools.stdout.splitlines()[0], 'publication_requested': args.publish,
        'candidate_inventory_or_proof_bytes_written': False}
    require(sha(args.source50_manifest) == SOURCE50_SHA, 'Original50 manifest changed during independent audit')
    write(args.audit_output, audit)
    if not selected:
        print(json.dumps({'complete': False, 'published': False, 'missing_accepted_slots': [index for index in SLOTS if not any(row.get('passed') is True and row.get('original_slot_index') == index for row in rows)], 'audit': str(args.audit_output)}, ensure_ascii=False))
        return 2
    if args.publish:
        print(json.dumps(publish_patch(selected, args.output, audit, args.source50_manifest, reader), ensure_ascii=False))
    else:
        print(json.dumps({'complete': True, 'published': False, 'count': 5, 'audit': str(args.audit_output), 'full50_randomized': False}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
