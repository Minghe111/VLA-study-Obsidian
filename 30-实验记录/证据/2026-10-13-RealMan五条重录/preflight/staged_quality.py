"""Independent bounded CPU readback after normal collect-only process exit.

This stage validates native RGB/SDK values and the frozen motion gates.  It does
not advance simulation, certify an interrupted cache, or accept a dataset before
the separate native-video stage has completed.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import math
from pathlib import Path
import resource
import shutil
import sys
import time
import traceback

import h5py

import streaming_export_staged as exporter

HERE = Path(__file__).resolve().parent
TASK = HERE.parent
NUMERIC_LIMIT = 256 * 1024**2
PUBLISH_HELPER_SHA256 = 'b3c9347c6bfa90f952cddb27faf77263fc5501cec4f9f92c2076731456420e3d'
DR = {'random_background': True, 'cluttered_table': True, 'clean_background_rate': .02,
      'random_head_camera_dis': 0, 'random_table_height': .03, 'random_light': True,
      'crazy_random_light_rate': .02}
HARDWARE_KEYS = ('frozen_sapien_urdf', 'frozen_sapien_urdf_sha256', 'actual_joint_names',
    'actual_arm_drives', 'actual_hand_drives', 'actual_tendon_installation_receipt',
    'current_native_collision_materials', 'current_native_collision_offsets_and_groups',
    'current_native_link_mass_and_inertia')
sha, require, read_json = exporter.base.sha, exporter.base.require, exporter.read_json


def put_json(path, value):
    """Never replace a previously produced physical/quality artifact."""
    path = Path(path)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if path.exists():
        require(path.is_file() and not path.is_symlink() and path.read_text() == payload,
                'Refusing to overwrite a different stage artifact: ' + str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as stream: stream.write(payload)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def option(command, name, *, required=True):
    values = []
    for index, token in enumerate(command):
        if token == name:
            require(index + 1 < len(command), 'Incomplete collector option ' + name)
            values.append(command[index + 1])
        elif token.startswith(name + '='):
            values.append(token.split('=', 1)[1])
    require(len(values) == 1 if required else len(values) <= 1,
            'Missing/duplicate collector option ' + name)
    return values[0] if values else None


def collected_output(command):
    explicit = option(command, '--output', required=False)
    if explicit is not None:
        return Path(explicit).resolve()
    # The guarded script uses an exact source-pinned argparse default. Read it
    # rather than assuming an absent option means an arbitrary known directory.
    entry = Path(command[1])
    tree = ast.parse(entry.read_text(encoding='utf-8'))
    defaults = [keyword.value for node in ast.walk(tree) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument'
        and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == '--output'
        for keyword in node.keywords if keyword.arg == 'default']
    require(len(defaults) == 1 and ast.unparse(defaults[0]) == "TASK_ROOT / 'new_physical_episodes'",
            'Absent collector --output has no unique exact reviewed parser default')
    return entry.resolve().parent.parent / 'new_physical_episodes'


def validate_collect_resource_receipt(path, candidate, spec):
    """Bind the successful external process exit to this exact immutable cache."""
    path, candidate = Path(path), Path(candidate).resolve()
    receipt = read_json(path)
    require(receipt.get('schema') == 'realman_guarded_stage_v1' and receipt.get('stage') == 'collect'
        and receipt.get('passed') is True and receipt.get('check_only') is False
        and receipt.get('cgroup_readback_verified') is True
        and receipt.get('launcher_returncode') == 0 and receipt.get('monitor_stop_reason') is None,
        'Collection resource receipt is not a verified successful bounded execution')
    command = receipt.get('command', [])
    require(isinstance(command, list) and len(command) >= 2
        and command[0] == '/home/admin123/liminghe/miniconda3/envs/robotwin/bin/python',
        'Unexpected collected process interpreter/command')
    entry = Path(command[1])
    require(entry.is_absolute() and entry.is_file() and not entry.is_symlink()
        and entry == entry.resolve() and entry.name == 'run_staged_collector.py',
        'Normal exit receipt does not identify the exact staged collection entry')
    runtime = entry.parent.parent
    index, attempt = int(option(command, '--index')), int(option(command, '--attempt'))
    output = collected_output(command)
    expected = output / f'slot_{index:07d}_randomized_attempt_{attempt:03d}'
    require(candidate == expected and candidate.is_relative_to(runtime)
        and index == spec['index'] and attempt == spec['rerecord_attempt']
        and command.count('--run') == 1 and '--prepare-runtime' not in command,
        'Collection process receipt belongs to another candidate or a static preflight')
    domain_seed = option(command, '--domain-seed', required=False)
    actual_seed = int(domain_seed) if domain_seed is not None else spec['source_seed'] + 10_000_000 * attempt
    require(actual_seed == spec['seed'], 'Actual collected random seed differs from process receipt')
    binding = receipt.get('source_binding', {})
    require(binding.get('entry_sha256') == sha(entry)
        and binding.get('guard_sha256') == sha(HERE / 'run_guarded_stage.py'),
        'Collected entry/resource guard changed after its verified execution')
    limits = receipt.get('limits', {})
    require(limits.get('MemoryMax') == str(12 * 1024**3)
        and limits.get('MemoryHigh') == str(10 * 1024**3)
        and limits.get('MemorySwapMax') == '0' and limits.get('CPUQuota') == '200%'
        and limits.get('KillMode') == 'control-group' and limits.get('OOMPolicy') == 'stop',
        'Original collection resource cap differs from the authorized desktop budget')
    worker = receipt.get('worker_terminal', {})
    require(worker.get('stage_returncode') == 0 and worker.get('source_unchanged') is True,
        'Actual collected worker did not exit successfully with unchanged source')
    events = worker.get('resource', {}).get('memory.events', {})
    require(all(events.get(name) == 0 for name in ('max', 'oom', 'oom_kill', 'oom_group_kill')),
        'Collected worker exceeded its hard memory cap or suffered OOM')
    unit = receipt.get('unit_terminal', {})
    require(unit.get('Result') == 'success' and unit.get('ExecMainStatus') == '0'
        and unit.get('ActiveState') == 'inactive', 'Task cgroup/unit has not completed normal cleanup')
    return {'passed': True, 'receipt_path': str(path.resolve()), 'receipt_sha256': sha(path),
            'candidate_directory': str(candidate), 'entry': str(entry), 'entry_sha256': sha(entry),
            'actual_external_process_exit_code': 0, 'normal_scene_close_is_separately_sealed': True,
            'limits': limits, 'worker_memory_events': events,
            'peak': receipt.get('peak'), 'unit_terminal': unit}


def collection_exit_proof(candidate, spec, resource_root):
    root = Path(resource_root).resolve()
    require(root.is_dir() and not root.is_symlink(), 'Actual completed collection resource ledger missing')
    found = []
    for path in sorted(root.glob('collect_*/receipt.json')):
        receipt = read_json(path)
        command = receipt.get('command', [])
        # Select candidates without treating unrelated failed trials as success.
        try:
            output = collected_output(command)
            index, attempt = int(option(command, '--index')), int(option(command, '--attempt'))
            match = output / f'slot_{index:07d}_randomized_attempt_{attempt:03d}' == candidate.resolve()
        except (ValueError, KeyError, TypeError):
            match = False
        if match and receipt.get('passed') is True:
            found.append(validate_collect_resource_receipt(path, candidate, spec))
    require(len(found) == 1, 'Exactly one successful external collection exit must bind this candidate')
    return found[0]


def numeric_budget(hdf):
    """Cap all numeric columns before unmodified numerical validators load them."""
    total, columns = 0, []
    with h5py.File(hdf, 'r') as handle:
        def visit(name, obj):
            nonlocal total
            if isinstance(obj, h5py.Dataset) and obj.dtype.kind in 'bifu':
                size = math.prod(obj.shape) * obj.dtype.itemsize
                require(size <= NUMERIC_LIMIT, 'One numeric HDF column exceeds fixed validator budget')
                total += size
                require(total <= NUMERIC_LIMIT, 'Numeric HDF columns exceed fixed validator budget')
                columns.append({'name': name, 'shape': list(obj.shape), 'bytes': size})
        handle.visititems(visit)
        require(0 < handle['state/left_arm_joint_states'].shape[0] <= exporter.base.MAX_FRAMES,
                'Original validator observation count exceeds reviewed budget')
    return {'passed': True, 'numeric_total_bytes': total, 'numeric_limit_bytes': NUMERIC_LIMIT,
            'columns': columns, 'RGB_columns_loaded_together': False}


def copy_exact(source, destination, expected):
    source, destination = Path(source), Path(destination)
    require(source.is_file() and not source.is_symlink() and sha(source) == expected,
            'Independent export artifact missing/changed before native candidate alias')
    if destination.exists():
        require(destination.is_file() and not destination.is_symlink() and sha(destination) == expected,
                'Refusing to overwrite another native candidate alias')
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source.open('rb') as origin, destination.open('xb') as target:
            shutil.copyfileobj(origin, target, length=1024**2)
        require(sha(destination) == expected, 'Byte-exact native candidate export alias differs')
    require(sha(source) == expected, 'Independent exporter artifact changed during alias creation')
    return {'source': str(source.resolve()), 'destination': str(destination.resolve()),
            'sha256': expected, 'copied_without_HDF_or_instruction_modification': True}


def upstream_proofs(candidate, report):
    before = read_json(candidate / 'actual_hardware_camera_clock_receipt.json')
    after = read_json(candidate / 'actual_hardware_camera_clock_after_receipt.json')
    for value in (before, after):
        require(value.get('passed') is True and value.get('hardware_io') is False
            and value.get('actual_joint_count') == 36 and value.get('actual_link_count') == 93
            and value.get('observation_physics_step_stride') == 15
            and value.get('vendor_inertia_preservation_enabled') is True
            and abs(value['actual_physics_timestep_s'] - .004) <= 1e-9,
            'Actual hardware/camera/4ms contract readback incomplete')
        require(sha(value['frozen_sapien_urdf']) == value['frozen_sapien_urdf_sha256'],
                'Frozen native hardware URDF changed')
    require(all(before[key] == after[key] for key in HARDWARE_KEYS),
            'Static actual hardware readback changed during original episode')
    scene = read_json(candidate / 'randomized_scene_receipt.json')
    spec = read_json(candidate / 'sample_spec.json')
    require(scene.get('passed') is True and scene.get('official_domain_randomization') == DR
        and scene.get('actual_domain_randomization') == DR and scene.get('actual_seed') == spec['seed']
        and scene.get('source_slot_index') == spec['index'] and scene.get('source_seed') == spec['source_seed']
        and scene.get('original_source_pose_restored') is False,
        'Actual official randomized parameters or fresh scene identity differs')
    coverage = read_json(candidate / 'randomized_environment_coverage.json')
    execution = read_json(candidate / 'randomized_environment_execution.json')
    count = scene.get('clutter_count')
    require(type(count) is int and count >= 0 and count == len(scene['official_clutter_records'])
        and coverage.get('passed') is True and coverage.get('all_collidable_clutter_covered') is True
        and coverage.get('expected_clutter_count') == coverage.get('covered_clutter_count') == count,
        'Actual randomized native clutter coverage incomplete')
    require(execution.get('passed') is True and execution.get('hardware_io') is False
        and execution.get('every_actual_physics_step_checked') is True
        and execution.get('complete_physics_telemetry_coverage_verified') is True
        and execution.get('steps_checked') == execution.get('original_report_physical_steps') == report['physical_steps']
        and execution.get('robot_clutter_active_contacts') == execution.get('bottle_clutter_active_contacts') == 0
        and 0 <= execution['max_clutter_vertex_displacement_m'] <= .001 + 1e-9,
        'Actual every-step clutter execution failed or incomplete')
    geometry = read_json(candidate / 'actual36_geometry_audit.json')
    require(geometry.get('passed') is True and geometry.get('physical_scene_unchanged') is True,
            'Complete live original actual36 geometry validation did not pass')
    def evidence(name, passed):
        path = candidate / name
        return {'passed': passed is True, 'evidence_path': str(path), 'evidence_sha256': sha(path)}
    return {
        'original_physical_success': evidence('expert_report.json', report.get('accepted') is True
            and report.get('result') == 'physics_success'),
        'original_airhold_acceptance': evidence('expert_report.json', report.get('hold_passed') is True
            and report.get('hold_validation_s', 0) >= 1 and report.get('minimum_hold_bottle_gap_m', 0) >= .01),
        'original_safety_acceptance': evidence('actual36_geometry_audit.json', geometry['passed']
            and report.get('canonical_sources_unchanged') is True),
        'official_demo_randomized_actual_parameters': evidence('randomized_scene_receipt.json', scene['passed']),
        'actual_clutter_collision_guard': evidence('randomized_environment_execution.json', execution['passed']),
        'hardware_camera_physics_clock_unchanged': evidence('actual_hardware_camera_clock_after_receipt.json', after['passed'])}


def separate_stage_paths(candidate, export_output, output):
    paths = [('candidate', Path(candidate).resolve()), ('export', Path(export_output).resolve()),
             ('quality', Path(output).resolve())]
    for index, (name, path) in enumerate(paths):
        for other_name, other in paths[index + 1:]:
            require(not path.is_relative_to(other) and not other.is_relative_to(path),
                'Immutable stage directories overlap in either direction: ' + name + '/' + other_name)
    return [path for _, path in paths]


def validate_quality(candidate, export_output, output, *, resource_root=None, benchmark_only=False):
    candidate, export_output, output = separate_stage_paths(candidate, export_output, output)
    require(candidate.is_dir() and export_output.is_dir() and not output.exists()
        and not output.is_symlink(),
        'Independent quality output must be new and outside input candidate/export')
    output.mkdir(parents=True)
    start = time.monotonic()
    result = {'schema': 'staged-independent-quality-v1', 'passed': False, 'accepted': False,
        'data_accepted': False, 'component_only': bool(benchmark_only), 'hardware_io': False,
        'task_or_scene_constructed': False, 'candidate': str(candidate), 'export_output': str(export_output),
        'pending_native_video_full_decode': True}
    try:
        exported = read_json(export_output / 'export_receipt.json')
        require(exported.get('export_complete') is True and exported.get('accepted') is False
            and exported.get('component_only') is benchmark_only
            and exported.get('candidate') == str(candidate) and exported.get('output') == str(export_output)
            and exported.get('mode') in ('hdf_only', 'sequential_hdf_then_video'),
            'Completed independent streaming HDF export identity/scope differs')
        hdf = export_output / 'data/episode_0000000.hdf5'
        hdf_output = exported['outputs']['hdf5']
        require(hdf_output['path'] == str(hdf) and hdf_output['sha256'] == sha(hdf),
                'Independent streamed HDF artifact differs from exporter receipt')
        input_hashes = exported.get('protected_raw_inputs_sha256', {})
        require(input_hashes and all(sha(name) == digest for name, digest in input_hashes.items()),
                'Original raw export inputs changed since isolated export')
        report = read_json(candidate / 'expert_report.json')
        spec = read_json(candidate / 'sample_spec.json')
        exit_proof = None
        if not benchmark_only:
            exporter.base.original_physics_admission(candidate, report)
            exit_proof = collection_exit_proof(candidate, spec, resource_root or HERE / 'resource_runs')
            aliases = []
            for name, relative in (('hdf5', 'data/episode_0000000.hdf5'),
                                   ('instruction', 'instruction/episode_0000000.json')):
                source = export_output / relative
                require(exported['outputs'][name]['path'] == str(source), 'Export alias metadata belongs elsewhere')
                aliases.append(copy_exact(source, candidate / relative, exported['outputs'][name]['sha256']))
            put_json(candidate / 'streaming_export_alias_receipt.json', {'passed': True,
                'accepted': False, 'export_receipt': str(export_output / 'export_receipt.json'),
                'export_receipt_sha256': sha(export_output / 'export_receipt.json'),
                'collection_external_exit': exit_proof, 'aliases': aliases})
            hdf = candidate / 'data/episode_0000000.hdf5'
        budget = numeric_budget(hdf)
        runtime = candidate.parent.parent
        validator_path = runtime / 'source/baseline_rgb20/rgb20_final_20261004/validate_rgb_three_camera.py'
        if benchmark_only:
            validator_path = TASK / 'source/baseline_rgb20/rgb20_final_20261004/validate_rgb_three_camera.py'
        baseline_validator = TASK / 'source/baseline_rgb20/rgb20_final_20261004/validate_rgb_three_camera.py'
        require(sha(validator_path) == sha(baseline_validator), 'Original frozen three-camera validator changed')
        validator = load_module('_staged_original_rgb_validator', validator_path)
        import cv2
        cv2.setNumThreads(2)
        native = validator.validate(hdf, exporter.base.ROOT, None)
        require(native.get('passed') is True, 'Original native RGB/SDK/full36 reader rejected HDF')
        put_json(output / 'dataset_validation.json', native)
        quality = None
        if not benchmark_only:
            put_json(candidate / 'dataset_validation.json', native)
            upstream = upstream_proofs(candidate, report)
            put_json(candidate / 'upstream_acceptance.json', upstream)
            branch = candidate / 'nearest_periodic_branch_receipt.json'
            for path in [branch, *candidate.glob('free_approach_*.json')]:
                read_json(path)  # Enforce the bounded JSON budget before frozen motion reads.
            motion_path = runtime / 'quality/motion_gate.py'
            thresholds = runtime / 'quality/thresholds_confirmed.json'
            require(sha(motion_path) == sha(TASK / 'quality/motion_gate.py')
                and sha(thresholds) == sha(TASK / 'quality/thresholds_confirmed.json'),
                'Frozen independent motion implementation/thresholds changed')
            motion = load_module('_staged_frozen_motion_gate', motion_path)
            quality = motion.evaluate_candidate(hdf, {key: spec[key] for key in
                ('index', 'arm', 'pose_class', 'brand', 'seed')}, config_path=thresholds,
                branch_receipt_path=branch, upstream_acceptance_path=candidate / 'upstream_acceptance.json')
            put_json(candidate / 'independent_motion_quality.json', quality)
            put_json(output / 'independent_motion_quality.json', quality)
            require(quality.get('overall_publish_gate_passed') is True,
                    'Independent actual motion/branch/source/physical upstream gate rejected candidate')
        require(all(sha(name) == digest for name, digest in input_hashes.items()),
                'Protected physical inputs changed during independent quality stage')
        require(sha(hdf) == hdf_output['sha256'], 'Exported/HDF alias bytes changed during quality validation')
        result.update(passed=True, gates_passed=not benchmark_only,
            native_dataset_readback_passed=True, independent_motion_quality_passed=not benchmark_only,
            collection_external_exit_proof=exit_proof, numeric_budget=budget,
            hdf5=str(hdf), hdf5_sha256=sha(hdf),
            native_validator_path=str(validator_path), native_validator_sha256=sha(validator_path),
            exporter_receipt_path=str(export_output / 'export_receipt.json'),
            exporter_receipt_sha256=sha(export_output / 'export_receipt.json'),
            quality_stage_producer_sha256=sha(__file__),
            dataset_validation_path=str(output / 'dataset_validation.json'),
            dataset_validation_sha256=sha(output / 'dataset_validation.json'),
            independent_motion_quality_path=str(output / 'independent_motion_quality.json') if quality is not None else None,
            independent_motion_quality_sha256=sha(output / 'independent_motion_quality.json') if quality is not None else None,
            candidate_recording_report_accepted_not_changed=True,
            scope='Original numerical/RGB readback component only; interrupted historical cache never accepted'
                if benchmark_only else 'Original physical+RGB+SDK+motion gates passed; native video stage still required')
    except BaseException as error:
        result.update(error=str(error), error_type=type(error).__name__, traceback=traceback.format_exc())
    result.update(elapsed_s=time.monotonic() - start,
        python_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    put_json(output / 'quality_receipt.json', result)
    return result


def dispatch_publish_five(request_path, output):
    """Use this registered, resource-capped entry for a separately pinned publisher."""
    helper = HERE / 'publish_staged_patch.py'
    require(isinstance(PUBLISH_HELPER_SHA256, str) and len(PUBLISH_HELPER_SHA256) == 64
        and helper.is_file() and not helper.is_symlink() and sha(helper) == PUBLISH_HELPER_SHA256,
        'Five-episode publisher is not the reviewed frozen producer; refuse import')
    publisher = load_module('_staged_frozen_five_publisher', helper)
    result = publisher.publish_request(Path(request_path), Path(output), publish=True)
    require(sha(helper) == PUBLISH_HELPER_SHA256, 'Frozen five-episode publisher changed during dispatch')
    require(result.get('complete') is True and result.get('accepted') is True and result.get('count') == 5
        and result.get('full50_randomized') is False and result.get('original50_replaced_or_merged') is False,
        'Publisher did not complete the independent five-episode patch contract')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--candidate', type=Path)
    mode.add_argument('--publish-five', type=Path, help='Frozen five-release request; runs the bounded publisher')
    parser.add_argument('--export-output', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resource-root', type=Path)
    parser.add_argument('--benchmark-only', action='store_true')
    args = parser.parse_args()
    if args.publish_five is not None:
        require(args.export_output is None and args.resource_root is None and not args.benchmark_only,
                'Do not mix publication with per-candidate component/quality options')
        result = dispatch_publish_five(args.publish_five, args.output)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 0
    require(args.export_output is not None, 'Per-candidate quality requires --export-output')
    result = validate_quality(args.candidate, args.export_output, args.output,
        resource_root=args.resource_root, benchmark_only=args.benchmark_only)
    print(json.dumps({key: result.get(key) for key in ('passed', 'accepted', 'component_only',
        'gates_passed', 'native_dataset_readback_passed', 'python_peak_RSS_bytes', 'error')},
        ensure_ascii=False), flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
