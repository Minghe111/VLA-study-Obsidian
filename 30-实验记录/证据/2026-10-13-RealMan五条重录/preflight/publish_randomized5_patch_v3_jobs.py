"""Source-bound V3 jobs admission; original publisher and collector stay frozen.

The default independently audits actual recording directories only. Publication
requires the original five complete physics/motion gates and an explicit flag.
One dtype-only private reader binding recognizes the SHA-pinned new preparation
producer's float64 heading arithmetic; every physical threshold stays intact.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import textwrap

import numpy as np

HERE = Path(__file__).resolve().parent
EVIDENCE = Path('/home/admin123/liminghe/VLA-WAM-RL/30-实验记录/证据/2026-10-13-RealMan五条重录')
COLLECTION = Path('/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/rgb50_motion_quality_20261013_v3_acquisition')
SO3_COLLECTION = Path('/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/rgb50_motion_quality_20261013_v4_so3')
BASE_SHA = 'd3c405a9af1d9a1e79c11ec07f570fd8e2e1d9dd4e4f530e84146e6007166eea'
IMMUTABLE_E_SHA = {
    'planning/randomized_environment_guard.py': '2ddd85b9e67624a2b73cfd8207caf3034737c33b30883020993ed51aec9700d2',
    'planning/conservative_ompl_overlay.py': '4e8e100a2c8ad63614e3d0912f3e0d5d6821719a10ffab441bebc4054a4e042c',
    'planning/auxiliary_window_overlay.py': 'bf6fe89c7d9a2e5cc117cbb5795bd2138ce25c0a1be1d6ed892d1d78b8695bf2',
    'planning/pre_recording_clutter_settle.py': 'ad25399cdb6b99509bd726a193b1d2ecfd827fa43dfe29240cac85c4b09cac14',
    'planning/auxiliary_joint_window.py': '19151529355d50bdb7baf35f185bf4abe0854adda1d30cb2978be9cb5b89130e',
    'planning/free_clearance_proxy_overlay.py': '23d3bd416f2c0d655ac83fff9c623ca47683798914e7023ee2ddc7e0f5dd6891',
    'planning/near_branch_quality.py': '68d9e5efd533a06f1b2d005c4467e775eeab561d1238cd940eeadee9be2799e7',
    'planning/grasp_acquisition_guard.py': '4d837e95c0f8fcf7da2b837b2a49b8a35d8e83ff9244eb2d987fac1df63fd8d3',
    'planning/collector_quality_overlay.py': '0489a134b6a99383c13f2a6ec60c24662aa89ef0df6115c911b9ba5c39b4a9e6',
    'planning/randomized_support_guard.py': '22c59b556d10dcec0b31b7537250375003d6ef4edd87c1b76702236365d0f646',
    'preflight/run_clearance_acquisition_collector.py': '765fbf62f8454c466caa61a07bde7a4b7000b2f39d5cdf4b0200db7ffc0f1923',
    'quality/thresholds_confirmed.json': '240d9aeb45d9c6d469941ef71abc9146238f874100470b0228c99d833ee017e7',
    'quality/motion_gate.py': 'c7a84a66bf9c126793fe58731aa98319a7dfbf7d78ebb4ee2220f02016804cbb',
}
SO3_E_SHA = {**{k: v for k, v in IMMUTABLE_E_SHA.items()
               if k != 'preflight/run_clearance_acquisition_collector.py'},
    'planning/strict_goal_pose_roundtrip_overlay.py': 'af40a2dd005b1e5d2407002605c801c823e208d6bd4c22033726c8d0a24d3582',
    'preflight/run_so3_acquisition_collector.py': '18cc0f5a4ff62ea2fae61691db02d8a6c925f186eb2493cbdb9d9452e4f9ac04'}


def load_base():
    path = HERE / 'publish_randomized5_patch.py'
    if hashlib.sha256(path.read_bytes()).hexdigest() != BASE_SHA:
        raise ValueError('Original publisher bytes changed')
    spec = importlib.util.spec_from_file_location('_frozen_randomized5_publisher_v1', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base()


def collection_gate(root):
    root = Path(root)
    base.require(not root.is_symlink() and root.resolve() in (COLLECTION.resolve(), SO3_COLLECTION.resolve()), 'Only the authorized independent V3/V4 collection roots are allowed')
    return root.resolve()


def runtime_roots(root):
    jobs = Path(root) / 'jobs'
    base.require(jobs.is_dir() and not jobs.is_symlink(), 'V3 jobs root missing or symlinked')
    roots = {slot: jobs / f'slot{slot}_runtime' for slot in base.SLOTS
             if (jobs / f'slot{slot}_runtime').exists() or (jobs / f'slot{slot}_runtime').is_symlink()}
    base.require(roots, 'No isolated authorized slot runtimes present')
    for path in roots.values():
        base.require(path.is_dir() and not path.is_symlink() and path.resolve().parent == jobs.resolve(), 'Isolated runtime root missing or escaped')
    return roots


def discover_candidates(root):
    found = []
    for slot, runtime in runtime_roots(root).items():
        area = runtime / 'new_physical_episodes'
        if not area.exists():
            continue
        base.require(area.is_dir() and not area.is_symlink() and area.resolve().parent == runtime.resolve(), 'Candidate area escaped its actual runtime')
        for path in sorted(area.rglob('slot_*_randomized_attempt_*')):
            base.require(not path.is_symlink(), 'Candidate discovery found a symlink')
            if not path.is_dir():
                continue
            base.require(path.resolve().is_relative_to(area.resolve()), 'Candidate discovery escaped actual runtime')
            for ancestor in path.parents:
                if ancestor == runtime:
                    break
                base.require(not ancestor.is_symlink(), 'Candidate ancestor is a symlink')
            found.append((slot, runtime.resolve(), path.resolve()))
    base.require(len({p for _, _, p in found}) == len(found), 'Duplicate source candidate discovered')
    return found


def hardware_source_pins():
    guard = base.read(base.ARCHIVE / 'hardware_guard.json')
    aliases = {
        '/tmp/realman-eval60k-20261011/local_eval/hardware/robot_head_down_fixed.urdf': base.ARCHIVE / 'shared_frozen_runtime/robot_head_down_fixed.urdf',
        '/tmp/realman-eval60k-20261011/local_eval/scripts/remote_base_task_snapshot.py': base.ARCHIVE / 'shared_frozen_runtime/remote_base_task_snapshot.py',
    }
    base.require(len(guard['expected_sha256']) == 116, 'Reviewed hardware source guard must have116 entries')
    pins = {str(aliases.get(old, Path(old)).resolve()): value for old, value in guard['expected_sha256'].items()}
    for path, digest in pins.items():
        base.require(Path(path).is_file() and base.sha(path) == digest, 'Hardware source missing or changed: ' + path)
    return pins


def producer_pins(runtime):
    runtime = Path(runtime).resolve()
    previous = base.SOURCE
    try:
        base.SOURCE = runtime / 'source'
        source_receipt = base.verify_source_bundle()
        original = base.read(base.SOURCE / 'download_receipt.json')['files']
        pins = {str((base.SOURCE / name).resolve()): row['sha256'] for name, row in original.items()}
    finally:
        base.SOURCE = previous
    pins.update(hardware_source_pins())
    source_pins = SO3_E_SHA if runtime.parent.parent == SO3_COLLECTION else IMMUTABLE_E_SHA
    for relative, digest in source_pins.items():
        for path in (EVIDENCE / relative, runtime / relative):
            base.require(path.is_file() and not path.is_symlink() and base.sha(path) == digest, 'Frozen V3 producer/E source changed: ' + str(path))
        # The original expert registers planning and collector producers; the
        # independent quality code is separately SHA-bound by its own receipt.
        if not relative.startswith('quality/'):
            pins[str((runtime / relative).resolve())] = digest
    return pins, source_receipt


def prepared_producer_gate(directory, runtime, expected_pins):
    directory, runtime = Path(directory).resolve(), Path(runtime).resolve()
    base.require(directory.is_relative_to(runtime / 'new_physical_episodes'), 'Prepared candidate outside its own actual runtime')
    spec = base.read(directory / 'sample_spec.json')
    base.require(spec.get('index') in base.SLOTS and runtime.name == f"slot{spec['index']}_runtime", 'Candidate is assigned to the wrong isolated runtime')
    scene = base.read(directory / 'scene_sample.json')
    report = base.read(directory / 'expert_report.json')
    stable_path = directory / 'pre_recording_clutter_settle_receipt.json'
    stable = base.read(stable_path)
    base.require(stable.get('version') == 'pre_recording_clutter_settle_v1' and stable.get('passed') is True and
        stable.get('hardware_io') is False and stable.get('recording_starts_after_preparation_at_true_new_t0') is True,
        'Actual SHA-bound prepared t0 receipt missing or rejected')
    base.require(stable.get('source_hashes_before') == stable.get('source_hashes_after') and
        all(stable['source_hashes_before'].get(p) == digest for p, digest in expected_pins.items()),
        'Preparation does not bind every original601/hardware116/frozen V3 producer')
    registry = report.get('source_hashes', {})
    base.require(all(registry.get(p) == digest for p, digest in expected_pins.items()), 'Expert source registry omits a protected producer')
    proof_names = ('pre_recording_clutter_settle_receipt.json', 'pre_recording_clutter_settle_initial.json', 'pre_recording_clutter_settle_steps.jsonl')
    for name in proof_names:
        path = directory / name
        base.require(path.is_file() and registry.get(str(path)) == base.sha(path), 'Fixed actual preparation proof SHA not registered: ' + name)
    base.require(scene.get('pre_recording_clutter_settle') == stable, 'Scene does not retain exact preparation receipt')
    base.require(all(scene.get(k) == value for k, value in stable['bottle_after'].items()), 'Scene actual t0 fields differ from final preparation readback')
    base.require(stable.get('extra_random_draws') == stable.get('wake_calls') == stable.get('native_actor_robot_drive_velocity_geometry_material_writes') == 0,
        'Preparation changed unsupported scene inputs')
    base.require(stable.get('execution_vertex_drift_budget_m_unchanged') == .001 and
        stable.get('historical_original_extra_settle_receipt_preserved_exactly') is True and
        stable.get('all_native_collision_component_identities_unchanged') is True, 'Preparation original geometry/safety proof differs')
    return {'passed': True, 'preparation_version': stable['version'], 'actual_preparation_receipt': str(stable_path),
        'actual_preparation_receipt_sha256': base.sha(stable_path), 'immutable_E_and_runtime_producers_verified': True,
        'protected_original_source_count': 601, 'protected_hardware_source_count': 116,
        'prepared_heading_producer_helper_sha256': IMMUTABLE_E_SHA['planning/pre_recording_clutter_settle.py'],
        'original_source_files_or_recorded_proof_bytes_changed': False}


def prepared_heading_reader(original, bindings):
    source, start = inspect.getsourcelines(original)
    tree = ast.parse(textwrap.dedent(''.join(source)))
    replacements = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'axis32':
            base.require(ast.unparse(node.value) == "np.asarray(scene['initial_pose'], dtype=np.float32)[:3, 1]", 'Frozen pose reader heading site changed')
            dtype = [kw for child in ast.walk(node.value) if isinstance(child, ast.Call) for kw in child.keywords if kw.arg == 'dtype']
            base.require(len(dtype) == 1, 'Exactly one prepared heading dtype binding required')
            dtype[0].value = ast.Attribute(value=ast.Name(id='np', ctx=ast.Load()), attr='float64', ctx=ast.Load())
            replacements.append(start + node.lineno - 1)
    base.require(len(replacements) == 1, 'Exactly one prepared heading arithmetic site required')
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__)
    exec(compile(tree, str(original.__code__.co_filename) + ':prepared_producer_heading_dtype_only', 'exec'), namespace)
    derived = namespace[original.__name__]
    bindings.append({'binding': 'SHA_bound_pre_recording_clutter_settle_v1_heading_dtype_only',
        'original_reader_source': original.__code__.co_filename, 'original_reader_source_sha256': base.sha(original.__code__.co_filename),
        'exact_changed_line': replacements[0], 'original_heading_dtype': 'float32', 'prepared_producer_heading_dtype': 'float64',
        'heading_match_tolerance_deg_unchanged': 1e-8,
        'compiled_ast_sha256': hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest(),
        'physics_thresholds_or_other_validation_changed': False, 'original_source_or_proof_bytes_changed': False})
    def call(scene):
        result = derived(scene)
        result['prepared_native_float64_heading_deg'] = result.pop('native_float32_heading_deg')
        result['pose_heading_producer'] = 'SHA_bound_pre_recording_clutter_settle_v1_float64'
        return result
    return call


def install_pose_dispatch(reader, bindings):
    original = reader.original_readback.audit_trial.__globals__['pose_receipt']
    prepared = prepared_heading_reader(original, bindings)
    context = {'validated_prepared_scene': None}
    def dispatch(scene):
        if context['validated_prepared_scene'] is not None:
            base.require(scene == context['validated_prepared_scene'], 'Prepared pose readback received a different scene')
            return prepared(scene)
        return original(scene)
    reader.original_readback.audit_trial.__globals__['pose_receipt'] = dispatch
    return context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection-root', type=Path, action='append',
        help='Repeat to independently audit V3 and V4 original physical recording areas')
    parser.add_argument('--audit-output', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--source50-manifest', type=Path, default=base.SOURCE50)
    parser.add_argument('--codec', type=Path, default=base.REPOSITORY / 'data/decode_image_bit.py')
    parser.add_argument('--ffmpeg', type=Path, default=Path('/home/admin123/liminghe/miniconda3/envs/robotwin-official-data/bin/ffmpeg'))
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    collections = [collection_gate(p) for p in (args.collection_root or [COLLECTION])]
    base.require(len(set(collections)) == len(collections), 'Duplicate collection roots')
    root = COLLECTION.resolve()
    collection_gate(root)
    base.TASK_ROOT, base.SOURCE = root, root / 'source'
    base.PACKAGE, base.LEGACY = base.SOURCE / 'rgb50_20261007_v1', base.SOURCE / 'adjust_bottle_20261002'
    baseline = base.original50(args.source50_manifest)
    collection_source = base.verify_source_bundle()
    pin_sets, guards = {}, {}
    for collection in collections:
        for slot, runtime in runtime_roots(collection).items():
            pin_sets[runtime.resolve()], guards[str(runtime.resolve())] = producer_pins(runtime)
    base.require(args.ffmpeg.is_file(), 'Actually installed ffmpeg required')
    tools = subprocess.run([str(args.ffmpeg), '-version'], capture_output=True, text=True, timeout=15)
    base.require(tools.returncode == 0, 'Installed ffmpeg cannot execute')
    bindings = []
    reader = base.install_original_reader(bindings)
    context = install_pose_dispatch(reader, bindings)
    found = [item for collection in collections for item in discover_candidates(collection)]
    audit_output = (args.audit_output or root / 'preflight/randomized5_v3_jobs_patch_validation.json').resolve()
    output = (args.output or root / 'published/realman_randomized_rerecord5_v3_patch/dataset').resolve()
    base.require(audit_output.is_relative_to(root / 'preflight'), 'Independent audit output must remain in V3 preflight')
    rows = []
    for slot, runtime, directory in found:
        row = {'passed': False, 'validated_source_candidate': str(directory), 'hardware_io': False}
        try:
            producer = prepared_producer_gate(directory, runtime, pin_sets[runtime])
            context['validated_prepared_scene'] = base.read(directory / 'scene_sample.json')
            row = base.audit_candidate(directory, baseline, reader, args.ffmpeg, args.codec)
            row['prepared_producer_binding'] = producer
        except Exception as error:
            row.update(error=str(error), error_type=type(error).__name__)
        finally:
            context['validated_prepared_scene'] = None
        rows.append(row)
    selected = base.choose_five(rows)
    audit = {'schema': 'randomized5-v3-jobs-independent-validation-v1', 'complete': len(selected) == 5,
        'scope': 'Only five independently validated new physical replacements; not full50 randomized',
        'full50_randomized': False, 'original_slot_indices': list(base.SLOTS), 'candidate_count': len(rows),
        'accepted_candidate_count': sum(r.get('passed') is True for r in rows), 'candidates': rows,
        'selected_source_candidates': [r['validated_source_candidate'] for r in selected],
        'original50_manifest': str(args.source50_manifest), 'original50_manifest_sha256': base.SOURCE50_SHA,
        'original_collection_source_bundle': collection_source, 'isolated_runtime_original601_source_guards': guards,
        'collection_roots': [str(p) for p in collections], 'immutable_V3_E_SHA': IMMUTABLE_E_SHA,
        'immutable_V4_E_SHA': SO3_E_SHA, 'offline_source_path_and_heading_bindings': bindings,
        'frozen_original_publisher_sha256': BASE_SHA, 'validator_script_sha256': base.sha(__file__),
        'all_source_candidates_retained': True, 'candidate_inventory_or_proof_bytes_written': False,
        'new_physics_episodes': 0, 'model_inference_runs': 0, 'hardware_io': False,
        'ffmpeg_version': tools.stdout.splitlines()[0], 'publication_requested': args.publish}
    base.require(base.sha(args.source50_manifest) == base.SOURCE50_SHA, 'Original50 manifest changed during audit')
    base.write(audit_output, audit)
    if not selected:
        print(json.dumps({'complete': False, 'published': False, 'audit': str(audit_output),
            'missing_accepted_slots': [i for i in base.SLOTS if not any(r.get('passed') and r.get('original_slot_index') == i for r in rows)]}))
        return 2
    if args.publish:
        result = base.publish_patch(selected, output, audit, args.source50_manifest, reader)
    else:
        result = {'complete': True, 'published': False, 'count': 5, 'audit': str(audit_output), 'full50_randomized': False}
    print(json.dumps(result))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
