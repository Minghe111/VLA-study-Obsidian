#!/usr/bin/env python3
"""Build a separate, source-bound 161 collection bundle; never connect remotely.

All modifications are made in a newly created local build directory. Protected
601 producer files and 115 hardware/runtime files remain byte exact. The one
URDF relocation changes only absolute mesh filename prefixes and is separately
proved, retaining the original URDF and guard. No raw captures or weights enter
the archive. A complete configuration is required to create a formal bundle.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import xml.etree.ElementTree as ET

from hybrid_release_patch import patch_publisher, patch_quality, patch_render

HERE = Path(__file__).resolve().parent
E = HERE.parent.parent
LOCAL_REPO = Path('/home/admin123/liminghe/vla-benchmark/RoboTwin')
LOCAL_PREFIX = LOCAL_REPO.parent / 'deploy/realman_gen4'
LOCAL_ARCHIVE = E.parent / '2026-10-11-RealMan专家回放'
LOCAL_VARIANT = 'rgb50_motion_quality_20261014_v5_staged'
SLOTS = (17, 18, 20, 29, 34)
FROZEN_TMP = '/tmp/realman-eval60k-20261011/local_eval/hardware/robot_head_down_fixed.urdf'
BASE_TMP = '/tmp/realman-eval60k-20261011/local_eval/scripts/remote_base_task_snapshot.py'
CORE = ('run_guarded_stage.py', 'guarded_stage_cpu_probe.py', 'test_guarded_stage.py',
        'run_staged_collector.py', 'collected_phase_seal.py',
        'streaming_export.py', 'streaming_export_staged.py',
        'staged_quality.py', 'publish_staged_patch.py',
        'render_staged_video.py', 'bounded_actual_trace_audit.py')
SPEED_NAME = 'conservative_ompl_overlay.py'
SPEED_SOURCE = HERE / 'speed_variant' / SPEED_NAME
SPEED_CPU_RECEIPT = HERE / 'speed_variant/cpu_validation_receipt.json'
SPEED_DOWNSTREAM_REVIEW = HERE / 'speed_variant/downstream_short_trajectory_review.json'
ASSET_PREPARER = HERE / 'prepare_remote_assets.py'
RENDERER_SMOKE_SOURCE = HERE / 'renderer_only_smoke_v7.py'
SYSTEM_TEMPLATES = {
    'run_guarded_stage.py': HERE / 'run_guarded_stage_system_v7.py',
    'guarded_stage_cpu_probe.py': HERE / 'guarded_stage_cpu_probe_system_v6.py',
    'test_guarded_stage.py': HERE / 'test_guarded_stage_system_v7.py',
}
HYBRID_RUNTIME_HELPERS = (HERE / 'hybrid_cgroup_adapter.py',
                          HERE / 'hybrid_release_contract_v7.py')
PLANNING = ('collector_quality_overlay.py', 'near_branch_quality.py',
    'randomized_environment_guard.py', 'randomized_support_guard.py',
    'auxiliary_joint_window.py', 'auxiliary_window_overlay.py',
    'conservative_ompl_overlay.py', 'free_clearance_proxy_overlay.py',
    'pre_recording_clutter_settle.py', 'grasp_acquisition_guard.py',
    'strict_goal_pose_roundtrip_overlay.py', 'bounded_trace_overlay.py',
    'bounded_trace_storage.py')
MAX_FILE = 64 * 1024**2


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def external_asset_inventory(root):
    """Compare the two official asset trees without loading 14 GiB of media."""
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), 'Local official asset root absent or linked')
    result = {}
    for name in ('background_texture', 'objects'):
        base = root / name
        require(base.is_dir() and not base.is_symlink(), 'Local official asset tree absent or linked: ' + name)
        files = []
        for path in base.rglob('*'):
            require(not path.is_symlink(), 'Local official asset contains a link: ' + str(path))
            if path.is_file():
                files.append((path.relative_to(base).as_posix(), path.stat().st_size))
        if name == 'background_texture':
            require(files and all(relative.startswith('seen/') for relative, _ in files),
                    'This recording must expose only the official seen backgrounds')
        digest = hashlib.sha256()
        for relative, size in sorted(files):
            digest.update(relative.encode('utf-8') + b'\0' + str(size).encode('ascii') + b'\n')
        result[name] = {'file_count': len(files), 'bytes': sum(size for _, size in files),
                        'relative_path_and_size_sha256': digest.hexdigest(),
                        'content_sha256_verified': False,
                        'selection_scope': 'seen_only' if name == 'background_texture' else 'full_directory'}
    return result


def verify_speed_variant():
    original = E / 'planning' / SPEED_NAME
    receipt = json.loads(SPEED_CPU_RECEIPT.read_text())
    review = json.loads(SPEED_DOWNSTREAM_REVIEW.read_text())
    require(receipt.get('passed') is True and receipt.get('tests_run', 0) >= 9
            and receipt.get('component_only') is True
            and receipt.get('physical_speedup_verified') is False,
            'Server speed variant lacks frozen CPU-only proof')
    require(receipt['source_sha256'][str(SPEED_SOURCE)] == sha(SPEED_SOURCE)
            and receipt['frozen_input_sha256'][str(original)] == sha(original)
            and review['variant_sha256'] == sha(SPEED_SOURCE)
            and review['original_thresholds_modified'] is False,
            'Frozen native speed source/proof differs')
    require(review.get('physical_speedup_verified') is False
            and review.get('component_only') is True,
            'Native speed review scope differs')
    tree = ast.parse(SPEED_SOURCE.read_text())
    assignment = [node for node in tree.body if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == 'DEFAULT_POLICY'
                          for target in node.targets)]
    require(len(assignment) == 1 and ast.literal_eval(assignment[0].value)
            == {'time': 30.0, 'range': .01, 'simplify': True},
            'Server speed policy differs from reviewed native-only change')
    return {'original_source': str(original), 'original_sha256': sha(original),
            'server_source': str(SPEED_SOURCE), 'server_sha256': sha(SPEED_SOURCE),
            'effective_native_policy': {'time': 30.0, 'range': .01, 'simplify': True},
            'cpu_validation_receipt_sha256': sha(SPEED_CPU_RECEIPT),
            'downstream_short_trajectory_review_sha256': sha(SPEED_DOWNSTREAM_REVIEW),
            'component_only': True, 'physical_speedup_verified': False}


def dump(path, obj):
    path = Path(path)
    require(not path.exists() and not path.is_symlink(), 'Refuse metadata overwrite: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def validate_config(obj):
    required = {'schema', 'run_root', 'variant', 'python', 'gpu_index',
                'collect_cpu_cores', 'existing_asset_root'}
    require(set(obj) == required, 'Configuration must contain exact reviewed keys')
    require(obj['schema'] == 'realman-161-bundle-config-v1', 'Unknown configuration schema')
    for key in ('run_root', 'python', 'existing_asset_root'):
        value = obj[key]
        require(isinstance(value, str) and value.startswith('/') and '\x00' not in value,
                'Absolute path required: ' + key)
        require(not any(x in value.lower() for x in ('placeholder', '<', '>', 'todo', '{', '}')),
                'No placeholder allowed in formal configuration')
        require('..' not in Path(value).parts and str(Path(value)) == value,
                'Normalized real path required: ' + key)
    root = Path(obj['run_root'])
    require(any(root.is_relative_to(prefix) for prefix in (
        Path('/bigdata/liminghe'), Path('/bigdata2/liminghe'))), 'Run root outside task-owned 161 area')
    require(len(root.parts) >= 5, 'Use a fresh dedicated task root')
    variant = obj['variant']
    require(isinstance(variant, str) and variant.startswith('rgb50_motion_quality_20261014_')
        and variant != LOCAL_VARIANT and '/' not in variant and '..' not in variant,
        'New scoped variant name required')
    require(type(obj['gpu_index']) is int and 0 <= obj['gpu_index'] <= 15,
            'Physical GPU index must be an integer0..15')
    require(type(obj['collect_cpu_cores']) is int and 2 <= obj['collect_cpu_cores'] <= 32,
            'Collect CPU budget must be integer2..32 cores')
    require(Path(obj['python']).name in ('python', 'python3'), 'Exact environment interpreter required')
    require(any(Path(obj['existing_asset_root']).is_relative_to(prefix) for prefix in (
        Path('/bigdata/liminghe'), Path('/bigdata2/liminghe'))),
        'External assets must remain in task-owned 161 area')
    return dict(obj)


def binding_paths(cfg):
    root = Path(cfg['run_root'])
    return {'root': root, 'repo': root / 'runtime_repo/RoboTwin',
            'evidence': root / 'evidence', 'archive': root / 'archive',
            'prefix': root / 'deploy/realman_gen4',
            'variant': root / 'deploy/realman_gen4' / cfg['variant']}


def unit_prefix(cfg):
    suffix = cfg['variant'].removeprefix('rgb50_motion_quality_20261014_v6_161_')
    require(suffix and suffix != cfg['variant'] and
            all(c.isalnum() or c == '_' for c in suffix),
            '161 variant has no safe unique transient unit suffix')
    return 'realman-rerecord-20261014-v6-' + suffix.replace('_', '-')


def mapped(value, bindings):
    for old, new in sorted(bindings.items(), key=lambda x: len(x[0]), reverse=True):
        if value == old or value.startswith(old + '/'):
            return new + value[len(old):]
    return value


def rewrite_literals(text, replacements):
    """Change only AST string tokens, preserving all unmodified source bytes."""
    tree = ast.parse(text)
    raw = text.encode('utf-8')
    starts = [0]
    for line in raw.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))
    edits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        new = mapped(node.value, replacements)
        if new != node.value:
            begin = starts[node.lineno - 1] + node.col_offset
            end = starts[node.end_lineno - 1] + node.end_col_offset
            edits.append((begin, end, repr(new).encode(), node.lineno, node.value, new))
    for begin, end, new, *_ in sorted(edits, reverse=True):
        raw = raw[:begin] + new + raw[end:]
    changed = raw.decode('utf-8')
    ast.parse(changed)
    return changed, [{'line': line, 'original': old, 'remote': new}
                     for _, _, _, line, old, new in edits]


def replace_once(text, old, new):
    require(text.count(old) == 1, 'Reviewed relocation site count differs: ' + repr(old))
    return text.replace(old, new, 1)


def semantic_xml(node):
    return (node.tag, tuple(sorted(node.attrib.items())), (node.text or '').strip(),
            (node.tail or '').strip(), tuple(semantic_xml(c) for c in node))


def relocate_urdf(source, remote_repo):
    raw = Path(source).read_bytes()
    old = str(LOCAL_REPO).encode()
    new = str(remote_repo).encode()
    original = ET.fromstring(raw)
    meshes = list(original.iter('mesh'))
    require(len(meshes) == 186, 'Expected exactly186 frozen visual/collision meshes')
    rows = []
    for mesh in meshes:
        value = mesh.attrib['filename']
        require(value.startswith(str(LOCAL_REPO) + '/assets/embodiments/realman-rm65b-inspire-gen4/'),
                'Frozen mesh escaped expected hardware asset directory')
        p = Path(value)
        require(p.is_file() and not p.is_symlink(), 'Protected mesh must be a real original file')
        target = str(remote_repo) + value[len(str(LOCAL_REPO)):]
        rows.append({'original': value, 'remote': target, 'mesh_sha256': sha(p)})
        mesh.attrib['filename'] = target
    require(raw.count(old) == 186, 'Path prefix occurs outside the186 reviewed mesh attributes')
    changed = raw.replace(old, new)
    relocated = ET.fromstring(changed)
    require(semantic_xml(original) == semantic_xml(relocated),
            'URDF XML differs beyond reviewed mesh filename relocation')
    return changed, rows


def copy_exact(source, dest, *, expected=None):
    source, dest = Path(source), Path(dest)
    require(source.is_file() and not source.is_symlink() and source.stat().st_size <= MAX_FILE,
            'Input missing, linked or exceeds small-bundle size: ' + str(source))
    before = sha(source)
    require(expected is None or before == expected, 'Original protected input SHA changed: ' + str(source))
    if dest.exists():
        require(not dest.is_symlink() and sha(dest) == before, 'Conflicting package destination: ' + str(dest))
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
    require(sha(source) == before and sha(dest) == before, 'Byte-exact copy failed')
    return {'source': str(source), 'sha256': before, 'bytes': source.stat().st_size}


def write_source(path, text):
    require(not Path(path).exists(), 'Refuse producer overwrite')
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text)
    compile(text, str(path), 'exec')


def adapt_core(name, text, cfg, paths):
    replacements = {
        str(LOCAL_PREFIX / LOCAL_VARIANT): str(paths['variant']),
        str(LOCAL_PREFIX): str(paths['prefix']), str(LOCAL_REPO): str(paths['repo']),
        str(LOCAL_ARCHIVE): str(paths['archive']), str(E): str(paths['evidence']),
        '/home/admin123/liminghe/miniconda3/envs/robotwin/bin/python': cfg['python'],
        LOCAL_VARIANT: cfg['variant'],
    }
    text, sites = rewrite_literals(text, replacements)
    cpu = cfg['collect_cpu_cores']
    if name == 'run_guarded_stage.py':
        text = replace_once(text, "'CPUQuota': '200%'", f"'CPUQuota': '{cpu * 100}%' if stage == 'collect' else '200%'")
        text = replace_once(text, "'cpu.max': '200000 100000'",
            "'cpu.max': str(int(limits['CPUQuota'][:-1]) * 1000) + ' 100000'")
        text = replace_once(text, "== '0')", f"== '{cfg['gpu_index']}')")
        text = replace_once(text, "'CUDA_VISIBLE_DEVICES': '0' if stage == 'collect' else ''",
            f"'CUDA_VISIBLE_DEVICES': '{cfg['gpu_index']}' if stage == 'collect' else ''")
        start = "'PATH': '/home/admin123/liminghe/miniconda3/envs/robotwin-official-data/bin:'\n                   '/home/admin123/liminghe/miniconda3/envs/robotwin/bin:/usr/bin:/bin'"
        text = replace_once(text, start, "'PATH': " + repr(str(Path(cfg['python']).parent) + ':/usr/bin:/bin'))
    elif name == 'staged_quality.py':
        text = replace_once(text, "limits.get('CPUQuota') == '200%'", f"limits.get('CPUQuota') == '{cpu * 100}%'")
    elif name == 'publish_staged_patch.py':
        text = replace_once(text, "limits.get('CPUQuota') == '200%'",
            f"limits.get('CPUQuota') == ('{cpu * 100}%' if stage == 'collect' else '200%')")
        text = replace_once(text, "actual.get('cpu.max') == '200000 100000'",
            f"actual.get('cpu.max') == ('{cpu * 100000} 100000' if stage == 'collect' else '200000 100000')")
        text = replace_once(text, "'source_policy': 'V5 genuine randomized physical captures only'",
            "'source_policy': 'V6 genuine randomized physical captures with native OMPL simplification only'")
    elif name == 'test_guarded_stage.py':
        text = replace_once(text, "self.assertEqual(r['CPUQuota'], '200%')",
            f"self.assertEqual(r['CPUQuota'], '{cpu * 100}%' if stage == 'collect' else '200%')")
        text = replace_once(text, "self.assertEqual(c['CUDA_VISIBLE_DEVICES'], '0')",
            f"self.assertEqual(c['CUDA_VISIBLE_DEVICES'], '{cfg['gpu_index']}')")
    compile(text, name, 'exec')
    return text, sites


def adapt_system_template(text, cfg, paths):
    """Freeze the separate 161 system-unit templates into one remote root."""
    replacements = {
        '@PYTHON@': cfg['python'],
        '@PREFIX@': str(paths['prefix']),
        '@VARIANT@': cfg['variant'],
        '@GPU_INDEX@': str(cfg['gpu_index']),
        '@COLLECT_CPU_PERCENT@': str(cfg['collect_cpu_cores'] * 100),
        '@UNIT_PREFIX@': unit_prefix(cfg),
    }
    sites = []
    for token, value in replacements.items():
        count = text.count(token)
        if count:
            text = text.replace(token, value)
            sites.append({'template_token': token, 'count': count, 'remote': value})
    require('@PYTHON@' not in text and '@PREFIX@' not in text
            and '@VARIANT@' not in text and '@GPU_INDEX@' not in text
            and '@COLLECT_CPU_PERCENT@' not in text and '@UNIT_PREFIX@' not in text,
            'Unresolved system guard template token')
    compile(text, 'system-guard-template', 'exec')
    return text, sites


def patch_assignment(text, name, value):
    tree = ast.parse(text)
    nodes = [node for node in tree.body if isinstance(node, ast.Assign)
             and any(isinstance(x, ast.Name) and x.id == name for x in node.targets)]
    require(len(nodes) == 1 and isinstance(nodes[0].value, ast.Constant),
            'Exactly one literal frozen binding required: ' + name)
    return rewrite_literals(text, {nodes[0].value.value: value})[0]


def build(config_path, output):
    cfg = validate_config(json.loads(Path(config_path).read_text()))
    paths = binding_paths(cfg)
    speed = verify_speed_variant()
    local_assets = external_asset_inventory(LOCAL_REPO / 'assets')
    output = Path(output).absolute()
    require(output.is_relative_to(HERE) and output.parent.is_dir() and not output.exists(),
            'Build must use a new local directory below remote_161_v6')
    output.mkdir()
    stage = output / 'payload'
    stage.mkdir()
    target = lambda remote: stage / Path(remote).relative_to(paths['root'])
    unchanged = {}
    adaptations = {}
    speed_deployments = {}
    new_helpers = {}
    source = E / 'source'
    downloaded = json.loads((source / 'download_receipt.json').read_text())
    require(len(downloaded['files']) == 601, 'Exactly601 original protected sources required')
    source_files = [p for p in source.rglob('*') if p.is_file()
        and '__pycache__' not in p.parts and p.suffix != '.pyc']
    for relative, row in downloaded['files'].items():
        rel = Path(relative)
        require(not rel.is_absolute() and '..' not in rel.parts and sha(source / rel) == row['sha256'],
                'Original601 source file changed or escaped')
    for base in [paths['evidence'], *[paths['variant'] / f'jobs/slot{x}_runtime' for x in SLOTS]]:
        for p in source_files:
            dest = target(base / 'source' / p.relative_to(source))
            unchanged[str(dest.relative_to(stage))] = copy_exact(p, dest)
        for name in PLANNING:
            p = SPEED_SOURCE if name == SPEED_NAME else E / 'planning' / name
            dest = target(base / 'planning' / name)
            row = copy_exact(p, dest)
            if name == SPEED_NAME:
                require(row['sha256'] == speed['server_sha256'], 'Server speed variant copy differs')
                speed_deployments[str(dest.relative_to(stage))] = row
            else:
                unchanged[str(dest.relative_to(stage))] = row
        for name in ('motion_gate.py', 'thresholds_confirmed.json'):
            p = E / 'quality' / name
            dest = target(base / 'quality' / name)
            unchanged[str(dest.relative_to(stage))] = copy_exact(p, dest)
        for directory in ('preflight', 'prepared', 'new_physical_episodes'):
            target(base / directory).mkdir(parents=True, exist_ok=True)
    # Repository code is separate real files; large shared task assets are bound
    # by the remote preparation gate, not bundled or silently downloaded.
    suffixes = {'.py', '.yml', '.yaml', '.json', '.xml', '.srdf', '.urdf', '.sh'}
    for folder in ('envs', 'integrations', 'env_cfg', 'data', 'scripts'):
        for p in (LOCAL_REPO / folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix in suffixes:
                dest = target(paths['repo'] / p.relative_to(LOCAL_REPO))
                unchanged[str(dest.relative_to(stage))] = copy_exact(p, dest)
    for p in (LOCAL_REPO / 'assets/files').rglob('*'):
        if p.is_file():
            dest = target(paths['repo'] / p.relative_to(LOCAL_REPO))
            unchanged[str(dest.relative_to(stage))] = copy_exact(p, dest)
    # Keep registered auxiliary media entry and all speed-variant proof files
    # available in the independent bundle, outside protected original sources.
    for name in ('render_accepted_expert.py',):
        p = E / 'media_scripts' / name
        dest = target(paths['evidence'] / 'media_scripts' / name)
        unchanged[str(dest.relative_to(stage))] = copy_exact(p, dest)
    speed_proofs = {}
    for p in (E / 'planning' / SPEED_NAME, SPEED_CPU_RECEIPT,
              SPEED_DOWNSTREAM_REVIEW, HERE / 'speed_variant/test_native_simplification_cpu.py'):
        dest = target(paths['evidence'] / 'preflight/native_speed_variant_evidence' / p.name)
        speed_proofs[str(dest.relative_to(stage))] = copy_exact(p, dest)
    asset_tool = target(paths['root'] / 'bundle_tools/prepare_remote_assets.py')
    new_helpers[str(asset_tool.relative_to(stage))] = copy_exact(ASSET_PREPARER, asset_tool)
    for helper in HYBRID_RUNTIME_HELPERS:
        remote_name = ('hybrid_release_contract.py' if helper.name == 'hybrid_release_contract_v7.py'
                       else helper.name)
        dest = target(paths['evidence'] / 'preflight' / remote_name)
        if helper.name == 'hybrid_release_contract_v7.py':
            source = helper.read_text()
            rendered = replace_once(source,
                "UNIT_PREFIX = 'realman-rerecord-20261014-v6-hybrid'",
                'UNIT_PREFIX = ' + repr(unit_prefix(cfg)))
            write_source(dest, rendered)
            new_helpers[str(dest.relative_to(stage))] = {
                'source': str(helper), 'original_sha256': sha(helper),
                'sha256': sha(dest), 'bytes': dest.stat().st_size,
                'unit_prefix': unit_prefix(cfg)}
        else:
            new_helpers[str(dest.relative_to(stage))] = copy_exact(helper, dest)
    smoke_dest = target(paths['evidence'] / 'preflight/renderer_only_smoke.py')
    new_helpers[str(smoke_dest.relative_to(stage))] = copy_exact(RENDERER_SMOKE_SOURCE, smoke_dest)
    guard = json.loads((LOCAL_ARCHIVE / 'hardware_guard.json').read_text())
    require(len(guard['expected_sha256']) == 116, 'Exactly116 protected hardware/runtime inputs required')
    for name in ('hardware_guard.json', 'source_manifest.json'):
        dest = target(paths['archive'] / ('original_' + name if name == 'hardware_guard.json' else name))
        unchanged[str(dest.relative_to(stage))] = copy_exact(LOCAL_ARCHIVE / name, dest)
    frozen = LOCAL_ARCHIVE / 'shared_frozen_runtime/robot_head_down_fixed.urdf'
    relocated, mesh_rows = relocate_urdf(frozen, paths['repo'])
    original_dest = target(paths['archive'] / 'shared_frozen_runtime/robot_head_down_fixed.original.urdf')
    unchanged[str(original_dest.relative_to(stage))] = copy_exact(frozen, original_dest, expected=guard['expected_sha256'][FROZEN_TMP])
    new_urdf = target(paths['archive'] / 'shared_frozen_runtime/robot_head_down_fixed.urdf')
    new_urdf.write_bytes(relocated)
    changes = {}
    remote_guard = dict(guard)
    remote_guard['expected_sha256'] = {}
    for old, digest in guard['expected_sha256'].items():
        if old == FROZEN_TMP:
            remote_guard['expected_sha256'][old] = sha(new_urdf)
            continue
        if old == BASE_TMP:
            source_file = LOCAL_ARCHIVE / 'shared_frozen_runtime/remote_base_task_snapshot.py'
            remote = paths['archive'] / 'shared_frozen_runtime/remote_base_task_snapshot.py'
            key = old
        else:
            source_file = Path(old)
            require(source_file.is_relative_to(LOCAL_REPO), 'Unknown hardware input relocation')
            remote = paths['repo'] / source_file.relative_to(LOCAL_REPO)
            key = str(remote)
        dest = target(remote)
        unchanged[str(dest.relative_to(stage))] = copy_exact(source_file, dest, expected=digest)
        remote_guard['expected_sha256'][key] = digest
        changes[old] = key
    # Descriptive paths move recursively; SHA fields remain original, except the
    # explicitly separately proved mesh-filename-only frozen URDF relocation.
    def map_json(v):
        if isinstance(v, dict):
            return {k: map_json(x) for k, x in v.items()}
        if isinstance(v, list):
            return [map_json(x) for x in v]
        return mapped(v, {str(LOCAL_REPO): str(paths['repo']), str(LOCAL_ARCHIVE): str(paths['archive'])}) if isinstance(v, str) else v
    remote_guard = map_json(remote_guard)
    dump(target(paths['archive'] / 'hardware_guard.json'), remote_guard)
    relocation = {'schema': 'mesh-filename-only-relocation-v1', 'passed': True,
        'original_urdf_sha256': sha(frozen), 'remote_urdf_sha256': sha(new_urdf),
        'mesh_attribute_count': len(mesh_rows), 'mesh_filename_only_changed': True,
        'all_other_XML_semantics_equal': True, 'original_hardware_guard_sha256': sha(LOCAL_ARCHIVE / 'hardware_guard.json'),
        'remote_hardware_guard_sha256': sha(target(paths['archive'] / 'hardware_guard.json')),
        'other115_hardware_sha256_unchanged': True, 'hardware_path_bindings': changes, 'meshes': mesh_rows}
    dump(target(paths['archive'] / 'relocation_proof.json'), relocation)
    # Adapt producer copies in order of their SHA dependencies.
    core_texts = {}
    for name in CORE:
        original = SYSTEM_TEMPLATES.get(name, E / 'preflight' / name)
        if name in SYSTEM_TEMPLATES:
            text, sites = adapt_system_template(original.read_text(), cfg, paths)
        else:
            text, sites = adapt_core(name, original.read_text(), cfg, paths)
        if name == 'staged_quality.py':
            text = patch_quality(text)
        elif name == 'publish_staged_patch.py':
            text = patch_publisher(text)
        elif name == 'render_staged_video.py':
            text = patch_render(text)
        core_texts[name] = text
        adaptations[name] = {'original_path': str(original), 'original_sha256': sha(original),
            'literal_path_sites': sites, 'server_resource_adapter': name in
            ('run_guarded_stage.py', 'guarded_stage_cpu_probe.py',
             'staged_quality.py', 'publish_staged_patch.py', 'render_staged_video.py')}
    remote_export_sha = hashlib.sha256(core_texts['streaming_export.py'].encode()).hexdigest()
    core_texts['render_staged_video.py'] = patch_assignment(core_texts['render_staged_video.py'], 'BASE_SHA', remote_export_sha)
    publisher_sha = hashlib.sha256(core_texts['publish_staged_patch.py'].encode()).hexdigest()
    core_texts['staged_quality.py'] = patch_assignment(core_texts['staged_quality.py'], 'PUBLISH_HELPER_SHA256', publisher_sha)
    for name, text in core_texts.items():
        dest = target(paths['evidence'] / 'preflight' / name)
        write_source(dest, text)
        adaptations[name]['remote_sha256'] = sha(dest)
        adaptations[name]['remote_path'] = str(paths['evidence'] / 'preflight' / name)
    diagnostic = E / 'preflight/stage_diagnostics/sitecustomize.py'
    dest = target(paths['evidence'] / 'preflight/stage_diagnostics/sitecustomize.py')
    unchanged[str(dest.relative_to(stage))] = copy_exact(diagnostic, dest)
    for slot in SLOTS:
        base = paths['variant'] / f'jobs/slot{slot}_runtime/preflight'
        for name in ('run_staged_collector.py', 'collected_phase_seal.py'):
            dest = target(base / name)
            write_source(dest, core_texts[name])
    manifest = {'schema': 'realman-161-portability-bundle-v1', 'complete_config': cfg,
        'remote_paths': {k: str(v) for k, v in paths.items()},
        'no_connection_or_remote_write_performed': True,
        'hardware_io': False, 'raw_capture_or_weights_included': False,
        'original601_source_sha_unchanged': True, 'protected_source_count': 601,
        'protected_hardware_count': 116, 'relocated_urdf_mesh_attributes': 186,
        'other115_hardware_SHA_unchanged': True,
        'producer_adaptations': adaptations, 'byte_exact_original_copies': unchanged,
        'new_helper_copies': new_helpers,
        'hybrid_cgroup_contract': {
            'mode_required': 'dedicated-system-unit-v1',
            'memory_soft_limit_scope': 'v1 best effort only',
            'memory_hard_limit_bytes_collect': 12 * 1024**3,
            'memory_plus_swap_total_limit_bytes_collect': 12 * 1024**3,
            'swap_claim': 'zero-swap observation required at each guarded stage; no hard swap-disable claim',
            'swap_observation_source': 'v1 memory.stat swap and total_swap, bracketed reads',
            'fuzzy_usage_and_memsw_difference': 'retained as signed diagnostic only',
            'nonzero_or_missing_explicit_swap_stat': 'fail closed',
            'adapter_sha256': sha(HERE / 'hybrid_cgroup_adapter.py'),
            'release_contract_sha256': sha(target(paths['evidence'] / 'preflight/hybrid_release_contract.py')),
            'release_patch_sha256': sha(HERE / 'hybrid_release_patch.py')},
        'renderer_loader_contract': {
            'scope': 'collect and optional renderer_smoke protected child only',
            'preload_only_after_cpu_probe_and_cgroup_gate': True,
            'guard_service_and_cpu_probe_preloaded': False,
            'gpucomp_path': '/lib/x86_64-linux-gnu/libnvidia-gpucomp.so.570.133.07',
            'gpucomp_sha256': '66178cf6eef724402b77193665ab1defede1b36a6333d81423b48ee32f0742dc',
            'nvidia_icd_path': '/usr/share/vulkan/icd.d/nvidia_icd.json',
            'nvidia_icd_sha256': '46cee75b42b43649839bc7f29820d36ac6e85782b792a97d5b91d9adbca8ca09',
            'running_kernel_and_all_eight_gpu_driver_version': '570.133.07',
            'renderer_smoke_entry_sha256': sha(smoke_dest),
            'renderer_smoke_default_loads_task_scene': False,
            'official_sapien_offscreen_example_required_before_collect': True,
            'constructor_and_offscreen_release_pair_same_boot_bundle_driver_required': True,
            'latest_same_boot_attempt_per_mode_only': True,
            'newer_failed_or_interrupted_smoke_blocks_collect': True,
            'explicit_mode_scoped_smoke_receipt_directory_required': True,
            'all_eight_gpu_driver_and_12_gib_free_floor_guarded': True},
        'native_speed_variant': {**speed, 'deployments': speed_deployments,
            'proof_files': speed_proofs, 'original_planning_copies_overridden': True,
            'producer_sources_byte_exact_with_original_v5': False},
        'external_asset_inventory': local_assets,
        'external_asset_binding_required_before_scene': {
            'background_texture/seen': str(Path(cfg['existing_asset_root']) / 'background_texture/seen'),
            'background_texture/unseen': 'excluded_from_independent_runtime',
            'objects': str(Path(cfg['existing_asset_root']) / 'objects')},
        'pending_remote_gates': ['owned realpath verification', 'external asset binding',
            'exact environment dependency/version readback', 'native CPU runtime prepare',
            'dedicated system-unit v1 positive/negative OOM probe',
            'guarded SAPIEN renderer constructor/destructor and official offscreen smoke release pair',
            'four-phase kernel controller readback and zero swap observation',
            'native physical recording'],
        'allowed_changes': ['server-owned path relocation', 'declared collect CPU quota',
            'declared physical GPU selection', 'dependent exporter/publisher SHA binding',
            'six independently pinned native OMPL simplify=True overlay copies',
            'task-scoped transient system-unit and v1 controller resource guard',
            'pinned NVIDIA Vulkan loader only in guarded GPU stage child',
            'optional scene-free renderer diagnostic and official 128x128 offscreen diagnostic'],
        'physics_dt_s': .004, 'observation_dt_s': .060,
        'SDK_action_dim': 24, 'RGB_shape': [480, 640, 3],
        'builder_sha256': sha(__file__)}
    files = {}
    for p in stage.rglob('*'):
        if p.is_file():
            require(not p.is_symlink() and p.stat().st_size <= MAX_FILE, 'Unexpected package file')
            files[str(p.relative_to(stage))] = {'sha256': sha(p), 'bytes': p.stat().st_size}
    manifest['package_files'] = files
    manifest['file_count'] = len(files)
    manifest['uncompressed_file_bytes'] = sum(v['bytes'] for v in files.values())
    dump(output / 'bundle_manifest.json', manifest)
    dump(stage / 'bundle_manifest.json', manifest)
    archive = output / 'realman_161_bundle.tar.gz'
    with tarfile.open(archive, 'w:gz', compresslevel=2, dereference=False) as tar:
        tar.add(stage, arcname='.', recursive=True)
    receipt = {'passed': True, 'static_bundle_only': True, 'remote_run_started': False,
        'archive_path': str(archive), 'archive_sha256': sha(archive),
        'archive_bytes': archive.stat().st_size, 'manifest_sha256': sha(output / 'bundle_manifest.json'),
        'file_count': len(files), 'uncompressed_file_bytes': manifest['uncompressed_file_bytes'],
        'complete_config': cfg}
    dump(output / 'build_receipt.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.config, args.output), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
