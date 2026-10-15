"""Read-only package/archive audit; write only the new validation receipt."""
from pathlib import Path
import hashlib
import json
import tarfile

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
BUNDLE = BASE / 'server_bundle_20261015_v8b_handoff'

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()

manifest_path = BUNDLE / 'bundle_manifest.json'
manifest = json.loads(manifest_path.read_text())
payload = BUNDLE / 'payload'
assert sha(manifest_path) == sha(payload / 'bundle_manifest.json')
expected = dict(manifest['package_files'])
expected['bundle_manifest.json'] = {'sha256': sha(manifest_path), 'bytes': manifest_path.stat().st_size}
actual = {str(p.relative_to(payload)) for p in payload.rglob('*') if p.is_file()}
assert actual == set(expected)
for name, row in expected.items():
    path = payload / name
    assert path.is_file() and not path.is_symlink()
    assert sha(path) == row['sha256'] and path.stat().st_size == row['bytes'], name
seen = set()
archive = BUNDLE / 'realman_161_bundle.tar.gz'
with tarfile.open(archive, 'r:gz') as tar:
    for member in tar:
        assert not member.issym() and not member.islnk()
        if member.isdir():
            continue
        name = member.name.removeprefix('./')
        assert member.isfile() and name in expected and name not in seen
        assert '..' not in Path(name).parts
        digest = hashlib.sha256()
        with tar.extractfile(member) as stream:
            for chunk in iter(lambda: stream.read(1024**2), b''):
                digest.update(chunk)
        assert digest.hexdigest() == expected[name]['sha256'] and member.size == expected[name]['bytes'], name
        seen.add(name)
assert seen == set(expected)
deployment = payload / 'deploy/realman_gen4' / manifest['complete_config']['variant']
runtimes = {'evidence': payload / 'evidence', **{f'slot{s}': deployment / f'jobs/slot{s}_runtime' for s in (17, 18, 20, 29, 34)}}
overlays = {label: {p.name: sha(p) for p in (root / 'planning').glob('*.py')} for label, root in runtimes.items()}
assert all(len(row) == 13 and row == overlays['evidence'] for row in overlays.values())
assert overlays['evidence']['grasp_acquisition_guard.py'] == 'f6751cc8204d85c4e23067994c79b14b5199df5274df0e564f5181980783583e'
before = json.loads((HERE / 'before.json').read_text())
for name, row in before.items():
    assert sha(name) == row['sha256'] and Path(name).stat().st_size == row['bytes'], name
build = json.loads((BUNDLE / 'build_receipt.json').read_text())
assert sha(archive) == build['archive_sha256'] and archive.stat().st_size == build['archive_bytes']
guard_grasp = json.loads((HERE / 'grasp_guard_tests_receipt.json').read_text())
quality = json.loads((HERE / 'quality_tests_receipt.json').read_text())
tests = guard_grasp['tests'] + [quality]
assert all(row['passed'] and sha(row['log']) == row['log_sha256'] for row in tests)
result = {'passed': True, 'static_offline_local_only': True, 'remote_upload_or_connection_performed_by_this_audit': False,
    'hardware_io': False, 'scene_or_episode_executed_by_this_audit': False,
    'archive': str(archive), 'archive_sha256': sha(archive), 'archive_bytes': archive.stat().st_size,
    'manifest_sha256': sha(manifest_path), 'all_manifest_files_verified_count': len(manifest['package_files']),
    'all_archive_file_SHA_verified_count_including_manifest': len(seen),
    'six_overlay_sets_byte_exact_equal': True, 'overlay_sets': overlays,
    'original_source_count': manifest['protected_source_count'], 'protected_hardware_count': manifest['protected_hardware_count'],
    'original_candidate7f_and_unreleased_candidate8_archive_metadata_unchanged': True,
    'active_final_helper_and_test_unchanged': True, 'tests': tests,
    'tests_run_total': sum(row['tests_run'] for row in tests),
    'quality_test_scope': 'Unmodified BoundedIO, ComponentBoundary, PublisherDispatch cases against packaged quality; test-process TASK points at local package evidence. Desktop-v2 ResourceFixture excluded; separate package guard suite tests v1 resource-contract code.',
    'limitations': ['CPU synthetic fixtures and byte checks do not prove physical grasp success.',
        'This audit did not execute 161 resource controls, a renderer or a physical scene.',
        'External scene assets are inventoried separately, not content-verified by this archive audit.'],
    'audit_script_sha256': sha(__file__)}
destination = HERE / 'validation_receipt.json'
assert not destination.exists()
destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({key: result[key] for key in ('passed', 'archive_sha256', 'archive_bytes', 'manifest_sha256', 'tests_run_total', 'all_archive_file_SHA_verified_count_including_manifest')}, indent=2))
