"""CPU-only exact-byte verification of the relocated history3000 task.

No model/torch imports, no training, no network. Reads only files inside the
fixed authorized task root, and writes one point-in-time JSON audit receipt.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import socket
import sys
import time

EXPECTED_TASK_ROOT = Path('/bigdata2/liminghe/rlinf_history3000_20261019_v1')
SOURCE_PREFIX = Path('/home/admin123/liminghe/vla-benchmark')
EXPECTED_USER = 'liminghe'
EXPECTED_HOST = 'user-4U-GPU-Server'
EXPECTED_FILE_COUNT = 3060
EXPECTED_STRUCTURED_COUNT = 16
EXPECTED_VENDOR_COUNT = 68
FIT_RELATIVE = 'deploy/realman_gen4/recap_memory_20261017/experiments/high_sft_history3000_20261019_v1/scripts/fit_comparison.py'
FIT_SHA = '6769f2f796daab0fdfccb4c14d1e7a31563e0fbbbfe6d611f352fa99b513235f'
FIT_BYTES = 29014
FULL_BASE_RELATIVE = 'deploy/realman_gen4/recap_memory_20261017/experiments/protocol_repair_20261019_v2/reports/continuation_v1/language_warmup_96_2200/checkpoint/full_weights.pt'
COMPACT_RELATIVE = 'deploy/realman_gen4/recap_memory_20261017/experiments/high_sft_history3000_20261019_v1/reports/migrated_s002000/checkpoint'
COMPACT_FORMAT = 'rm65b_history3000_language_lora_compact_v1'
SHA = re.compile(r'^[0-9a-f]{64}$')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def stat_identity(path):
    s = Path(path).stat()
    return {'device': s.st_dev, 'inode': s.st_ino, 'size': s.st_size,
            'mtime_ns': s.st_mtime_ns, 'ctime_ns': s.st_ctime_ns}


def safe_path(path, task, *, require_file=True):
    p = Path(path)
    if not p.is_absolute() or '..' in p.parts:
        raise ValueError('Expected a canonical absolute task-local path')
    # Reject links at every component, even if they happen to resolve in task.
    if not p.is_relative_to(task):
        raise ValueError('Path outside authorized task root: ' + str(p))
    current = task
    for part in p.relative_to(task).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError('Symlink runtime input is forbidden: ' + str(current))
    resolved = p.resolve(strict=require_file)
    if not resolved.is_relative_to(task):
        raise ValueError('Resolved input outside authorized task root')
    if require_file and not p.is_file():
        raise FileNotFoundError(p)
    return p


def check_sha(value):
    if not isinstance(value, str) or SHA.fullmatch(value) is None:
        raise ValueError('Expected complete SHA256 digest')
    return value


def verify_one(path, expected_sha, expected_bytes, *, task, kind):
    p = safe_path(path, task)
    before = stat_identity(p)
    actual = sha(p)
    after = stat_identity(p)
    result = {'path': str(p), 'kind': kind, 'expected_sha256': check_sha(expected_sha),
              'actual_sha256': actual, 'expected_bytes': expected_bytes,
              'actual_bytes': after['size'], 'stat_before': before, 'stat_after': after,
              'stable_during_hash': before == after}
    result['passed'] = (actual == expected_sha and before == after and
                        (expected_bytes is None or after['size'] == expected_bytes))
    return result


def read_json(path, task):
    p = safe_path(path, task)
    before = stat_identity(p)
    value = json.loads(p.read_text())
    after = stat_identity(p)
    if before != after:
        raise ValueError('Audit input changed during parse: ' + str(p))
    return value


def identity_guard(task, expected_hostname):
    if not task.is_absolute() or task != EXPECTED_TASK_ROOT:
        raise ValueError('Task must equal the fixed authorized remote prefix')
    if task.is_symlink() or task.resolve(strict=True) != task:
        raise ValueError('Authorized task root must be a real canonical directory')
    uid = os.getuid()
    username = pwd.getpwuid(uid).pw_name
    hostname = socket.gethostname()
    if uid == 0 or username != EXPECTED_USER or task.stat().st_uid != uid:
        raise ValueError('Host user/UID/task ownership mismatch')
    if not expected_hostname or hostname != expected_hostname:
        raise ValueError('Unexpected host: ' + hostname)
    return {'hostname': hostname, 'username': username, 'uid': uid,
            'task_owner_uid': task.stat().st_uid, 'task_root_resolved': str(task.resolve()),
            'expected_hostname': expected_hostname}


def audit(args):
    task = Path(args.task_root)
    identity = identity_guard(task, args.expected_hostname)
    inventory_path = Path(args.inventory or task/'migration_tools/package_inventory.json')
    vendor_path = Path(args.vendor_manifest or task/'migration_tools/vendor_manifest.json')
    receipt_path = Path(args.relocation_receipt or task/'reports/structured_relocation_v1/receipt.json')
    compact_path = Path(args.compact_checkpoint or task/'source_tree'/COMPACT_RELATIVE)
    output = Path(args.output or task/'reports/live_runtime_asset_verification_v2.json')
    safe_path(output, task, require_file=False)
    for p in (inventory_path, vendor_path, receipt_path):
        safe_path(p, task)
    safe_path(compact_path, task, require_file=False)
    report = {'status': 'running', 'scope': 'point_in_time_live_runtime_bytes_CPU_only',
              'identity': identity, 'started_unix': time.time(), 'GPU_used': False,
              'model_loaded': False, 'training_executed': False, 'input_manifests': {},
              'counts': {}, 'issues': [], 'files': [], 'compact': {}}
    issues = report['issues']
    def add(result):
        report['files'].append(result)
        if not result['passed']:
            issues.append({'kind': 'file_mismatch_or_concurrent_mutation', 'path': result['path']})
        return result
    def write():
        output.parent.mkdir(parents=True, exist_ok=True)
        safe_path(output, task, require_file=False)
        temporary = output.with_name(output.name+'.staging')
        safe_path(temporary, task, require_file=False)
        temporary.write_text(json.dumps(report, sort_keys=True, indent=2)+'\n')
        temporary.replace(output)
    try:
        inventory = read_json(inventory_path, task)
        vendor = read_json(vendor_path, task)
        relocation = read_json(receipt_path, task)
        for label, p in [('inventory', inventory_path), ('vendor', vendor_path), ('relocation', receipt_path)]:
            report['input_manifests'][label] = {'path': str(p), 'sha256': sha(p)}
        target_prefix = task / 'source_tree'
        if inventory['source_prefix'] != str(SOURCE_PREFIX) or inventory['target_prefix'] != str(target_prefix):
            raise ValueError('Inventory prefix differs from fixed migration contract')
        rows = inventory['entries']
        if len(rows) != EXPECTED_FILE_COUNT:
            raise ValueError('Inventory must contain exactly 3060 files')
        by_source = {}
        for row in rows:
            relative = Path(row['relative'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Noncanonical inventory relative path')
            if row['path'] != str(SOURCE_PREFIX/relative) or row['target'] != str(target_prefix/relative):
                raise ValueError('Inventory source/target/relative does not agree')
            if row['path'] in by_source:
                raise ValueError('Duplicate inventory source')
            by_source[row['path']] = row
            check_sha(row['sha256'])
        if relocation.get('status') != 'relocated_structured_copies_only' or relocation.get('path_map') != {str(SOURCE_PREFIX): str(target_prefix)}:
            raise ValueError('Structured receipt prefix or status differs')
        changed = {}
        for entry in relocation['files']:
            source = entry['source']
            if source not in by_source or source in changed:
                raise ValueError('Unexpected or duplicate structured receipt source')
            original = by_source[source]
            if (not original.get('structured_rewrite') or entry['sha256'] != original['sha256'] or
                    entry['target'] != original['target']):
                raise ValueError('Structured receipt does not seal the original inventory')
            check_sha(entry['target_sha256'])
            add(verify_one(entry['original_backup'], original['sha256'], original['size'], task=task, kind='preserved_structured_original'))
            changed[source] = entry
        if len(changed) != EXPECTED_STRUCTURED_COUNT or set(changed) != {r['path'] for r in rows if r.get('structured_rewrite')}:
            raise ValueError('Expected all and only 16 structured copies to be relocated')
        first_full = None
        for row in rows:
            expected_sha, expected_bytes, kind = row['sha256'], row['size'], 'original_exact_bytes'
            if row['path'] in changed:
                expected_sha, expected_bytes, kind = changed[row['path']]['target_sha256'], None, 'structured_relocated_sha'
            if row['relative'] == FIT_RELATIVE:
                expected_sha, expected_bytes, kind = FIT_SHA, FIT_BYTES, 'approved_task_CardMonitor_UUID_patch'
            checked = add(verify_one(row['target'], expected_sha, expected_bytes, task=task, kind=kind))
            if row['relative'] == FULL_BASE_RELATIVE:
                first_full = checked
        report['counts']['inventory_checked'] = len(rows)
        report['counts']['structured_overrides'] = len(changed)
        report['counts']['CardMonitor_overrides'] = sum(r['relative'] == FIT_RELATIVE for r in rows)
        if report['counts']['CardMonitor_overrides'] != 1 or first_full is None:
            raise ValueError('Expected exactly one CardMonitor and one C2200 base weight')
        if vendor.get('status') != 'exact_local_runtime_python_and_resources_vendored' or len(vendor['entries']) != EXPECTED_VENDOR_COUNT:
            raise ValueError('Vendor manifest must contain exactly 68 files')
        seen_vendor = set()
        for row in vendor['entries']:
            relative = Path(row['relative'])
            if relative.is_absolute() or '..' in relative.parts or relative.parts[0] not in ('openpi', 'openpi_client'):
                raise ValueError('Unexpected vendor package path')
            if str(relative) in seen_vendor:
                raise ValueError('Duplicate vendor file')
            seen_vendor.add(str(relative))
            add(verify_one(task/'vendor'/relative, row['sha256'], row['bytes'], task=task, kind='exact_local_OpenPi_vendor'))
        report['counts']['vendor_checked'] = len(seen_vendor)
        version_path = compact_path/'version.json'
        version = read_json(version_path, task)
        if version.get('checkpoint_format') != COMPACT_FORMAT or version.get('global_step') != 2000:
            raise ValueError('Expected the derived strict step2000 compact checkpoint')
        if set(version['files']) != {'adapter_weights.pt', 'trainer_state.pt'}:
            raise ValueError('Unexpected compact binary file set')
        for name, entry in version['files'].items():
            add(verify_one(compact_path/name, entry['sha256'], entry['bytes'], task=task, kind='derived_compact_version_seal'))
        base = version['base_checkpoint']
        base_path = Path(base['path'])
        if base_path != (target_prefix/FULL_BASE_RELATIVE).parent:
            raise ValueError('Derived compact points to an unexpected C2200 base')
        for p, expected, kind in [
                (base_path/'full_weights.pt', base['full_weights_sha256'], 'derived_compact_base_full_seal'),
                (base_path/'version.json', base['version_sha256'], 'derived_compact_base_version_seal'),
                (base_path.parent/'effective_config.yaml', base['effective_config_sha256'], 'derived_compact_base_config_seal')]:
            # Base full is checked again last below rather than rehashing here.
            if p.name != 'full_weights.pt':
                add(verify_one(p, expected, None, task=task, kind=kind))
            elif expected != first_full['expected_sha256']:
                raise ValueError('Derived compact base full SHA differs from original inventory')
        report['compact'] = {'path': str(compact_path), 'version_sha256': sha(version_path),
                             'global_step': version['global_step'], 'base_checkpoint': base,
                             'binary_seals_checked': 2, 'policy_loaded': False}
        # Rehash the repaired full base at the end and compare inode/clock as well.
        last_full = add(verify_one(first_full['path'], first_full['expected_sha256'],
                                   first_full['expected_bytes'], task=task, kind='terminal_C2200_rehash_race_guard'))
        stable = first_full['stat_before'] == last_full['stat_after']
        report['C2200_stable_across_complete_audit'] = stable
        if not stable:
            issues.append({'kind': 'C2200_replaced_or_modified_during_complete_audit', 'path': first_full['path']})
        for label, p in [('inventory', inventory_path), ('vendor', vendor_path), ('relocation', receipt_path)]:
            if sha(p) != report['input_manifests'][label]['sha256']:
                issues.append({'kind': 'audit_input_manifest_changed', 'path': str(p)})
        report['status'] = 'passed' if not issues else 'failed'
    except Exception as error:
        issues.append({'kind': 'audit_exception', 'type': type(error).__name__, 'message': str(error)})
        report['status'] = 'failed'
    report['finished_unix'] = time.time()
    report['elapsed_seconds'] = report['finished_unix'] - report['started_unix']
    report['counts']['checked_hash_records'] = len(report['files'])
    report['counts']['issues'] = len(issues)
    report['limitation'] = 'A point-in-time audit cannot prevent a writer replacing files after completion; stop all old transfer receivers before model loading.'
    write()
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task-root', type=Path, default=EXPECTED_TASK_ROOT)
    p.add_argument('--inventory', type=Path)
    p.add_argument('--vendor-manifest', type=Path)
    p.add_argument('--relocation-receipt', type=Path)
    p.add_argument('--compact-checkpoint', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--expected-hostname', default=EXPECTED_HOST)
    args = p.parse_args()
    try:
        result = audit(args)
    except Exception as error:
        print(json.dumps({'status': 'refused_before_task_read', 'type': type(error).__name__, 'message': str(error)}))
        return 2
    print(json.dumps({k: result[k] for k in ('status', 'identity', 'counts', 'elapsed_seconds')}))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
