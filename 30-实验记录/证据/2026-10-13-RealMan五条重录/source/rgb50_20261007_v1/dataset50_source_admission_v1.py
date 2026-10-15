"""Explicit original-root admission using the frozen v3 independent validator.

CPU readback only. Never moves a candidate, edits a report, imports an expert,
steps physics, or writes existing observers' outputs. Baseline receipts are
reused only after a complete current SHA inventory equals their frozen proof.
Subsequent reuse requires unchanged all-file stats, completion proof hashes,
and current expert-source hashes. Final publication always reruns full v3.
"""
from __future__ import annotations
import argparse
from collections import Counter
import datetime
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
REMOTE_ROOT = Path('/bigdata2/liminghe/VLA-benchmark/deploy/realman_gen4/rgb50_20261007_v1')
V3_REGISTRY_SHA = 'a14f0dda04df6290e26778226739a3a2747c2c8c212ea29ed6e10a727a09f96a'
PLAN_SHA = 'a146e9cb966bf8d029317dcce9f248e5f32af56a4c3b2080afbc9d12880b904e'
APPROVED_SOURCES = {
    'full_native_approach_waypoint_v4.py': 'f20f53c7b4cf464ac0f21bdbe61ad008b0054d0e9bcbf2640f4b340ab5c8ae66',
    'expert_generation_v9_waypoint_v4.py': '1314c9de08b2929108614d0ff36151a048283ec16f59721c5087e37dab14fb76',
}
TARGETS = {'arm': {'left': 25, 'right': 25}, 'model_id': {'13': 20, '16': 20, '4': 10},
           'pose_class': {'upright': 20, 'horizontal': 15, 'diagonal': 15}}


def require(value, message):
    if not bool(value): raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024*1024), b''): digest.update(part)
    return digest.hexdigest()


def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def token(value): return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic(path, value):
    path = Path(path); tmp = path.with_name('.'+path.name+'.tmp-'+str(os.getpid()))
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    os.replace(tmp, path)


def within(root, directory, *, allow_root=False):
    root = Path(root); directory = Path(directory)
    require(root.is_absolute() and directory.is_absolute(), 'Collection paths must be absolute')
    for path in [directory, *directory.parents]: require(not path.is_symlink(), 'Collection symlink forbidden')
    root = root.resolve(); directory = directory.resolve()
    require(directory.is_relative_to(root) and (allow_root or directory != root), 'Candidate escapes explicit admitted root')
    return directory


def check_required_sources(actual, expected):
    require(isinstance(actual, dict) and bool(actual), 'Missing report source registry')
    for path, digest in expected.items():
        require(actual.get(path) == digest, 'Required admitted expert/helper source pin differs: '+path)
    for path, digest in actual.items():
        require(Path(path).is_absolute() and Path(path).is_file() and not Path(path).is_symlink(), 'Expert source unavailable: '+path)
        require(sha(path) == digest, 'Current expert source SHA differs: '+path)


def same_snapshot(before, after):
    require(before == after, 'Completion/source snapshot changed during independent verification')


def file_stamps(directory):
    result = {}
    for path in sorted(Path(directory).rglob('*')):
        require(not path.is_symlink(), 'Trial symlink forbidden')
        if path.is_file():
            stat = path.stat(); result[str(path.relative_to(directory))] = [stat.st_size, stat.st_mtime_ns]
    return result


def formal_identity(identity, expected):
    index = identity.get('spec_index', identity.get('index'))
    require(type(index) is int and index in expected, 'Not a formal0..49 slot')
    require(type(identity.get('seed')) is int and identity['seed'] >= 0, 'Invalid actual seed')
    require(all(identity[k] == expected[index][k] for k in ['model_id', 'arm', 'pose_class']), 'Formal slot brand/arm/pose differs')
    return index


def select_unique(rows, expected):
    selected = []; excluded = []; slots = set(); seeds = set()
    for row in sorted(rows, key=lambda x: (x['qualified_utc'], x['trial_dir'])):
        try: index = formal_identity(row['identity'], expected)
        except Exception as error:
            excluded.append({'trial_dir': row['trial_dir'], 'reason': str(error)}); continue
        seed = row['identity']['seed']
        if index in slots:
            excluded.append({'trial_dir': row['trial_dir'], 'index': index, 'reason': 'Later independently qualifying duplicate slot'}); continue
        if seed in seeds:
            excluded.append({'trial_dir': row['trial_dir'], 'index': index, 'reason': 'Duplicate actual seed across distinct slots'}); continue
        slots.add(index); seeds.add(seed); selected.append(row)
    selected.sort(key=lambda x: x['identity']['spec_index'])
    counts = {key: dict(Counter(str(row['identity'][key]) for row in selected)) for key in TARGETS}
    for key, counts_for_key in counts.items():
        require(all(value <= TARGETS[key].get(name, 0) for name, value in counts_for_key.items()), 'Formal quota exceeded')
    complete = len(selected) == 50 and slots == set(range(50)) and counts == TARGETS and len(seeds) == 50
    return selected, excluded, counts, complete


def validate_config(config):
    require(ROOT == REMOTE_ROOT, 'This admission is frozen to the166 dataset root')
    require(config.get('schema_version') == 1 and config.get('hardware_io') is False
            and config.get('new_physics_runs') == 0 and config.get('gpu_runs') == 0, 'Admission scope differs')
    expected_root = ROOT/'smoke_worker_right_upright_waypoint_v4'
    roots = config.get('admitted_roots')
    require(isinstance(roots, list) and len(roots) == 1, 'Only one explicitly reviewed additional root is admitted')
    require(roots[0]['path'] == str(expected_root), 'Unknown or broadened admitted root')
    required = {str(ROOT/name): digest for name, digest in APPROVED_SOURCES.items()}
    require(roots[0]['required_report_source_sha256'] == required, 'Reviewed waypoint source pins changed')
    require(config['baseline_accepted_list'] == str(ROOT/'accepted_list_v2.json'), 'Baseline list path changed')
    require(config['frozen_v3_registry_sha256'] == V3_REGISTRY_SHA and config['formal_plan_sha256'] == PLAN_SHA, 'Original freeze/plan differs')
    within(ROOT, expected_root)
    check_required_sources(required, required)
    return expected_root, required


def verify_environment(freeze, own_registry, own_digest, config_digest):
    require(sha(ROOT/'dataset50_source_admission_v1_freeze.json') == own_digest, 'Admission registry changed')
    require(sha(ROOT/'source_admission_waypoint_v4.json') == config_digest, 'Admission config changed')
    for name, digest in own_registry['source_sha256'].items(): require(sha(ROOT/name) == digest, 'Admission source changed: '+name)
    require(sha(ROOT/'dataset50_validator_v3_source_freeze.json') == V3_REGISTRY_SHA, 'Frozen v3 registry changed')
    for name, digest in freeze['source_sha256'].items(): require(sha(ROOT/name) == digest, 'Frozen v3 source changed: '+name)
    require(sha(ROOT/'sample_plan.json') == PLAN_SHA, 'Formal sampling plan changed')
    for name, digest in APPROVED_SOURCES.items(): require(sha(ROOT/name) == digest, 'Admitted waypoint source changed: '+name)


def load_runtime():
    registry_path = ROOT/'dataset50_source_admission_v1_freeze.json'
    own = read_json(registry_path); own_digest = sha(registry_path)
    config_path = ROOT/'source_admission_waypoint_v4.json'; config_digest = sha(config_path)
    freeze = read_json(ROOT/'dataset50_validator_v3_source_freeze.json')
    verify_environment(freeze, own, own_digest, config_digest)
    config = read_json(config_path); admitted, required = validate_config(config)
    # Import only after verifying every frozen validator byte. No expert import.
    import dataset50_audit_v3 as audit
    import dataset50_progress_v3 as gates
    plan = read_json(ROOT/'sample_plan.json'); expected = {s['index']: s for s in plan['specs']}
    require(set(expected) == set(range(50)), 'Formal plan lacks all50 distinct slots')
    return config, admitted, required, freeze, own, own_digest, config_digest, audit, gates, expected


def snapshot(directory, audit, gates, expected_sources=None):
    spec, key = gates.ready_snapshot(directory)
    report = read_json(directory/'expert_report.json')
    check_required_sources(report['source_hashes'], expected_sources or {})
    key['all_source_file_stats'] = file_stamps(directory)
    return spec, key


def original_collection(directory):
    directory = Path(directory)
    roots = [ROOT/'batch_candidates']+[p for p in ROOT.glob('smoke_worker*_v9') if p.is_dir()]
    require(any(directory.resolve().is_relative_to(p.resolve()) for p in roots), 'Baseline is outside original explicit v3 collections')
    within(ROOT, directory)


def proof_identity(row, item, freeze, expected):
    require(row.get('passed') is True and row.get('validator_version') == 3 and row.get('boundary_used_verified') is True,
            'Not a fully passed independent v3 receipt')
    require(row.get('synthetic_fixture_only') is not True, 'Synthetic receipt forbidden')
    require(row['frozen_validator_source_sha256'] == freeze['source_sha256'], 'Independent receipt freeze pins differ')
    require(row['private_reader_sha256'] == freeze['source_sha256']['dataset50_reader_v3.py'], 'Private v3 reader pin differs')
    require(Path(row['trial_dir']).resolve() == Path(item['trial_dir']).resolve(), 'Proof trial identity differs')
    require(row['identity'] == item['identity'] and row['boundary_used'] == item['boundary_used'], 'Proof/list identity differs')
    formal_identity(row['identity'], expected)


def observe(watch, interval):
    config, admitted, required, freeze, own, own_digest, config_digest, audit, gates, expected = load_runtime()
    proof_dir = ROOT/'formal_independent_v3_admission_receipts'; proof_dir.mkdir(exist_ok=True)
    baseline_cache = {}; candidate_cache = {}; first_seen = {}; cycle = 0
    while True:
        cycle += 1; started = time.monotonic(); pending = []; rejected = []; qualified = []
        verify_environment(freeze, own, own_digest, config_digest)
        # Admitted candidates are checked first, so a large baseline rehash cannot hide completion.
        for receipt in sorted(admitted.rglob('worker_receipt.json'), key=lambda p: (p.stat().st_mtime_ns, str(p))):
            try:
                directory = within(admitted, receipt.parent)
                spec, key = snapshot(directory, audit, gates, required); formal_identity(spec, expected)
            except Exception as error:
                rejected.append({'trial_dir': str(receipt.parent), 'reason': str(error)}); continue
            name = str(directory); previous = candidate_cache.get(name)
            if previous and previous['key'] == key:
                if previous['passed']:
                    qualified.append({**previous['item'], 'current_unchanged_stats_and_proofs_verified_utc': now()})
                else: rejected.append({'trial_dir': name, 'reason': previous['error']})
                continue
            candidate_cache.pop(name, None)
            seen = first_seen.get(name)
            if not seen or seen['key'] != key:
                first_seen[name] = {'key': key, 'time': time.time()}
                pending.append({'trial_dir': name, 'index': spec['index'], 'reason': 'Waiting30s for unchanged original completion files'}); continue
            if time.time()-seen['time'] < 30:
                pending.append({'trial_dir': name, 'index': spec['index'], 'reason': 'Completion stability interval not yet30s'}); continue
            print('ADMISSION_FULL_V3_AUDIT_START', spec['index'], name, flush=True)
            row = audit.audit_trial(directory, codec_path=ROOT.parents[2]/'RoboTwin/data/decode_image_bit.py')
            try:
                _, after = snapshot(directory, audit, gates, required); same_snapshot(key, after)
                verify_environment(freeze, own, own_digest, config_digest)
                if row['passed']:
                    formal_identity(row['identity'], expected)
                    require(audit.inventory(directory) == row['source_files'], 'Admitted complete source SHA inventory changed after v3 audit')
            except Exception as error: row.update(passed=False, error=str(error))
            generation = token(key)[:16]; trial_id = hashlib.sha256(name.encode()).hexdigest()[:12]
            path = proof_dir/(f"slot{spec['index']:07d}_admission_{trial_id}_{generation}_{time.time_ns()}.json")
            row.update(frozen_validator_source_sha256=freeze['source_sha256'], worker_receipt_sha256=key['small_proof_hashes']['worker_receipt.json'],
                       source_admission_config_sha256=config_digest, source_admission_registry_sha256=own_digest,
                       source_admission_original_root=str(admitted), source_admission_new_physics_runs=0)
            with path.open('x', encoding='utf-8') as stream: stream.write(json.dumps(row, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
            entry = {'key': key, 'passed': row['passed'], 'error': row.get('error')}
            if row['passed']:
                item = {'trial_dir': name, 'identity': row['identity'], 'boundary_used': row['boundary_used'], 'qualified_utc': now(),
                        'audit_receipt_path': str(path), 'audit_receipt_sha256': sha(path),
                        'worker_receipt_sha256': key['small_proof_hashes']['worker_receipt.json'],
                        'expert_report_sha256': key['small_proof_hashes']['expert_report.json'],
                        'admission_kind': 'new_explicit_original_root_full_v3_audit', 'complete_source_rehash_utc': now(),
                        'current_unchanged_stats_and_proofs_verified_utc': now()}
                entry['item'] = item; qualified.append(item)
            else: rejected.append({'trial_dir': name, 'reason': row.get('error'), 'audit_receipt_path': str(path)})
            candidate_cache[name] = entry
            print('ADMISSION_FULL_V3_AUDIT_RESULT', spec['index'], row['passed'], row.get('error'), flush=True)
        baseline_path = Path(config['baseline_accepted_list']); baseline_bytes = baseline_path.read_bytes()
        baseline_digest = hashlib.sha256(baseline_bytes).hexdigest(); baseline = json.loads(baseline_bytes)
        require(baseline.get('reader_version') == 3 and baseline.get('validator_freeze_sha256') == V3_REGISTRY_SHA,
                'Baseline accepted list is not the frozen v3 independently qualified list')
        for item in baseline['trials']:
            try:
                directory = Path(item['trial_dir']).resolve(); original_collection(directory)
                path = within(ROOT/'formal_independent_v3_receipts', Path(item['audit_receipt_path']))
                proof_digest = sha(path); require(proof_digest == item['audit_receipt_sha256'], 'Baseline immutable receipt SHA changed')
                row = read_json(path); proof_identity(row, item, freeze, expected)
                spec, key = snapshot(directory, audit, gates); formal_identity(spec, expected)
                require(key['small_proof_hashes']['worker_receipt.json'] == item['worker_receipt_sha256']
                        and key['small_proof_hashes']['expert_report.json'] == item['expert_report_sha256'], 'Baseline current report/worker SHA differs')
                require(row['source_files']['worker_receipt.json']['sha256'] == item['worker_receipt_sha256'], 'Original v3 source inventory lacks exact worker binding')
                cached = baseline_cache.get(proof_digest)
                if not cached or cached['key'] != key:
                    print('BASELINE_CURRENT_FULL_SHA_START', row['identity']['spec_index'], str(directory), flush=True)
                    require(audit.inventory(directory) == row['source_files'], 'Baseline current full source SHA inventory differs from frozen receipt')
                    _, after = snapshot(directory, audit, gates); same_snapshot(key, after)
                    require(sha(path) == proof_digest, 'Baseline receipt changed during complete SHA reread')
                    cached = {'key': key, 'rehash_utc': now()}; baseline_cache[proof_digest] = cached
                    print('BASELINE_CURRENT_FULL_SHA_PASSED', row['identity']['spec_index'], flush=True)
                qualified.append({**item, 'admission_kind': 'baseline_frozen_v3_receipt_current_complete_SHA_rechecked',
                                  'complete_source_rehash_utc': cached['rehash_utc'], 'current_unchanged_stats_and_proofs_verified_utc': now()})
            except Exception as error: rejected.append({'trial_dir': item.get('trial_dir'), 'reason': str(error), 'admission_kind': 'baseline_recheck'})
        # Refresh proof snapshots at publication time; invalidate any file change during a long scan.
        current = []
        for item in qualified:
            try:
                directory = Path(item['trial_dir']); _, key = snapshot(directory, audit, gates, required if item['admission_kind'].startswith('new_') else {})
                cache = candidate_cache[str(directory)] if item['admission_kind'].startswith('new_') else baseline_cache[item['audit_receipt_sha256']]
                same_snapshot(cache['key'], key)
                require(sha(item['audit_receipt_path']) == item['audit_receipt_sha256'], 'Qualification receipt changed before publishing progress')
                current.append(item)
            except Exception as error: rejected.append({'trial_dir': item['trial_dir'], 'reason': str(error)})
        selected, conflicts, counts, complete = select_unique(current, expected)
        verify_environment(freeze, own, own_digest, config_digest)
        summary = {'schema_version': 1, 'reader_version': 3, 'source_admission_version': 1, 'updated_utc': now(), 'cycle': cycle,
                   'complete': complete, 'dataset_published': False, 'accepted_count': len(selected), 'actual_counts': counts, 'target_counts': TARGETS,
                   'unique_actual_seeds': len({r['identity']['seed'] for r in selected}) == len(selected),
                   'status': 'all50_independently_qualified_not_published' if complete else 'partial_independent_qualification_not50complete',
                   'boundary_used_count': sum(r['boundary_used'] is True for r in selected),
                   'boundary_used_spec_indices': [r['identity']['spec_index'] for r in selected if r['boundary_used'] is True],
                   'original_extra_collection_root': str(admitted), 'baseline_accepted_list_snapshot_sha256': baseline_digest,
                   'baseline_list_changed_during_cycle': sha(baseline_path) != baseline_digest,
                   'source_admission_config_sha256': config_digest, 'source_admission_registry_sha256': own_digest,
                   'validator_freeze_sha256': V3_REGISTRY_SHA, 'formal_plan_sha256': PLAN_SHA,
                   'resampled_seed_spec_indices': [r['identity']['spec_index'] for r in selected if r['identity']['seed'] != expected[r['identity']['spec_index']]['seed']],
                   'new_physics_runs': 0, 'gpu_runs': 0, 'hardware_io': False, 'last_cycle_elapsed_s': time.monotonic()-started,
                   'verification_scope': 'Every baseline first/rechanged admission rehashes every source file against immutable full v3 receipt. Unchanged cache reuse still checks all file size/mtime, completion proof SHA and every current expert source SHA. New explicit-root candidates run full frozen v3 after30s stability. Final publisher reruns full v3 for all50.',
                   'qualification_rule': 'First qualifying exact formal slot with a unique actual seed; conflicting duplicates excluded with reasons; no candidate movement or proof rebinding.'}
        atomic(ROOT/'accepted_list_v3_admission.json', {**summary, 'trials': selected})
        atomic(ROOT/'formal_accepted_progress_v3_admission.json', {**summary, 'accepted': selected, 'pending': pending,
                                                               'rejected_or_changed': rejected, 'duplicates_excluded': conflicts})
        print('ADMISSION_PROGRESS', json.dumps({k: summary[k] for k in ['accepted_count', 'actual_counts', 'boundary_used_count', 'complete']}), flush=True)
        if complete or not watch: return 0
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--watch', action='store_true'); parser.add_argument('--interval', type=float, default=30)
    args = parser.parse_args(); require(1 <= args.interval <= 60, 'CPU observer interval must be1..60s')
    return observe(args.watch, args.interval)


if __name__ == '__main__': raise SystemExit(main())
