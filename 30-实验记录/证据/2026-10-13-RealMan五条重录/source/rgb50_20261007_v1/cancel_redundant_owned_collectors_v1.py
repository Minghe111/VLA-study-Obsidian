"""Cancel only exact owned queues whose unfinished identities are formal.

Default is a read-only probe. --execute sends SIGTERM only after checking the
exact worker/jobs/output/GPU/UID and independent non-synthetic v3 receipts.
Cancellation records are external; original candidate files are never edited.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import signal
import time


WORK = Path(__file__).resolve().parent
EXPERT = 'expert_generation_v9_equivalent_dense_waypoint_v8'
GROUPS = {
    1: ('coke_right_upright_formal33_micro_parallel_v8_gpu1_jobs.json',
        'batch_candidates/v9_coke_right_upright_formal33_micro_v8_gpu1_attempt07'),
    2: ('coke_right_upright_formal33_neighbours_v8_jobs.json',
        'batch_candidates/v9_coke_right_upright_formal33_neighbours_v8_attempt06'),
    3: ('coke_right_upright_gain01_waypoint_v8_matched_jobs.json',
        'batch_candidates/v9_coke_right_upright_gain01_waypoint_v8_attempt05'),
    6: ('coke_right_upright_formal33_micro_parallel_v8_gpu6_jobs.json',
        'batch_candidates/v9_coke_right_upright_formal33_micro_v8_gpu6_attempt08'),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flag(argv, name):
    return argv[argv.index(name) + 1] if name in argv else None


def owned_processes(gpu, jobs, output):
    matches = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdecimal():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            argv = [value.decode() for value in (proc / 'cmdline').read_bytes().split(b'\0') if value]
            if str(WORK / 'generation_worker_v2.py') not in argv:
                continue
            if flag(argv, '--jobs') != str(jobs) or flag(argv, '--output') != str(output):
                continue
            if flag(argv, '--expert-module') != EXPERT or flag(argv, '--backend') != 'full_native':
                continue
            environ = (proc / 'environ').read_bytes().split(b'\0')
            assert f'CUDA_VISIBLE_DEVICES={gpu}'.encode() in environ
            assert (proc / 'exe').resolve() == Path('/bigdata2/liminghe/envs/robotwin/bin/python').resolve()
            matches.append({'pid': int(proc.name), 'argv': argv, 'CUDA_VISIBLE_DEVICES': str(gpu)})
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    assert len(matches) <= 1, 'Ambiguous owned queue process'
    return matches


def qualified_proof(index):
    for path in (WORK / 'accepted_list_v3_admission.json', WORK / 'accepted_list_v2.json'):
        current = json.loads(path.read_bytes())
        assert current['reader_version'] == 3
        rows = current.get('trials', current.get('accepted', []))
        for row in rows:
            identity = row['identity']
            if identity['spec_index'] != index:
                continue
            assert (identity['model_id'], identity['arm'], identity['pose_class']) == (13, 'right', 'upright')
            receipt_path = Path(row['audit_receipt_path'])
            assert sha(receipt_path) == row['audit_receipt_sha256']
            proof = json.loads(receipt_path.read_bytes())
            assert proof['passed'] is True and proof['readback_only'] is True
            assert proof['synthetic_fixture_only'] is False and proof['new_physics_runs'] == 0
            assert proof['identity'] == identity
            trial = Path(proof['trial_dir'])
            assert trial.resolve() == Path(row['trial_dir']).resolve()
            assert sha(trial / 'expert_report.json') == proof['source_files']['expert_report.json']['sha256']
            return {'index': index, 'trial_dir': str(trial), 'identity': identity,
                    'formal_receipt_path': str(receipt_path), 'formal_receipt_sha256': sha(receipt_path)}
    return None


ap = argparse.ArgumentParser()
ap.add_argument('--gpu', type=int, choices=sorted(GROUPS), required=True)
ap.add_argument('--execute', action='store_true')
opt = ap.parse_args()
jobs_name, output_name = GROUPS[opt.gpu]
jobs_path, output = WORK / jobs_name, WORK / output_name
jobs = json.loads(jobs_path.read_bytes())
status_path = output / 'worker_status.json'
status = json.loads(status_path.read_bytes()) if status_path.is_file() else {}
finished = {row['name'] for row in status.get('receipts', [])}
pending_indices = sorted({job['spec']['index'] for job in jobs if job['name'] not in finished})
proofs = {index: qualified_proof(index) for index in pending_indices}
processes = owned_processes(opt.gpu, jobs_path, output)
record = {'scope': 'External cancellation only, no episode success or source changes',
          'gpu': opt.gpu, 'jobs_path': str(jobs_path), 'jobs_sha256': sha(jobs_path),
          'output': str(output), 'pending_spec_indices': pending_indices,
          'independent_formal_qualification': proofs, 'owned_processes': processes,
          'read_only_probe': not opt.execute, 'original_candidates_not_modified_by_this_helper': True}
if opt.execute:
    assert pending_indices and all(proofs.values()), 'Every unfinished identity must already be independently formal'
    assert processes, 'No exact active owned worker to cancel'
    assert owned_processes(opt.gpu, jobs_path, output) == processes, 'Process scope changed before signal'
    pid = processes[0]['pid']
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 10.
    while owned_processes(opt.gpu, jobs_path, output) and time.monotonic() < deadline:
        time.sleep(.1)
    record['owned_worker_stopped'] = not owned_processes(opt.gpu, jobs_path, output)
    record['signal'] = 'SIGTERM'
    record['cancelled_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    target_dir = WORK / 'collector_cancellations_v1'
    target_dir.mkdir(exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    receipt_path = target_dir / f'gpu{opt.gpu}_pid{pid}_{stamp}.json'
    with receipt_path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(record, indent=2) + '\n')
    record['external_cancel_receipt'] = str(receipt_path)
    record['external_cancel_receipt_sha256'] = sha(receipt_path)
print(json.dumps(record, indent=2), flush=True)
