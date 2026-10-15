#!/usr/bin/env python3
"""External serial controller; never modifies the frozen bundle or sudo policy.
Run with the bundle Python in the already authenticated SSH PTY. Default is
plan-only. Failed stages stop this batch and preserve all outputs for diagnosis.
"""
import argparse
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pwd
import re
import signal
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path('/bigdata/liminghe/realman_rerecord/20261015_v8_handoff')
VARIANT = 'rgb50_motion_quality_20261014_v6_161_hybrid_v8_handoff'
PYTHON = '/bigdata/liminghe/VLA-benchmark/deploy/realman_gen4/simulation_conda/robotwin/bin/python'
PREFLIGHT = ROOT / 'evidence/preflight'
RUNS = PREFLIGHT / 'resource_runs'
RUNTIME = ROOT / 'deploy/realman_gen4' / VARIANT / 'jobs'
CONTROL = ROOT / 'evidence/continuous_v8'
PLAN = {34: (8, 31400034), 17: (9, 81400017), 18: (2, 10400018),
        20: (2, 11400020), 29: (2, 10100029)}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            value.update(block)
    return value.hexdigest()


def safe(path):
    path = Path(path)
    require(path.is_absolute() and path.is_relative_to(ROOT) and path == path.resolve(),
            'Path is outside the exact real task root: ' + str(path))
    return path


def write_new(path, value):
    path = safe(path)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def emit(event, **values):
    print(json.dumps({'event': event, **values}, ensure_ascii=False), flush=True)


class Authentication:
    """Refresh only this same-terminal credential cache while controller lives."""
    def __init__(self):
        self.stop = threading.Event()
        self.error = None
        self.thread = None

    def refresh(self):
        require(self.error is None, 'Task sudo refresh failed: ' + str(self.error))
        result = subprocess.run(['sudo', '-n', '-v'], stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                text=True, timeout=10)
        require(result.returncode == 0, 'Same-PTY sudo authentication is unavailable')

    def start(self):
        self.refresh()
        def heartbeat():
            while not self.stop.wait(60):
                try:
                    self.refresh()
                except Exception as error:
                    self.error = str(error)
                    return
        self.thread = threading.Thread(target=heartbeat, name='task-sudo-refresh', daemon=True)
        self.thread.start()

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=12)


def selected(args):
    slots = args.slots
    require(len(slots) == len(set(slots)), 'Repeated slots are forbidden')
    if args.stop_after_slot is not None:
        require(args.stop_after_slot in slots, '--stop-after-slot is not in --slots')
        slots = slots[:slots.index(args.stop_after_slot) + 1]
    return slots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--slots', nargs='+', type=int, choices=PLAN, default=list(PLAN))
    parser.add_argument('--stop-after-slot', type=int, choices=PLAN)
    parser.add_argument('--reuse-renderer-smokes', action='store_true',
                        help='Use latest valid same-boot smoke pair; collect guard independently validates it')
    parser.add_argument('--batch-id', default=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8])
    args = parser.parse_args()
    slots = selected(args)
    require(re.fullmatch(r'[a-zA-Z0-9_-]{1,64}', args.batch_id), 'Unsafe batch id')
    summary = {'run_root': str(ROOT), 'variant': VARIANT, 'slots': slots,
               'attempt_seed': {str(i): PLAN[i] for i in slots},
               'reuse_renderer_smokes': args.reuse_renderer_smokes,
               'source_policy': 'Frozen stages; no ordinary-failure retry in this controller',
               'batch_id': args.batch_id, 'publication_performed': False}
    if not args.run:
        emit('plan_only', **summary)
        return 0
    require(os.getuid() == pwd.getpwnam('liminghe').pw_uid and
            os.getgid() == pwd.getpwnam('liminghe').pw_gid, 'Run controller as liminghe, not root')
    require(os.isatty(0), 'Controller requires the already authenticated SSH PTY')
    require(Path(sys.executable).resolve() == Path(PYTHON).resolve(), 'Use exact bundle Python')
    safe(ROOT)
    require(ROOT.is_dir() and PREFLIGHT.is_dir(), 'Bundle root/preflight is absent')
    manifest_path = ROOT / 'bundle_manifest.json'
    manifest_sha = sha(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    def pinned(path):
        path = safe(path)
        relative = path.relative_to(ROOT).as_posix()
        row = manifest['package_files'][relative]
        require(path.stat().st_size == row['bytes'] and sha(path) == row['sha256'],
                'Frozen bundle file differs: ' + str(path))
        require(sha(manifest_path) == manifest_sha, 'Bundle manifest changed')
    for filename in ['run_guarded_stage.py', 'hybrid_release_contract.py',
                     'hybrid_cgroup_adapter.py', 'publish_staged_patch.py']:
        pinned(PREFLIGHT / filename)
    CONTROL.mkdir(exist_ok=True)
    safe(CONTROL)
    with safe(CONTROL / 'controller.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        accepted = CONTROL / 'verified_slots'
        accepted.mkdir(exist_ok=True)
        safe(accepted)
        batch = safe(CONTROL / args.batch_id)
        require(not batch.exists(), 'Existing batch/partial state must not be overwritten')
        for slot in slots:
            require(not (accepted / f'slot_{slot}.json').exists(),
                    'Slot was already verified; exclude it instead of recollecting: ' + str(slot))
            attempt, _ = PLAN[slot]
            candidate = RUNTIME / f'slot{slot}_runtime/new_physical_episodes' / f'slot_{slot:07d}_randomized_attempt_{attempt:03d}'
            require(not safe(candidate).exists(), 'Candidate exists; never overwrite or automatically reuse')
        batch.mkdir()
        write_new(batch / 'request.json', summary)
        auth = Authentication()
        rows = []
        current = {}
        active = None
        def interrupted(signum, frame):
            raise KeyboardInterrupt('Controller received signal ' + str(signum))
        for signum in (signal.SIGHUP, signal.SIGTERM):
            signal.signal(signum, interrupted)
        def stage(name, label, command):
            nonlocal active
            auth.refresh()
            pinned(PREFLIGHT / 'run_guarded_stage.py')
            pinned(command[1])
            # The quality scanner treats collect_*/receipt.json as actual runs.
            # A CPU prepare must not enter that namespace for the same slot/attempt.
            prefix = 'prepare' if label.endswith('_prepare') else name
            receipt_dir = safe(RUNS / (prefix + '_continuous_' + args.batch_id + '_' + label))
            require(not receipt_dir.exists(), 'Resource receipt already exists')
            launch = ['/usr/bin/python3', str(PREFLIGHT / 'run_guarded_stage.py'),
                      '--stage', name, '--run', '--receipt-dir', str(receipt_dir), '--', *command]
            current.clear()
            current.update(stage=name, label=label, receipt_directory=str(receipt_dir))
            write_new(batch / (label + '_launch.json'), {'command': launch, **current})
            with (batch / (label + '_console.log')).open('xb') as log:
                active = subprocess.Popen(launch, stdout=log, stderr=subprocess.STDOUT)
                emit('stage_started', pid=active.pid, **current)
                reported_unit = None
                while active.poll() is None:
                    if auth.error:
                        raise RuntimeError('Authentication heartbeat failed: ' + auth.error)
                    spec = receipt_dir / 'launch_spec.json'
                    if reported_unit is None and spec.is_file():
                        reported_unit = json.loads(spec.read_text()).get('unit')
                        emit('stage_unit', unit=reported_unit, pid=active.pid, **current)
                    time.sleep(1)
                code = active.returncode
                active = None
            receipt_path = receipt_dir / 'receipt.json'
            require(receipt_path.is_file(), 'Stage produced no complete receipt; inspect retained console/probe files')
            receipt = json.loads(receipt_path.read_text())
            emit('stage_terminal', returncode=code, passed=receipt.get('passed'),
                 unit=receipt.get('unit'), receipt=str(receipt_path), **current)
            require(code == 0 and receipt.get('passed') is True, 'Stage rejected or failed; batch stopped for diagnosis')
            return str(receipt_path)
        try:
            auth.start()
            emit('controller_started', pid=os.getpid(), tty=os.ttyname(0), session_id=os.getsid(0), **summary)
            if not args.reuse_renderer_smokes:
                # Required explicit prefixes are enforced by the original guard.
                for mode, suffix in [('ctor', []), ('offscreen', ['--offscreen-example'])]:
                    # stage() naming is specialized below to match the frozen smoke path contract.
                    smoke_name = 'renderer_smoke_' + mode + '_continuous_' + args.batch_id
                    smoke_dir = RUNS / smoke_name
                    require(not smoke_dir.exists(), 'Renderer receipt already exists')
                    auth.refresh()
                    command = ['/usr/bin/python3', str(PREFLIGHT / 'run_guarded_stage.py'), '--stage',
                               'renderer_smoke', '--run', '--receipt-dir', str(smoke_dir), '--',
                               PYTHON, str(PREFLIGHT / 'renderer_only_smoke.py'), *suffix]
                    pinned(PREFLIGHT / 'renderer_only_smoke.py')
                    current.clear(); current.update(stage='renderer_smoke', label=mode, receipt_directory=str(smoke_dir))
                    with (batch / (smoke_name + '_console.log')).open('xb') as log:
                        active = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                        emit('stage_started', pid=active.pid, **current)
                        while active.poll() is None:
                            require(auth.error is None, 'Authentication heartbeat failed')
                            time.sleep(1)
                        code = active.returncode; active = None
                    receipt = json.loads((smoke_dir / 'receipt.json').read_text())
                    require(code == 0 and receipt.get('passed') is True, 'Renderer smoke failed; stop batch')
                    emit('stage_terminal', passed=True, unit=receipt.get('unit'), **current)
            for slot in slots:
                attempt, seed = PLAN[slot]
                runtime = RUNTIME / f'slot{slot}_runtime'
                command = [PYTHON, str(runtime / 'preflight/run_staged_collector.py'), '--index',
                           str(slot), '--attempt', str(attempt), '--domain-seed', str(seed)]
                tag = f'slot{slot}_a{attempt}'
                stage('collect', tag + '_prepare', command + ['--prepare-runtime'])
                receipts = {'collect': stage('collect', tag + '_collect', command + ['--run'])}
                candidate = runtime / 'new_physical_episodes' / f'slot_{slot:07d}_randomized_attempt_{attempt:03d}'
                outputs = {name: batch / (tag + '_' + name) for name in ('export', 'quality', 'render')}
                for name, script, extra in [
                    ('export', 'streaming_export_staged.py', ['--hdf-only']),
                    ('quality', 'staged_quality.py', ['--export-output', str(outputs['export']), '--resource-root', str(RUNS)]),
                    ('render', 'render_staged_video.py', [])]:
                    receipts[name] = stage(name, tag + '_' + name, [PYTHON, str(PREFLIGHT / script),
                        '--candidate', str(candidate), '--output', str(outputs[name]), *extra])
                row = {'slot': slot, 'candidate': str(candidate), 'resource_receipts': receipts,
                       **{k + '_output': str(v) for k, v in outputs.items()}}
                # This is the frozen full per-candidate publication verifier; no copy/publication.
                pinned(PREFLIGHT / 'publish_staged_patch.py')
                sys.path.insert(0, str(PREFLIGHT))
                spec = importlib.util.spec_from_file_location('_v8_frozen_publisher', PREFLIGHT / 'publish_staged_patch.py')
                publisher = importlib.util.module_from_spec(spec); spec.loader.exec_module(publisher)
                publisher.pinned_file(publisher.ORIGINAL50, publisher.ORIGINAL50_SHA)
                originals = publisher.read_json(publisher.ORIGINAL50)
                require(originals.get('complete') is True and originals.get('count') == 50, 'Original50 reference incomplete')
                old = {entry['index']: entry for entry in originals['episodes']}
                proof = publisher.verify_candidate(row, old[slot])
                pinned(PREFLIGHT / 'publish_staged_patch.py')
                require(proof.get('all_original_physical_RGB_SDK_motion_DR_video_resource_gates_passed') is True,
                        'Full per-candidate verifier did not pass')
                write_new(batch / (tag + '_verification.json'), proof)
                write_new(accepted / f'slot_{slot}.json', {'row': row, 'proof_path': str(batch / (tag + '_verification.json')),
                          'proof_sha256': sha(batch / (tag + '_verification.json')), 'bundle_manifest_sha256': manifest_sha})
                rows.append(row)
                emit('slot_verified', slot=slot, verified_slots=[r['slot'] for r in rows], publication_performed=False)
            write_new(batch / 'terminal.json', {'complete_selected_slots': True, 'rows': rows, **summary})
            emit('batch_completed', selected_slots=slots, verified_count=len(rows), publication_performed=False)
            return 0
        except BaseException as error:
            if active is not None and active.poll() is None:
                # SIGINT enters the existing guard's bounded exact-unit cleanup path.
                active.send_signal(signal.SIGINT)
                try:
                    active.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    emit('guard_still_live', pid=active.pid, **current)
            write_new(batch / 'terminal.json', {'complete_selected_slots': False, 'error': str(error),
                      'error_type': type(error).__name__, 'current': current, 'verified_rows': rows,
                      'automatic_retry_performed': False, 'publication_performed': False})
            emit('batch_stopped', error=str(error), **current)
            return 1
        finally:
            auth.close()


if __name__ == '__main__':
    raise SystemExit(main())
