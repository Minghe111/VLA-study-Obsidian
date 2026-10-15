"""Run exactly one task stage in a verified, bounded user cgroup.

Default is a read-only preflight. Execution is argv-only; there is no shell.
The original guard is left intact and shares this guard's lock and unit name.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

GIB = 1024 ** 3
E = Path(__file__).resolve().parent
TASK_E = E.parent
PYTHON = '/home/admin123/liminghe/miniconda3/envs/robotwin/bin/python'
CPU_PYTHON = '/usr/bin/python3'
PREFIX = Path('/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4')
VARIANT = 'rgb50_motion_quality_20261014_v5_staged'
UNIT = 'realman-rerecord-protected.service'
LOCK = E / 'resource_guard.lock'
RUNS = E / 'resource_runs'
CGROOT = Path('/sys/fs/cgroup')
MIN_AVAILABLE = 18 * GIB
FLOOR_AVAILABLE = 6 * GIB
GPU_MIN_FREE = 12 * GIB
MIN_DISK_FREE = 32 * GIB
FLOOR_DISK_FREE = 16 * GIB
SAMPLE_SEC = 2.
STAGES = {'collect': (10, 12, 35 * 60), 'export': (3, 4, 20 * 60),
          'quality': (2, 3, 20 * 60), 'render': (2, 3, 20 * 60)}
TOOLS = {'export': (E / 'streaming_export.py', E / 'streaming_export_staged.py'),
         'quality': (E / 'staged_quality.py', E / 'bounded_actual_trace_audit.py'),
         'render': (TASK_E / 'media_scripts/render_accepted_expert.py',
                    E / 'render_staged_video.py')}
COLLECTOR_NAMES = {'run_randomized_collector.py', 'run_conservative_collector.py',
                   'run_clearance_acquisition_collector.py', 'run_so3_acquisition_collector.py',
                   'run_staged_collector.py', 'scratch_collector_entry.py',
                   'generation_worker_v2.py', 'worker_settle_wake_v5.py'}
ALL_STAGE_NAMES = COLLECTOR_NAMES | {p.name for ps in TOOLS.values() for p in ps}
WORKER = E / 'run_guarded_stage.py'
PROBE = E / 'guarded_stage_cpu_probe.py'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    temp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    os.replace(temp, path)


def append_jsonl(path, obj):
    # A single append syscall per bounded sample; no image/trace payloads.
    data = (json.dumps(obj, ensure_ascii=False, separators=(',', ':')) + '\n').encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        n = os.write(fd, data)
        if n != len(data):
            raise RuntimeError('Short resource sample write')
    finally:
        os.close(fd)


def memory_available():
    for row in Path('/proc/meminfo').read_text().splitlines():
        if row.startswith('MemAvailable:'):
            return int(row.split()[1]) * 1024
    raise RuntimeError('MemAvailable unavailable')


def output_filesystems(runtime, command):
    paths = [Path(runtime), TASK_E]
    for flag in ('--output', '--export-output', '--candidate'):
        if flag in command:
            i = command.index(flag)
            if i + 1 >= len(command):
                raise ValueError('Missing output path')
            paths.append(Path(command[i + 1]).resolve())
    devices = {}
    for path in paths:
        while not path.exists():
            path = path.parent
        devices.setdefault(path.stat().st_dev, path)
    return tuple(devices.values())


def disk_free(paths):
    return {str(path): shutil.disk_usage(path).free for path in paths}


def integer_file(path):
    try:
        value = Path(path).read_text().strip()
        return value if value == 'max' else int(value)
    except FileNotFoundError:
        return None


def flat_numbers(path):
    try:
        return {row.split()[0]: int(row.split()[1]) for row in Path(path).read_text().splitlines()}
    except FileNotFoundError:
        return None


def limits_for(stage, timeout_sec=None):
    high, maximum, timeout = STAGES[stage]
    if timeout_sec is not None:
        if not math.isfinite(timeout_sec) or not 30 <= timeout_sec <= 35 * 60:
            raise ValueError('Timeout must be finite and between 30 and 2100 seconds')
        timeout = int(timeout_sec)
    return {'MemoryAccounting': 'yes', 'MemoryHigh': str(high * GIB),
            'MemoryMax': str(maximum * GIB), 'MemorySwapMax': '0',
            'CPUAccounting': 'yes', 'CPUQuota': '200%', 'TasksMax': '128',
            'Nice': '10', 'KillMode': 'control-group', 'TimeoutStopSec': '10s',
            'RuntimeMaxSec': str(timeout), 'OOMPolicy': 'stop',
            'IOWeight': '25'}


def validate_command(stage, command):
    if len(command) < 2 or command[0] != PYTHON:
        raise ValueError('Only exact task Python and registered stage script are allowed')
    if any('\x00' in arg for arg in command):
        raise ValueError('NUL in argument')
    original = Path(command[1])
    resolved = original.resolve(strict=True)
    if not original.is_absolute() or original != resolved:
        raise ValueError('Script must use its exact absolute real path')
    if stage == 'collect':
        rel = resolved.relative_to(PREFIX)
        if (len(rel.parts) != 5 or rel.parts[0] != VARIANT or rel.parts[1] != 'jobs'
            or rel.parts[2] not in {f'slot{x}_runtime' for x in (17, 18, 20, 29, 34)}
            or rel.parts[3:] != ('preflight', 'run_staged_collector.py')):
            raise ValueError('Only new per-slot V5 collection entry is allowed')
        runtime = resolved.parent.parent
    else:
        if resolved not in TOOLS[stage]:
            raise ValueError('Script is not registered for selected stage')
        runtime = TASK_E
    return runtime, resolved


def process_rows():
    rows = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            uid = p.stat().st_uid
            args = [a.decode(errors='replace') for a in (p / 'cmdline').read_bytes().split(b'\0') if a]
            if not args:
                continue
            cg = next(x[3:] for x in (p / 'cgroup').read_text().splitlines() if x.startswith('0::'))
            status = (p / 'status').read_text()
            rss = next((int(x.split()[1]) * 1024 for x in status.splitlines() if x.startswith('VmRSS:')), 0)
        except (FileNotFoundError, PermissionError, ProcessLookupError, StopIteration):
            continue
        # Do not publish arbitrary desktop command lines or environment values.
        rows.append({'pid': int(p.name), 'uid': uid, 'args': args,
                     'cgroup': cg, 'rss_bytes': rss})
    return rows


def known_task_processes(rows=None):
    found = []
    for row in rows if rows is not None else process_rows():
        if row['pid'] == os.getpid():
            continue
        args = row['args']
        script = next((arg for arg in args[1:] if Path(arg).name in ALL_STAGE_NAMES), None)
        # Restrict identification to the task, never an unrelated similarly named job.
        scoped = any(str(PREFIX / 'rgb50_motion_quality_2026101') in x
                     or str(TASK_E) in x for x in args)
        isolated_python = (Path(args[0]).name.startswith('python') and any(
            str(PREFIX / 'rgb50_motion_quality_2026101') in a and a.endswith('.py')
            for a in args[1:]))
        task_media = Path(args[0]).name == 'ffmpeg' and scoped
        if scoped and (script or isolated_python or task_media):
            script = script or next((a for a in args[1:] if a.endswith('.py')), 'ffmpeg')
            found.append({'pid': row['pid'], 'uid': row['uid'], 'script': script,
                          'cgroup': row['cgroup'], 'rss_bytes': row['rss_bytes']})
    return found


def systemctl(*args, check=False):
    return subprocess.run(['systemctl', '--user', *args], check=check, text=True,
                          capture_output=True, timeout=15)


def unit_info():
    properties = ('LoadState', 'ActiveState', 'SubState', 'ControlGroup',
                  'ExecMainCode', 'ExecMainStatus', 'Result', 'Description',
                  'MemoryCurrent', 'MemoryPeak', 'CPUUsageNSec')
    r = systemctl('show', UNIT, '--property=' + ','.join(properties))
    if r.returncode:
        raise RuntimeError('User systemd manager unavailable; refuse unbounded execution: ' + r.stderr.strip())
    return dict(line.split('=', 1) for line in r.stdout.splitlines() if '=' in line)


def scoped_cgroup(info):
    value = info.get('ControlGroup', '')
    if not value:
        return None
    root = f'/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/'
    if not value.startswith(root) or Path(value).name != UNIT:
        raise RuntimeError('Unexpected unit cgroup; refuse touching it')
    return CGROOT / value.lstrip('/')


def safe_stop(info=None):
    info = info or unit_info()
    cg = scoped_cgroup(info)
    if cg is None:
        return
    relative = '/' + str(cg.relative_to(CGROOT))
    rows = [row for row in process_rows() if row['cgroup'] == relative or row['cgroup'].startswith(relative + '/')]
    if any(row['uid'] != os.getuid() for row in rows):
        raise RuntimeError('Cgroup contains foreign-owned process; refuse cleanup')
    try:
        r = systemctl('stop', UNIT)
        if r.returncode == 0:
            return
    except subprocess.TimeoutExpired:
        r = None
    # User manager failure must not leave this job running after guard failure.
    # cgroup.kill acts on this exact validated user task group, including children.
    killfile = cg / 'cgroup.kill'
    if killfile.exists():
        killfile.write_text('1')
        return
    raise RuntimeError('Task unit stop failed and bounded task cgroup cleanup unavailable')


def cgroup_sample(cg, *, rows=None):
    if cg is None:
        return {}
    relative = '/' + str(cg.relative_to(CGROOT))
    members = [r for r in rows if r['cgroup'] == relative or r['cgroup'].startswith(relative + '/')]
    result = {name: integer_file(cg / name) for name in ('memory.current', 'memory.peak',
              'memory.swap.current', 'memory.swap.peak', 'memory.swap.max', 'pids.current', 'pids.peak')}
    for name in ('memory.events', 'memory.events.local', 'memory.swap.events', 'cpu.stat'):
        result[name] = flat_numbers(cg / name)
    result['rss_sum_bytes'] = sum(r['rss_bytes'] for r in members)
    result['member_pids'] = sorted(r['pid'] for r in members)
    return result


def gpu_sample():
    # Small driver query only. No simulator, CUDA context or allocation is created.
    try:
        r = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu',
                            '--format=csv,noheader,nounits'], text=True, capture_output=True, timeout=3)
        if r.returncode:
            return {'healthy': False, 'reason': 'nvidia-smi exit ' + str(r.returncode)}
        row = next(x for x in r.stdout.splitlines() if x.split(',')[0].strip() == '0')
        values = [int(x.strip()) for x in row.split(',')]
        return {'healthy': True, 'index': values[0], 'total_bytes': values[1] * 1024**2,
                'used_bytes': values[2] * 1024**2, 'free_bytes': values[3] * 1024**2,
                'utilization_percent': values[4], 'hard_memory_cap': False}
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError, StopIteration):
        return {'healthy': False, 'reason': 'GPU readback unavailable or timed out'}


def env_for(stage, runtime):
    return {'PATH': '/home/admin123/liminghe/miniconda3/envs/robotwin-official-data/bin:'
                   '/home/admin123/liminghe/miniconda3/envs/robotwin/bin:/usr/bin:/bin',
            'PYTHONDONTWRITEBYTECODE': '1', 'OPENBLAS_NUM_THREADS': '2', 'OMP_NUM_THREADS': '2',
            'MKL_NUM_THREADS': '2', 'NUMEXPR_NUM_THREADS': '2',
            'CUDA_VISIBLE_DEVICES': '0' if stage == 'collect' else '',
            'MPLCONFIGDIR': str(runtime / 'preflight/protected_mplcache')}


def verify_current_cgroup(limits):
    relative = next(x[3:] for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'))
    if Path(relative).name != UNIT or not relative.startswith(f'/user.slice/user-{os.getuid()}.slice/'):
        raise RuntimeError('Worker is not inside expected task cgroup')
    cg = CGROOT / relative.lstrip('/')
    actual = {name: (cg / name).read_text().strip() for name in
              ('memory.high', 'memory.max', 'memory.swap.max', 'cpu.max', 'pids.max', 'memory.oom.group')}
    required = {'memory.high': limits['MemoryHigh'], 'memory.max': limits['MemoryMax'],
                'memory.swap.max': '0', 'cpu.max': '200000 100000', 'pids.max': '128'}
    if any(actual[key] != value for key, value in required.items()):
        raise RuntimeError('Cgroup resource readback differs from requested limits')
    if os.getpriority(os.PRIO_PROCESS, 0) != 10:
        raise RuntimeError('Worker priority readback differs from Nice=10')
    return cg, actual


def internal_worker(spec_path):
    path = Path(spec_path).resolve(strict=True)
    path.relative_to(RUNS.resolve(strict=True))
    if path.name != 'launch_spec.json':
        raise ValueError('Unregistered worker spec')
    spec = json.loads(path.read_text())
    runtime, script = validate_command(spec['stage'], spec['command'])
    if sha(script) != spec['source_binding']['entry_sha256'] or sha(WORKER) != spec['source_binding']['guard_sha256']:
        raise RuntimeError('Stage source changed before execution')
    cg, readback = verify_current_cgroup(spec['limits'])
    run_dir = path.parent
    atomic_json(run_dir / 'cgroup_readback.json', {'limits_verified': True, 'actual': readback,
                                                'cgroup': str(cg), 'pid': os.getpid()})
    proc = subprocess.Popen(spec['command'], cwd=runtime, start_new_session=False)
    def forward(sig, _frame):
        if proc.poll() is None:
            proc.send_signal(sig)
    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    code = proc.wait()
    snapshot = {'stage_returncode': code, 'wall_time_unix': time.time(),
                'resource': cgroup_sample(cg, rows=process_rows()),
                'source_unchanged': sha(script) == spec['source_binding']['entry_sha256']}
    atomic_json(run_dir / 'worker_terminal.json', snapshot)
    return code


def execute(stage, command, *, run=False, timeout_sec=None, receipt_dir=None, cleanup_orphan=False):
    runtime, script = validate_command(stage, command)
    limits = limits_for(stage, timeout_sec)
    RUNS.mkdir(exist_ok=True)
    directory = Path(receipt_dir) if receipt_dir else RUNS / f'{stage}_{time.time_ns()}'
    if not directory.is_absolute():
        raise ValueError('Receipt directory must be absolute')
    directory = directory.resolve()
    directory.relative_to(RUNS.resolve())
    with LOCK.open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another task stage holds shared lock; refuse overlap')
        info = unit_info()
        old_cg = scoped_cgroup(info)
        active = info.get('ActiveState') in {'active', 'activating', 'deactivating'}
        if active:
            if not cleanup_orphan:
                raise RuntimeError('Task unit still active; refuse overlap. Explicit --cleanup-orphan required.')
            # Exact task unit only; never global process killing or desktop cleanup.
            members = [r for r in process_rows() if old_cg and
                       r['cgroup'].startswith('/' + str(old_cg.relative_to(CGROOT)))]
            owned = any(any(str(WORKER) == a or (Path(a).name in ALL_STAGE_NAMES and
                        (str(PREFIX) in a or str(TASK_E) in a)) for a in r['args']) for r in members)
            if not owned or any(r['uid'] != os.getuid() for r in members):
                raise RuntimeError('Cannot prove active unit is an orphan from this task')
            if not run:
                raise RuntimeError('Read-only preflight never cleans an orphan; use --run explicitly')
            orphan_receipt = {'previous_unit': info, 'pids': [r['pid'] for r in members],
                              'before': cgroup_sample(old_cg, rows=process_rows())}
            safe_stop(info)
            info = unit_info()
        else:
            orphan_receipt = None
        existing = known_task_processes()
        if existing:
            raise RuntimeError('Existing task stage/collector outside guarded unit: ' + json.dumps(existing))
        available = memory_available()
        if available < MIN_AVAILABLE:
            raise RuntimeError('Less than 18 GiB host memory available; refuse start')
        disk_paths = output_filesystems(runtime, command)
        disk_before = disk_free(disk_paths)
        if min(disk_before.values()) < MIN_DISK_FREE:
            raise RuntimeError('Less than 32 GiB disk free on task destination; refuse start')
        gpu = gpu_sample() if stage == 'collect' else None
        if gpu and (not gpu['healthy'] or gpu.get('free_bytes', 0) < GPU_MIN_FREE):
            raise RuntimeError('GPU 0 readback unhealthy or less than 12 GiB free; refuse collect')
        receipt = {'schema': 'realman_guarded_stage_v1', 'stage': stage, 'check_only': not run,
                   'command': command, 'runtime': str(runtime), 'unit': UNIT, 'limits': limits,
                   'environment_overrides': env_for(stage, runtime), 'start_minimum_available_bytes': MIN_AVAILABLE,
                   'stop_available_floor_bytes': FLOOR_AVAILABLE, 'low_samples_required': 3,
                   'sample_seconds': SAMPLE_SEC, 'gpu_hard_memory_cap': False, 'auto_retry': False,
                   'available_before_bytes': available, 'gpu_before': gpu, 'disk_before_bytes': disk_before,
                   'disk_start_minimum_bytes': MIN_DISK_FREE, 'disk_stop_floor_bytes': FLOOR_DISK_FREE, 'orphan_cleanup': orphan_receipt,
                   'source_binding': {'entry_sha256': sha(script), 'guard_sha256': sha(WORKER)}}
        if not run:
            return receipt, 0
        directory.mkdir(exist_ok=False)
        atomic_json(directory / 'receipt.json', receipt)
        atomic_json(directory / 'launch_spec.json', receipt)
        # Clean stale failed transient state only after the global lock and scope checks.
        systemctl('reset-failed', UNIT)
        launch = ['systemd-run', '--user', '--wait', '--pipe', '--quiet', '--unit=' + UNIT,
                  '--description=RealMan rerecord guarded stage ' + stage,
                  '--working-directory=' + str(runtime)]
        launch += ['--property=' + key + '=' + value for key, value in limits.items()]
        launch += ['--setenv=' + key + '=' + value for key, value in env_for(stage, runtime).items()]
        launch += [CPU_PYTHON, str(WORKER), '--internal-worker', str(directory / 'launch_spec.json')]
        begin = time.monotonic()
        peak = {'cgroup_bytes': 0, 'rss_sum_bytes': 0, 'minimum_available_bytes': available}
        low = 0
        low_disk = 0
        bad_gpu = 0
        reason = None
        latest_gpu = gpu
        next_gpu = 0.
        with (directory / 'stage.log').open('wb') as log:
            proc = subprocess.Popen(launch, stdout=log, stderr=subprocess.STDOUT)
            try:
                while True:
                    elapsed = time.monotonic() - begin
                    info = unit_info()
                    cg = scoped_cgroup(info)
                    resources = cgroup_sample(cg, rows=process_rows())
                    avail = memory_available()
                    sample = {'elapsed_seconds': elapsed, 'time_unix': time.time(),
                              'host_available_bytes': avail, 'cgroup': resources, 'unit_state': info,
                              'disk_free_bytes': disk_free(disk_paths)}
                    if stage == 'collect' and elapsed >= next_gpu:
                        latest_gpu = gpu_sample()
                        bad_gpu = bad_gpu + 1 if not latest_gpu['healthy'] else 0
                        next_gpu = elapsed + 10.
                    sample['gpu'] = latest_gpu
                    append_jsonl(directory / 'resources.jsonl', sample)
                    peak['minimum_available_bytes'] = min(peak['minimum_available_bytes'], avail)
                    peak['cgroup_bytes'] = max(peak['cgroup_bytes'], resources.get('memory.peak') or 0,
                                                resources.get('memory.current') or 0)
                    peak['rss_sum_bytes'] = max(peak['rss_sum_bytes'], resources.get('rss_sum_bytes', 0))
                    low = low + 1 if avail < FLOOR_AVAILABLE else 0
                    low_disk = low_disk + 1 if min(sample['disk_free_bytes'].values()) < FLOOR_DISK_FREE else 0
                    if low_disk >= 3:
                        reason = 'Task destination disk free below 16 GiB for three samples'
                    if low >= 3:
                        reason = 'Host MemAvailable below 6 GiB for three 2-second samples'
                    if bad_gpu >= 3:
                        reason = 'GPU driver readback failed across three monitored queries'
                    if elapsed > int(limits['RuntimeMaxSec']) + 15:
                        reason = 'Guard wall timeout beyond systemd RuntimeMaxSec'
                    if reason:
                        safe_stop(info)
                        break
                    if proc.poll() is not None:
                        break
                    time.sleep(SAMPLE_SEC)
                code = proc.wait(timeout=20)
            except BaseException:
                try:
                    safe_stop(info)
                finally:
                    proc.wait(timeout=20)
                    receipt['monitor_stop_reason'] = 'Guard interrupted or monitor failure; task unit stopped'
                    receipt['wall_seconds'] = time.monotonic() - begin
                    receipt['launcher_returncode'] = proc.poll()
                    atomic_json(directory / 'receipt.json', receipt)
                raise
        terminal_path = directory / 'worker_terminal.json'
        terminal = json.loads(terminal_path.read_text()) if terminal_path.exists() else None
        if terminal:
            peak['cgroup_bytes'] = max(peak['cgroup_bytes'], terminal['resource'].get('memory.peak') or 0)
            peak['rss_sum_bytes'] = max(peak['rss_sum_bytes'], terminal['resource'].get('rss_sum_bytes') or 0)
        info = unit_info()
        receipt.update({'wall_seconds': time.monotonic() - begin, 'launcher_returncode': code,
                        'monitor_stop_reason': reason, 'available_after_bytes': memory_available(),
                        'disk_after_bytes': disk_free(disk_paths),
                        'peak': peak, 'unit_terminal': info, 'worker_terminal': terminal,
                        'cgroup_readback_verified': (directory / 'cgroup_readback.json').exists(),
                        'receipt_directory': str(directory)})
        receipt['passed'] = (code == 0 and not reason and terminal is not None
                             and terminal['stage_returncode'] == 0 and terminal['source_unchanged']
                             and receipt['cgroup_readback_verified'])
        atomic_json(directory / 'receipt.json', receipt)
        return receipt, 0 if receipt['passed'] else (code if code else 1)


def self_test_cgroup():
    """Two tiny CPU-only units: readback and intentional 32 MiB cap violation."""
    with LOCK.open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        info = unit_info()
        if info.get('ActiveState') in {'active', 'activating', 'deactivating'} or known_task_processes():
            raise RuntimeError('Task active; no concurrent resource probe')
        directory = RUNS / ('cpu_probe_' + str(time.time_ns()))
        directory.mkdir(parents=True, exist_ok=False)
        results = []
        for negative in (False, True):
            unit = 'realman-rerecord-cpu-cap-probe-' + str(time.time_ns()) + '.service'
            output = directory / ('negative_readback.json' if negative else 'positive_readback.json')
            launch = ['systemd-run', '--user', '--wait', '--pipe', '--quiet', '--unit=' + unit,
                      '--property=MemoryHigh=' + ('33554432' if negative else '25165824'), '--property=MemoryMax=33554432',
                      '--property=MemorySwapMax=0', '--property=CPUQuota=100%',
                      '--property=TasksMax=32', '--property=Nice=10',
                      '--property=KillMode=control-group', '--property=OOMPolicy=stop',
                      '--property=RuntimeMaxSec=30', '--setenv=OPENBLAS_NUM_THREADS=1',
                      '--setenv=OMP_NUM_THREADS=1', '--setenv=CUDA_VISIBLE_DEVICES=',
                      CPU_PYTHON, str(PROBE), str(output)]
            if negative:
                launch.append('--overrun')
            begin = time.monotonic()
            r = subprocess.run(launch, capture_output=True, text=True, timeout=45)
            properties = subprocess.run(['systemctl', '--user', 'show', unit,
                           '--property=Result,ExecMainCode,ExecMainStatus,MemoryPeak,CPUUsageNSec'],
                           text=True, capture_output=True, timeout=15)
            observed = dict(x.split('=', 1) for x in properties.stdout.splitlines() if '=' in x)
            readback = json.loads(output.read_text()) if output.exists() else None
            result = {'negative': negative, 'unit': unit, 'returncode': r.returncode,
                      'wall_seconds': time.monotonic() - begin, 'readback': readback,
                      'unit_terminal': observed, 'log': r.stdout + r.stderr}
            result['passed'] = (readback is not None and readback.get('limits_verified') is True
                                and (r.returncode != 0 and observed.get('Result') == 'oom-kill'
                                     if negative else r.returncode == 0))
            results.append(result)
            # Exact probe unit only; no service/configuration or unrelated task changes.
            subprocess.run(['systemctl', '--user', 'stop', unit], capture_output=True, timeout=15)
            subprocess.run(['systemctl', '--user', 'reset-failed', unit], capture_output=True, timeout=15)
        receipt = {'schema': 'realman_tiny_cgroup_probe_v1', 'scope': 'CPU-only 32 MiB task probe',
                   'results': results, 'passed': all(x['passed'] for x in results),
                   'guard_sha256': sha(WORKER), 'probe_sha256': sha(PROBE)}
        atomic_json(directory / 'receipt.json', receipt)
        print(json.dumps({'passed': receipt['passed'], 'receipt': str(directory / 'receipt.json')}, indent=2))
        return 0 if receipt['passed'] else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=STAGES)
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--timeout-sec', type=float)
    ap.add_argument('--receipt-dir')
    ap.add_argument('--cleanup-orphan', action='store_true')
    ap.add_argument('--internal-worker', help=argparse.SUPPRESS)
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--probe-cgroup', action='store_true')
    ap.add_argument('command', nargs=argparse.REMAINDER)
    args = ap.parse_args()
    if args.internal_worker:
        return internal_worker(args.internal_worker)
    if args.self_test:
        if args.stage or args.command or args.run:
            ap.error('--self-test cannot combine stage commands')
        tests = E / 'test_guarded_stage.py'
        code = subprocess.call([CPU_PYTHON, str(tests)])
        return code if code or not args.probe_cgroup else self_test_cgroup()
    if args.probe_cgroup:
        ap.error('--probe-cgroup requires --self-test')
    if not args.stage:
        ap.error('--stage is required')
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    receipt, code = execute(args.stage, command, run=args.run, timeout_sec=args.timeout_sec,
                            receipt_dir=args.receipt_dir, cleanup_orphan=args.cleanup_orphan)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
