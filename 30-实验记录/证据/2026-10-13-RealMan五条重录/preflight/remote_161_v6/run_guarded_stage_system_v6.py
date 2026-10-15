"""Server-only RealMan stage guard for the 161 hybrid cgroup host.

Rendered by build_remote_bundle.py as evidence/preflight/run_guarded_stage.py.
No stage process starts before a dedicated system unit has kernel-verified
memory, CPU and PID limits. The v1 soft limit is best effort, and a combined
RAM+swap cap is not a hard swap-off switch; zero swap is observed throughout.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import pwd
import grp
import shutil
import signal
import subprocess
import sys
import time
import uuid

import hybrid_cgroup_adapter as cgroup

GIB = 1024 ** 3
E = Path(__file__).resolve().parent
TASK_E = E.parent
PYTHON = '@PYTHON@'
CPU_PYTHON = '/usr/bin/python3'
PREFIX = Path('@PREFIX@')
VARIANT = '@VARIANT@'
GPU_INDEX = '@GPU_INDEX@'
COLLECT_CPU_PERCENT = '@COLLECT_CPU_PERCENT@'
UNIT_PREFIX = '@UNIT_PREFIX@'
RUN_USER = 'liminghe'
RUN_GROUP = 'liminghe'
LOCK = E / 'resource_guard.lock'
RUNS = E / 'resource_runs'
WORKER = E / 'run_guarded_stage.py'
PROBE = E / 'guarded_stage_cpu_probe.py'
ADAPTER = E / 'hybrid_cgroup_adapter.py'
RELEASE = E / 'hybrid_release_contract.py'
DIAGNOSTICS = E / 'stage_diagnostics/sitecustomize.py'
MIN_AVAILABLE = 18 * GIB
FLOOR_AVAILABLE = 6 * GIB
GPU_MIN_FREE = 12 * GIB
MIN_DISK_FREE = 32 * GIB
FLOOR_DISK_FREE = 16 * GIB
SAMPLE_SEC = 2.
GATE_TIMEOUT_SEC = 45.
STAGES = {'collect': (10, 12, 35 * 60), 'export': (3, 4, 20 * 60),
          'quality': (2, 3, 20 * 60), 'render': (2, 3, 20 * 60)}
TOOLS = {'export': (E / 'streaming_export.py', E / 'streaming_export_staged.py'),
         'quality': (E / 'staged_quality.py', E / 'bounded_actual_trace_audit.py'),
         'render': (TASK_E / 'media_scripts/render_accepted_expert.py',
                    E / 'render_staged_video.py')}
COLLECTOR_NAMES = {'run_staged_collector.py'}
ALL_STAGE_NAMES = COLLECTOR_NAMES | {p.name for ps in TOOLS.values() for p in ps}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    temp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    os.replace(temp, path)


def append_jsonl(path, obj):
    data = (json.dumps(obj, ensure_ascii=False, separators=(',', ':')) + '\n').encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        if os.write(fd, data) != len(data):
            raise RuntimeError('Short resource sample write')
    finally:
        os.close(fd)


def _boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def _identity():
    user = pwd.getpwnam(RUN_USER)
    group = grp.getgrnam(RUN_GROUP)
    if os.getuid() != user.pw_uid or os.getgid() != group.gr_gid:
        raise RuntimeError('Guard must run as exact RealMan owner')


def _new_unit(stage):
    if not UNIT_PREFIX.startswith('realman-rerecord-') or not UNIT_PREFIX.replace('-', '').isalnum():
        raise RuntimeError('Unexpected dedicated unit prefix')
    return f'{UNIT_PREFIX}-{stage}-{uuid.uuid4().hex}.service'


def _systemctl(*args):
    return subprocess.run(['systemctl', '--system', *args], capture_output=True,
                          text=True, timeout=15)


def _sudo_systemctl(*args):
    return subprocess.run(['sudo', '-n', 'systemctl', '--system', *args],
                          capture_output=True, text=True, timeout=15)


def unit_info(unit):
    props = ('LoadState,ActiveState,SubState,ControlGroup,MainPID,User,Group,'
             'InvocationID,Result,ExecMainCode,ExecMainStatus,MemoryCurrent,MemoryPeak,CPUUsageNSec')
    result = _systemctl('show', unit, '--property=' + props)
    if result.returncode:
        raise RuntimeError('System manager status unavailable: ' + result.stderr.strip())
    return dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)


def _unit_inactive(info):
    return info.get('MainPID') in ('0', '', None) and info.get('ActiveState') not in {
        'active', 'activating', 'deactivating'}


def _assert_unit_absent(unit):
    info = unit_info(unit)
    if (info.get('LoadState') != 'not-found' or info.get('ActiveState') != 'inactive'
        or info.get('MainPID') not in ('0', '') or info.get('ControlGroup')
        or info.get('InvocationID')):
        raise RuntimeError('Generated task unit name is already occupied')


def _pid_starttime(pid):
    raw = (Path('/proc') / str(pid) / 'stat').read_text()
    closing = raw.rfind(')')
    if closing < 0:
        raise RuntimeError('Cannot parse worker PID start time')
    fields = raw[closing + 2:].split()
    if len(fields) < 20 or not fields[19].isdecimal():
        raise RuntimeError('Worker PID start time unavailable')
    return fields[19]


def _attest_unit(pid, unit, binding):
    info = unit_info(unit)
    invocation = info.get('InvocationID', '')
    if (info.get('MainPID') != str(pid) or info.get('ControlGroup') != '/system.slice/' + unit
        or info.get('User') != RUN_USER or info.get('Group') != RUN_GROUP
        or len(invocation) != 32 or any(x not in '0123456789abcdef' for x in invocation.lower())):
        raise RuntimeError('Cannot attest exact system unit invocation')
    uid = (Path('/proc') / str(pid)).stat().st_uid
    if uid != pwd.getpwnam(RUN_USER).pw_uid:
        raise RuntimeError('Cannot attest worker UID')
    paths = {key: str(binding[key]) for key in ('memory', 'cpu', 'pids')}
    if binding.get('mode') != 'v1' or any(Path(path).name != unit for path in paths.values()):
        raise RuntimeError('Cannot attest three dedicated v1 controller leaves')
    return {'unit': unit, 'pid': pid, 'uid': uid, 'invocation_id': invocation,
            'pid_starttime': _pid_starttime(pid), 'controller_paths': paths,
            'boot_id': _boot_id()}


def _attest_unready_unit(unit, script, spec_path, expected_source):
    """Read-only ownership proof when the worker never wrote gate_ready.

    A random unit name by itself is never enough to authorize a stop. The
    service PID must still execute this exact frozen task script and spec.
    """
    info = unit_info(unit)
    raw_pid = info.get('MainPID', '')
    if not raw_pid.isdecimal() or int(raw_pid) <= 0:
        raise RuntimeError('No live MainPID for unready unit attestation')
    pid = int(raw_pid)
    binding = _binding_for_pid(pid, unit)
    owner = _attest_unit(pid, unit, binding)
    raw = (Path('/proc') / str(pid) / 'cmdline').read_bytes()
    argv = [piece.decode(errors='replace') for piece in raw.split(b'\0') if piece]
    flag = '--internal-worker' if script == WORKER else '--internal-probe'
    if (len(argv) != 4 or Path(argv[0]).resolve() != Path(CPU_PYTHON).resolve()
        or argv[1:] != [str(script), flag, str(spec_path)]
        or sha(script) != expected_source):
        raise RuntimeError('Unready unit MainPID does not run exact frozen task command')
    spec = json.loads(Path(spec_path).read_text())
    if spec.get('unit') != unit:
        raise RuntimeError('Unready unit launch spec identity differs')
    return owner


def stop_exact_unit(unit, owner):
    # Never kill by name pattern or system-wide process search.
    if (not isinstance(owner, dict) or owner.get('unit') != unit or '/' in unit
        or not unit.startswith(UNIT_PREFIX + '-') or not unit.endswith('.service')):
        raise RuntimeError('Stop target is outside attested task unit')
    before = unit_info(unit)
    if _unit_inactive(before):
        return {'stop_returncode': None, 'terminal': before, 'already_inactive': True}
    current_binding = _binding_for_pid(owner['pid'], unit)
    current = _attest_unit(owner['pid'], unit, current_binding)
    if current != owner:
        raise RuntimeError('Task unit invocation or PID identity changed; refuse stop')
    result = _sudo_systemctl('stop', unit)
    deadline = time.monotonic() + 10.
    while time.monotonic() < deadline:
        info = unit_info(unit)
        if _unit_inactive(info):
            return {'stop_returncode': result.returncode, 'terminal': info}
        time.sleep(.2)
    raise RuntimeError('Cannot prove exact task unit stopped; rc=' + str(result.returncode))


def memory_available():
    for row in Path('/proc/meminfo').read_text().splitlines():
        if row.startswith('MemAvailable:'):
            return int(row.split()[1]) * 1024
    raise RuntimeError('MemAvailable unavailable')


def disk_free(paths):
    return {str(path): shutil.disk_usage(path).free for path in paths}


def output_filesystems(runtime, command):
    paths = [Path(runtime), TASK_E]
    for flag in ('--output', '--export-output', '--candidate'):
        if flag in command:
            index = command.index(flag)
            if index + 1 >= len(command):
                raise ValueError('Output path missing')
            paths.append(Path(command[index + 1]).resolve())
    devices = {}
    for path in paths:
        while not path.exists():
            path = path.parent
        devices.setdefault(path.stat().st_dev, path)
    return tuple(devices.values())


def gpu_sample():
    try:
        query = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu',
                                '--format=csv,noheader,nounits'], capture_output=True,
                               text=True, timeout=3)
        if query.returncode:
            return {'healthy': False, 'reason': 'nvidia-smi exit ' + str(query.returncode)}
        row = next(x for x in query.stdout.splitlines() if x.split(',')[0].strip() == GPU_INDEX)
        values = [int(x.strip()) for x in row.split(',')]
        return {'healthy': True, 'index': values[0], 'total_bytes': values[1] * 1024**2,
                'used_bytes': values[2] * 1024**2, 'free_bytes': values[3] * 1024**2,
                'utilization_percent': values[4], 'hard_memory_cap': False}
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError, StopIteration):
        return {'healthy': False, 'reason': 'GPU readback unavailable or timed out'}


def limits_for(stage, timeout_sec=None):
    high, maximum, timeout = STAGES[stage]
    if timeout_sec is not None:
        if not math.isfinite(timeout_sec) or not 30 <= timeout_sec <= 35 * 60:
            raise ValueError('Timeout must be finite and between 30 and 2100 seconds')
        timeout = int(timeout_sec)
    cpu = int(COLLECT_CPU_PERCENT) if stage == 'collect' else 200
    return {'MemoryAccounting': 'yes', 'MemoryHigh': str(high * GIB),
            'MemoryMax': str(maximum * GIB),
            'MemorySwapMax': 'not-enforceable-on-v1',
            'CPUAccounting': 'yes', 'CPUQuota': str(cpu) + '%', 'TasksMax': '128',
            'Nice': '10', 'KillMode': 'control-group', 'TimeoutStopSec': '10s',
            'RuntimeMaxSec': str(timeout), 'OOMPolicy': 'stop'}


def validate_command(stage, command):
    if stage not in STAGES or len(command) < 2 or command[0] != PYTHON:
        raise ValueError('Only the exact task Python and stage are allowed')
    if any('\x00' in arg for arg in command):
        raise ValueError('NUL in command')
    original = Path(command[1])
    resolved = original.resolve(strict=True)
    if not original.is_absolute() or original != resolved:
        raise ValueError('Script must use its exact absolute real path')
    if stage == 'collect':
        rel = resolved.relative_to(PREFIX)
        if (len(rel.parts) != 5 or rel.parts[0] != VARIANT or rel.parts[1] != 'jobs'
            or rel.parts[2] not in {f'slot{x}_runtime' for x in (17, 18, 20, 29, 34)}
            or rel.parts[3:] != ('preflight', 'run_staged_collector.py')):
            raise ValueError('Only isolated per-slot V6 collector is allowed')
        runtime = resolved.parent.parent
    else:
        if resolved not in TOOLS[stage]:
            raise ValueError('Script is not registered for stage')
        runtime = TASK_E
    return runtime, resolved


def _known_task_processes():
    found = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            raw = (entry / 'cmdline').read_bytes()
            argv = [x.decode(errors='replace') for x in raw.split(b'\0') if x]
            if not argv:
                continue
            task = any(str(PREFIX / VARIANT) in a or str(TASK_E) in a for a in argv)
            stage = (any(Path(a).name in ALL_STAGE_NAMES for a in argv[1:])
                     or ('--internal-worker' in argv and str(WORKER) in argv)
                     or ('--internal-probe' in argv and str(PROBE) in argv))
            if task and stage:
                found.append({'pid': int(entry.name), 'uid': entry.stat().st_uid})
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return found


def env_for(stage, runtime):
    return {'PATH': str(Path(PYTHON).parent) + ':/usr/bin:/bin',
            'PYTHONDONTWRITEBYTECODE': '1', 'OPENBLAS_NUM_THREADS': '2',
            'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'NUMEXPR_NUM_THREADS': '2',
            'CUDA_VISIBLE_DEVICES': GPU_INDEX if stage == 'collect' else '',
            'MPLCONFIGDIR': str(runtime / 'preflight/protected_mplcache')}


def _binding_for_pid(pid, unit):
    info = unit_info(unit)
    if (info.get('MainPID') != str(pid) or info.get('ControlGroup') != '/system.slice/' + unit
        or info.get('User') != RUN_USER or info.get('Group') != RUN_GROUP):
        raise RuntimeError('System unit identity, owner or exact cgroup differs')
    proc = Path('/proc') / str(pid)
    if proc.stat().st_uid != pwd.getpwnam(RUN_USER).pw_uid:
        raise RuntimeError('Worker PID owner differs')
    binding = cgroup.discover((proc / 'cgroup').read_text(),
                              (proc / 'mountinfo').read_text(), unit)
    if binding['mode'] != 'v1':
        raise RuntimeError('161 hybrid runtime requires independent v1 memory/cpu/pids leaves')
    return binding


def _sudo_write_exact(path, value, binding, unit):
    path = Path(path)
    if (path.parent != binding['memory'] or path.parent.name != unit
        or path.is_symlink() or not path.is_file()):
        raise RuntimeError('Refuse write outside own v1 memory unit leaf')
    result = subprocess.run(['sudo', '-n', 'tee', str(path)], input=str(value) + '\n',
                            capture_output=True, text=True, timeout=10)
    if result.returncode:
        raise RuntimeError('Kernel cgroup write rejected: ' + path.name + ': ' + result.stderr.strip())
    if path.read_text().strip() != str(value):
        raise RuntimeError('Kernel cgroup write did not read back: ' + path.name)


def _install_limits(binding, unit, limits):
    memory = binding['memory']
    _sudo_write_exact(memory / 'memory.soft_limit_in_bytes', limits['MemoryHigh'], binding, unit)
    _sudo_write_exact(memory / 'memory.memsw.limit_in_bytes', limits['MemoryMax'], binding, unit)
    _sudo_write_exact(memory / 'memory.swappiness', '0', binding, unit)
    # Parent cannot impersonate the worker's Nice=10. The worker performs
    # adapter.verify_limits itself before launching any stage process.
    if (int((memory / 'memory.limit_in_bytes').read_text()) != int(limits['MemoryMax'])
        or (binding['pids'] / 'pids.max').read_text().strip() != limits['TasksMax']):
        raise RuntimeError('Systemd hard memory or PID cap not applied')
    quota = int((binding['cpu'] / 'cpu.cfs_quota_us').read_text())
    period = int((binding['cpu'] / 'cpu.cfs_period_us').read_text())
    if quota <= 0 or quota * 100 != int(limits['CPUQuota'][:-1]) * period:
        raise RuntimeError('Systemd CPU CFS quota not applied')


def _member_memory(binding):
    # A child can fork or exit between reads of separate v1 controllers.
    # Retry that transient race only; persistent disagreement fails closed.
    for attempt in range(3):
        groups = []
        for key in ('memory', 'cpu', 'pids'):
            path = binding[key] / 'cgroup.procs'
            groups.append({int(x) for x in path.read_text().split()})
        if groups[0] and groups[0] == groups[1] == groups[2]:
            break
        if attempt < 2:
            time.sleep(.05)
    else:
        raise RuntimeError('v1 task-unit controller PID sets differ')
    pid_swap = {}
    rss = 0
    for pid in groups[0]:
        proc = Path('/proc') / str(pid)
        try:
            if proc.stat().st_uid != pwd.getpwnam(RUN_USER).pw_uid:
                raise RuntimeError('Foreign process in task unit')
            status = (proc / 'status').read_text().splitlines()
            value = next((int(row.split()[1]) * 1024 for row in status
                          if row.startswith('VmSwap:')), None)
            resident = next((int(row.split()[1]) * 1024 for row in status
                             if row.startswith('VmRSS:')), None)
        except FileNotFoundError as error:
            raise RuntimeError('Task PID vanished during swap readback') from error
        if value is None or resident is None:
            raise RuntimeError('VmSwap or VmRSS unavailable for task PID')
        pid_swap[str(pid)] = value
        rss += resident
    return pid_swap, rss


def _sample(binding):
    resource = cgroup.sample(binding)
    swaps, rss = _member_memory(binding)
    resource['member_pids'] = sorted(int(x) for x in swaps)
    resource['member_vm_swap_bytes'] = swaps
    resource['rss_sum_bytes'] = rss
    resource['swap_zero_observed'] = (resource['memory.swap.current'] == 0
                                       and all(value == 0 for value in swaps.values()))
    return resource


def _wait_gate(proc, directory, unit, limits, *, expected_source, proof_box):
    ready = directory / 'gate_ready.json'
    deadline = time.monotonic() + GATE_TIMEOUT_SEC
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError('Task unit exited before startup gate')
        if ready.exists():
            claim = json.loads(ready.read_text())
            pid = claim.get('pid')
            if (claim.get('unit') != unit or type(pid) is not int
                or claim.get('source_sha256') != expected_source
                or claim.get('adapter_sha256') != sha(ADAPTER)):
                raise RuntimeError('Startup gate identity or source differs')
            binding = _binding_for_pid(pid, unit)
            owner = _attest_unit(pid, unit, binding)
            proof_box['owner'] = owner
            _install_limits(binding, unit, limits)
            release = {'unit': unit, 'pid': pid, 'source_sha256': expected_source,
                       'adapter_sha256': sha(ADAPTER), 'limits': limits}
            atomic_json(directory / 'gate_release.json', release)
            return binding, owner, time.monotonic()
        time.sleep(.1)
    raise RuntimeError('Worker did not reach bounded startup gate')


def _launch(unit, runtime, script, spec, limits, env, *, allow_expected_oom=False):
    command = ['systemd-run', '--system', '--wait', '--pipe', '--quiet', '--unit=' + unit,
               '--description=RealMan bounded rerecord stage', '--working-directory=' + str(runtime),
               '--property=User=' + RUN_USER, '--property=Group=' + RUN_GROUP,
               '--property=MemoryAccounting=yes', '--property=MemoryMax=' + limits['MemoryMax'],
               '--property=CPUAccounting=yes', '--property=CPUQuota=' + limits['CPUQuota'],
               '--property=TasksMax=' + limits['TasksMax'], '--property=Nice=10',
               '--property=KillMode=control-group', '--property=TimeoutStopSec=10s',
               '--property=RuntimeMaxSec=' + limits['RuntimeMaxSec'],
               '--property=OOMPolicy=' + ('continue' if allow_expected_oom else 'stop')]
    command += ['--setenv=' + key + '=' + value for key, value in env.items()]
    command += [CPU_PYTHON, str(script), '--internal-worker' if script == WORKER else '--internal-probe',
                str(spec)]
    return ['sudo', '-n', *command]


def _run_bounded_unit(unit, runtime, script, spec, limits, env, directory,
                      *, stage=None, allow_expected_oom=False):
    _assert_unit_absent(unit)
    directory.mkdir(parents=True, exist_ok=False, mode=0o700)
    source = sha(script)
    atomic_json(directory / 'launch_spec.json', spec)
    start = time.monotonic()
    peak = {'cgroup_bytes': 0, 'rss_sum_bytes': 0,
            'minimum_available_bytes': memory_available()}
    low_memory = low_disk = 0
    reason = None
    binding = None
    final_sample = None
    sample_count = 0
    last_sample_at = None
    maximum_sample_gap = 0.
    tail_sample_gap = 0.
    gate_released_at = None
    unit_exit_at = None
    proof_box = {}
    gpu = gpu_sample() if stage == 'collect' else None
    next_gpu = 0.
    disk_paths = output_filesystems(runtime, spec.get('command', []))
    with (directory / 'stage.log').open('wb') as log:
        proc = subprocess.Popen(_launch(unit, runtime, script, directory / 'launch_spec.json',
                                         limits, env, allow_expected_oom=allow_expected_oom),
                                stdout=log, stderr=subprocess.STDOUT)
        try:
            binding, owner, gate_released_at = _wait_gate(
                proc, directory, unit, limits, expected_source=source, proof_box=proof_box)
            while True:
                if proc.poll() is not None:
                    unit_exit_at = time.monotonic()
                    tail_sample_gap = unit_exit_at - (last_sample_at or gate_released_at)
                    maximum_sample_gap = max(maximum_sample_gap, tail_sample_gap)
                    if tail_sample_gap > 6.:
                        reason = 'Resource monitoring tail gap exceeded 6 seconds'
                    break
                resource = _sample(binding)
                final_sample = resource
                now = time.monotonic()
                elapsed = now - start
                maximum_sample_gap = max(maximum_sample_gap,
                                         now - (last_sample_at or gate_released_at))
                if maximum_sample_gap > 6.:
                    reason = 'Resource monitoring gap exceeded 6 seconds'
                last_sample_at = now
                sample_count += 1
                available = memory_available()
                free = disk_free(disk_paths)
                if stage == 'collect' and elapsed >= next_gpu:
                    gpu = gpu_sample()
                    next_gpu = elapsed + 10.
                sample = {'elapsed_seconds': elapsed, 'time_unix': time.time(),
                          'host_available_bytes': available, 'cgroup': resource,
                          'unit_state': unit_info(unit), 'disk_free_bytes': free, 'gpu': gpu}
                append_jsonl(directory / 'resources.jsonl', sample)
                peak['minimum_available_bytes'] = min(peak['minimum_available_bytes'], available)
                peak['cgroup_bytes'] = max(peak['cgroup_bytes'], resource['memory.peak'],
                                           resource['memory.current'])
                peak['rss_sum_bytes'] = max(peak['rss_sum_bytes'], resource['rss_sum_bytes'])
                low_memory = low_memory + 1 if available < FLOOR_AVAILABLE else 0
                low_disk = low_disk + 1 if min(free.values()) < FLOOR_DISK_FREE else 0
                if not resource['swap_zero_observed']:
                    reason = 'Task swap observed in cgroup or VmSwap'
                if not allow_expected_oom and any(resource['memory.events'].get(key, -1) != 0
                                                  for key in ('max', 'oom', 'oom_kill', 'oom_group_kill')):
                    reason = 'Memory hard cap or OOM event observed'
                if low_memory >= 3:
                    reason = 'Host MemAvailable below 6 GiB for three samples'
                if low_disk >= 3:
                    reason = 'Task destination disk below 16 GiB for three samples'
                if stage == 'collect' and (not gpu['healthy'] or gpu.get('free_bytes', 0) < GPU_MIN_FREE):
                    reason = 'GPU readback unhealthy or free memory below 12 GiB'
                if elapsed > int(limits['RuntimeMaxSec']) + 15:
                    reason = 'Guard wall timeout beyond systemd RuntimeMaxSec'
                if reason:
                    stop_exact_unit(unit, owner)
                    break
                time.sleep(SAMPLE_SEC if stage else .1)
            try:
                code = proc.wait(timeout=20)
                if unit_exit_at is None:
                    unit_exit_at = time.monotonic()
            except subprocess.TimeoutExpired:
                stop_exact_unit(unit, owner)
                code = proc.wait(timeout=20)
                unit_exit_at = time.monotonic()
                reason = reason or 'Launcher did not terminate after unit exit'
        except BaseException as error:
            try:
                owner = proof_box.get('owner')
                if owner is None:
                    try:
                        owner = _attest_unready_unit(unit, script,
                                  directory / 'launch_spec.json', source)
                        proof_box['owner'] = owner
                    except (OSError, RuntimeError, ValueError, KeyError) as proof_error:
                        try:
                            state = unit_info(unit)
                        except (OSError, RuntimeError, subprocess.TimeoutExpired) as status_error:
                            state = {'status_error': str(status_error)}
                        atomic_json(directory / 'unattributed_startup_failure.json', {
                            'unit': unit, 'startup_error': str(error),
                            'ownership_proof_error': str(proof_error),
                            'unit_state': state,
                            'runtime_max_sec': limits['RuntimeMaxSec'],
                            'action': 'no_stop_without_ownership_proof'})
                if owner is not None:
                    stop_exact_unit(unit, owner)
            finally:
                if proof_box.get('owner') is not None:
                    try:
                        proc.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        pass
            raise
    return {'launcher_returncode': code, 'unit_terminal': unit_info(unit),
            'wall_seconds': time.monotonic() - start, 'monitor_stop_reason': reason,
            'unit_owner_attestation': owner,
            'peak': peak, 'last_resource_sample': final_sample,
            'swap_observation_count': sample_count,
            'maximum_sample_gap_seconds': maximum_sample_gap,
            'tail_sample_gap_seconds': tail_sample_gap,
            'gate_release_elapsed_seconds': gate_released_at - start,
            'unit_exit_elapsed_seconds': unit_exit_at - start,
            'resource_samples_sha256': sha(directory / 'resources.jsonl') if sample_count else None,
            'swap_zero_observed': final_sample is not None
                                  and final_sample['swap_zero_observed']
                                  and reason != 'Task swap observed in cgroup or VmSwap'}


def _verify_source(binding, script, stage):
    if (sha(script) != binding['entry_sha256'] or sha(WORKER) != binding['guard_sha256']
        or sha(ADAPTER) != binding['cgroup_adapter_sha256']
        or sha(RELEASE) != binding['release_contract_sha256']):
        raise RuntimeError('Stage or guard source changed')
    if stage == 'collect' and sha(DIAGNOSTICS) != binding['diagnostics_sha256']:
        raise RuntimeError('Diagnostic producer changed')


def internal_worker(spec_path):
    path = Path(spec_path).resolve(strict=True)
    path.relative_to(RUNS.resolve(strict=True))
    if path.name != 'launch_spec.json':
        raise ValueError('Unregistered worker spec')
    spec = json.loads(path.read_text())
    unit = spec['unit']
    if not unit.startswith(UNIT_PREFIX + '-') or not unit.endswith('.service'):
        raise RuntimeError('Unexpected unit in launch spec')
    runtime, script = validate_command(spec['stage'], spec['command'])
    _verify_source(spec['source_binding'], script, spec['stage'])
    binding = cgroup.discover_self(unit)
    if binding['mode'] != 'v1' or os.getuid() != pwd.getpwnam(RUN_USER).pw_uid:
        raise RuntimeError('Worker is outside dedicated v1 system unit or wrong user')
    run_dir = path.parent
    atomic_json(run_dir / 'gate_ready.json', {'unit': unit, 'pid': os.getpid(),
                'source_sha256': sha(WORKER), 'adapter_sha256': sha(ADAPTER)})
    deadline = time.monotonic() + GATE_TIMEOUT_SEC
    gate = run_dir / 'gate_release.json'
    while not gate.exists() and time.monotonic() < deadline:
        time.sleep(.1)
    if not gate.exists():
        raise RuntimeError('Parent did not release protected startup gate')
    release = json.loads(gate.read_text())
    if (release.get('unit') != unit or release.get('pid') != os.getpid()
        or release.get('source_sha256') != sha(WORKER)
        or release.get('adapter_sha256') != sha(ADAPTER)
        or release.get('limits') != spec['limits']):
        raise RuntimeError('Startup gate release differs')
    limits = spec['limits']
    readback = cgroup.verify_limits(binding, high=int(limits['MemoryHigh']),
              maximum=int(limits['MemoryMax']), cpu_percent=int(limits['CPUQuota'][:-1]),
              pids=int(limits['TasksMax']), nice=int(limits['Nice']))
    readback.update({'unit': unit, 'pid': os.getpid(), 'boot_id': _boot_id()})
    atomic_json(run_dir / 'cgroup_readback.json', readback)
    _verify_source(spec['source_binding'], script, spec['stage'])
    child_env = dict(os.environ)
    if spec['stage'] == 'collect':
        child_env['PYTHONPATH'] = str(DIAGNOSTICS.parent)
        child_env['REALMAN_STAGE_DIAGNOSTIC'] = '1'
    proc = subprocess.Popen(spec['command'], cwd=runtime, env=child_env)

    def forward(sig, _frame):
        if proc.poll() is None:
            proc.send_signal(sig)
    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)
    code = proc.wait()
    resource = cgroup.sample(binding)
    clean = True
    reason = None
    try:
        cgroup.require_clean_terminal(resource)
    except cgroup.CgroupProofError as error:
        clean, reason = False, str(error)
    unchanged = (sha(script) == spec['source_binding']['entry_sha256']
                 and sha(WORKER) == spec['source_binding']['guard_sha256']
                 and sha(ADAPTER) == spec['source_binding']['cgroup_adapter_sha256']
                 and sha(RELEASE) == spec['source_binding']['release_contract_sha256']
                 and (spec['stage'] != 'collect' or sha(DIAGNOSTICS) ==
                      spec['source_binding']['diagnostics_sha256']))
    atomic_json(run_dir / 'worker_terminal.json', {'stage_returncode': code,
                'wall_time_unix': time.time(), 'resource': resource,
                'resource_clean': clean, 'resource_failure': reason,
                'source_unchanged': unchanged})
    return code if code else (0 if clean and unchanged else 1)


def _run_probe_once(directory, *, negative):
    unit = _new_unit('probe-negative' if negative else 'probe-positive')
    limits = {'MemoryHigh': str(24 * 1024**2), 'MemoryMax': str(32 * 1024**2),
              'CPUQuota': '100%', 'TasksMax': '32', 'Nice': '10', 'RuntimeMaxSec': '30'}
    spec = {'unit': unit, 'negative': negative, 'limits': limits,
            'source_binding': {'guard_sha256': sha(WORKER),
                               'probe_sha256': sha(PROBE),
                               'cgroup_adapter_sha256': sha(ADAPTER)}}
    result = _run_bounded_unit(unit, E, PROBE, spec, limits,
              {'PATH': '/usr/bin:/bin', 'CUDA_VISIBLE_DEVICES': '', 'OPENBLAS_NUM_THREADS': '1'},
              directory, allow_expected_oom=negative)
    terminal_file = directory / 'probe_terminal.json'
    terminal = json.loads(terminal_file.read_text()) if terminal_file.exists() else None
    readback_file = directory / 'cgroup_readback.json'
    readback = json.loads(readback_file.read_text()) if readback_file.exists() else None
    result.update({'unit': unit, 'negative': negative, 'probe_terminal': terminal,
                   'cgroup_readback': readback,
                   'passed': result['launcher_returncode'] == 0
                             and result['monitor_stop_reason'] is None
                             and result['swap_zero_observed']
                             and terminal is not None and terminal.get('passed') is True
                             and readback is not None and readback.get('limits_verified') is True})
    return result


def run_probe():
    """Positive kernel readback plus child OOM; no simulator or GPU context."""
    _identity()
    RUNS.mkdir(exist_ok=True)
    directory = RUNS / ('cpu_probe_' + str(time.time_ns()))
    directory.mkdir(mode=0o700)
    positive = _run_probe_once(directory / 'positive', negative=False)
    negative = _run_probe_once(directory / 'negative', negative=True) if positive['passed'] else None
    receipt = {'schema': 'realman_tiny_cgroup_probe_v1', 'scope': 'CPU-only 32 MiB task probe',
               'mode': 'v1', 'results': [x for x in (positive, negative) if x],
               'passed': positive['passed'] and negative is not None and negative['passed'],
               'guard_sha256': sha(WORKER), 'probe_sha256': sha(PROBE),
               'adapter_sha256': sha(ADAPTER), 'boot_id': _boot_id(),
               'time_unix': time.time()}
    path = directory / 'receipt.json'
    atomic_json(path, receipt)
    return {'path': str(path), 'sha256': sha(path), 'passed': receipt['passed'],
            'guard_sha256': receipt['guard_sha256'],
            'adapter_sha256': receipt['adapter_sha256'],
            'probe_sha256': receipt['probe_sha256']}


def execute(stage, command, *, run=False, timeout_sec=None, receipt_dir=None, cleanup_orphan=False):
    _identity()
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
        except BlockingIOError as error:
            raise RuntimeError('Another task stage holds guard lock') from error
        if cleanup_orphan:
            raise RuntimeError('Unique system units never permit broad orphan cleanup')
        existing = _known_task_processes()
        if existing:
            raise RuntimeError('Existing unguarded task process: ' + json.dumps(existing))
        available = memory_available()
        if available < MIN_AVAILABLE:
            raise RuntimeError('Less than 18 GiB host memory available')
        disk_paths = output_filesystems(runtime, command)
        free = disk_free(disk_paths)
        if min(free.values()) < MIN_DISK_FREE:
            raise RuntimeError('Less than 32 GiB free at destination')
        gpu = gpu_sample() if stage == 'collect' else None
        if gpu and (not gpu['healthy'] or gpu['free_bytes'] < GPU_MIN_FREE):
            raise RuntimeError('GPU readback unhealthy or less than 12 GiB free')
        unit = _new_unit(stage)
        receipt = {'schema': 'realman_guarded_stage_v1', 'stage': stage, 'check_only': not run,
                   'command': command, 'runtime': str(runtime), 'unit': unit,
                   'limits': limits, 'environment_overrides': env_for(stage, runtime),
                   'sample_seconds': SAMPLE_SEC, 'gpu_hard_memory_cap': False,
                   'auto_retry': False, 'available_before_bytes': available,
                   'gpu_before': gpu, 'disk_before_bytes': free,
                   'start_minimum_available_bytes': MIN_AVAILABLE,
                   'stop_available_floor_bytes': FLOOR_AVAILABLE, 'low_samples_required': 3,
                   'disk_start_minimum_bytes': MIN_DISK_FREE,
                   'disk_stop_floor_bytes': FLOOR_DISK_FREE,
                   'source_binding': {'entry_sha256': sha(script), 'guard_sha256': sha(WORKER),
                                      'cgroup_adapter_sha256': sha(ADAPTER),
                                      'release_contract_sha256': sha(RELEASE),
                                      'diagnostics_sha256': sha(DIAGNOSTICS) if stage == 'collect' else None},
                   'cgroup_mode_required': 'v1', 'swap_guarantee': 'zero_observed_not_hard_disabled'}
        if not run:
            return receipt, 0
        probe = run_probe()
        receipt['preflight_probe'] = probe
        if not probe['passed'] or sha(probe['path']) != probe['sha256']:
            raise RuntimeError('CPU-only positive/negative kernel probe did not pass')
        receipt['source_binding']['cgroup_adapter_sha256'] = sha(ADAPTER)
        if (probe['guard_sha256'] != receipt['source_binding']['guard_sha256']
            or probe['adapter_sha256'] != receipt['source_binding']['cgroup_adapter_sha256']
            or probe['probe_sha256'] != sha(PROBE)):
            raise RuntimeError('Probe source differs before stage launch')
        # The CPU-only probe can take time; none of the original host, disk,
        # GPU or task-process preconditions may be carried across it blindly.
        if _known_task_processes():
            raise RuntimeError('Task process appeared after CPU probe')
        available_after_probe = memory_available()
        free_after_probe = disk_free(disk_paths)
        gpu_after_probe = gpu_sample() if stage == 'collect' else None
        receipt.update({'available_after_probe_bytes': available_after_probe,
                        'disk_after_probe_bytes': free_after_probe,
                        'gpu_after_probe': gpu_after_probe})
        if available_after_probe < MIN_AVAILABLE:
            raise RuntimeError('Host memory fell below 18 GiB after CPU probe')
        if min(free_after_probe.values()) < MIN_DISK_FREE:
            raise RuntimeError('Task destination fell below 32 GiB after CPU probe')
        if gpu_after_probe and (not gpu_after_probe['healthy']
                                or gpu_after_probe.get('free_bytes', 0) < GPU_MIN_FREE):
            raise RuntimeError('GPU 0 fell below 12 GiB or became unreadable after CPU probe')
        receipt['boot_id'] = _boot_id()
        receipt['probe_binding_verified'] = True
        spec = dict(receipt)
        result = _run_bounded_unit(unit, runtime, WORKER, spec, limits,
                                   env_for(stage, runtime), directory, stage=stage)
        terminal_file = directory / 'worker_terminal.json'
        terminal = json.loads(terminal_file.read_text()) if terminal_file.exists() else None
        readback_file = directory / 'cgroup_readback.json'
        readback = json.loads(readback_file.read_text()) if readback_file.exists() else None
        resource = terminal.get('resource') if terminal else None
        receipt.update(result)
        receipt.update({'available_after_bytes': memory_available(),
                        'disk_after_bytes': disk_free(disk_paths),
                        'worker_terminal': terminal, 'cgroup_readback_verified':
                        readback is not None and readback.get('limits_verified') is True,
                        'cgroup_readback': readback, 'receipt_directory': str(directory)})
        clean = False
        if resource is not None:
            try:
                cgroup.verify_recorded(readback, resource, unit=unit,
                    high=int(limits['MemoryHigh']), maximum=int(limits['MemoryMax']),
                    cpu_percent=int(limits['CPUQuota'][:-1]), pids=int(limits['TasksMax']))
                clean = True
            except (cgroup.CgroupProofError, TypeError) as error:
                receipt['resource_rejection'] = str(error)
        receipt['swap_zero_observed'] = result['swap_zero_observed'] and clean
        receipt['passed'] = (result['launcher_returncode'] == 0
                             and result['monitor_stop_reason'] is None
                             and result['swap_observation_count'] >= 1
                             and terminal is not None and terminal['stage_returncode'] == 0
                             and terminal['source_unchanged'] and terminal['resource_clean']
                             and receipt['cgroup_readback_verified'] and clean
                             and receipt['swap_zero_observed'] and receipt['probe_binding_verified'])
        atomic_json(directory / 'receipt.json', receipt)
        return receipt, 0 if receipt['passed'] else (result['launcher_returncode'] or 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=STAGES)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--timeout-sec', type=float)
    parser.add_argument('--receipt-dir')
    parser.add_argument('--cleanup-orphan', action='store_true')
    parser.add_argument('--internal-worker', help=argparse.SUPPRESS)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--probe-cgroup', action='store_true')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.internal_worker:
        return internal_worker(args.internal_worker)
    if args.self_test:
        tests = E / 'test_guarded_stage.py'
        code = subprocess.call([CPU_PYTHON, str(tests)]) if tests.exists() else 0
        if code or not args.probe_cgroup:
            return code
        with LOCK.open('a') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            if _known_task_processes():
                raise RuntimeError('Another task process exists; no concurrent cgroup probe')
            probe = run_probe()
        print(json.dumps(probe, ensure_ascii=False, indent=2))
        return 0 if probe['passed'] else 1
    if args.probe_cgroup:
        parser.error('--probe-cgroup requires --self-test')
    if not args.stage:
        parser.error('--stage is required')
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    receipt, code = execute(args.stage, command, run=args.run,
                            timeout_sec=args.timeout_sec, receipt_dir=args.receipt_dir,
                            cleanup_orphan=args.cleanup_orphan)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return code


if __name__ == '__main__':
    sys.exit(main())
