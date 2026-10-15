"""Independent release check for the 161 dedicated-v1 stage evidence.

The v1 soft memory limit is best effort.  ``swap_zero_observed`` describes
sampling evidence, not a kernel guarantee that swap was disabled.
"""

from pathlib import Path
import hashlib
import json
import math

import hybrid_cgroup_adapter as cgroup


GIB = 1024 ** 3
STAGES = {'collect': (10, 12, 800), 'export': (3, 4, 200),
          'quality': (2, 3, 200), 'render': (2, 3, 200)}
UNIT_PREFIX = 'realman-rerecord-20261014-v6-hybrid'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 64 * 1024**2,
            'Bounded proof JSON missing, linked or oversized: ' + str(path))
    return json.loads(path.read_text())


def validate_probe(receipt, preflight):
    """Bind a successful separate positive/negative probe to frozen sources."""
    root = Path(preflight).resolve()
    row = receipt.get('preflight_probe', {})
    path = Path(row.get('path', '/'))
    require(path.is_absolute() and path == path.resolve() and path.name == 'receipt.json'
            and path.parent.parent == root / 'resource_runs'
            and path.parent.name.startswith('cpu_probe_'),
            'Stage has no task-local CPU-only probe receipt')
    proof = read_json(path)
    require(row.get('sha256') == sha(path) and row.get('passed') is True
            and proof.get('passed') is True,
            'Probe receipt changed or did not pass')
    for key, name in (('guard_sha256', 'run_guarded_stage.py'),
                      ('adapter_sha256', 'hybrid_cgroup_adapter.py'),
                      ('probe_sha256', 'guarded_stage_cpu_probe.py')):
        require(row.get(key) == sha(root / name) and proof.get(key) == row.get(key),
                'CPU-only OOM probe source changed: ' + key)
    results = proof.get('results', [])
    require(len(results) == 2 and all(item.get('passed') is True for item in results)
            and {item.get('negative') for item in results} == {False, True},
            'Positive and negative OOM probes were not both witnessed')
    units = set()
    for item in results:
        negative = item['negative']
        unit = item.get('unit', '')
        require(unit.startswith(UNIT_PREFIX + '-probe-')
                and unit.endswith('.service') and unit not in units,
                'Probe did not use two distinct scoped system units')
        units.add(unit)
        phase = path.parent / ('negative' if negative else 'positive')
        readback = read_json(phase / 'cgroup_readback.json')
        terminal = read_json(phase / 'probe_terminal.json')
        require(item.get('cgroup_readback') == readback
                and item.get('probe_terminal') == terminal
                and item.get('launcher_returncode') == 0
                and item.get('monitor_stop_reason') is None
                and item.get('swap_zero_observed') is True
                and terminal.get('passed') is True
                and terminal.get('source_unchanged') is True,
                'Probe kernel/terminal files differ from reported successful run')
        cgroup.verify_recorded_limits(readback, unit=unit, high=24 * 1024**2,
                                     maximum=32 * 1024**2, cpu_percent=100, pids=32)
        resource = terminal.get('resource', {})
        require(resource.get('mode') == 'v1'
                and resource.get('memory.swap.current') == 0,
                'Probe was not a zero-swap dedicated v1 unit')
        if negative:
            events = resource.get('memory.events', {})
            hard_fail, memsw_fail = events.get('max'), events.get('memsw_max')
            oom_kill = events.get('oom_kill')
            require(terminal.get('child_returncode') == -9
                    and type(hard_fail) is int and hard_fail >= 0
                    and type(memsw_fail) is int and memsw_fail >= 0
                    and (hard_fail > 0 or memsw_fail > 0)
                    and type(oom_kill) is int and oom_kill > 0
                    and resource.get('memory.oom_control', {}).get('oom_kill') == oom_kill,
                    'Negative probe did not prove a RAM or memory+swap cap OOM kill')
        else:
            require(terminal.get('child_returncode') is None,
                    'Positive probe unexpectedly spawned an OOM child')
            events = resource.get('memory.events', {})
            require(events.get('max') == 0 and events.get('memsw_max') == 0,
                    'Positive probe touched a RAM or memory+swap cap')
            cgroup.require_clean_terminal(resource)
    return {'path': str(path), 'sha256': sha(path), 'passed': True}


def validate_stage(receipt, receipt_path, stage, preflight):
    """Read back a real phase's bounded kernel, exit and probe evidence."""
    require(stage in STAGES and receipt.get('stage') == stage,
            'Resource stage identity differs')
    preflight = Path(preflight).resolve()
    receipt_path = Path(receipt_path).resolve()
    require(receipt_path.parent.parent == preflight / 'resource_runs'
            and receipt_path.name == 'receipt.json',
            'Resource receipt is outside this task')
    binding = receipt.get('source_binding', {})
    require(binding.get('guard_sha256') == sha(preflight / 'run_guarded_stage.py')
            and binding.get('cgroup_adapter_sha256') == sha(preflight / 'hybrid_cgroup_adapter.py')
            and binding.get('release_contract_sha256') == sha(preflight / 'hybrid_release_contract.py'),
            'Resource guard, cgroup adapter or release contract changed after stage execution')
    probe = validate_probe(receipt, preflight)
    unit = receipt.get('unit', '')
    require(unit.startswith(UNIT_PREFIX + '-')
            and unit.endswith('.service') and '/' not in unit,
            'Resource unit is not the scoped unique system unit')
    high, maximum, cpu = STAGES[stage]
    readback = read_json(receipt_path.parent / 'cgroup_readback.json')
    worker = read_json(receipt_path.parent / 'worker_terminal.json')
    require(worker == receipt.get('worker_terminal') and worker.get('stage_returncode') == 0
            and worker.get('source_unchanged') is True,
            'Worker terminal file does not prove normal frozen-source exit')
    resource = worker.get('resource', {})
    cgroup.verify_recorded(readback, resource, unit=unit,
                           high=high * GIB, maximum=maximum * GIB,
                           cpu_percent=cpu, pids=128)
    samples = receipt_path.parent / 'resources.jsonl'
    require(samples.is_file() and not samples.is_symlink()
            and samples.stat().st_size <= 64 * 1024**2
            and receipt.get('resource_samples_sha256') == sha(samples),
            'Resource observation stream missing, linked, oversized or changed')
    def finite_seconds(value):
        return type(value) in (int, float) and math.isfinite(value)

    gate = receipt.get('gate_release_elapsed_seconds')
    unit_exit = receipt.get('unit_exit_elapsed_seconds')
    wall = receipt.get('wall_seconds')
    require(all(finite_seconds(value) for value in (gate, unit_exit, wall))
            and 0 <= gate <= unit_exit <= wall,
            'Stage gate, unit exit and wall times are absent or out of order')
    count = 0
    previous_elapsed = gate
    largest_observed_gap = 0.
    with samples.open() as stream:
        for line in stream:
            require(len(line) < 1024**2, 'Oversized resource observation')
            row = json.loads(line)
            observed = row.get('cgroup', {})
            members = observed.get('member_vm_swap_bytes', {})
            require(observed.get('mode') == 'v1'
                    and observed.get('swap_zero_observed') is True
                    and observed.get('memory.swap.current') == 0
                    and isinstance(members, dict) and members
                    and all(type(value) is int and value == 0 for value in members.values())
                    and row.get('unit_state', {}).get('ControlGroup') == '/system.slice/' + unit,
                    'A sampled unit membership or swap observation is invalid')
            cgroup.require_clean_terminal(observed)
            elapsed = row.get('elapsed_seconds')
            require(finite_seconds(elapsed) and previous_elapsed <= elapsed <= unit_exit,
                    'Resource observation is outside the live stage interval or out of order')
            largest_observed_gap = max(largest_observed_gap, elapsed - previous_elapsed)
            previous_elapsed = elapsed
            count += 1
    tail_gap = unit_exit - previous_elapsed
    largest_observed_gap = max(largest_observed_gap, tail_gap)
    reported_gap = receipt.get('maximum_sample_gap_seconds')
    reported_tail = receipt.get('tail_sample_gap_seconds')
    require(count >= 1 and count == receipt.get('swap_observation_count')
            and finite_seconds(reported_gap) and finite_seconds(reported_tail)
            and abs(reported_gap - largest_observed_gap) <= 1e-6
            and abs(reported_tail - tail_gap) <= 1e-6
            and largest_observed_gap <= 6.,
            'Resource monitor omitted samples, changed timing evidence, or exceeded its bounded interval')
    require(receipt.get('swap_zero_observed') is True
            and resource.get('memory.swap.current') == 0,
            'A non-zero or unknown swap observation invalidates stage')
    require(receipt.get('monitor_stop_reason') is None
            and receipt.get('launcher_returncode') == 0
            and receipt.get('passed') is True,
            'Stage monitor or launcher failed')
    return {'passed': True, 'cgroup_mode': 'v1', 'unit': unit,
            'readback_sha256': sha(receipt_path.parent / 'cgroup_readback.json'),
            'probe': probe, 'swap_zero_observed': True}
