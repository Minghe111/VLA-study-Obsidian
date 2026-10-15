"""Server-only CPU probe for a dedicated hybrid-cgroup system service.

Rendered into evidence/preflight/guarded_stage_cpu_probe.py. The negative
probe allocates only inside its 32 MiB task unit and never opens the simulator.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import hybrid_cgroup_adapter as cgroup

E = Path(__file__).resolve().parent
RUNS = E / 'resource_runs'
SELF = E / 'guarded_stage_cpu_probe.py'
GUARD = E / 'run_guarded_stage.py'
ADAPTER = E / 'hybrid_cgroup_adapter.py'
UNIT_PREFIX = '@UNIT_PREFIX@'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    temp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    temp.write_text(json.dumps(obj, indent=2) + '\n')
    os.replace(temp, path)


def allocate():
    # Force actual page touches, so kernel—not Python's allocator—tests the cap.
    blocks = []
    for _ in range(64):
        block = bytearray(1024 * 1024)
        for offset in range(0, len(block), 4096):
            block[offset] = 1
        if sum(block[::4096]) != 256:
            raise RuntimeError('Probe allocation pages not resident and readable')
        blocks.append(block)
    raise RuntimeError('32 MiB unit failed to kill a 64 MiB allocation')


def negative_oom_proof(resource, child_returncode):
    control = resource.get('memory.oom_control', {})
    oom_kill = control.get('oom_kill')
    if type(oom_kill) is not int:
        return False, 'Kernel memory.oom_control omitted oom_kill counter; negative probe unprovable'
    events = resource.get('memory.events', {})
    hard_fail = events.get('max')
    memsw_fail = events.get('memsw_max')
    if (type(hard_fail) is not int or hard_fail < 0
        or type(memsw_fail) is not int or memsw_fail < 0
        or events.get('oom_kill') != oom_kill):
        return False, 'Kernel RAM/memory+swap limit counters are missing or inconsistent'
    # On cgroup-v1 a memsw limit can kill the child with memory.failcnt=0.
    # Either kernel limit counter is evidence only when the OOM kill also rose.
    proved = (child_returncode == -9
              and (hard_fail > 0 or memsw_fail > 0)
              and oom_kill > 0 and control.get('under_oom') == 0)
    return proved, None if proved else 'Child OOM kill or kernel counter did not prove memory cap'


def run(spec_path):
    path = Path(spec_path).resolve(strict=True)
    path.relative_to(RUNS.resolve(strict=True))
    if path.name != 'launch_spec.json':
        raise ValueError('Unregistered probe spec')
    spec = json.loads(path.read_text())
    unit, limits, binding_sha = spec['unit'], spec['limits'], spec['source_binding']
    if not unit.startswith(UNIT_PREFIX + '-probe-') or not unit.endswith('.service'):
        raise RuntimeError('Probe unit identity differs')
    if (sha(SELF) != binding_sha['probe_sha256'] or sha(GUARD) != binding_sha['guard_sha256']
        or sha(ADAPTER) != binding_sha['cgroup_adapter_sha256']):
        raise RuntimeError('Probe source changed before gate')
    binding = cgroup.discover_self(unit)
    if binding['mode'] != 'v1':
        raise RuntimeError('Probe must run in own hybrid v1 unit')
    directory = path.parent
    atomic_json(directory / 'gate_ready.json', {'unit': unit, 'pid': os.getpid(),
                'source_sha256': sha(SELF), 'adapter_sha256': sha(ADAPTER)})
    gate_path = directory / 'gate_release.json'
    deadline = time.monotonic() + 45.
    while not gate_path.exists() and time.monotonic() < deadline:
        time.sleep(.1)
    if not gate_path.exists():
        raise RuntimeError('Probe gate not released')
    gate = json.loads(gate_path.read_text())
    if (gate.get('unit') != unit or gate.get('pid') != os.getpid()
        or gate.get('source_sha256') != sha(SELF)
        or gate.get('adapter_sha256') != sha(ADAPTER)
        or gate.get('limits') != limits):
        raise RuntimeError('Probe gate release identity differs')
    readback = cgroup.verify_limits(binding, high=int(limits['MemoryHigh']),
        maximum=int(limits['MemoryMax']), cpu_percent=int(limits['CPUQuota'][:-1]),
        pids=int(limits['TasksMax']), nice=int(limits['Nice']))
    readback.update({'unit': unit, 'pid': os.getpid(),
                     'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()})
    atomic_json(directory / 'cgroup_readback.json', readback)
    negative = spec['negative']
    child_returncode = None
    if negative:
        child = subprocess.Popen([sys.executable, str(SELF), '--allocate'])
        child_returncode = child.wait(timeout=20)
    # Let the external monitor observe the live unit, including zero swap.
    time.sleep(.5)
    resource = cgroup.sample(binding)
    zero_swap = resource.get('memory.swap.current') == 0
    unchanged = (sha(SELF) == binding_sha['probe_sha256']
                 and sha(GUARD) == binding_sha['guard_sha256']
                 and sha(ADAPTER) == binding_sha['cgroup_adapter_sha256'])
    rejection = None
    if negative:
        passed_probe, rejection = negative_oom_proof(resource, child_returncode)
    else:
        try:
            cgroup.require_clean_terminal(resource)
            passed_probe = True
        except cgroup.CgroupProofError as error:
            passed_probe = False
            rejection = str(error)
    terminal = {'negative': negative, 'child_returncode': child_returncode,
                'resource': resource, 'swap_zero_observed': zero_swap,
                'source_unchanged': unchanged,
                'rejection': rejection,
                'passed': passed_probe and zero_swap and unchanged}
    atomic_json(directory / 'probe_terminal.json', terminal)
    return 0 if terminal['passed'] else 1


if __name__ == '__main__':
    if sys.argv[1:] == ['--allocate']:
        sys.exit(allocate())
    if len(sys.argv) == 3 and sys.argv[1] == '--internal-probe':
        sys.exit(run(sys.argv[2]))
    raise SystemExit('Only --internal-probe is allowed')
