"""Independent release check for the 161 dedicated-v1 stage evidence.

The v1 soft memory limit is best effort.  ``swap_zero_observed`` describes
sampling evidence, not a kernel guarantee that swap was disabled.
"""

from pathlib import Path
import hashlib
import json
import math
import re
import subprocess

import hybrid_cgroup_adapter as cgroup


GIB = 1024 ** 3
STAGES = {'collect': (10, 12, 800), 'export': (3, 4, 200),
          'quality': (2, 3, 200), 'render': (2, 3, 200),
          'renderer_smoke': (10, 12, 800)}
GPU_STAGES = {'collect', 'renderer_smoke'}
GPUCOMP = Path('/lib/x86_64-linux-gnu/libnvidia-gpucomp.so.570.133.07')
GPUCOMP_REAL = Path('/usr/lib/x86_64-linux-gnu/libnvidia-gpucomp.so.570.133.07')
GPUCOMP_SHA256 = '66178cf6eef724402b77193665ab1defede1b36a6333d81423b48ee32f0742dc'
NVIDIA_ICD = Path('/usr/share/vulkan/icd.d/nvidia_icd.json')
NVIDIA_ICD_SHA256 = '46cee75b42b43649839bc7f29820d36ac6e85782b792a97d5b91d9adbca8ca09'
NVIDIA_DRIVER_VERSION = '570.133.07'
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


def driver_version_readback():
    kernel_line = Path('/proc/driver/nvidia/version').read_text().splitlines()[0]
    kernel_versions = re.findall(r'\b\d{3}\.\d{1,3}\.\d{1,3}\b', kernel_line)
    result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version',
                             '--format=csv,noheader'], capture_output=True, text=True, timeout=5)
    versions = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    require(kernel_versions == [NVIDIA_DRIVER_VERSION] and result.returncode == 0
            and versions == [NVIDIA_DRIVER_VERSION] * 8,
            'Running NVIDIA kernel/user driver changed from pinned 570.133.07')
    return {'kernel_module_version': NVIDIA_DRIVER_VERSION,
            'nvidia_smi_versions': versions}


def validate_eight_gpu_sample(sample):
    """The GPU floor covers every enumerated card, not just CUDA index zero."""
    require(isinstance(sample, dict) and sample.get('healthy') is True
            and sample.get('all_eight_gpu_floor_required') is True
            and sample.get('driver_version') == NVIDIA_DRIVER_VERSION,
            'Eight-GPU resource sample is absent or unhealthy')
    cards = sample.get('cards')
    require(isinstance(cards, list) and len(cards) == 8
            and [card.get('index') for card in cards] == list(range(8))
            and len({card.get('uuid') for card in cards}) == 8
            and len({card.get('pci_bus_id') for card in cards}) == 8,
            'Eight unique ordered GPU identities were not sampled')
    for card in cards:
        require(card.get('driver_version') == NVIDIA_DRIVER_VERSION
                and isinstance(card.get('uuid'), str) and card['uuid']
                and isinstance(card.get('pci_bus_id'), str) and card['pci_bus_id']
                and type(card.get('total_bytes')) is int and card['total_bytes'] > 0
                and type(card.get('free_bytes')) is int
                and 12 * GIB <= card['free_bytes'] <= card['total_bytes']
                and type(card.get('used_bytes')) is int and card['used_bytes'] >= 0,
                'An NVIDIA GPU driver or 12 GiB free-memory floor failed')
    minimum = min(cards, key=lambda card: card['free_bytes'])
    require(sample.get('free_bytes') == minimum['free_bytes']
            and sample.get('minimum_free_index') == minimum['index']
            and sample.get('index') == 0 and sample.get('uuid') == cards[0]['uuid'],
            'GPU aggregate does not match the eight raw cards')
    return True


def latest_renderer_attempt(preflight, mode):
    """The latest same-boot attempt wins even if it failed before a receipt."""
    preflight = Path(preflight).resolve()
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    require(mode in ('constructor', 'offscreen'), 'Unknown renderer attempt mode')
    runs = preflight / 'resource_runs'
    require(runs.is_dir() and not runs.is_symlink(), 'Renderer attempt directory absent or linked')
    prefix = 'renderer_smoke_attempt_' + mode + '_' + boot + '_'
    markers = sorted(runs.glob(prefix + '*.json'))
    require(markers and len(markers) <= 1000,
            'Latest same-boot renderer attempt absent or unbounded')
    path = markers[-1]
    require(not path.is_symlink() and path.is_file() and path.stat().st_size <= 64 * 1024
            and re.fullmatch(re.escape(prefix) + r'\d{20}_[0-9a-f]{32}\.json', path.name),
            'Latest renderer attempt marker is invalid')
    marker = read_json(path)
    suffix = [] if mode == 'constructor' else ['--offscreen-example']
    command = marker.get('command', [])
    directory = Path(marker.get('receipt_dir', '/'))
    dir_prefix = 'renderer_smoke_' + ('ctor_' if mode == 'constructor' else 'offscreen_')
    require(marker.get('schema') == 'realman_renderer_smoke_attempt_v1'
            and marker.get('mode') == mode and marker.get('boot_id') == boot
            and marker.get('bundle_manifest_sha256') ==
                sha(preflight.parents[1] / 'bundle_manifest.json')
            and marker.get('guard_sha256') == sha(preflight / 'run_guarded_stage.py')
            and marker.get('entry_sha256') == sha(preflight / 'renderer_only_smoke.py')
            and len(command) == 2 + len(suffix)
            and command[1:] == [str(preflight / 'renderer_only_smoke.py'), *suffix]
            and directory.is_absolute() and directory.parent == runs
            and directory.name.startswith(dir_prefix) and not directory.is_symlink(),
            'Latest renderer attempt identity, source or bundle differs')
    receipt_path = directory / 'receipt.json'
    require(receipt_path.is_file() and not receipt_path.is_symlink(),
            'Latest renderer attempt ended without a complete receipt')
    return {'marker_path': str(path), 'marker_sha256': sha(path),
            'receipt_path': str(receipt_path), 'marker': marker}


def validate_renderer_gate(receipt, preflight):
    gate = receipt.get('renderer_smoke_gate', {})
    require(isinstance(gate, dict)
            and gate.get('boot_id') == receipt.get('boot_id')
            and gate.get('bundle_manifest_sha256') == receipt.get('bundle_manifest_sha256')
            and gate.get('vulkan_runtime') == receipt.get('vulkan_runtime'),
            'Collect receipt lacks same-boot, same-bundle renderer gate')
    units = set()
    for mode, suffix in (('constructor', []), ('offscreen', ['--offscreen-example'])):
        latest = latest_renderer_attempt(preflight, mode)
        row = gate.get(mode, {})
        path = Path(row.get('path', '/'))
        require(path == Path(latest['receipt_path'])
                and row.get('attempt_path') == latest['marker_path']
                and row.get('attempt_sha256') == latest['marker_sha256']
                and path.is_absolute() and path == path.resolve(strict=True)
                and path.parent.parent == preflight / 'resource_runs'
                and path.parent.name.startswith('renderer_smoke_')
                and path.name == 'receipt.json',
                'Renderer gate proof path escaped this task')
        proof = read_json(path)
        require(row.get('sha256') == sha(path) and row.get('unit') == proof.get('unit')
                and row.get('boot_id') == proof.get('boot_id') == receipt.get('boot_id')
                and proof.get('bundle_manifest_sha256') == receipt.get('bundle_manifest_sha256')
                and proof.get('vulkan_runtime') == receipt.get('vulkan_runtime')
                and proof.get('command') == [receipt['command'][0],
                    str(preflight / 'renderer_only_smoke.py'), *suffix]
                and proof.get('unit') not in units,
                'Renderer gate proof is stale, changed or not independent')
        validate_stage(proof, path, 'renderer_smoke', preflight)
        units.add(proof['unit'])
    return True


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
            and proof.get('passed') is True
            and proof.get('boot_id') == receipt.get('boot_id'),
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
                and readback.get('boot_id') == receipt.get('boot_id')
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
        cgroup.require_zero_swap(resource)
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
    require(receipt.get('swap_accounting_source') ==
                'v1 memory.stat swap and total_swap, bracketed'
            and receipt.get('memory_usage_and_memsw_difference_role') ==
                'diagnostic_only_fuzzy',
            'Stage did not bind explicit v1 swap accounting semantics')
    preflight = Path(preflight).resolve()
    receipt_path = Path(receipt_path).resolve()
    require(receipt_path.parent.parent == preflight / 'resource_runs'
            and receipt_path.name == 'receipt.json',
            'Resource receipt is outside this task')
    require(receipt.get('boot_id') == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            and receipt.get('bundle_manifest_sha256') ==
                sha(preflight.parents[1] / 'bundle_manifest.json'),
            'Stage boot or independent bundle manifest changed')
    binding = receipt.get('source_binding', {})
    require(binding.get('guard_sha256') == sha(preflight / 'run_guarded_stage.py')
            and binding.get('cgroup_adapter_sha256') == sha(preflight / 'hybrid_cgroup_adapter.py')
            and binding.get('release_contract_sha256') == sha(preflight / 'hybrid_release_contract.py'),
            'Resource guard, cgroup adapter or release contract changed after stage execution')
    if stage in GPU_STAGES:
        runtime = receipt.get('vulkan_runtime', {})
        require(GPUCOMP.resolve(strict=True) == GPUCOMP_REAL
                and sha(GPUCOMP_REAL) == GPUCOMP_SHA256
                and sha(NVIDIA_ICD) == NVIDIA_ICD_SHA256
                and runtime == {
                    'gpucomp': {'path': str(GPUCOMP), 'real_path': str(GPUCOMP_REAL),
                                'sha256': GPUCOMP_SHA256, 'bytes': 63745800},
                    'nvidia_icd': {'path': str(NVIDIA_ICD), 'real_path': str(NVIDIA_ICD),
                                   'sha256': NVIDIA_ICD_SHA256, 'bytes': 140},
                    'driver': driver_version_readback()}
                and receipt.get('renderer_child_environment') ==
                    {'LD_PRELOAD': str(GPUCOMP), 'VK_ICD_FILENAMES': str(NVIDIA_ICD)},
                'Renderer child loader environment or fixed host files differ')
        command = receipt.get('command', [])
        require(len(command) >= 2 and binding.get('entry_sha256') == sha(command[1]),
                'Protected GPU stage entry changed before release')
        if stage == 'renderer_smoke':
            require(command[1] == str(preflight / 'renderer_only_smoke.py')
                    and command[2:] in ([], ['--offscreen-example']),
                    'Renderer diagnostic command is not the registered entry')
            mode = 'offscreen' if command[2:] else 'constructor'
            latest = latest_renderer_attempt(preflight, mode)
            require(receipt_path == Path(latest['receipt_path'])
                    and receipt.get('renderer_smoke_attempt') ==
                        {'path': latest['marker_path'], 'sha256': latest['marker_sha256']},
                    'Renderer stage is not the latest same-boot attempt')
        require(receipt.get('all_eight_gpu_floor_required') is True,
                'GPU stage lacks an eight-card memory floor')
        validate_eight_gpu_sample(receipt.get('gpu_before'))
        validate_eight_gpu_sample(receipt.get('gpu_after_probe'))
    if stage == 'collect':
        validate_renderer_gate(receipt, preflight)
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
    if stage == 'renderer_smoke':
        smoke_path = receipt_path.parent / 'renderer_smoke_result.json'
        smoke = read_json(smoke_path)
        require(receipt.get('renderer_smoke_result_sha256') == sha(smoke_path)
                and smoke.get('schema') == 'realman_renderer_only_smoke_v7'
                and smoke.get('passed') is True
                and smoke.get('robot_or_task_scene_loaded') is False
                and smoke.get('physical_episode_started') is False
                and smoke.get('loader_proof', {}).get('gpucomp_mapped_before_sapien_import') is True
                and smoke.get('loader_proof', {}).get('kernel_module_version') == NVIDIA_DRIVER_VERSION
                and smoke.get('scope') == ('official SAPIEN 128x128 offscreen example'
                    if receipt['command'][2:] == ['--offscreen-example'] else 'constructor/destructor only')
                and all(len(smoke.get(key, [])) == 8
                        for key in ('gpu_before', 'gpu_during', 'gpu_after')),
                'Renderer-only diagnostic result is missing or inconsistent')
        require(all(all(row.get('driver_version') == NVIDIA_DRIVER_VERSION for row in smoke[key])
                    for key in ('gpu_before', 'gpu_during', 'gpu_after')),
                'Renderer diagnostic observed a different driver on one of eight GPUs')
        if receipt['command'][2:] == ['--offscreen-example']:
            frame = smoke.get('frame', {})
            png = receipt_path.parent / 'official_offscreen_example/sapien_offscreen.png'
            require(frame.get('path') == str(png) and frame.get('sha256') == sha(png)
                    and frame.get('width') == frame.get('height') == 128
                    and type(frame.get('distinct_colors')) is int
                    and frame['distinct_colors'] >= 3,
                    'Official offscreen frame is missing or does not match the diagnostic receipt')
    resource = worker.get('resource', {})
    swap_fields = ('memory.stat.swap_before_bytes', 'memory.stat.total_swap_before_bytes',
                   'memory.stat.swap_after_bytes', 'memory.stat.total_swap_after_bytes',
                   'memory.usage_in_bytes.raw', 'memory.memsw.usage_in_bytes.raw',
                   'memory.memsw_minus_memory.diagnostic_bytes')
    require(receipt.get('swap_accounting_last') ==
            {key: resource.get(key) for key in swap_fields},
            'Stage receipt does not match final explicit swap and raw usage readback')
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
            if stage in GPU_STAGES:
                validate_eight_gpu_sample(row.get('gpu'))
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
