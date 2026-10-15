"""Fail-closed kernel cgroup readback for a dedicated RealMan transient unit.

This is a server-only source file.  A systemd property is never accepted as
proof: the worker must belong to the named unit in every active controller.
The same code can read unified v2 or a v1 memory/cpu/pids hybrid host.
"""

from pathlib import Path, PurePosixPath
import os


class CgroupProofError(RuntimeError):
    pass


def _require(condition, reason):
    if not condition:
        raise CgroupProofError(reason)


def _unescape_mount(value):
    # mountinfo escapes spaces, tabs, newlines and backslashes as octal codes.
    for encoded, plain in ((r'\040', ' '), (r'\011', '\t'),
                           (r'\012', '\n'), (r'\134', '\\')):
        value = value.replace(encoded, plain)
    return value


def parse_membership(raw):
    membership = {}
    for line in raw.splitlines():
        fields = line.split(':', 2)
        _require(len(fields) == 3, 'Malformed /proc/PID/cgroup row')
        _hierarchy, controllers, relative = fields
        _require(relative.startswith('/') and '..' not in PurePosixPath(relative).parts,
                 'Unsafe cgroup membership path')
        keys = controllers.split(',') if controllers else ('unified',)
        for key in keys:
            _require(key not in membership, 'Duplicate cgroup controller membership')
            membership[key] = PurePosixPath(relative)
    return membership


def parse_mounts(raw):
    mounts = {}
    for line in raw.splitlines():
        if ' - ' not in line:
            continue
        left, right = line.split(' - ', 1)
        primary, other = left.split(), right.split()
        if len(primary) < 5 or len(other) < 3 or other[0] not in ('cgroup', 'cgroup2'):
            continue
        root = PurePosixPath(_unescape_mount(primary[3]))
        mountpoint = Path(_unescape_mount(primary[4]))
        _require(root.is_absolute() and mountpoint.is_absolute(), 'Unsafe cgroup mount')
        if other[0] == 'cgroup2':
            keys = ('unified',)
        else:
            keys = tuple(x for x in other[2].split(',') if x in ('memory', 'cpu', 'pids'))
        for key in keys:
            _require(key not in mounts, 'Ambiguous cgroup controller mount')
            mounts[key] = (root, mountpoint)
    return mounts


def _directory(key, membership, mounts, unit):
    _require(key in membership and key in mounts, 'Missing ' + key + ' controller')
    relative = membership[key]
    _require(relative == PurePosixPath('/system.slice') / unit,
             key + ' controller is not in exact dedicated system task unit')
    mount_root, mountpoint = mounts[key]
    try:
        suffix = relative.relative_to(mount_root)
    except ValueError as error:
        raise CgroupProofError(key + ' controller escapes mounted hierarchy') from error
    directory = mountpoint.joinpath(*suffix.parts)
    _require(directory.is_dir() and not directory.is_symlink(),
             key + ' controller task directory absent or linked')
    return directory


def discover(proc_cgroup, mountinfo, unit):
    """Return exact system-unit controller directories, never an ancestor."""
    _require(unit.startswith('realman-rerecord-') and unit.endswith('.service')
             and '/' not in unit, 'Unexpected task unit identity')
    membership, mounts = parse_membership(proc_cgroup), parse_mounts(mountinfo)
    if any(key in membership for key in ('memory', 'cpu', 'pids')):
        # A hybrid process may also have a decorative cgroup2 unit path.
        # Never accept that path in place of its actual v1 controllers.
        return {'mode': 'v1', **{key: _directory(key, membership, mounts, unit)
                               for key in ('memory', 'cpu', 'pids')}}
    if 'unified' in membership and 'unified' in mounts:
        unified = _directory('unified', membership, mounts, unit)
        if all((unified / name).is_file() for name in
               ('memory.high', 'memory.max', 'memory.swap.max', 'cpu.max', 'pids.max')):
            return {'mode': 'v2', 'unified': unified}
    raise CgroupProofError('No complete kernel controller hierarchy for task unit')


def discover_self(unit):
    return discover(Path('/proc/self/cgroup').read_text(),
                    Path('/proc/self/mountinfo').read_text(), unit)


def _read(directory, name):
    path = directory / name
    _require(path.is_file() and not path.is_symlink(), 'Missing kernel cgroup file: ' + str(path))
    return path.read_text().strip()


def _integer(directory, name):
    raw = _read(directory, name)
    _require(raw.isdecimal(), 'Non-numeric kernel cgroup value: ' + name)
    return int(raw)


def _flat(directory, name):
    result = {}
    for row in _read(directory, name).splitlines():
        parts = row.split()
        _require(len(parts) == 2 and parts[1].isdecimal(), 'Malformed cgroup counter: ' + name)
        result[parts[0]] = int(parts[1])
    return result


def verify_limits(binding, *, high, maximum, cpu_percent, pids, nice=10):
    """Return a portable actual readback; fail if any promised cap is absent."""
    _require(all(type(x) is int and x > 0 for x in (high, maximum, cpu_percent, pids)),
             'Non-positive resource contract')
    _require(high < maximum and cpu_percent % 100 == 0,
             'Resource contract is not a bounded whole-core budget')
    mode = binding['mode']
    if mode == 'v2':
        directory = binding['unified']
        actual = {name: _read(directory, name) for name in
                  ('memory.high', 'memory.max', 'memory.swap.max', 'cpu.max', 'pids.max')}
        _require(actual == {'memory.high': str(high), 'memory.max': str(maximum),
                  'memory.swap.max': '0', 'cpu.max': f'{cpu_percent * 1000} 100000',
                  'pids.max': str(pids)}, 'Unified kernel cgroup limits differ')
    elif mode == 'v1':
        memory, cpu, process = (binding[key] for key in ('memory', 'cpu', 'pids'))
        soft = _integer(memory, 'memory.soft_limit_in_bytes')
        hard = _integer(memory, 'memory.limit_in_bytes')
        combined = _integer(memory, 'memory.memsw.limit_in_bytes')
        swappiness = _integer(memory, 'memory.swappiness')
        quota = int(_read(cpu, 'cpu.cfs_quota_us'))
        period = _integer(cpu, 'cpu.cfs_period_us')
        actual = {'memory.soft_limit_in_bytes': str(soft),
                  'memory.limit_in_bytes': str(hard),
                  'memory.memsw.limit_in_bytes': str(combined),
                  'memory.swappiness': str(swappiness),
                  'cpu.cfs_quota_us': str(quota),
                  'cpu.cfs_period_us': str(period),
                  'pids.max': _read(process, 'pids.max')}
        # v1 soft is best-effort.  memsw=hard caps the combined RAM+swap
        # charge, but does *not* independently prove that no page is swapped.
        _require(soft == high and hard == maximum and combined == maximum
                 and swappiness == 0,
                 'v1 memory soft/hard/combined/swappiness kernel settings differ')
        _require(1000 <= period <= 1000000 and quota > 0
                 and quota * 100 == cpu_percent * period,
                 'v1 CPU CFS quota differs')
        _require(actual['pids.max'] == str(pids), 'v1 process cap differs')
    else:
        raise CgroupProofError('Unsupported cgroup controller layout')
    _require(os.getpriority(os.PRIO_PROCESS, 0) == nice, 'Actual process nice differs')
    return {'mode': mode, 'actual': actual,
            'controller_paths': {key: str(value) for key, value in binding.items()
                                 if key != 'mode'},
            'limits_verified': True}


def sample(binding):
    """Normalize counters used by the existing guard and release gates."""
    mode = binding['mode']
    if mode == 'v2':
        directory = binding['unified']
        return {'mode': mode, 'memory.current': _integer(directory, 'memory.current'),
                'memory.peak': _integer(directory, 'memory.peak'),
                'memory.swap.current': _integer(directory, 'memory.swap.current'),
                'pids.current': _integer(directory, 'pids.current'),
                'memory.events': _flat(directory, 'memory.events'),
                'cpu.stat': _flat(directory, 'cpu.stat')}
    _require(mode == 'v1', 'Unsupported cgroup sample layout')
    memory, cpu, process = (binding[key] for key in ('memory', 'cpu', 'pids'))
    failcnt = _integer(memory, 'memory.failcnt')
    memsw_failcnt = _integer(memory, 'memory.memsw.failcnt')
    oom = _flat(memory, 'memory.oom_control')
    _require(oom.get('under_oom') == 0 and oom.get('oom_kill_disable') == 0,
             'v1 OOM state is not safe')
    # Do not manufacture an OOM-kill count on older kernels which omit it.
    # The release gate rejects an unknown count unless a separate trustworthy
    # kernel/systemd source is explicitly added and proved.
    oom_kill = oom.get('oom_kill')
    events = {'max': failcnt, 'oom': failcnt, 'oom_kill': oom_kill,
              'oom_group_kill': 0, 'memsw_max': memsw_failcnt}
    # v1 usage_in_bytes and memsw.usage_in_bytes are separately-read, fuzzy
    # counters; their subtraction is not a trustworthy swap measurement.
    # Bracket those diagnostic reads with the kernel's explicit swap stats.
    stat_before = _flat(memory, 'memory.stat')
    used = _integer(memory, 'memory.usage_in_bytes')
    memsw = _integer(memory, 'memory.memsw.usage_in_bytes')
    stat_after = _flat(memory, 'memory.stat')
    swap_stats = {}
    for key, stat in (('before', stat_before), ('after', stat_after)):
        for field in ('swap', 'total_swap'):
            value = stat.get(field)
            _require(type(value) is int and value >= 0,
                     'Missing or malformed v1 memory.stat ' + field)
            swap_stats['memory.stat.' + field + '_' + key + '_bytes'] = value
        _require(stat['total_swap'] >= stat['swap'],
                 'v1 hierarchical swap is below local swap')
    swap_current = max(swap_stats.values())
    return {'mode': mode, 'memory.current': used,
            'memory.peak': _integer(memory, 'memory.max_usage_in_bytes'),
            'memory.swap.current': swap_current,
            **swap_stats,
            'memory.usage_in_bytes.raw': used,
            'memory.memsw.usage_in_bytes.raw': memsw,
            'memory.memsw_minus_memory.diagnostic_bytes': memsw - used,
            'memory.memsw.max_usage': _integer(memory, 'memory.memsw.max_usage_in_bytes'),
            'pids.current': _integer(process, 'pids.current'),
            'memory.events': events, 'memory.oom_control': oom,
            'cpu.stat': _flat(cpu, 'cpu.stat')}


def require_zero_swap(resource):
    """Require explicit v1 swap stats; never infer zero from fuzzy usage."""
    if resource.get('mode') == 'v1':
        values = [resource.get('memory.stat.' + field + '_' + point + '_bytes')
                  for point in ('before', 'after')
                  for field in ('swap', 'total_swap')]
        _require(all(type(value) is int and value >= 0 for value in values)
                 and values[1] >= values[0] and values[3] >= values[2]
                 and resource.get('memory.swap.current') == max(values),
                 'Explicit v1 memory.stat swap proof is inconsistent or missing')
        used = resource.get('memory.usage_in_bytes.raw')
        memsw = resource.get('memory.memsw.usage_in_bytes.raw')
        _require(type(used) is int and used >= 0 and type(memsw) is int and memsw >= 0
                 and resource.get('memory.memsw_minus_memory.diagnostic_bytes') == memsw - used,
                 'Fuzzy v1 usage diagnostic readback is missing or inconsistent')
        _require(all(value == 0 for value in values),
                 'Explicit v1 memory.stat swap is non-zero or missing')
    _require(resource.get('memory.swap.current') == 0,
             'Observed cgroup swap is non-zero or unknown')
    return True


def require_clean_terminal(resource):
    events = resource.get('memory.events', {})
    _require(all(events.get(name) == 0 for name in
                 ('max', 'oom', 'oom_kill', 'oom_group_kill')),
             'Memory hard cap or OOM event occurred')
    require_zero_swap(resource)
    if resource.get('mode') == 'v1':
        _require(events.get('memsw_max') == 0 and
                 resource.get('memory.oom_control', {}).get('under_oom') == 0
                 and resource.get('memory.swap.current') == 0,
                 'Memory+swap cap, observed swap or active OOM occurred')
    return True


def verify_recorded_limits(readback, *, unit, high, maximum, cpu_percent, pids):
    """Validate frozen v1 kernel limits after systemd removed the unit.

    The worker wrote the kernel readback while the unit was alive; its source
    hash and the receipt/file identity must be pinned separately by callers.
    """
    _require(readback.get('limits_verified') is True and readback.get('mode') == 'v1',
             'No verified dedicated v1 kernel readback in stage receipt')
    actual = readback.get('actual', {})
    paths = readback.get('controller_paths', {})
    expected = {
        'memory': f'/sys/fs/cgroup/memory/system.slice/{unit}',
        'cpu': f'/sys/fs/cgroup/cpu,cpuacct/system.slice/{unit}',
        'pids': f'/sys/fs/cgroup/pids/system.slice/{unit}',
    }
    _require(all(paths.get(key) == value for key, value in expected.items()),
             'Recorded controller paths are not dedicated task unit paths')
    _require(actual.get('memory.soft_limit_in_bytes') == str(high)
             and actual.get('memory.limit_in_bytes') == str(maximum)
             and actual.get('memory.memsw.limit_in_bytes') == str(maximum)
             and actual.get('memory.swappiness') == '0'
             and actual.get('pids.max') == str(pids),
             'Recorded memory, swap or PID kernel limit differs')
    try:
        quota = int(actual.get('cpu.cfs_quota_us', ''))
        period = int(actual.get('cpu.cfs_period_us', ''))
    except ValueError as error:
        raise CgroupProofError('Recorded CPU CFS values missing') from error
    _require(1000 <= period <= 1000000 and quota > 0
             and quota * 100 == cpu_percent * period,
             'Recorded CPU quota differs')
    return True


def verify_recorded(readback, resource, *, unit, high, maximum, cpu_percent, pids):
    """Validate frozen limits plus a clean non-OOM unit terminal sample."""
    verify_recorded_limits(readback, unit=unit, high=high, maximum=maximum,
                           cpu_percent=cpu_percent, pids=pids)
    _require(resource.get('mode') == 'v1', 'Terminal resource uses different cgroup mode')
    require_clean_terminal(resource)
    return True
