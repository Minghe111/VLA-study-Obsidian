"""Tiny self-contained readback/negative OOM probe; no simulator or media."""
from pathlib import Path
import json
import sys
import os
relative = next(x[3:] for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'))
assert Path(relative).name.startswith('realman-rerecord-cpu-cap-probe-')
cg = Path('/sys/fs/cgroup') / relative.lstrip('/')
actual = {n: (cg / n).read_text().strip() for n in ('memory.high', 'memory.max', 'memory.swap.max', 'cpu.max', 'pids.max')}
required = {'memory.high': '33554432' if '--overrun' in sys.argv else '25165824', 'memory.max': '33554432', 'memory.swap.max': '0',
            'cpu.max': '100000 100000', 'pids.max': '32'}
assert actual == required
assert os.getpriority(os.PRIO_PROCESS, 0) == 10
Path(sys.argv[1]).write_text(json.dumps({'limits_verified': True, 'actual': actual,
                                     'nice': 10, 'pid': os.getpid(), 'negative': '--overrun' in sys.argv}, indent=2) + '\n')
if '--overrun' in sys.argv:
    data = bytearray(64 * 1024 * 1024)
    raise RuntimeError('32 MiB memory cap failed to terminate 64 MiB allocation')
