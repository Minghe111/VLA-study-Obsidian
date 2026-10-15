from pathlib import Path
import json,sys
line=next(x for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'))
cg=Path('/sys/fs/cgroup')/line[3:].lstrip('/')
proof={n:(cg/n).read_text().strip() for n in ('memory.high','memory.max','memory.swap.max','cpu.max','pids.max')}
proof['scope']='Tiny CPU-only task; no simulator/arrays/media loaded'
Path(sys.argv[1]).write_text(json.dumps(proof,indent=2)+'\n')
assert proof['memory.max']=='33554432'
assert proof['memory.swap.max']=='0'
assert proof['cpu.max']=='100000 100000'
assert proof['pids.max']=='32'
if '--overrun' in sys.argv:
    data=bytearray(64*1024*1024)
    raise RuntimeError('Memory cap did not terminate allocation')
