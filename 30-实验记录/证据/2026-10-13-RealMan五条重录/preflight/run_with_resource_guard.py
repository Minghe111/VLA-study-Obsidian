"""Single, cgroup-limited task runner; defaults to check only, never auto-retry."""
from __future__ import annotations
import argparse, fcntl, json, os, subprocess, time
from pathlib import Path

GIB=1024**3
E=Path(__file__).resolve().parent
PYTHON='/home/admin123/liminghe/miniconda3/envs/robotwin/bin/python'
PREFIX=Path('/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4')
VARIANTS=('rgb50_motion_quality_20261013_v3_acquisition','rgb50_motion_quality_20261013_v4_so3')
ENTRIES=('run_clearance_acquisition_collector.py','run_so3_acquisition_collector.py')
UNIT='realman-rerecord-protected'
LIMITS={'MemoryAccounting':'yes','MemoryHigh':'10G','MemoryMax':'12G','MemorySwapMax':'0',
        'CPUQuota':'200%','TasksMax':'128','Nice':'10','KillMode':'control-group',
        'TimeoutStopSec':'10s','RuntimeMaxSec':'35min','OOMPolicy':'stop'}

def memory_available():
    return int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines()
                    if x.startswith('MemAvailable:')))*1024

def active_collectors():
    rows=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name)==os.getpid():continue
        try:parts=(p/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        args=[x.decode(errors='replace') for x in parts if x]
        if args and args[0]==PYTHON and any(Path(x).name in ENTRIES for x in args[1:]):
            rows.append({'pid':int(p.name),'argv':args})
    return rows

def validate_command(command):
    if len(command)<2 or command[0]!=PYTHON:raise ValueError('Only known task Python and collector supported')
    entry=Path(command[1]);resolved=entry.resolve(strict=True)
    if resolved.name not in ENTRIES:raise ValueError('Wrong collector entry')
    rel=resolved.relative_to(PREFIX)
    if len(rel.parts)!=5 or rel.parts[0] not in VARIANTS or rel.parts[1]!='jobs' or rel.parts[3]!='preflight':
        raise ValueError('Only isolated task runtimes allowed')
    if rel.parts[2] not in tuple(f'slot{x}_runtime' for x in (17,18,20,29,34)):
        raise ValueError('Wrong task slot')
    if entry!=resolved:raise ValueError('Entry must use exact absolute real path')
    return resolved.parent.parent

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',action='store_true')
    ap.add_argument('command',nargs=argparse.REMAINDER);a=ap.parse_args()
    cmd=a.command[1:] if a.command[:1]==['--'] else a.command
    runtime=validate_command(cmd)
    lock=(E/'resource_guard.lock').open('a')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    if active_collectors():raise RuntimeError('Existing task collector; refuse parallel work')
    status=subprocess.run(['systemctl','--user','is-active',UNIT],text=True,capture_output=True)
    if status.returncode==0:raise RuntimeError('Previous bounded job still active')
    available=memory_available()
    if available<18*GIB:raise RuntimeError('Less than 18 GiB available; refuse start')
    launch=['systemd-run','--user','--wait','--collect','--pipe',f'--unit={UNIT}',f'--working-directory={runtime}']
    launch += [f'--property={k}={v}' for k,v in LIMITS.items()]
    env={'PATH':'/home/admin123/liminghe/miniconda3/envs/robotwin-official-data/bin:/home/admin123/liminghe/miniconda3/envs/robotwin/bin:/usr/bin:/bin',
         'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2','CUDA_VISIBLE_DEVICES':'0',
         'MPLCONFIGDIR':str(runtime/'preflight/protected_mplcache')}
    launch += [f'--setenv={k}={v}' for k,v in env.items()]
    launch += cmd
    receipt={'check_only':not a.run,'available_before_bytes':available,'single_worker':True,
             'limits':LIMITS,'start_minimum_available_GiB':18,'stop_available_floor_GiB':6,
             'gpu_hard_memory_cap':False,'unit':UNIT,'command':launch,'auto_retry':False}
    if not a.run:print(json.dumps(receipt,ensure_ascii=False,indent=2));return 0
    path=E/f'resource_guard_launch_{time.time_ns()}.json'
    path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    proc=subprocess.Popen(launch);low=0;reason=None
    try:
        while proc.poll() is None:
            low=low+1 if memory_available()<6*GIB else 0
            if low>=3:
                reason='Host available memory below 6 GiB across 3 samples'
                subprocess.run(['systemctl','--user','stop',UNIT],check=False);break
            time.sleep(2)
        code=proc.wait()
    except BaseException:
        subprocess.run(['systemctl','--user','stop',UNIT],check=False)
        raise
    finally:
        receipt['monitor_stop_reason']=reason
        receipt['launcher_returncode']=proc.poll()
        receipt['available_after_bytes']=memory_available()
        path.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    return code

if __name__=='__main__':raise SystemExit(main())
