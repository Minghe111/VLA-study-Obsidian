"""Bounded simulation diagnostic session; executes only scripts in this package."""
from pathlib import Path
import sys,json,traceback,time,os,faulthandler
base=Path(__file__).resolve().parent
inbox=base/'commands';inbox.mkdir(exist_ok=True)
state={'__name__':'__main__','__file__':str(base/'run_adjust_bottle.py')}
sys.argv=[state['__file__'],'--seed','0','--label','official_ready','--align-table','--collect']
(base/'worker_pid.txt').write_text(str(os.getpid()))
try:
 exec(compile(Path(state['__file__']).read_text(),state['__file__'],'exec'),state)
except BaseException as exc:
 print('BASELINE_STOP',repr(exc),flush=True)
faulthandler.cancel_dump_traceback_later()
print('DIAGNOSTIC_READY',state.get('report',{}).get('result'),flush=True)
(base/'worker_status.json').write_text(json.dumps({'state':'ready','baseline_result':state.get('report',{}).get('result')}))
start=time.monotonic()
while time.monotonic()-start<3600:
 pending=sorted(inbox.glob('*.command.json'))
 if not pending:
  time.sleep(.25);continue
 command=pending[0];request=json.loads(command.read_text())
 if request.get('stop'):
  command.rename(command.with_suffix('.done'));break
 name=request['file'];path=(base/name).resolve()
 assert path.parent==base, 'Only package-local diagnostic scripts are accepted'
 print('COMMAND_START',name,flush=True)
 try:
  exec(compile(path.read_text(),str(path),'exec'),state)
  status='ok'
 except BaseException:
  traceback.print_exc();status='error'
 (inbox/(command.stem+'.result.json')).write_text(json.dumps({'file':name,'status':status}))
 command.rename(command.with_suffix('.done'))
 print('COMMAND_DONE',name,status,flush=True)
(base/'worker_status.json').write_text(json.dumps({'state':'stopped'}))