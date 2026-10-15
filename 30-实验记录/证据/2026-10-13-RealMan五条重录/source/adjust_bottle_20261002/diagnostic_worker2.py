from pathlib import Path
import sys,json,traceback,time,os,faulthandler,ast
base=Path(__file__).resolve().parent
state={'__name__':'__main__','__file__':str(base/'run_adjust_bottle.py')}
sys.argv=[state['__file__'],'--seed','7','--label','resume_init','--align-table']
source=(base/'run_adjust_bottle.py').read_text().split('\ntry:\n Task=')[0].replace("BASE/'ready_home.json'","BASE/'ready_home_elevated.json'")
exec(compile(source,'resume_prelude','exec'),state)
exec('import transforms3d as t3d\nfrom types import MethodType\nfrom envs._base_task import Base_Task\nfrom envs.robot.robot import Robot',state)
state['original_bottle_state']=state['bottle_state']
for file,names in [('try_physical_grasp.py',{'audited_bottle_state'}),('try_horizontal.py',{'horizontal_grasp'})]:
 tree=ast.parse((base/file).read_text());tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names];exec(compile(tree,file,'exec'),state)
state['bottle_state']=state['audited_bottle_state']
exec("Task=getattr(importlib.import_module('envs.adjust_bottle'),'adjust_bottle');task=Task();print('RESUME_SETUP_START',flush=True);task.setup_demo(**args);print('RESUME_SETUP_DONE',flush=True)",state)
faulthandler.cancel_dump_traceback_later()
(base/'worker2_pid.txt').write_text(str(os.getpid()));(base/'worker2_status.json').write_text(json.dumps({'state':'ready'}))
inbox=base/'commands2';inbox.mkdir(exist_ok=True);start=time.monotonic()
while time.monotonic()-start<3000:
 pending=sorted(inbox.glob('*.command.json'))
 if not pending:time.sleep(.25);continue
 command=pending[0];request=json.loads(command.read_text())
 if request.get('stop'):command.rename(command.with_suffix('.done'));break
 path=(base/request['file']).resolve();assert path.parent==base
 print('COMMAND_START',path.name,flush=True)
 try:exec(compile(path.read_text(),str(path),'exec'),state);status='ok'
 except BaseException:traceback.print_exc();status='error'
 (inbox/(command.stem+'.result.json')).write_text(json.dumps({'file':path.name,'status':status}));command.rename(command.with_suffix('.done'));print('COMMAND_DONE',path.name,status,flush=True)
(base/'worker2_status.json').write_text(json.dumps({'state':'stopped'}))