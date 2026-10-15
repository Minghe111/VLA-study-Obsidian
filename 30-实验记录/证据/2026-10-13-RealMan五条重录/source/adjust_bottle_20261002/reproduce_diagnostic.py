"""Reproduce one bounded, explicitly diagnostic official adjust_bottle attempt."""
from pathlib import Path
import sys,ast,argparse,re,time,faulthandler
p=argparse.ArgumentParser();p.add_argument('--label',default='diagnostic_'+time.strftime('%Y%m%d_%H%M%S'));opt=p.parse_args()
if not re.fullmatch(r'[A-Za-z0-9_-]+',opt.label):raise SystemExit('Use only letters, digits, dash, underscore for label')
base=Path(__file__).resolve().parent
state={'__name__':'__main__','__file__':str(base/'run_adjust_bottle.py')}
sys.argv=[state['__file__'],'--seed','7','--label',opt.label+'_init','--align-table']
source=(base/'run_adjust_bottle.py').read_text();assert '\ntry:\n Task=' in source
source=source.split('\ntry:\n Task=')[0].replace("BASE/'ready_home.json'","BASE/'ready_home_elevated.json'")
exec(compile(source,'diagnostic_prelude','exec'),state)
exec('import transforms3d as t3d\nfrom types import MethodType\nfrom envs._base_task import Base_Task\nfrom envs.robot.robot import Robot',state)
state['original_bottle_state']=state['bottle_state']
for file,names in [('try_physical_grasp.py',{'audited_bottle_state'}),('try_horizontal.py',{'horizontal_grasp'})]:
 tree=ast.parse((base/file).read_text());tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names];exec(compile(tree,file,'exec'),state)
state['bottle_state']=state['audited_bottle_state']
exec("Task=getattr(importlib.import_module('envs.adjust_bottle'),'adjust_bottle');task=Task();task.setup_demo(**args)",state)
faulthandler.cancel_dump_traceback_later()
video=(base/'record_diagnostic_v2.py').read_text().replace("BASE/'official_adjust_bottle_diagnostic_v2_seed7'",'BASE/'+repr(opt.label))
exec(compile(video,'record_diagnostic_v2.py','exec'),state)
print('Output:',state['OUT'])
raise SystemExit(0 if state['report']['result']=='physics_success_not_collected' else 2)