"""Stdlib CPU-only replay binding tests; no simulator/GPU or accepted episodes.
Module doubles execute only source function DEFINITIONS, never planner/LP bodies.
"""
from pathlib import Path
import ast,copy,hashlib,os,subprocess,sys,types,unittest
from typing import Sequence,Mapping
from unittest.mock import patch
BASE=Path(__file__).resolve().parent
OLD=BASE/'accepted_reset_replay_v29.py';NEW=BASE/'accepted_reset_replay_v32.py'
TARGETS={'_function_code_signature','_validate_live_solver_bindings','_extract_function_defs','_require_explicit_helper_import'}
def helpers():
 context={'Path':Path,'ast':ast,'copy':copy,'hashlib':hashlib,'CodeType':types.CodeType}
 t=ast.parse(NEW.read_text(encoding='utf-8'));nodes=[n for n in t.body if isinstance(n,ast.FunctionDef) and n.name in TARGETS]
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(NEW),'exec'),context)
 return context
H=helpers();SIGN=H['_function_code_signature'];VALIDATE=H['_validate_live_solver_bindings']
def function(source,name='f'):
 c={};exec(compile(source,'cpu_fixture','exec'),c);return c[name]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def doubles():
 modules={};runtime={str(NEW.resolve()):sha(NEW)}
 for name in ('purecontact_lp_v26','planned_carry_wrench_v26','full_native_carry_planner_v31'):
  path=BASE/(name+'.py');module=types.ModuleType(name);module.__file__=str(path)
  module.__dict__.update(Sequence=Sequence,Mapping=Mapping,DEFAULT_CAPS_N=(0.,0.,1.5,1.5,2.5))
  nodes=[n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef)]
  exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),module.__dict__)
  modules[name]=module;runtime[str(path.resolve())]=sha(path)
 original=next(n for n in ast.parse((BASE/'try_coupled_side_entry_v29.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='precise_path')
 selected=copy.deepcopy(original);calls=[n for n in ast.walk(selected) if isinstance(n,ast.Call) and ast.unparse(n.func)=='ncp.plan_carry_to_task_pose'];assert len(calls)==1
 calls[0].func=ast.copy_location(ast.Name(id='batch_plan_carry',ctx=ast.Load()),calls[0].func)
 worker={'__name__':'cpu_source_double','batch_plan_carry':modules['full_native_carry_planner_v31'].plan_carry_to_task_pose}
 exec(compile(ast.fix_missing_locations(ast.Module(body=[selected],type_ignores=[])),'cpu_dispatch','exec'),worker)
 return modules,runtime,worker
class CanonicalBindingBoundaries(unittest.TestCase):
 def test_frozenset_layout_and_nested_tuple_same_signature(self):
  template=function('def f():\n return 1\n');sig=[]
  for seq in [('alpha','beta','gamma','delta','epsilon'),('epsilon','delta','gamma','beta','alpha')]:
   const=(('nested',frozenset(seq)),frozenset((('a',1),('b',2))))
   code=template.__code__.replace(co_consts=(None,const));sig.append(SIGN(types.FunctionType(code,{})))
  self.assertEqual(sig[0],sig[1])
 def test_constants_types_tuple_order_and_bytecode_all_checked(self):
  same=function('def f(x):\n return (x+1, "fixed", (1,2))\n')
  for source in ['def f(x):\n return (x+2, "fixed", (1,2))\n','def f(x):\n return (x+1, "changed", (1,2))\n','def f(x):\n return (x+1, "fixed", (2,1))\n','def f(x):\n return (x-1, "fixed", (1,2))\n']:
   self.assertNotEqual(SIGN(same),SIGN(function(source)))
  a=function('def f():\n return True\n');b=function('def f():\n return 1\n');self.assertNotEqual(SIGN(a),SIGN(b))
 def test_nested_code_and_freevars_argument_layout_not_ignored(self):
  a=function('def f(x):\n def inner():\n  return x+1\n return inner\n');b=function('def f(x):\n def inner():\n  return x+2\n return inner\n')
  self.assertNotEqual(SIGN(a),SIGN(b));self.assertNotEqual(SIGN(function('def f(x):\n return x\n')),SIGN(function('def f(x,/,y=None):\n return x\n')))
 def test_cross_hashseed_canonical_signature_stable(self):
  code='''from pathlib import Path
import ast,hashlib,types
p=Path(%r);t=ast.parse(p.read_text());n=next(x for x in t.body if isinstance(x,ast.FunctionDef) and x.name=='_function_code_signature')
c={'CodeType':types.CodeType,'hashlib':hashlib};exec(compile(ast.Module(body=[n],type_ignores=[]),str(p),'exec'),c)
d={};exec('def f(x):\\n return (x in {"alpha","beta","gamma","delta","epsilon"},("constant",23))\\n',d)
print(c['_function_code_signature'](d['f']))
'''%str(NEW)
  sig=[]
  for seed in ('0','5','9'):
   env=dict(os.environ,PYTHONHASHSEED=seed);sig.append(subprocess.check_output([sys.executable,'-B','-c',code],env=env,text=True).strip())
  self.assertEqual(len(set(sig)),1)
 def test_exact_source_dispatch_binding_passes(self):
  modules,runtime,worker=doubles()
  with patch.dict(sys.modules,modules):proof=VALIDATE(worker,BASE,runtime,sha)
  self.assertEqual(proof['planner_fork']['module'],'full_native_carry_planner_v31');self.assertEqual(proof['planner_fork']['authorized_dispatch_count'],1)
  self.assertTrue(proof['planner_fork']['all_other_precise_source_ast_unchanged']);self.assertIn('frozenset',proof['normalized_signature_model']);self.assertEqual(proof['normalized_signature_source_sha256'],sha(NEW))
 def test_stale_dispatch_object_rejected(self):
  modules,runtime,worker=doubles();worker['batch_plan_carry']=function('def f(*a,**k):\n return None\n')
  with patch.dict(sys.modules,modules):
   with self.assertRaises(RuntimeError):VALIDATE(worker,BASE,runtime,sha)
 def test_changed_precise_code_and_undispatched_original_rejected(self):
  for mutation in ('changed','original'):
   modules,runtime,worker=doubles()
   if mutation=='changed':
    co=worker['precise_path'].__code__;const=list(co.co_consts);at=next(i for i,x in enumerate(const) if isinstance(x,str) and x!='');const[at]+=' altered';worker['precise_path']=types.FunctionType(co.replace(co_consts=tuple(const)),worker)
   else:H['_extract_function_defs'](BASE/'try_coupled_side_entry_v29.py',{'precise_path'},worker)
   with patch.dict(sys.modules,modules):
    with self.assertRaises(RuntimeError):VALIDATE(worker,BASE,runtime,sha)
 def test_altered_loaded_planner_function_rejected(self):
  modules,runtime,worker=doubles();m=modules['full_native_carry_planner_v31'];co=m.plan_carry_to_task_pose.__code__;const=list(co.co_consts);at=next(i for i,x in enumerate(const) if isinstance(x,str));const[at]+=' changed'
  m.plan_carry_to_task_pose=types.FunctionType(co.replace(co_consts=tuple(const)),m.__dict__);worker['batch_plan_carry']=m.plan_carry_to_task_pose
  with patch.dict(sys.modules,modules):
   with self.assertRaises(RuntimeError):VALIDATE(worker,BASE,runtime,sha)
 def test_changed_runtime_origin_and_missing_actual32_source_rejected(self):
  for kind in ('hash','path','actual32'):
   modules,runtime,worker=doubles()
   if kind=='hash':runtime[str((BASE/'full_native_carry_planner_v31.py').resolve())]='0'*64
   if kind=='path':modules['full_native_carry_planner_v31'].__file__=str(BASE/'full_native_carry_planner_v21.py')
   if kind=='actual32':runtime.pop(str(NEW.resolve()))
   with patch.dict(sys.modules,modules):
    with self.assertRaises(RuntimeError):VALIDATE(worker,BASE,runtime,sha)
 def test_only_two_functions_and_actual_helper_hash_list_changed(self):
  old=ast.parse(OLD.read_text());new=ast.parse(NEW.read_text());names={'_function_code_signature','_validate_live_solver_bindings'}
  for t in (old,new):t.body=[n for n in t.body if not isinstance(n,ast.FunctionDef) or n.name not in names]
  removed=0
  for node in ast.walk(new):
   if isinstance(node,ast.Tuple):
    keep=[]
    for item in node.elts:
     if ast.unparse(item)=="base / 'accepted_reset_replay_v32.py'":removed+=1
     else:keep.append(item)
    node.elts=keep
  self.assertEqual(removed,1);self.assertEqual(ast.dump(old,include_attributes=False),ast.dump(new,include_attributes=False));self.assertEqual(sha(OLD),'8a4191456cffa7664806ac0b738fbb177ca8cf19a4a1694372665d20ce16fb46')

if __name__=='__main__':unittest.main(verbosity=2)
