"""I/O-only native-audit derivation; every actual step and predicate retained.

This protects recording history and the v13/v14 full native geometry audit from
full-file JSON/string copies. The live original scene, FCL, limits, contacts,
closure boundary and all physical decision arithmetic remain unchanged.
"""
from __future__ import annotations
import ast,copy,hashlib,importlib,inspect,json,textwrap
from pathlib import Path
from bounded_trace_storage import (LazyJSONList, PendingTraceList, DiskJSONAppender,
    copy_with_overrides, write_json_stream, stream_sha)

_INSTALLED=None

def _dump(node):return ast.dump(node,include_attributes=False)

def derive(function, mode, directory):
 source=textwrap.dedent(inspect.getsource(function));tree=ast.parse(source);counts={}
 def count(key):counts[key]=counts.get(key,0)+1
 class IO(ast.NodeTransformer):
  def visit_Assign(self,node):
   self.generic_visit(node)
   name=ast.unparse(node.targets[0]) if len(node.targets)==1 else ''
   value=ast.unparse(node.value)
   if name=='trace' and value.startswith('json.loads(Path(trace_path).read_text('):
    node.value=ast.parse('LazyJSONList(trace_path)',mode='eval').body;count('lazy_trace_read')
   if mode=='full' and name=='rows' and isinstance(node.value,ast.List) and not node.value.elts:
    node.value=ast.parse("DiskJSONAppender(Path(trace_path).with_name('actual36_geometry_rows.compact.json'))",mode='eval').body;count('disk_audit_rows')
   if mode=='reconcile' and name=="result['samples']" and value=='list(samples)':
    node.value=ast.parse('copy_with_overrides(samples)',mode='eval').body;count('bounded_sample_copy')
   if mode=='full' and name=='result' and isinstance(node.value,ast.Dict):
    count('seal_audit_rows');return [ast.parse('rows = rows.close()').body[0],node]
   return node
  def visit_Call(self,node):
   self.generic_visit(node)
   if (isinstance(node.func,ast.Attribute) and node.func.attr=='hexdigest' and isinstance(node.func.value,ast.Call)
       and ast.unparse(node.func.value.func)=='hashlib.sha256' and len(node.func.value.args)==1):
    inner=node.func.value.args[0]
    if isinstance(inner,ast.Call) and isinstance(inner.func,ast.Attribute) and inner.func.attr=='read_bytes':
     count('stream_source_sha');return ast.copy_location(ast.Call(func=ast.Name(id='stream_sha',ctx=ast.Load()),args=[inner.func.value],keywords=[]),node)
   return node
  def visit_Expr(self,node):
   self.generic_visit(node)
   if mode=='save' and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='write_text':
    args=node.value.args
    if args and isinstance(args[0],ast.Call) and ast.unparse(args[0].func)=='json.dumps' and args[0].args and ast.unparse(args[0].args[0])=='self.traces':
     count('stream_trace_save');return ast.parse("self.traces.flush_json(self.out / 'physics_trace.json')").body[0]
   if mode in ('full','boundary_wrapper') and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='write_text':
    text=ast.unparse(node.value)
    if 'json.dumps(result' in text:
     count('stream_audit_write');return ast.parse('write_audit(output_path, result)').body[0]
   return node
  def visit_With(self,node):
   if mode=='retained' and len(node.items)==1 and "Path(output_path).open('x'" in ast.unparse(node.items[0].context_expr) and 'json.dumps(result' in ast.unparse(node):
    count('stream_audit_write');return ast.parse("write_audit(output_path, result, newline=True)").body[0]
   return self.generic_visit(node)
 tree=IO().visit(tree);ast.fix_missing_locations(tree)
 expected={'loader':{'lazy_trace_read':1},'save':{'stream_trace_save':1},
   'full':{'disk_audit_rows':1,'seal_audit_rows':1,'stream_source_sha':4,'stream_audit_write':1},
   'reconcile':{'bounded_sample_copy':1},'boundary_wrapper':{'lazy_trace_read':1,'stream_audit_write':1},
   'retained':{'stream_audit_write':1}}[mode]
 if counts!=expected:raise RuntimeError(f'Exact bounded I/O sites changed {mode}: {counts} != {expected}')
 path=Path(directory)/f'{function.__module__}_{function.__name__}_bounded_{mode}_generated.py'
 path.write_text(ast.unparse(tree)+'\n')
 namespace=dict(function.__globals__)
 namespace.update(LazyJSONList=LazyJSONList,DiskJSONAppender=DiskJSONAppender,copy_with_overrides=copy_with_overrides,
                  write_audit=write_audit,stream_sha=stream_sha)
 exec(compile(tree,str(path),'exec'),namespace)
 return namespace[function.__name__],{'mode':mode,'original_source':inspect.getsourcefile(function),
  'original_function_ast_sha256':hashlib.sha256(_dump(ast.parse(source)).encode()).hexdigest(),
  'derived_ast_sha256':hashlib.sha256(_dump(tree).encode()).hexdigest(),'exact_IO_sites':counts,
  'generated_source':str(path.resolve()),'generated_source_sha256':stream_sha(path),
  'native_physical_geometry_thresholds_and_predicates_changed':False}

def write_audit(path,data,*,newline=False):
 # Full audit whitespace only: every original key, row and value remains.
 write_json_stream(path,data,indent=None,allow_nan=False)
 if newline:
  with Path(path).open('a') as f:f.write('\n')
 if isinstance(data,dict) and 'samples' in data:
  summary={k:v for k,v in data.items() if k!='samples'}
  summary['bounded_full_audit_file']={'path':str(Path(path).resolve()),'sha256':stream_sha(path),
      'all_samples_written':len(data['samples'])}
  target=Path(path).with_name(Path(path).stem+'_summary.json')
  write_json_stream(target,summary,indent=2,allow_nan=False)


def install_bounded_trace_overlay(source_package,legacy_directory):
 global _INSTALLED
 if _INSTALLED is not None:raise RuntimeError('Bounded trace installs once per independent worker')
 package=Path(source_package);legacy=Path(legacy_directory);directory=Path(__file__).resolve().parent
 loader=importlib.import_module('observed_trace_audit_v13')
 full=importlib.import_module('observed_full_trace_audit_v13')
 boundary=importlib.import_module('closure_boundary_audit_v14')
 retained=importlib.import_module('expert_generation_v9')
 expert=importlib.import_module('expert_generation')
 receipt={'schema':'bounded-native-trace-IO-v1','only_residency_and_serialization_changed':True,
  'no_recorded_rows_or_contacts_removed':True,'mutable_tail_rows':50,'physics_dt_and_observation_stride_unchanged':True,
  'native_v13_v14_full_geometry_checks_retained':True,'live_original_scene_audited':True,
  'full_audit_json_whitespace_layout':'compact; canonical JSON values unchanged','derivations':[]}
 # v9 write() hashes the complete final audit: preserve SHA256 bytes while streaming.
 retained._sha=stream_sha
 receipt['final_report_audit_SHA256_streamed']=True
 original_loader=loader._load_trace
 loader._load_trace,r=derive(original_loader,'loader',directory);receipt['derivations'].append(r)
 full._load_trace=loader._load_trace
 full.audit_accepted_actual_trace,r=derive(full.audit_accepted_actual_trace,'full',directory);receipt['derivations'].append(r)
 boundary.reconcile_original_result,r=derive(boundary.reconcile_original_result,'reconcile',directory);receipt['derivations'].append(r)
 boundary.audit_accepted_actual_trace_v14,r=derive(boundary.audit_accepted_actual_trace_v14,'boundary_wrapper',directory);receipt['derivations'].append(r)
 retained.audit_with_retained_report,r=derive(retained.audit_with_retained_report,'retained',directory);receipt['derivations'].append(r)
 expert.ExpertSession.save_evidence,r=derive(expert.ExpertSession.save_evidence,'save',directory);receipt['derivations'].append(r)
 original_init=expert.ExpertSession.__init__
 pins={str(Path(__file__).resolve()):stream_sha(__file__),
       str(Path(importlib.import_module('bounded_trace_storage').__file__).resolve()):stream_sha(importlib.import_module('bounded_trace_storage').__file__)}
 pins.update({r['generated_source']:r['generated_source_sha256'] for r in receipt['derivations']})
 def init(session,*args,**kwargs):
  original_init(session,*args,**kwargs)
  if len(session.traces)!=0:raise RuntimeError('Bounded history must install before first actual step')
  session.traces=PendingTraceList(session.out/'physics_trace.json',tail_rows=50,default=expert._json)
  session.sources.update(pins);session.report['source_hashes']=session.sources
  session.report['bounded_trace_IO_overlay']=copy.deepcopy(receipt)
 expert.ExpertSession.__init__=init;_INSTALLED=copy.deepcopy(receipt)
 return receipt
