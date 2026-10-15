"""Small CPU proof for proposed future-only full-audit whitespace compaction."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
from bounded_trace_storage import LazyJSONList, write_json_stream, stream_sha

ROOT=Path(__file__).resolve().parent.parent
CURRENT=ROOT/'planning/bounded_trace_overlay.py'
PROPOSAL=ROOT/'preflight/proposals/audit_compact_future4_20261014/bounded_trace_overlay.py'

def writer(path):
 tree=ast.parse(path.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='write_audit')
 ns={'Path':Path,'write_json_stream':write_json_stream,'stream_sha':stream_sha}
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
 return ns['write_audit']

class CompactionTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.p=Path(self.tmp.name)
  self.pretty=writer(CURRENT);self.compact=writer(PROPOSAL)
 def tearDown(self):self.tmp.cleanup()
 def fixture(self):
  rows=[]
  for i in range(1,31):
   contacts=[{'links':['l_link1','l_finger'], 'normal_impulse_Ns':0., 'active':False,
      'points':[{'impulse_Ns':[0.,-0.,1e-20], 'normal_world':[0.,1.,0.],
          'point_world_m':[.001*j,.002,-.0], 'separation_m':1e-9}]} for j in range(6)]
   rows.append({'trace_index':i-1,'step':i,'phase':{'name':'close','gripper':True,'dense_call':3},
     'self_collisions':[],'robot_table_collisions':[],'bottle_non_touch_collisions':[],
     'allowed_exact_grasp_touch_links':['l_inspire_left_thumb_2'],
     'bottle_table_geometry':{'distance_m':.001,'collision':False},
     'actual_self_contacts':contacts,'relative_translation_drift_m':.0001,
     'unicode':'中文引号 " { } \\ é','none':None,'negative_zero':-0.})
  return {'passed':True,'recorded_samples':len(rows),'samples':rows,'source_hashes':{'original_source':'a'*64},
          'physical_scene_unchanged':True,'elapsed_s':123.456,'small':{'false':False,'int':3,'float':3.0}}
 def test_only_layout_literal_and_receipt_marker_ast_changed(self):
  old=ast.parse(CURRENT.read_text());new=ast.parse(PROPOSAL.read_text());sites=0;marker=0
  for node in ast.walk(new):
   if isinstance(node,ast.FunctionDef) and node.name=='write_audit':
    for call in ast.walk(node):
     if isinstance(call,ast.Call) and isinstance(call.func,ast.Name) and call.func.id=='write_json_stream':
      for keyword in call.keywords:
       if keyword.arg=='indent' and isinstance(keyword.value,ast.Constant) and keyword.value.value is None:
        keyword.value=ast.Constant(value=2);sites+=1
   if isinstance(node,ast.Dict):
    keep=[i for i,key in enumerate(node.keys) if not(isinstance(key,ast.Constant) and key.value=='full_audit_json_whitespace_layout')]
    marker+=len(node.keys)-len(keep);node.keys=[node.keys[i] for i in keep];node.values=[node.values[i] for i in keep]
  self.assertEqual((sites,marker),(1,1))
  self.assertEqual(ast.dump(old,include_attributes=False),ast.dump(new,include_attributes=False))
 def test_complete_values_and_row_order(self):
  data=self.fixture();old=self.p/'old.json';new=self.p/'new.json';self.pretty(old,data);self.compact(new,data)
  self.assertEqual(json.loads(old.read_text()),json.loads(new.read_text()))
  self.assertEqual(json.loads(new.read_text()),data)
  self.assertLess(new.stat().st_size,old.stat().st_size*.60)
 def test_original_dtypes_and_signed_float_exact(self):
  data=self.fixture();target=self.p/'audit.json';self.compact(target,data);result=json.loads(target.read_text())
  self.assertIs(type(result['passed']),bool);self.assertIs(type(result['small']['int']),int)
  self.assertIs(type(result['small']['float']),float);self.assertIsNone(result['samples'][0]['none'])
  self.assertEqual(struct.pack('d',result['samples'][0]['negative_zero']),struct.pack('d',-0.))
 def test_lazy_list_all_rows_preserved_and_summary_sha(self):
  data=self.fixture();rows=data['samples'];source=self.p/'rows.json';source.write_text(json.dumps(rows))
  data['samples']=LazyJSONList(source,chunk_bytes=17);target=self.p/'actual36_geometry_audit.json';self.compact(target,data)
  actual=json.loads(target.read_text());self.assertEqual(actual['samples'],rows)
  summary=json.loads(target.with_name('actual36_geometry_audit_summary.json').read_text())
  binding=summary.pop('bounded_full_audit_file')
  self.assertEqual(binding,{'path':str(target.resolve()),'sha256':stream_sha(target),'all_samples_written':30})
  self.assertEqual(summary,{k:v for k,v in actual.items() if k!='samples'})
 def test_summary_indentation_unchanged(self):
  target=self.p/'audit.json';data=self.fixture();self.compact(target,data)
  summary=target.with_name('audit_summary.json');values=json.loads(summary.read_text())
  self.assertEqual(summary.read_text(),json.dumps(values,indent=2,allow_nan=False))
 def test_newline_and_original_object_not_mutated(self):
  data=self.fixture();original=copy.deepcopy(data);target=self.p/'audit.json';self.compact(target,data,newline=True)
  self.assertTrue(target.read_bytes().endswith(b'\n'));self.assertEqual(data,original)
 def test_nonfinite_rejection_same_and_partial_not_published(self):
  for number in (float('nan'),float('inf'),-float('inf')):
   for index,method in enumerate((self.pretty,self.compact)):
    target=self.p/f'bad{index}_{repr(number)}.json'
    with self.assertRaises(ValueError):method(target,{'nonfinite':number})
    self.assertFalse(target.exists())
 def test_actual_staged_summary_loader_reads_new_compact_file(self):
  data=self.fixture();target=self.p/'actual36_geometry_audit.json';self.compact(target,data)
  helper=self.p/'bounded_trace_overlay.py';helper.write_bytes(PROPOSAL.read_bytes())
  report={'actual36_geometry_audit':{k:v for k,v in data.items() if k!='samples'},
          'source_hashes':{str(helper):stream_sha(helper)}}
  (self.p/'expert_report.json').write_text(json.dumps(report))
  spec=importlib.util.spec_from_file_location('_compact_fixture_staged_export',ROOT/'preflight/streaming_export_staged.py')
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  self.assertEqual(module.read_json(target,limit=1),report['actual36_geometry_audit'])
 def test_corrupt_summary_sha_is_rejected_by_staged_loader(self):
  data=self.fixture();target=self.p/'actual36_geometry_audit.json';self.compact(target,data)
  summary=target.with_name('actual36_geometry_audit_summary.json');value=json.loads(summary.read_text())
  value['bounded_full_audit_file']['sha256']='0'*64;summary.write_text(json.dumps(value))
  spec=importlib.util.spec_from_file_location('_bad_compact_fixture_staged_export',ROOT/'preflight/streaming_export_staged.py')
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  with self.assertRaises((ValueError,AssertionError)):module.read_json(target,limit=1)

if __name__=='__main__':unittest.main(verbosity=2)
