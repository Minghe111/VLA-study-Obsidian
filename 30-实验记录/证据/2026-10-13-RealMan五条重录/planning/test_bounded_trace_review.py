"""Small regressions for independent override ownership and bounded writes."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from bounded_trace_storage import LazyJSONList, copy_with_overrides, write_json_stream

class ReviewTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
 def tearDown(self):self.temp.cleanup()
 def rows(self):
  path=self.root/'rows.json';path.write_text(json.dumps([{'step':1},{'step':2}]))
  return LazyJSONList(path,chunk_bytes=9)
 def test_existing_nested_override_independent(self):
  original=self.rows();original[0]={'step':1,'nested':{'items':[3,4]}}
  clone=copy_with_overrides(original);clone[0]['nested']['items'].append(9)
  self.assertEqual(original[0]['nested']['items'],[3,4])
  original[0]['nested']['items'].append(8)
  self.assertEqual(clone[0]['nested']['items'],[3,4,9])
 def test_supplied_override_copied(self):
  original=self.rows();supplied={1:{'step':2,'nested':{'items':[5]}}}
  clone=copy_with_overrides(original,supplied);supplied[1]['nested']['items'].append(6)
  self.assertEqual(clone[1]['nested']['items'],[5]);self.assertEqual(original[1],{'step':2})
 def test_large_single_encoder_chunk_has_bounded_writes(self):
  target=self.root/'audit.json';data={'large':'x'*230000+'中文','remaining':[1,2]}
  sizes=[];original_open=Path.open
  class CountWrites:
   def __init__(self,stream):self.stream=stream
   def __enter__(self):self.stream.__enter__();return self
   def __exit__(self,*args):return self.stream.__exit__(*args)
   def __getattr__(self,name):return getattr(self.stream,name)
   def write(self,text):sizes.append(len(text.encode('utf-8')));return self.stream.write(text)
  def tracked_open(path,*args,**kwargs):
   stream=original_open(path,*args,**kwargs)
   return CountWrites(stream) if path.name.endswith('.writing') else stream
  with patch.object(Path,'open',tracked_open):write_json_stream(target,data)
  self.assertTrue(sizes);self.assertLessEqual(max(sizes),65536)
  self.assertEqual(target.read_text(),json.dumps(data,indent=2,allow_nan=False))
 def test_clone_lazy_slice_overlay_mapping(self):
  source=self.rows();source[1]={'step':2,'review':{'x':[7]}}
  view=source[1:];clone=copy_with_overrides(view);clone[0]['review']['x'].append(8)
  self.assertEqual(source[1]['review']['x'],[7])
  self.assertEqual(list(clone),[{'step':2,'review':{'x':[7,8]}}])

if __name__=='__main__':unittest.main(verbosity=2)
