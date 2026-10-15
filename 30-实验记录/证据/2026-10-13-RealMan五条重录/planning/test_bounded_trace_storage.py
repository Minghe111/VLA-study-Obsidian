import json,tempfile,unittest
from pathlib import Path
from bounded_trace_storage import *
class StorageTests(unittest.TestCase):
 def setUp(self):self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
 def tearDown(self):self.temp.cleanup()
 def make(self,rows):
  p=self.root/'data.json';p.write_text(json.dumps(rows,ensure_ascii=False));return p
 def test_unicode_small_chunks_offsets(self):
  rows=[{'step':i,'s':'中文 } \\" [ 和é','x':[1.,2.]} for i in range(1,15)]
  data=LazyJSONList(self.make(rows),chunk_bytes=7)
  self.assertIsInstance(data,list);self.assertEqual(list(data),rows);self.assertEqual(data[-1],rows[-1]);self.assertEqual(list(data[2:-1]),rows[2:-1])
 def test_pending_late_mutation_and_exact_bytes(self):
  rows=[];p=self.root/'data.json';data=PendingTraceList(p,tail_rows=50)
  for i in range(80):
   row={'step':i,'v':0.};rows.append(row);data.append(row);data[-1]['late_environment']={'x':i}
   if i>=49:self.assertEqual(data[-50:],[r for r in rows[-50:]])
  data.flush_json();self.assertEqual(p.read_text(),json.dumps(rows));self.assertEqual(list(data),rows);data.flush_json()
  with self.assertRaises(RuntimeError):data.append({})
 def test_cow_not_mutate_original(self):
  original=LazyJSONList(self.make([{'step':1},{'step':2}]),chunk_bytes=9);clone=copy_with_overrides(original);clone[1]={'step':2,'reviewed':True}
  self.assertEqual(original[1],{'step':2});self.assertTrue(clone[1]['reviewed'])
 def test_stream_indent_exact(self):
  rows=[{'step':i,'k':[1,2],'unicode':'中文'} for i in range(10)];lazy=LazyJSONList(self.make(rows),chunk_bytes=9);p=self.root/'audit.json';data={'passed':True,'samples':lazy,'after':'value'}
  write_json_stream(p,data);self.assertEqual(p.read_text(),json.dumps({'passed':True,'samples':rows,'after':'value'},indent=2,allow_nan=False))
 def test_truncated_and_trailing_fail(self):
  for value in ('[{"a":1}', '[{"a":1},]', '[{"a":1}] junk','{"a":1}'):
   p=self.root/'bad.json';p.write_text(value)
   with self.assertRaises(ValueError):LazyJSONList(p,chunk_bytes=3)
 def test_record_limit(self):
  p=self.make([{'blob':'x'*128}])
  with self.assertRaises(ValueError):LazyJSONList(p,max_record_bytes=32,chunk_bytes=9)
 def test_disk_audit_every_row(self):
  p=self.root/'auditrows.json';a=DiskJSONAppender(p)
  for i in range(300):a.append({'step':i})
  self.assertEqual(len(a.tail),50);b=a.close();self.assertEqual(len(b),300);self.assertEqual(b[-1]['step'],299)
 def test_unsealed_retains_part(self):
  p=self.root/'data.json';a=PendingTraceList(p);a.append({'step':1});a.append({'step':2});self.assertFalse(p.exists());self.assertTrue(a.part.exists());a.flush_json()
if __name__=='__main__':unittest.main()
