import json,tempfile,unittest
from pathlib import Path
import streaming_export_staged as m
class BoundedAuditBindingTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
  self.audit=self.root/'actual36_geometry_audit.json';self.audit.write_text(json.dumps({'samples':[{'value':'x'*300}]}))
  helper=self.root/'bounded_trace_overlay.py';helper.write_text('# synthetic producer binding only')
  self.meta={'passed':True,'recorded_samples':1}
  self.report={'actual36_geometry_audit':self.meta,'source_hashes':{str(helper):m.base.sha(helper)}}
  self.summary={**self.meta,'bounded_full_audit_file':{'path':str(self.audit),'sha256':m.base.sha(self.audit),'all_samples_written':1}}
  self.save()
 def tearDown(self):self.tmp.cleanup()
 def save(self):
  (self.root/'expert_report.json').write_text(json.dumps(self.report))
  self.audit.with_name(self.audit.stem+'_summary.json').write_text(json.dumps(self.summary))
 def test_valid_complete_binding(self):self.assertEqual(m.read_json(self.audit,limit=32),self.meta)
 def test_full_file_mutation_rejected(self):
  self.audit.write_text(self.audit.read_text()+' ')
  with self.assertRaises(Exception):m.read_json(self.audit,limit=32)
 def test_incomplete_rows_rejected(self):
  self.summary['bounded_full_audit_file']['all_samples_written']=0;self.save()
  with self.assertRaises(Exception):m.read_json(self.audit,limit=32)
 def test_report_mismatch_rejected(self):
  self.summary['passed']=False;self.save()
  with self.assertRaises(Exception):m.read_json(self.audit,limit=32)
 def test_unregistered_producer_rejected(self):
  self.report['source_hashes']={};self.save()
  with self.assertRaises(Exception):m.read_json(self.audit,limit=32)
 def test_original_small_read_unchanged(self):
  self.assertEqual(m.read_json(self.audit),json.loads(self.audit.read_text()))
if __name__=='__main__':unittest.main()
