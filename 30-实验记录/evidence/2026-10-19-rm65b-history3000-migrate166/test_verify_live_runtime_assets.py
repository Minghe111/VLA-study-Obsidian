from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import tempfile,json,hashlib,importlib.util,unittest,os
HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('verify',HERE/'verify_live_runtime_assets.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def digest(b):return hashlib.sha256(b).hexdigest()
def write(path,b):path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b);return path
class VerificationTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(prefix='live_runtime_audit_fixture_');self.task=Path(self.tmp.name)
  self.source=self.task/'source_tree';self.inputs=self.task/'migration_tools';self.inputs.mkdir()
  base=self.source/m.FULL_BASE_RELATIVE;self.base=base
  cfg=base.parent.parent/'effective_config.yaml';fit=self.source/m.FIT_RELATIVE
  self.rows=[]
  for p,old,new,structured in [(base,b'complete-full-weights',None,False),(base.parent/'version.json',b'{"base":true}',None,False),(cfg,b'original-config',b'relocated-config',True),(fit,b'original-monitor',b'approved-uuid-monitor',False)]:
   write(p,new if new is not None else old);rel=p.relative_to(self.source)
   self.rows.append({'path':str(m.SOURCE_PREFIX/rel),'relative':str(rel),'target':str(p),'sha256':digest(old),'size':len(old),'structured_rewrite':structured})
  cfgrow=self.rows[2];backup=write(self.task/'reports/structured_relocation_v1/original_structured/000000.yaml',b'original-config')
  self.reloc={'status':'relocated_structured_copies_only','path_map':{str(m.SOURCE_PREFIX):str(self.source)},'files':[{'source':cfgrow['path'],'target':cfgrow['target'],'sha256':cfgrow['sha256'],'target_sha256':digest(b'relocated-config'),'original_backup':str(backup)}]}
  self.inventory={'source_prefix':str(m.SOURCE_PREFIX),'target_prefix':str(self.source),'entries':self.rows}
  self.vendor={'status':'exact_local_runtime_python_and_resources_vendored','entries':[]}
  for rel,data in [('openpi/__init__.py',b'openpi'),('openpi_client/__init__.py',b'client')]:
   write(self.task/'vendor'/rel,data);self.vendor['entries'].append({'relative':rel,'sha256':digest(data),'bytes':len(data)})
  compact=self.source/m.COMPACT_RELATIVE;self.compact=compact
  files={}
  for name,data in [('adapter_weights.pt',b'adapter'),('trainer_state.pt',b'state')]:
   write(compact/name,data);files[name]={'bytes':len(data),'sha256':digest(data)}
  self.version={'checkpoint_format':m.COMPACT_FORMAT,'global_step':2000,'files':files,'base_checkpoint':{'path':str(base.parent),'full_weights_sha256':self.rows[0]['sha256'],'version_sha256':self.rows[1]['sha256'],'effective_config_sha256':digest(b'relocated-config')}}
  self.args=SimpleNamespace(task_root=self.task,expected_hostname='fake-test',inventory=None,vendor_manifest=None,relocation_receipt=None,compact_checkpoint=None,output=None)
  self.patches=[patch.object(m,'identity_guard',return_value={'scope':'CPU-test-fixture'}),patch.object(m,'EXPECTED_FILE_COUNT',4),patch.object(m,'EXPECTED_STRUCTURED_COUNT',1),patch.object(m,'EXPECTED_VENDOR_COUNT',2),patch.object(m,'FIT_SHA',digest(b'approved-uuid-monitor')),patch.object(m,'FIT_BYTES',len(b'approved-uuid-monitor'))]
  for p in self.patches:p.start()
  self.sync()
 def tearDown(self):
  for p in reversed(self.patches):p.stop()
  self.tmp.cleanup()
 def sync(self):
  for p,v in [(self.inputs/'package_inventory.json',self.inventory),(self.inputs/'vendor_manifest.json',self.vendor),(self.task/'reports/structured_relocation_v1/receipt.json',self.reloc),(self.compact/'version.json',self.version)]:write(p,json.dumps(v).encode())
 def run_audit(self):self.sync();return m.audit(self.args)
 def test_all_expected_assets_and_terminal_rehash_pass(self):
  r=self.run_audit();self.assertEqual(r['status'],'passed',r['issues']);self.assertEqual(r['counts']['inventory_checked'],4);self.assertTrue(r['C2200_stable_across_complete_audit']);self.assertFalse(r['GPU_used']);self.assertFalse(r['model_loaded'])
 def test_truncated_original_weight_fails(self):
  write(self.base,b'complete');r=self.run_audit();self.assertEqual(r['status'],'failed');self.assertTrue(any(x['path']==str(self.base) for x in r['issues'] if 'path'in x))
 def test_wrong_structured_original_seal_fails(self):
  self.reloc['files'][0]['sha256']='0'*64;self.assertEqual(self.run_audit()['status'],'failed')
 def test_external_receipt_target_rejected(self):
  self.reloc['files'][0]['target']='/etc/passwd';r=self.run_audit();self.assertEqual(r['status'],'failed');self.assertIn('does not seal',r['issues'][-1]['message'])
 def test_external_vendor_relative_rejected(self):
  self.vendor['entries'][0]['relative']='../../etc/passwd';r=self.run_audit();self.assertEqual(r['status'],'failed');self.assertIn('vendor package',r['issues'][-1]['message'])
 def test_wrong_derived_trainer_seal_fails(self):
  write(self.compact/'trainer_state.pt',b'badstate');self.assertEqual(self.run_audit()['status'],'failed')
 def test_compact_base_path_escape_rejected(self):
  self.version['base_checkpoint']['path']='/tmp/other-model';r=self.run_audit();self.assertEqual(r['status'],'failed');self.assertIn('unexpected C2200 base',r['issues'][-1]['message'])
 def test_symlink_input_rejected(self):
  p=self.task/'link';p.symlink_to('/etc/passwd')
  with self.assertRaises(ValueError):m.safe_path(p,self.task)
 def test_original_backup_tamper_fails(self):
  write(Path(self.reloc['files'][0]['original_backup']),b'altered original');self.assertEqual(self.run_audit()['status'],'failed')
 def test_original_readonly_bytes_not_mutated(self):
  before={p:p.read_bytes() for p in self.task.rglob('*') if p.is_file()};r=self.run_audit();self.assertEqual(r['status'],'passed');self.assertTrue(all(p.read_bytes()==b for p,b in before.items()))
 def test_identity_guard_refuses_wrong_authorized_root(self):
  # Call the unpatched function by loading a fresh module.
  sp=importlib.util.spec_from_file_location('fresh_verify',HERE/'verify_live_runtime_assets.py');fresh=importlib.util.module_from_spec(sp);sp.loader.exec_module(fresh)
  with self.assertRaisesRegex(ValueError,'fixed authorized'):fresh.identity_guard(self.task,'fake-test')
if __name__=='__main__':
 s=unittest.defaultTestLoader.loadTestsFromTestCase(VerificationTests);r=unittest.TextTestRunner(verbosity=2).run(s)
 receipt={'status':'passed' if r.wasSuccessful() else 'failed','tests_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'GPU_used':False,'remote_actions':False,'production_inventory_full_hash_executed':False}
 (HERE/'verify_live_runtime_assets_cpu_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');raise SystemExit(not r.wasSuccessful())
