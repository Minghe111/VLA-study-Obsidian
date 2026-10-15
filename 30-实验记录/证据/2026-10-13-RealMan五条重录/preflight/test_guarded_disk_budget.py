"""CPU-only disk reserve tests; all manager/process calls are mocked.

No running unit, collector, source or real destination is modified.
"""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('guard_disk_review',HERE/'run_guarded_stage.py')
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)

class DiskBudgetTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
  self.runtime=self.root/'runtime';self.runtime.mkdir()
  self.vault=self.root/'vault';self.vault.mkdir()
  self.script=self.root/'tool.py';self.script.write_text('pass\n')
 def tearDown(self):self.temp.cleanup()
 def test_exact_reserves_and_interval(self):
  self.assertEqual(g.MIN_DISK_FREE,32*1024**3)
  self.assertEqual(g.FLOOR_DISK_FREE,16*1024**3)
  self.assertEqual(g.SAMPLE_SEC,2.)
 def test_nonexisting_destination_resolves_existing_ancestor(self):
  destination=self.root/'new'/'nested'/'output'
  with patch.object(g,'TASK_E',self.vault):
   paths=g.output_filesystems(self.runtime,['tool','--output',str(destination)])
  self.assertEqual(paths,(self.runtime,))  # same device is checked only once
  self.assertTrue(all(path.exists() for path in paths))
 def test_different_mount_and_dedup(self):
  output=self.root/'other_mount'/'future'/'out';export=self.root/'third_mount'/'future'
  mounts={self.runtime:1,self.vault:1,self.root/'other_mount':2,self.root/'third_mount':3}
  original_exists=Path.exists;original_stat=Path.stat
  def exists(path):return path in mounts or original_exists(path)
  def stat(path,*args,**kwargs):
   return SimpleNamespace(st_dev=mounts[path]) if path in mounts else original_stat(path,*args,**kwargs)
  with patch.object(g,'TASK_E',self.vault),patch.object(Path,'exists',exists),patch.object(Path,'stat',stat):
   paths=g.output_filesystems(self.runtime,['tool','--output',str(output),'--candidate',str(output.parent),'--export-output',str(export)])
  self.assertEqual(paths,(self.runtime,self.root/'other_mount',self.root/'third_mount'))
 def test_symlink_destination_ancestor(self):
  target=self.root/'actual_destination';target.mkdir();alias=self.root/'alias';alias.symlink_to(target,target_is_directory=True)
  mounts={self.runtime:1,self.vault:1,target:2};original_stat=Path.stat
  def stat(path,*args,**kwargs):
   return SimpleNamespace(st_dev=mounts[path]) if path in mounts else original_stat(path,*args,**kwargs)
  with patch.object(g,'TASK_E',self.vault),patch.object(Path,'stat',stat):
   paths=g.output_filesystems(self.runtime,['tool','--output',str(alias/'new'/'out')])
  self.assertEqual(paths,(self.runtime,target))
 def test_missing_path_is_rejected(self):
  with self.assertRaisesRegex(ValueError,'Missing output path'):
   g.output_filesystems(self.runtime,['tool','--output'])
 def test_disk_free_each_filesystem(self):
  calls=[]
  def usage(path):calls.append(path);return SimpleNamespace(free=32*g.GIB if path==self.runtime else 40*g.GIB)
  with patch.object(g.shutil,'disk_usage',usage):values=g.disk_free((self.runtime,self.vault))
  self.assertEqual(calls,[self.runtime,self.vault])
  self.assertEqual(values,{str(self.runtime):32*g.GIB,str(self.vault):40*g.GIB})
 def common(self):
  inactive={'LoadState':'not-found','ActiveState':'inactive','ControlGroup':''}
  return [patch.object(g,'LOCK',self.root/'private_test.lock'),patch.object(g,'RUNS',self.root/'private_runs'),
          patch.object(g,'validate_command',return_value=(self.runtime,self.script)),
          patch.object(g,'unit_info',return_value=inactive),patch.object(g,'known_task_processes',return_value=[]),
          patch.object(g,'memory_available',return_value=24*g.GIB),
          patch.object(g,'output_filesystems',return_value=(self.runtime,self.vault))]
 def test_preflight_insufficient_other_destination_refuses_before_launch(self):
  from contextlib import ExitStack
  with ExitStack() as stack:
   for p in self.common():stack.enter_context(p)
   stack.enter_context(patch.object(g,'disk_free',return_value={str(self.runtime):100*g.GIB,str(self.vault):32*g.GIB-1}))
   launch=stack.enter_context(patch.object(g.subprocess,'Popen',side_effect=AssertionError('Must never launch')))
   with self.assertRaisesRegex(RuntimeError,'32 GiB disk free'):
    g.execute('export',[g.PYTHON,str(self.script)],run=True)
   self.assertFalse(launch.called)
 def test_exact_start_reserve_passes_readonly_preflight(self):
  from contextlib import ExitStack
  with ExitStack() as stack:
   for p in self.common():stack.enter_context(p)
   stack.enter_context(patch.object(g,'disk_free',return_value={str(self.runtime):100*g.GIB,str(self.vault):32*g.GIB}))
   launch=stack.enter_context(patch.object(g.subprocess,'Popen',side_effect=AssertionError('Read-only')))
   receipt,code=g.execute('export',[g.PYTHON,str(self.script)],run=False)
   self.assertEqual(code,0);self.assertEqual(receipt['disk_start_minimum_bytes'],32*g.GIB)
   self.assertEqual(receipt['disk_stop_floor_bytes'],16*g.GIB);self.assertFalse(launch.called)
 def monitor(self,free_sequence,expected_samples):
  from contextlib import ExitStack
  states=[{str(self.runtime):100*g.GIB,str(self.vault):value*g.GIB} for value in free_sequence]
  disk_reads=iter([{str(self.runtime):100*g.GIB,str(self.vault):40*g.GIB},*states,states[-1]])
  fake=SimpleNamespace(stopped=False)
  fake.poll=lambda:143 if fake.stopped else None
  fake.wait=lambda **kw:143
  def stop(_info):fake.stopped=True
  with ExitStack() as stack:
   for p in self.common():stack.enter_context(p)
   stack.enter_context(patch.object(g,'disk_free',side_effect=lambda _paths:next(disk_reads)))
   stack.enter_context(patch.object(g,'systemctl',return_value=SimpleNamespace(returncode=0)))
   stack.enter_context(patch.object(g.subprocess,'Popen',return_value=fake))
   stack.enter_context(patch.object(g,'process_rows',return_value=[]))
   stack.enter_context(patch.object(g.time,'sleep',return_value=None))
   stopped=stack.enter_context(patch.object(g,'safe_stop',side_effect=stop))
   receipt,code=g.execute('export',[g.PYTHON,str(self.script)],run=True)
  self.assertEqual(code,143);self.assertFalse(receipt['passed']);self.assertEqual(stopped.call_count,1)
  self.assertEqual(receipt['monitor_stop_reason'],'Task destination disk free below 16 GiB for three samples')
  lines=Path(receipt['receipt_directory'],'resources.jsonl').read_text().splitlines()
  self.assertEqual(len(lines),expected_samples)
  self.assertTrue(all('disk_free_bytes' in json.loads(line) for line in lines))
 def test_three_low_samples_stop_only_own_mocked_unit(self):self.monitor([15,15,15],3)
 def test_floor_equality_does_not_count_as_low(self):self.monitor([16,15,15,15],4)
 def test_recovery_resets_low_streak(self):self.monitor([15,17,15,15,15],5)

if __name__=='__main__':unittest.main(verbosity=2)
