import sys,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,'/tmp')
import c2200_closedloop_runner_v3 as driver
sys.path.insert(0,str(driver.DEFAULT_ROOT/'code/RLinf'))
from rlinf.workers.rollout.hf.hierarchical_recap_worker import GeneratedText,HierarchicalReCapRolloutWorker

class FakeActor:
 version='fake_readonly';versions={}
 tokenizer=SimpleNamespace(encode_target=lambda text:np.array([37,1]))
 def generate(self,obs,mode,context):
  if mode=='READ':text=json.dumps({'read_ids':['sample_identity']})
  elif mode=='WRITE':text=json.dumps({'operations':[{'op':'WRITE','slot_id':'sample_identity','value':'First sample left','confirmed':True}]})
  else:text='Pick up and hold the sample initially on the left.'
  return GeneratedText(text,[37,1],True)
 def actions(self,obs,context):return np.zeros((16,24),np.float32)
class FakeEnv:
 def __init__(self):self.i=0;self.closed=False
 def reset(self,**kw):
  self.i=0;return [{'state':np.zeros(24,np.float32),**{k:np.zeros((480,640,3),np.uint8) for k in ['full_image','left_wrist_image','right_wrist_image']}}]
 def step(self,actions):
  self.i+=1;obs=self.reset()[0] # restore i after fresh pixels
  self.i=getattr(self,'count',0)+1;self.count=self.i
  info={'physics_step_count':self.i*15,'executed_action_count':1,'observation_valid':True,'episode_success':False,'sim_time_s':self.i*.06,'step_count':self.i,
   'episode_metrics':{'verified_count':0,'returned_count':0,'task_events':[]},'action_trace':[{'raw_model_action24':np.asarray(actions)[0,0],'actual_final_drive_command24':np.asarray(actions)[0,0],'executed_physics_steps':15,'endpoint_command_written':True}]}
  return [obs],np.zeros((1,1)),np.zeros((1,1),bool),np.zeros((1,1),bool),[info]
 def close(self):self.closed=True
class Tests(unittest.TestCase):
 def test_cases_order_and_goal(self):
  cases=driver.cases(4);self.assertEqual(len(cases),4)
  for case in cases:self.assertIn('initially on the '+case['first_side'],case['goal']);self.assertLess(case['home_xy'][0][0],case['home_xy'][1][0])
  self.assertNotEqual(cases[0]['home_xy'],cases[2]['home_xy'])
 def test_config_override_does_not_mutate_original(self):
  cfg={'model':{'openpi':{}},'environment':{'task_config':{}},'rollout':{},'memory':{}}
  copy=driver.config_for_case(cfg,'/tmp/root','/tmp/ckpt',driver.cases()[1],'/tmp/out',128)
  self.assertEqual(cfg['environment']['task_config'],{})
  self.assertEqual(copy['environment']['task_config']['memory_task']['execution_order'],['sample_b','sample_a'])
  self.assertEqual(copy['rollout']['max_steps'],128)
 def test_privileged_policy_keys_blocked(self):
  with self.assertRaises(ValueError):driver.ensure_no_privileged({'state':0,'episode_metrics':{}})
 def test_malformed_motor_shape_refused(self):
  class Bad(FakeActor):
   def actions(self,*args):return np.zeros((1,24))
  with tempfile.TemporaryDirectory() as tmp:
   wrapper=driver.LoggedActor(Bad(),Path(tmp)/'p.jsonl',driver.cases()[0])
   with self.assertRaises(ValueError):wrapper.actions(FakeEnv().reset()[0], {})
 def test_oracle_does_not_repair_missing_eos(self):
  class Missing(FakeActor):
   def generate(self,*args):return GeneratedText('bad',[37],False)
  with tempfile.TemporaryDirectory() as tmp:
   wrapped=driver.LoggedActor(Missing(),Path(tmp)/'p.jsonl',driver.cases()[0],True)
   result=wrapped.generate({'state':np.zeros(24)},'SUBTASK',{})
   self.assertFalse(result.ended_with_eos)
   self.assertNotIn('oracle_subtask_override',json.loads((Path(tmp)/'p.jsonl').read_text()))
 def test_real_memory_tools_and_two_K1_receipts(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp); actor=driver.LoggedActor(FakeActor(),root/'policy.jsonl',driver.cases()[0]);env=driver.LoggedEnv(FakeEnv(),root/'info.jsonl')
   worker=HierarchicalReCapRolloutWorker(actor,env,{'sample_identity':'Observed identity and role of requested sample.'},max_steps=2,history_num_frames=1,strict_memory_protocol=True)
   episode=worker.collect_episode(root/'episode',episode_id='cpu_contract',goal=driver.cases()[0]['goal'],seed=1,evaluate=True)
   self.assertEqual(len(episode.steps),2);self.assertEqual(env.controls,2)
   self.assertEqual(episode.decisions[0].read_ids,['sample_identity'])
   self.assertEqual(episode.decisions[0].actor_memory['selected'][0]['value'],'First sample left')
   self.assertEqual(len((root/'info.jsonl').read_text().splitlines()),2)
   self.assertEqual(episode.steps[-1].physics_steps,15)
   self.assertTrue(episode.steps[-1].truncated)
class ExtraV2Tests(unittest.TestCase):
 def test_process_environment_local_tokenizer_cache(self):
  import os
  from unittest.mock import patch
  with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{},clear=True):
   tokenizer=Path(tmp)/'openpi/big_vision/paligemma_tokenizer.model';tokenizer.parent.mkdir(parents=True);tokenizer.write_bytes(b'local')
   receipt=driver.configure_process_environment('/tmp/testroot',tokenizer)
   self.assertEqual(receipt['OPENPI_DATA_HOME'],str(tokenizer.parent.parent))
   self.assertEqual(receipt['OMP_NUM_THREADS'],'4');self.assertEqual(receipt['OPENBLAS_NUM_THREADS'],'4')
   self.assertEqual(receipt['HF_HUB_OFFLINE'],'1');self.assertIn('/tmp/testroot/code/RLinf',receipt['PYTHONPATH'])
 def test_compare_loaded_adapter_exact_and_mismatch(self):
  import torch
  value=torch.nn.Parameter(torch.tensor([[.25,.5]],dtype=torch.float32))
  self.assertEqual(driver.compare_adapter_tensors({'lora':value},{'lora':value.detach().clone()}),1)
  with self.assertRaises(ValueError):driver.compare_adapter_tensors({'lora':value},{'lora':torch.zeros_like(value)})
  with self.assertRaises(ValueError):driver.compare_adapter_tensors({'lora':value},{})
 def test_inference_no_grad_guard(self):
  import torch
  with tempfile.TemporaryDirectory() as tmp:
   actor=FakeActor();actor.model=SimpleNamespace(training=False)
   wrapper=driver.LoggedActor(actor,Path(tmp)/'p.jsonl',driver.cases()[0],enforce_no_grad=True)
   with self.assertRaises(RuntimeError):wrapper.generate({'state':np.zeros(24)},'READ',{})
   with torch.no_grad():result=wrapper.generate({'state':np.zeros(24)},'READ',{})
   self.assertTrue(result.ended_with_eos)
   receipt=json.loads((Path(tmp)/'p.jsonl').read_text())['inference_guard']
   self.assertFalse(receipt['grad_enabled']);self.assertFalse(receipt['model_training'])
 def test_source_receipt_detects_changed_bytes(self):
  with tempfile.TemporaryDirectory() as tmp:
   file=Path(tmp)/'source';file.write_bytes(b'original')
   before=driver.source_files_receipt({'selected':file});file.write_bytes(b'changed')
   self.assertNotEqual(driver.source_files_receipt({'selected':file}),before)
class ExtraV3Tests(unittest.TestCase):
 def test_seed_helper_pairs_cpu_cuda_and_resets(self):
  calls=[]
  fake=SimpleNamespace(manual_seed=lambda seed:calls.append(('cpu',seed)),cuda=SimpleNamespace(manual_seed_all=lambda seed:calls.append(('cuda',seed))))
  self.assertEqual([driver.seed_motor_query(i,fake) for i in (0,1,0)],[7,8,7])
  self.assertEqual(calls,[('cpu',7),('cuda',7),('cpu',8),('cuda',8),('cpu',7),('cuda',7)])
  with self.assertRaises(ValueError):driver.seed_motor_query(-1,fake)
 def test_seed_reset_per_case_actual_CPU_noise(self):
  import torch
  class RandomActor(FakeActor):
   def actions(self,*args):return torch.randn((16,24)).numpy()
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);obs=FakeEnv().reset()[0]
   left=driver.LoggedActor(RandomActor(),root/'left.jsonl',driver.cases()[0]);right=driver.LoggedActor(RandomActor(),root/'right.jsonl',driver.cases()[1])
   before=torch.cuda.is_initialized()
   first=left.actions(obs,{});second=left.actions(obs,{});paired=right.actions(obs,{})
   self.assertTrue(np.array_equal(first,paired));self.assertFalse(np.array_equal(first,second))
   self.assertEqual(torch.cuda.is_initialized(),before)
   self.assertEqual([json.loads(x)['motor_noise_seed'] for x in (root/'left.jsonl').read_text().splitlines()],[7,8])
   self.assertEqual(json.loads((root/'right.jsonl').read_text())['motor_noise_seed'],7)
 def test_actual_failed_episode_metadata_transferred(self):
  class BadRead(FakeActor):
   def generate(self,*args):return GeneratedText('{"read_ids":"invalid"}',[37,1],True)
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);actor=driver.LoggedActor(BadRead(),root/'policy.jsonl',driver.cases()[0]);env=driver.LoggedEnv(FakeEnv(),root/'info.jsonl')
   worker=HierarchicalReCapRolloutWorker(actor,env,{'sample_identity':'Observed identity.'},max_steps=2,history_num_frames=1,strict_memory_protocol=True)
   with self.assertRaises(ValueError):worker.collect_episode(root/'episode',episode_id='badread',goal=driver.cases()[0]['goal'],seed=1,evaluate=True)
   receipt=driver.episode_evidence(root/'episode');self.assertEqual(receipt['kind'],'failed_episode.json')
   self.assertIn('READ schema',receipt['metadata']['collection_error'])
   diagnosis=driver.classify({'episode_metadata':receipt['metadata'],'episode_evidence':receipt},[],None)
   self.assertEqual(diagnosis['protocol_or_execution_error'],receipt['metadata']['collection_error'])
   self.assertEqual(diagnosis['all_failure_evidence']['episode_metadata_errors']['collection_error'],receipt['metadata']['collection_error'])
 def test_normal_terminal_reason_and_full_metadata_preserved(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);payload={'metadata':{'success':False,'custom_original_field':'preserved','failure_reason':'wrong_sample_order'},'decisions':[],'steps':[{'terminated':True,'truncated':False,'termination_reason':'wrong_sample_order','physics_steps':15}]}
   (root/'episode.json').write_text(json.dumps(payload));receipt=driver.episode_evidence(root)
   self.assertEqual(receipt['metadata'],payload['metadata'])
   diagnosis=driver.classify({'episode_metadata':receipt['metadata'],'episode_evidence':receipt},[],{})
   self.assertEqual(diagnosis['protocol_or_execution_error'],'wrong_sample_order')
   self.assertEqual(diagnosis['all_failure_evidence']['last_step_termination_reason'],'wrong_sample_order')
if __name__=='__main__':unittest.main()


