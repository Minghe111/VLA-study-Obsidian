import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('runner',Path(__file__).with_name('old_task_paired_eval_v2.py'))
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
ROOT=runner.DEFAULT_ROOT
sys.path.insert(0,str(ROOT/'code/RoboTwin'))
from integrations.realman_rm65b_gen4.rl_contract import ChunkDriver,EnvConfig

class Session:
    def __init__(self,success_at=None):
        self.physics_step_count=0;self.metrics={'success':False,'failure_reason':None};self.contract={'seed':2010};self.calls=0;self.success_at=success_at
    def capture(self):return {'state':np.zeros(24,dtype=np.float32)}
    def apply(self,action):
        start=self.physics_step_count;self.calls+=1
        ticks=3 if self.calls==self.success_at else 15
        self.physics_step_count+=ticks
        self.metrics['success']=self.calls==self.success_at
        return {'requested_physics_steps':15,'executed_physics_steps':ticks,'physics_step_start':start,'physics_step_end':self.physics_step_count,'raw_model_action24':np.asarray(action).tolist()}

class Tests(unittest.TestCase):
    def test_plan_k16_never_creates_output_or_executes(self):
        with tempfile.TemporaryDirectory() as parent:
            p=Path(parent)/'not-created';result=runner.main(['--variant','baseline','--output',str(p),'--execute-actions','16'])
            self.assertFalse(p.exists());self.assertFalse(result['execute']);self.assertEqual(result['K_exec'],16)
            self.assertEqual(result['max_targets_per_episode'],400);self.assertEqual(result['optimizer_updates'],0)

    def test_real_chunk_driver_400_targets_is25queries_6000ticks(self):
        session=Session();driver=ChunkDriver(session,EnvConfig(max_episode_steps=400));old_ticks=old_steps=queries=0
        while old_steps<400:
            chunk=runner.validate_chunk(np.zeros((1,16,24)),execute_actions=16,remaining=400-old_steps)
            obs,rewards,terms,truncs,info=driver.step(chunk[0]);queries+=1
            receipt=runner.validate_step(info,seed=2010,previous_ticks=old_ticks,previous_targets=old_steps,requested_k=chunk.shape[1],budget=400)
            self.assertEqual(receipt['executed_target_count'],16);self.assertEqual(receipt['padding_target_count'],0)
            old_steps=info['step_count'];old_ticks=info['physics_step_count']
        self.assertEqual((old_steps,old_ticks,queries),(400,6000,25));self.assertTrue(truncs[-1]);self.assertFalse(terms.any())

    def test_real_driver_terminal_midchunk_masks_tail_and_keeps_partial_tick(self):
        driver=ChunkDriver(Session(success_at=17),EnvConfig(max_episode_steps=400))
        driver.step(np.zeros((16,24)))
        _,rewards,terms,truncs,info=driver.step(np.zeros((16,24)))
        receipt=runner.validate_step(info,seed=2010,previous_ticks=240,previous_targets=16,requested_k=16,budget=400)
        self.assertEqual(receipt['executed_target_count'],1);self.assertEqual(receipt['padding_target_count'],15)
        self.assertEqual(receipt['per_target_physics_ticks'],[3]);self.assertTrue(receipt['partial_final_target'])
        self.assertEqual(info['step_count'],17);self.assertEqual(info['physics_step_count'],243)
        np.testing.assert_array_equal(rewards,[1]+[0]*15);np.testing.assert_array_equal(terms,[True]+[False]*15)

    def test_nonmultiple_remaining_budget_is_cropped_not_overrequested(self):
        driver=ChunkDriver(Session(),EnvConfig(max_episode_steps=397));old_steps=old_ticks=0
        while old_steps<397:
            chunk=runner.validate_chunk(np.ones((1,16,24)),execute_actions=16,remaining=397-old_steps)
            _,_,_,truncs,info=driver.step(chunk[0]);runner.validate_step(info,seed=2010,previous_ticks=old_ticks,previous_targets=old_steps,requested_k=chunk.shape[1],budget=397)
            old_steps=info['step_count'];old_ticks=info['physics_step_count']
        self.assertEqual(chunk.shape,(1,13,24));self.assertEqual(old_steps,397);self.assertTrue(truncs[-1])

    def test_k1_still_executes_only_first_and_source_not_mutated(self):
        source=np.arange(384,dtype=np.float32).reshape(1,16,24);actual=runner.validate_chunk(source,execute_actions=1,remaining=400)
        self.assertEqual(actual.shape,(1,1,24));np.testing.assert_array_equal(actual,source[:,:1]);actual[0,0,0]=-99;self.assertEqual(source[0,0,0],0)

    def test_bad_valid_mask_or_fake_tail_reward_is_rejected(self):
        driver=ChunkDriver(Session(success_at=1),EnvConfig(max_episode_steps=400));*_,info=driver.step(np.zeros((16,24)))
        info['valid_action_mask'][0]=False;info['valid_action_mask'][1]=True
        with self.assertRaises(ValueError):runner.validate_step(info,seed=2010,previous_ticks=0,requested_k=16)
        info['valid_action_mask'][0]=True;info['valid_action_mask'][1]=False;info['chunk_rewards'][1]=1
        with self.assertRaises(ValueError):runner.validate_step(info,seed=2010,previous_ticks=0,requested_k=16)

    def test_bad_shapes_nonfinite_or_exhausted_budget_are_rejected(self):
        for x in [np.zeros((1,1,24)),np.zeros((1,16,32)),np.full((1,16,24),float('nan'))]:
            with self.assertRaises(ValueError):runner.validate_chunk(x,execute_actions=16)
        with self.assertRaises(ValueError):runner.validate_chunk(np.zeros((1,16,24)),execute_actions=16,remaining=0)

if __name__=='__main__':unittest.main()
