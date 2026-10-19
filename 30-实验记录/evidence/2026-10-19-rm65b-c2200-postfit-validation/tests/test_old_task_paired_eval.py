import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('runner',Path(__file__).with_name('old_task_paired_eval.py'))
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)

class Tests(unittest.TestCase):
    def test_default_plan_has_no_files_or_training(self):
        with tempfile.TemporaryDirectory() as parent:
            out=Path(parent)/'not-created'
            result=runner.main(['--variant','baseline','--output',str(out)])
            self.assertFalse(out.exists());self.assertFalse(result['execute']);self.assertEqual(result['optimizer_updates'],0)
            self.assertEqual(result['seeds'],[2010,2013]);self.assertEqual(result['nominal_max_sim_seconds_per_episode'],24.)

    def test_chunk_executes_only_first_without_modifying_source(self):
        value=np.arange(384,dtype=np.float32).reshape(1,16,24);copy=value.copy()
        result=runner.validate_chunk(value)
        np.testing.assert_array_equal(result,value[:,:1]);result[0,0,0]=-100
        np.testing.assert_array_equal(value,copy)

    def test_bad_shape_and_nonfinite_chunk_fail(self):
        for value in (np.zeros((1,1,24)),np.zeros((1,16,32)),np.full((1,16,24),float('nan'))):
            with self.assertRaises(ValueError):runner.validate_chunk(value)

    def test_noise_is_paired_by_query_and_wraps(self):
        self.assertEqual([runner.noise_seed(7,i) for i in range(3)],[7,8,9])
        self.assertEqual(runner.noise_seed(2**63-1,1),0)
        with self.assertRaises(ValueError):runner.noise_seed(7,-1)

    def test_partial_final_physics_tick_allowed_but_not_nonterminal(self):
        info={'executed_action_count':1,'valid_action_mask':[True],'episode_contract':{'seed':2010},'physics_step_count':18,'sim_time_s':.072,'failure_reason':None,'episode_terminated':True,'episode_truncated':False}
        self.assertEqual(runner.validate_step(info,seed=2010,previous_ticks=15),3)
        info['episode_terminated']=False
        with self.assertRaises(ValueError):runner.validate_step(info,seed=2010,previous_ticks=15)

    def test_wrong_seed_and_extra_executed_actions_rejected(self):
        info={'executed_action_count':1,'valid_action_mask':[True],'episode_contract':{'seed':2010},'physics_step_count':15,'sim_time_s':.06,'failure_reason':None,'episode_terminated':False,'episode_truncated':False}
        with self.assertRaises(ValueError):runner.validate_step(info,seed=2013,previous_ticks=0)
        info['executed_action_count']=2
        with self.assertRaises(ValueError):runner.validate_step(info,seed=2010,previous_ticks=0)

    def test_limits_refuse_changed_protocol(self):
        for extra in (['--budget','401'],['--seeds','2010','2010'],['--seeds','1001','1002']):
            with self.assertRaises(SystemExit):runner.parser_args(['--variant','baseline','--output','/unused',*extra])

if __name__=='__main__':unittest.main()
