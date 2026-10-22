"""CPU-only failure controls for the isolated full collection entry.

Fixtures are in-memory or temporary AST files, never robot data or HDF5.
"""
import copy
import json
from pathlib import Path
import tempfile
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import full_cap_collection as entry

def main():
    cfg=entry.load_config();results=[]
    def check(name,fn):
        fn();results.append({'check':name,'passed':True})
    def rejects(fn,kind=entry.FullCollectionRejected):
        try:fn()
        except kind:return
        raise AssertionError('Expected fail-closed rejection')
    metrics={key:True for key in cfg['required_true_final_metrics']}
    metrics.update(cfg['required_exact_final_metrics']);metrics.update(cfg['minimum_final_metrics'])
    metrics.update({key:0. for key in cfg['maximum_exclusive_final_metrics']})
    metrics.update(failure_reason=None,cap_state={'released':True})
    def gate_false(value,error=None):assert not entry.final_gate(value,error,cfg)['passed']
    check('stage_only_success_rejected',lambda:gate_false({'success':True,'stage_success':True}))
    check('rollout_exception_rejected_even_with_final_flags',lambda:gate_false(metrics,{'type':'Interrupted'}))
    eleven=copy.deepcopy(metrics);eleven['receiver_complete_capsules']=11
    check('eleven_complete_capsules_rejected',lambda:gate_false(eleven))
    center_only=copy.deepcopy(metrics);center_only.pop('receiver_complete_capsules')
    check('center_only_counts_rejected',lambda:gate_false(center_only))
    force=copy.deepcopy(metrics);force['tablet_bottle_force_N']=.1
    check('retained_bottle_contact_rejected',lambda:gate_false(force))
    unsafe=copy.deepcopy(metrics);unsafe['cap_physically_clear']=False
    check('actual_whole_cap_not_clear_rejected',lambda:gate_false(unsafe))
    moving=copy.deepcopy(metrics);moving['receiver_speed_m_s']=.02
    check('unsettled_dynamic_receiver_rejected',lambda:gate_false(moving))
    guard=copy.deepcopy(metrics);guard['guard_failure']='collision'
    check('active_physical_guard_rejected',lambda:gate_false(guard))
    def valid_boolean_only():assert entry.final_gate(metrics,None,cfg)['passed']
    check('positive_boolean_fixture_only_no_export',valid_boolean_only)
    def blocked_current_source():
        report=entry.preflight(cfg)
        assert not report['ready_to_attempt_full_collection']and'full_workflow_not_declared_ready_in_source'in report['reasons']
    check('current_full_workflow_false_blocks_execution',blocked_current_source)
    absent=copy.deepcopy(cfg);absent['task_module']='/tmp/no-such-full-cap-task-module-20261022.py'
    check('missing_task_has_no_stage_fallback',lambda:rejects(lambda:entry.load_task_class(absent)))
    with tempfile.TemporaryDirectory(prefix='full-cap-entry-negative-')as directory:
        root=Path(directory)
        missing_cfg=root/'missing_full_config.json';missing_cfg.write_text(json.dumps(absent))
        def cli_reject_before_output():
            target=root/'must_not_be_created'
            result=subprocess.run([sys.executable,str(Path(entry.__file__).resolve()),'--execute','--config',str(missing_cfg),'--output',str(target)],capture_output=True,text=True)
            assert result.returncode!=0 and'missing_full_task_module'in result.stderr and not target.exists()
        check('actual_cli_missing_task_rejected_before_output_or_physics',cli_reject_before_output)
        bad_clock=root/'wrong_clock.json';clockcfg=copy.deepcopy(cfg);clockcfg['physics_ticks_per_action']=14;bad_clock.write_text(json.dumps(clockcfg))
        check('wrong_clock_config_refused',lambda:rejects(lambda:entry.load_config(bad_clock)))
        valid=root/'official_wrapper.py'
        text=(f'from envs._base_task import Base_Task\nfrom envs.{entry.TASK} import {entry.TASK}\nfrom envs.utils import *\nimport sapien\n'
              f'class gpt_{entry.TASK}({entry.TASK}):\n    def play_once(self):\n        return self.info\n')
        valid.write_text(text)
        check('declared_full_official_wrapper_ast',lambda:entry.verify_candidate_ast(valid))
        wrong=root/'empty_task.py';wrong.write_text(text.replace(entry.TASK,'labware_pick_place'))
        check('empty_task_agent_wrapper_rejected',lambda:rejects(lambda:entry.verify_candidate_ast(wrong)))
        changed=root/'changed.py';changed.write_text(text.replace('return self.info','raise RuntimeError("changed")'))
        check('parser_changed_tested_method_rejected',lambda:rejects(lambda:entry.verify_candidate_ast(changed,valid)))
        replay=SimpleNamespace(info={'info':{}},calls=[],observed=[])
        def execute(action,phase):
            replay.calls.append((np.asarray(action).copy(),phase));replay.observed.append({'success':False})
        replay.apply_sdk24=execute
        rows=[{'requested_action':np.full(24,i).tolist(),'action':np.full(24,i).tolist(),'phase':str(i),'metrics':{'success':True}}for i in range(3)]
        def once():
            entry.central_replay(replay,rows)
            assert len(replay.calls)==3 and all(r['success']is False for r in replay.observed)
        check('central_replay_executes_three_rows_once_no_copied_success',once)
        check('second_full_replay_call_rejected',lambda:rejects(lambda:entry.central_replay(replay,rows)))
        check('reset_initial_state_mismatch_rejected',lambda:rejects(lambda:entry.same_initial_state({'q':[1.]},{'q':[1.000001]})))
        before={'state':np.zeros(24),'physics_tick':15};after={'state':np.full(24,.1),'physics_tick':30}
        drive={'executed_physics_steps':15,'endpoint_command_written':True,'physics_step_start':15,'physics_step_end':30,'actual_final_drive_command24':np.ones(24).tolist()}
        task=SimpleNamespace(clock=SimpleNamespace(actions=[drive]));row={}
        def actual_drive():
            saved=entry.bind_actual_transition(task,before,np.ones(24),after,row)
            assert np.array_equal(saved,np.ones(24))and not np.array_equal(saved,after['state'])and'actual_drive_record'in row
        check('stores_actual_drive_not_next_measured_state',actual_drive)
        bad=copy.deepcopy(drive);bad['executed_physics_steps']=14
        check('wrong_clock_rejected',lambda:rejects(lambda:entry.bind_actual_transition(SimpleNamespace(clock=SimpleNamespace(actions=[bad])),before,np.ones(24),after,{})))
        info={'info':json.loads((Path(cfg['metadata_root'])/'episode_language_bindings.json').read_text())}
        check('unresolved_full_language_rejected',lambda:rejects(lambda:entry.validate_language(info,['{A}']*12,cfg)))
        empty={'info':{'{A}':'bottle','{a}':'right'}}
        check('empty_only_language_schema_rejected',lambda:rejects(lambda:entry.validate_language(empty,['instruction']*12,cfg)))
        # This tests hook lifecycle only; no actual success receipt/file is
        # generated. Base.close intentionally destroys live metrics/info.
        class FakeBase:
            def check_success(self):return True
            def close_env(self,*args,**kwargs):self.metrics.clear();self.info.clear()
            def setup_demo(self,**kwargs):raise RuntimeError('synthetic_setup_failure')
        attempts=[];validations=[];cli=SimpleNamespace(max_seed_attempts=1,keep_cache=True,extra_sources=[])
        cls=entry.make_collection_task(FakeBase,cfg,root,{}, {},cli,SimpleNamespace(),attempts,validations)
        closed=cls();closed.collection_receipt={'success':True};closed.metrics=copy.deepcopy(metrics);closed.info=copy.deepcopy(info)
        def close_lifecycle():
            closed.close_env();assert closed.check_success()and closed._full_frozen_info==info and closed._full_frozen_metrics==metrics and closed._full_env_closed
        check('close_preserves_immutable_verdict_metrics_language_fixture',close_lifecycle)
        failed=cls()
        check('setup_error_counts_as_one_attempt',lambda:rejects(lambda:failed.setup_demo(need_plan=True,seed=0,now_ep_num=0),RuntimeError))
        check('seed_budget_stops_repeated_setup_failures',lambda:rejects(lambda:failed.setup_demo(need_plan=True,seed=1,now_ep_num=0),entry.SeedBudgetExceeded))
        def exceptional_cleanup():
            fake=SimpleNamespace(_full_env_closed=False,calls=0)
            def close():fake.calls+=1;fake._full_env_closed=True
            fake.close_env=close
            assert entry.close_unclosed_task(fake)is None and fake.calls==1
            assert entry.close_unclosed_task(fake)is None and fake.calls==1
        check('exceptional_run_cleanup_only_once',exceptional_cleanup)
        def cleanup_preserves_error():
            fake=SimpleNamespace(_full_env_closed=False)
            def close():raise RuntimeError('cleanup_failed')
            fake.close_env=close
            error=entry.close_unclosed_task(fake)
            assert error['type']=='RuntimeError'and error['message']=='cleanup_failed'
        check('cleanup_error_returned_without_overwriting_primary_exception',cleanup_preserves_error)
        assert len(attempts)==1 and not validations and not list(root.rglob('*.hdf5'))
    manifest=entry.source_manifest(cfg)
    check('actual_source_manifest_binds_full_implementation_profile_geometry',lambda:all_manifest(manifest))
    report={'scope':'CPU-only in-memory failure controls; no robot/scene step or success trajectory/HDF5 created',
        'passed':True,'checks':results,'count':len(results),'source_sha256':entry.sha(entry.__file__),
        'checks_sha256':entry.sha(__file__),'physics_steps':0,'hdf5_or_success_demo_created':False}
    report['source_manifest_files']=len(manifest)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return report

def all_manifest(manifest):
    assert str(entry.CAP/'labware_cap115_unscrew_pour_task.py')in manifest
    assert str(entry.CAP/'profiles/bottle115/physics_spec115.json')in manifest
    assert len(manifest)>=100 and all(len(value)==64for value in manifest.values())

if __name__=='__main__':main()
