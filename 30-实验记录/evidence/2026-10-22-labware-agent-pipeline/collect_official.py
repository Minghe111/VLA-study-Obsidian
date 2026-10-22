#!/usr/bin/env python3
"""Unchanged RoboTwin collect_data.run + isolated RealMan task/export hooks."""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import traceback
import numpy as np

ROOT=Path(__file__).resolve().parent
ROBOTWIN=Path('/home/admin123/liminghe/vla-benchmark/RoboTwin')


class SeedBudgetExceeded(BaseException): pass


def load_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec); sys.modules[name]=module
    spec.loader.exec_module(module); return module


def verify_parser_equivalence(path,tested_path):
    """Accept only the unchanged official wrapper and the actually tested method."""
    actual=ast.parse(Path(path).read_text()); tested=ast.parse(Path(tested_path).read_text())
    template=ast.parse('from envs._base_task import Base_Task\nfrom envs.labware_pick_place import labware_pick_place\nfrom envs.utils import *\nimport sapien\n')
    assert len(actual.body)==5 and all(ast.dump(a)==ast.dump(b) for a,b in zip(actual.body[:4],template.body))
    cls=actual.body[4]
    assert isinstance(cls,ast.ClassDef) and cls.name=='gpt_labware_pick_place'
    assert not cls.decorator_list and not cls.keywords and len(cls.bases)==1
    assert ast.dump(cls.bases[0])==ast.dump(ast.Name(id='labware_pick_place',ctx=ast.Load()))
    assert len(cls.body)==1 and isinstance(cls.body[0],ast.FunctionDef) and cls.body[0].name=='play_once'
    old_cls=next(v for v in tested.body if isinstance(v,ast.ClassDef) and v.name==cls.name)
    old_method=next(v for v in old_cls.body if isinstance(v,ast.FunctionDef) and v.name=='play_once')
    value=ast.dump(cls.body[0],include_attributes=False)
    assert value==ast.dump(old_method,include_attributes=False),'Official response parser changed the tested method'
    return hashlib.sha256(value.encode()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--episodes',type=int,default=1)
    p.add_argument('--max-seed-attempts',type=int,default=3)
    p.add_argument('--expert-code',type=Path,help='Actual official Agent output gpt_labware_pick_place.py; absent = native reference expert')
    p.add_argument('--agent-receipt',type=Path,help='Real official session-test receipt binding candidate SHA and physical test')
    p.add_argument('--parser-receipt',type=Path,help='Original generate_code post-response parser receipt, if using its byte-different but method-identical wrapper')
    p.add_argument('--replay-source',type=Path,help='Use actual validated prior planning trajectories with original use_seed=True; no new planning is claimed')
    p.add_argument('--camera-framing',type=Path,help='Private nominal simulation camera preset; changes rendering only')
    p.add_argument('--asset',choices=['reagent_bottle_body'],default='reagent_bottle_body',help='This registered task is validated only for the empty open reagent bottle')
    p.add_argument('--arm',choices=['right'],default='right',help='The current generated task uses the right arm')
    p.add_argument('--keep-cache',action='store_true')
    cli=p.parse_args()
    for field in ('output','expert_code','agent_receipt','parser_receipt','replay_source','camera_framing'):
        value=getattr(cli,field)
        if value is not None:setattr(cli,field,value.resolve())
    if cli.episodes<1 or cli.max_seed_attempts<cli.episodes: p.error('Require 1 <= episodes <= max-seed-attempts')
    output=cli.output.resolve()
    if output.exists(): raise FileExistsError('Use a fresh output directory; prior evidence is never overwritten: '+str(output))
    output.mkdir(parents=True)
    sys.path.insert(0,str(ROOT))
    # Load the verified native Base_Task before official code-generation imports.
    from labware_task import labware_pick_place
    sys.modules['envs.labware_pick_place']=sys.modules['labware_task']
    from sdk24_export import dump,save_transition,export_episode
    base=labware_pick_place
    if cli.expert_code:
        path=cli.expert_code.resolve()
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        expert={'kind':'provided_candidate_provenance_unverified','source':str(path),
                'sha256':digest,'agent_generated':False}
        if cli.agent_receipt:
            evidence=json.loads(cli.agent_receipt.read_text())
            if evidence['candidate_sha256']!=digest:
                assert cli.parser_receipt,'Candidate differs from tested source and has no parser binding'
                parsed=json.loads(cli.parser_receipt.read_text())
                assert parsed['generated_sha256']==digest and parsed['input_candidate_sha256']==evidence['candidate_sha256']
                assert parsed['method_ast_equal'] and parsed['original_generate_code_post_response_executed']
                assert parsed['new_physical_trials']==0
                original=Path(evidence['frozen_candidate'])
                assert hashlib.sha256(original.read_bytes()).hexdigest()==evidence['candidate_sha256']
                method_sha=verify_parser_equivalence(path,original)
                expert.update(parser_receipt=str(cli.parser_receipt.resolve()),
                              method_ast_sha256=method_sha,byte_identity_to_tested_candidate=False)
            assert evidence['task_name']=='labware_pick_place'
            assert evidence['original_test_gen_code_run_called'] is True
            assert evidence['status']=='physical_test_complete' and evidence['official_threshold_0_5_met'] is True
            assert evidence['source_unchanged'] and evidence['candidate_unchanged']
            expert.update(kind='current_session_candidate_officially_tested',agent_generated=True,
                          tested_candidate_sha256=evidence['candidate_sha256'],
                          generation_transport=evidence['generation_transport'],
                          official_test_receipt=str(cli.agent_receipt.resolve()),
                          official_test_success_rate=evidence['success_rate'])
        generated=load_file('envs_gen.gpt_labware_pick_place',path)
        base=generated.gpt_labware_pick_place
        if not issubclass(base,labware_pick_place): raise TypeError('Agent class must extend the registered labware task')
    else:
        expert={'kind':'native_reference_expert','agent_generated':False,
                'source':str(ROOT/'robot_expert.py'),
                'sha256':hashlib.sha256((ROOT/'robot_expert.py').read_bytes()).hexdigest()}
    replay_provenance=None
    if cli.replay_source:
        source=cli.replay_source.resolve()
        prior=json.loads((source/'collection_receipt.json').read_text())
        prior_launch=json.loads((source/'launch.json').read_text())
        assert prior['status']=='passed' and prior['validated_episodes']>=cli.episodes
        assert prior_launch['expert']['sha256']==expert['sha256']
        assert prior_launch['args']['asset_kind']==cli.asset and prior_launch['args']['arm']==cli.arm
        for name in ('labware_task.py','robot_expert.py'):
            path=ROOT/name
            assert prior_launch['source_sha256'][str(path)]==hashlib.sha256(path.read_bytes()).hexdigest(),'Physics/expert changed since the actual source plan'
        seeds=(source/'seed.txt').read_text().split()
        assert len(seeds)>=cli.episodes
        (output/'seed.txt').write_text(' '.join(seeds[:cli.episodes])+' ')
        (output/'_traj_data').mkdir()
        fingerprints={}
        for i in range(cli.episodes):
            path=source/'_traj_data'/f'episode{i}.json'
            trajectory=json.loads(path.read_text());assert trajectory['success_receipt']['success']
            assert trajectory['success_receipt']['error'] is None
            shutil.copyfile(path,output/'_traj_data'/path.name)
            fingerprints[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        replay_provenance={'source_collection':str(source),'source_collection_sha256':hashlib.sha256((source/'collection_receipt.json').read_bytes()).hexdigest(),
                           'trajectory_sha256':fingerprints,'new_planning_performed':False}
    os.chdir(ROBOTWIN)
    official=load_file('labware_original_collect_data',ROBOTWIN/'scripts/collect_data.py')
    import description.utils.generate_episode_instructions as language
    # These original language functions resolve templates via parent_directory.
    # Bind only their data root; no function source or official files are edited.
    language.parent_directory=str(ROOT/'metadata/description/utils')
    args=dict(task_name='labware_pick_place',task_config='labware_realman_sdk24',
              save_path=str(output),episode_num=cli.episodes,use_seed=bool(cli.replay_source),collect_data=True,
              clear_cache_freq=1,render_freq=0,save_data=False,language_num=12,
              asset_kind=cli.asset,arm=cli.arm)
    attempts=[]; validations=[]

    class CollectionTask(base):
        def setup_demo(self,**kwargs):
            self.collection_need_plan=bool(kwargs.get('need_plan',True))
            seed=int(kwargs.get('seed',0)); ep=int(kwargs.get('now_ep_num',0))
            if self.collection_need_plan and len(attempts)>=cli.max_seed_attempts:
                raise SeedBudgetExceeded('Finite seed budget exhausted; inspect retained failures')
            label=('plan' if self.collection_need_plan else 'capture')+f'_ep{ep:07d}_seed{seed:05d}'
            if self.collection_need_plan:attempts.append({'seed':seed,'episode':ep,'label':label})
            self.collection_run=output/'runs'/label
            if self.collection_run.exists():raise FileExistsError(self.collection_run)
            self.collection_cache=self.collection_run/'transitions'
            self.capture_counter=0
            # SDK24 callback replaces the generic PKL writer, whose action
            # semantics and scalar gripper layout differ from this contract.
            kwargs.update(runtime_path=str(self.collection_run),capture_rgb=not self.collection_need_plan,save_data=False)
            kwargs['on_transition']=None if self.collection_need_plan else self.capture_transition
            super().setup_demo(**kwargs)
            if cli.camera_framing:
                from camera_framing import apply_camera_framing
                self.camera_framing_receipt=apply_camera_framing(self,json.loads(cli.camera_framing.read_text()))
                dump(self.collection_run/'camera_framing_receipt.json',self.camera_framing_receipt)
            if self.collection_need_plan:attempts[-1]['path']=str(self.collection_run)
            return self.info

        def capture_transition(self,before,command,after,row):
            drive=self.clock.actions[-1]
            assert drive['executed_physics_steps']==15 and drive['endpoint_command_written']
            assert drive['physics_step_start']==before['physics_tick'] and drive['physics_step_end']==after['physics_tick']
            actual=np.asarray(drive['actual_final_drive_command24'],dtype=np.float64)
            assert np.allclose(actual,command,rtol=0,atol=1e-3),'Executor altered the requested target'
            stored=dict(row,requested_action=np.asarray(command).tolist(),action=actual.tolist(),
                        actual_drive_record=drive)
            save_transition(self.collection_cache,self.capture_counter,before,actual,after,stored)
            self.capture_counter+=1

        def play_once(self):
            try:
                result=super().play_once()
                self.persist_run(None)
                return result
            except Exception as exc:
                self.persist_run({'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
                raise

        def persist_run(self,error):
            self.collection_receipt={'success':bool(error is None and self.check_success()),'seed':int(self.seed),
                'phase':'planning' if self.collection_need_plan else 'capture',
                'expert':expert,'metrics':dict(self.metrics),'actions':len(self.transition_log),
                'actual_robot_driven':True,'object_state_writes_after_reset':0,
                'control_dt_seconds':.06,'physics_dt_seconds':.004,'error':error}
            if cli.camera_framing:self.collection_receipt['camera_framing']=self.camera_framing_receipt
            dump(self.collection_run/'rollout_receipt.json',self.collection_receipt)
            dump(self.collection_run/'actual_transitions.json',self.transition_log)
            dump(self.collection_run/'physics_trace.json',self.physics_trace)

        def planned_joints_legal(self):
            values=np.asarray([r['action'] for r in self.transition_log],dtype=np.float64)
            if values.ndim!=2 or values.shape[1]!=24 or not np.isfinite(values).all():return False,float('inf')
            self.left_joint_path=[{'position':values[:,:6]}]
            self.right_joint_path=[{'position':values[:,12:18]}]
            legal,excess=super().planned_joints_legal()
            hand=values[:,[6,7,8,9,10,11,18,19,20,21,22,23]]
            sdk_excess=max(0.,float(-hand.min()),float(hand.max()-1000))
            return bool(legal and sdk_excess<=1e-4),max(float(excess),sdk_excess)

        def save_traj_data(self,idx):
            if not self.check_success() or not self.collection_receipt['success'] or self.collection_receipt['error'] is not None:
                raise RuntimeError('Cannot save a failed or incomplete expert trajectory')
            dump(output/'_traj_data'/f'episode{idx}.json',{
                'seed':int(self.seed),'schema':'actual_sdk24_commands_v1','rows':self.transition_log,
                'source':'actual issued targets from physically successful native simulation',
                'success_receipt':self.collection_receipt})

        def load_tran_data(self,idx):
            path=output/'_traj_data'/f'episode{idx}.json'
            self.loaded_trajectory=json.loads(path.read_text())
            assert int(self.loaded_trajectory['seed'])==int(self.seed)
            values=np.asarray([r['action'] for r in self.loaded_trajectory['rows']])
            return {'left_joint_path':[{'position':values[:,:6]}],
                    'right_joint_path':[{'position':values[:,12:18]}]}

        def set_path_lst(self,config):
            self.need_plan=False
            self.left_joint_path=config['left_joint_path'];self.right_joint_path=config['right_joint_path']
            self.replay_commands=self.loaded_trajectory['rows']

        def merge_pkl_to_hdf5_video(self,instructions=None):
            if not self.check_success() or not self.collection_receipt['success'] or self.collection_receipt['error'] is not None:
                raise RuntimeError('Replayed robot trajectory failed actual physical success check')
            assert len(instructions or [])==args['language_num'],'Refuse generic fallback or missing task language'
            assert set(self.info['info'])=={'{A}','{a}'},'Wrong episode language binding'
            assert self.info['info']['{a}']==self.arm
            assert Path(self.info['info']['{A}']+'.json').is_file(),'Missing object language description'
            episode=f'episode_{self.ep_num:07d}'
            result=export_episode(self.collection_cache,output/'data'/f'{episode}.hdf5',instructions,
                                  self.collection_receipt,output/'video'/f'{episode}.mp4')
            validations.append(result)

        def remove_data_cache(self):
            # Original run invokes this only after a successful merge. Preserve
            # failed runs and all state/target/physics receipts in every case.
            if not cli.keep_cache and validations and validations[-1]['passed']:
                import shutil
                shutil.rmtree(self.collection_cache)

    task=CollectionTask()
    source_files=[ROBOTWIN/'scripts/collect_data.py',ROOT/'labware_task.py',ROOT/'robot_expert.py',Path(__file__),ROOT/'sdk24_export.py']
    if cli.camera_framing:source_files.extend([ROOT/'camera_framing.py',cli.camera_framing.resolve()])
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    snapshots={}
    (output/'source_snapshot').mkdir()
    for index,path in enumerate(source_files):
        target=output/'source_snapshot'/f'{index:02d}_{path.name}'
        shutil.copyfile(path,target)
        assert hashlib.sha256(target.read_bytes()).hexdigest()==sources[str(path)]
        snapshots[str(path)]=str(target)
    dump(output/'launch.json',{'args':args,'expert':expert,'source_sha256':sources,
          'source_snapshots':snapshots,
          'unchanged_official_entry':'scripts.collect_data.run','max_seed_attempts':cli.max_seed_attempts,
          'local_extensions':['RM65-B SDK24 expert','actual target trajectory hooks','SDK24 exporter','language data root'],
          'replay_source':replay_provenance,'camera_framing':str(cli.camera_framing.resolve()) if cli.camera_framing else None})
    status='running'
    try:
        official.run(task,args)
        assert len(validations)==cli.episodes,'Did not read back every requested episode'
        status='passed'
    except BaseException as exc:
        status='failed'
        dump(output/'failure.json',{'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
        raise
    finally:
        dump(output/'collection_receipt.json',{'status':status,'attempts':attempts,
             'requested_episodes':cli.episodes,'validated_episodes':len(validations),
             'validations':validations,'expert':expert,
             'replay_source':replay_provenance,
             'official_source_unchanged':sources[str(ROBOTWIN/'scripts/collect_data.py')]==hashlib.sha256((ROBOTWIN/'scripts/collect_data.py').read_bytes()).hexdigest()})
    print(json.dumps({'status':status,'output':str(output),'episodes':len(validations)},ensure_ascii=False))


if __name__=='__main__':main()
