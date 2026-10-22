#!/usr/bin/env python3
"""Isolated final-only cap/pour collector wiring; no empty-task fallback.

The real complete task is intentionally required, never synthesized here.
--preflight imports no simulator/task, --execute calls unchanged collect.run.
"""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import traceback
import xml.etree.ElementTree as ET
import numpy as np

CAP=Path(__file__).resolve().parent
ROOT=CAP.parent
TASK='labware_cap115_unscrew_pour'
CONFIG=CAP/'full_cap_collection_config.json'

class FullCollectionRejected(RuntimeError):pass
class SeedBudgetExceeded(BaseException):pass

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def load_config(path=CONFIG):
    cfg=json.loads(Path(path).read_text())
    if cfg.get('task_name')!=TASK or cfg.get('task_class')!=TASK:raise FullCollectionRejected('Wrong full task identity')
    if (cfg['control_dt_s'],cfg['physics_dt_s'],cfg['physics_ticks_per_action'])!=(.06,.004,15):
        raise FullCollectionRejected('This exporter requires exactly15x4ms=60ms')
    for key in ('task_module','metadata_root'):
        p=Path(cfg[key]);cfg[key]=str((CAP/p).resolve()if not p.is_absolute()else p.resolve())
    cfg['config_path']=str(Path(path).resolve())
    return cfg

def final_gate(metrics,error,cfg):
    """Pure final-only rejection; a stage success is never promoted."""
    reasons=[]
    if error is not None:reasons.append('rollout_exception')
    for key in cfg['required_true_final_metrics']:
        if metrics.get(key)is not True:reasons.append('not_final_true:'+key)
    for key,value in cfg['required_exact_final_metrics'].items():
        if type(metrics.get(key))is not int or metrics[key]!=value:reasons.append('wrong_final_count:'+key)
    for key,value in cfg['minimum_final_metrics'].items():
        actual=metrics.get(key)
        if not isinstance(actual,(int,float))or not np.isfinite(actual)or actual<value:reasons.append('insufficient_final_interval:'+key)
    for key,value in cfg['maximum_exclusive_final_metrics'].items():
        actual=metrics.get(key)
        if not isinstance(actual,(int,float))or not np.isfinite(actual)or actual<0 or actual>=value:reasons.append('invalid_final_force_or_speed:'+key)
    for key in cfg['required_clear_errors']:
        if key not in metrics or metrics[key]is not None:reasons.append('missing_or_active_guard:'+key)
    if metrics.get('guard_failure')is not None:reasons.append('active_guard_failure')
    if metrics.get('cap_state',{}).get('released')is not True:reasons.append('cap_not_physically_released')
    return {'passed':not reasons,'reasons':reasons}

def verify_candidate_ast(path,tested_path=None):
    tree=ast.parse(Path(path).read_text())
    template=ast.parse(f'from envs._base_task import Base_Task\nfrom envs.{TASK} import {TASK}\nfrom envs.utils import *\nimport sapien\n')
    if len(tree.body)!=5 or any(ast.dump(a)!=ast.dump(b)for a,b in zip(tree.body[:4],template.body)):
        raise FullCollectionRejected('Require exact official full-task import wrapper')
    cls=tree.body[4]
    if (not isinstance(cls,ast.ClassDef)or cls.name!='gpt_'+TASK or cls.decorator_list or cls.keywords
            or len(cls.bases)!=1 or not isinstance(cls.bases[0],ast.Name)or cls.bases[0].id!=TASK
            or len(cls.body)!=1 or not isinstance(cls.body[0],ast.FunctionDef)or cls.body[0].name!='play_once'):
        raise FullCollectionRejected('Wrong generated class, inheritance or method structure')
    value=ast.dump(cls.body[0],include_attributes=False)
    if tested_path is not None:
        old=ast.parse(Path(tested_path).read_text())
        classes=[x for x in old.body if isinstance(x,ast.ClassDef)and x.name=='gpt_'+TASK]
        if len(classes)!=1:raise FullCollectionRejected('Missing physically tested full candidate class')
        methods=[x for x in classes[0].body if isinstance(x,ast.FunctionDef)and x.name=='play_once']
        if len(methods)!=1 or ast.dump(methods[0],include_attributes=False)!=value:
            raise FullCollectionRejected('Parser changed the physically tested play_once AST')
    return hashlib.sha256(value.encode()).hexdigest()

def validate_language(info,instructions,cfg):
    expected=json.loads((Path(cfg['metadata_root'])/'episode_language_bindings.json').read_text())
    if info.get('info')!=expected:raise FullCollectionRejected('Wrong full body/cap/dish/right/left language binding')
    if len(instructions or[])!=cfg['language_num']or any(not isinstance(s,str)or not s.strip()or'{'in s or'}'in s for s in instructions):
        raise FullCollectionRejected('Missing, generic fallback, or unresolved full task instructions')
    for key in ('{A}','{B}','{C}'):
        if not Path(expected[key]+'.json').is_file():raise FullCollectionRejected('Missing profile-specific object description')
    return expected

def local_source_paths(cfg):
    paths=[Path(__file__),Path(cfg['config_path']),ROOT/'collect_official.py',ROOT/'sdk24_export.py',ROOT/'robot_cap_assembly.py',ROOT/'labware_task.py',ROOT/'robot_expert.py',
        CAP/'labware_cap_task.py',CAP/'labware_cap115_task.py',CAP/'labware_cap_workflow.py',CAP/'petri_receiver.py',CAP/'capsule_geometry.py',
        CAP/'cap_contact_controller.py',CAP/'held_left_planner.py',CAP/'contact_turn_geometry.py',Path(cfg['source_root'])/'scripts/collect_data.py',
        Path(cfg['source_root'])/'code_gen/task_generation.py',Path(cfg['source_root'])/'code_gen/test_gen_code.py',
        Path(cfg['native_root'])/'envs/_base_task.py']
    taskpath=Path(cfg['task_module'])
    if taskpath.is_file():paths.append(taskpath)
    # Follow only concrete local Python imports. String-based runtime/native
    # loads are separately captured below; absence is reported, never invented.
    seen=set(paths);pending=[p for p in paths if p.suffix=='.py'and p.is_file()]
    while pending:
        path=pending.pop()
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node,ast.ImportFrom)and node.module:
                name=node.module.replace('.','/')+'.py'
                for parent in (CAP,ROOT):
                    found=parent/name
                    if found.is_file()and found not in seen:seen.add(found);pending.append(found)
    for directory in (CAP/'profiles/bottle115',Path(cfg['metadata_root'])):
        for p in directory.rglob('*'):
            if p.is_file()and'__pycache__'not in p.parts and p.suffix not in('.pyc',):seen.add(p)
    native=Path(cfg['native_root'])
    for directory in (native/'integrations/realman_rm65b_gen4',native/'envs/robot/realman_rm65b'):
        for p in directory.rglob('*'):
            if p.is_file()and p.suffix in('.py','.json','.yml','.yaml','.urdf','.srdf')and'__pycache__'not in p.parts:seen.add(p)
    for directory in (native/'envs',native/'data',Path(cfg['source_root'])/'data',Path(cfg['source_root'])/'code_gen'):
        for p in directory.rglob('*.py'):
            if '__pycache__'not in p.parts:seen.add(p)
    vendor=Path(cfg['source_root'])/'assets/embodiments/realman-rm65b-inspire-gen4'
    for p in vendor.rglob('*'):
        if p.is_file()and p.suffix in('.json','.yml','.yaml','.urdf','.srdf','.obj','.stl','.glb'):seen.add(p)
    urdf=vendor/'robot_sapien.urdf'
    if urdf.is_file():
        for mesh in ET.parse(urdf).getroot().iter('mesh'):
            source=Path(mesh.get('filename'));seen.add(source if source.is_absolute()else vendor/source)
    frozen=ROOT.parent/'2026-10-22-labware-integration'
    seen.update([frozen/'physics_spec.json',frozen/'lab_physics.py',frozen/'prepared/reagent_bottle_body/model_data0.json',frozen/'prepared/reagent_bottle_cap/model_data0.json'])
    # Physical assets are source-linked, not copied into new fake geometry.
    registry=json.loads((Path(cfg['metadata_root'])/'registry.json').read_text())
    for entry in registry.get('assets',{}).values():
        if isinstance(entry,dict)and entry.get('absolute_modelname'):
            folder=Path(entry['absolute_modelname'])
            for p in folder.rglob('*'):
                if p.is_file()and p.suffix in('.json','.obj','.glb','.stl'):seen.add(p)
    return sorted(seen,key=str)

def source_manifest(cfg,extra=()):
    paths=local_source_paths(cfg)+[Path(p)for p in extra if p is not None]
    missing=[str(p)for p in paths if not p.is_file()]
    if missing:raise FullCollectionRejected('Missing source inputs: '+str(missing))
    return {str(p.resolve()):sha(p)for p in paths}

def preflight(cfg):
    module=Path(cfg['task_module']);reasons=[]
    if not module.is_file():reasons.append('missing_full_task_module:'+str(module))
    else:
        tree=ast.parse(module.read_text());names=[x.name for x in tree.body if isinstance(x,ast.ClassDef)]
        if TASK not in names:reasons.append('missing_declared_full_task_class')
        readiness=[x.value.value for x in tree.body if isinstance(x,ast.Assign)and isinstance(x.value,ast.Constant)and any(isinstance(t,ast.Name)and t.id=='FULL_WORKFLOW_READY'for t in x.targets)]
        if readiness!=[True]:reasons.append('full_workflow_not_declared_ready_in_source')
    required=['task_info_template.json','episode_language_bindings.json','registry.json','description/task_instruction/'+TASK+'.json']
    for relative in required:
        if not(Path(cfg['metadata_root'])/relative).is_file():reasons.append('missing_metadata:'+relative)
    language=json.loads((Path(cfg['metadata_root'])/'episode_language_bindings.json').read_text())
    if set(language)!={'{A}','{B}','{C}','{a}','{b}'}or language['{a}']!='right'or language['{b}']!='left':reasons.append('invalid_full_language_schema')
    return {'scope':'CPU source/metadata wiring preflight only; no task import, physics, Agent success or demonstration',
        'ready_to_attempt_full_collection':not reasons,'reasons':reasons,
        'task_name':TASK,'task_module':str(module),'full_task_fallback_allowed':False,
        'default_episode_num':cfg['episode_num'],'default_max_seed_attempts':cfg['max_seed_attempts'],
        'default_official_agent_physical_trials':1,'control_contract':'SDK24,actual state+drive,three RGB,15x4ms=60ms',
        'full_task_success_data_created':False,'actual_physics_steps':0}

def load_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

def load_task_class(cfg):
    report=preflight(cfg)
    if not report['ready_to_attempt_full_collection']:raise FullCollectionRejected('; '.join(report['reasons']))
    sys.path[:0]=[str(CAP),str(ROOT)]
    name='envs.'+TASK;path=Path(cfg['task_module']);module=sys.modules.get(name)
    if module is not None and Path(getattr(module,'__file__','')).resolve()==path.resolve():
        digest=sha(path)
        if getattr(module,'_full_registered_sha256',digest)!=digest:raise FullCollectionRejected('Registered full source changed')
        module._full_registered_sha256=digest
    else:
        module=load_file(name,path);module._full_registered_sha256=sha(path)
    base=getattr(module,TASK)
    if base.__name__!=TASK:raise FullCollectionRejected('Wrong task class; stage-task aliases are forbidden')
    return base

def create_task(task_name=TASK,runtime_config=None,run_dir=None,**kwargs):
    """Original Agent setup_factory; requires the actual future complete task."""
    if task_name!=TASK:raise FullCollectionRejected('Factory only registers the complete full task')
    cfg=load_config((runtime_config or{}).get('full_collection_config',CONFIG));base=load_task_class(cfg)
    candidate=Path(run_dir)/'envs_gen'/('gpt_'+TASK+'.py')if run_dir is not None else None
    if candidate is None or not candidate.is_file():raise FullCollectionRejected('Agent factory requires the actual generated full candidate; no reference fallback')
    verify_candidate_ast(candidate)
    generated=load_file('envs_gen.gpt_'+TASK,candidate);cls=getattr(generated,'gpt_'+TASK)
    if not issubclass(cls,base):raise FullCollectionRejected('Generated expert has wrong full task base')
    base=cls
    options=copy.deepcopy(cfg['runtime_task_args']);options.update(kwargs)
    options.setdefault('capture_rgb',True)
    if run_dir is not None:options.setdefault('runtime_path',str(Path(run_dir)/'simulation'))
    return base(),options

def actual_initial_state(task):
    """Read exact reset state; never writes or rounds away replay differences."""
    art=task.robot.left_entity
    def pq(entity):return np.r_[entity.pose.p,entity.pose.q].tolist()
    dish=task.dish.actor if hasattr(task.dish,'actor')else task.dish
    return {'seed':int(task.seed),'joint_names':[j.name for j in art.get_active_joints()],
        'qpos':np.asarray(art.get_qpos()).tolist(),'qvel':np.asarray(art.get_qvel()).tolist(),
        'root_matrix':np.asarray(art.root_pose.to_transformation_matrix()).tolist(),
        'body_pose':pq(task.assembly.body),'cap_pose':pq(task.assembly.cap),
        'tablet_names':[e.name for e in task.assembly.tablets],
        'tablet_poses':[pq(e)for e in task.assembly.tablets],'dish_pose':pq(dish),
        'passive_assembly_state':task.assembly.state(),
        'physics_tick':int(task.physics_step_count),'scene_dt_s':float(task.scene.get_timestep())}

def same_initial_state(current,expected):
    # Exact stored float readback, not labels or success copied from planning.
    if json.dumps(current,sort_keys=True,separators=(',',':'))!=json.dumps(expected,sort_keys=True,separators=(',',':')):
        raise FullCollectionRejected('Reset physical initial state differs from the actual source plan')

def bind_actual_transition(task,before,command,after,row):
    drive=copy.deepcopy(task.clock.actions[-1])
    if (drive['executed_physics_steps']!=15 or not drive['endpoint_command_written']
            or drive['physics_step_start']!=before['physics_tick']or drive['physics_step_end']!=after['physics_tick']
            or after['physics_tick']-before['physics_tick']!=15):raise FullCollectionRejected('Wrong actual60ms clock/drive readback')
    actual=np.asarray(drive['actual_final_drive_command24'],dtype=float);requested=np.asarray(command,dtype=float)
    if actual.shape!=(24,)or not np.isfinite(actual).all()or not np.allclose(actual,requested,rtol=0,atol=1e-3):
        raise FullCollectionRejected('Executor target differs from actual final SDK24 drive')
    row.update(requested_action=requested.tolist(),action=actual.tolist(),actual_drive_record=drive)
    return actual

def central_replay(task,rows):
    """One owner, one traversal. Never invokes each stage helper in replay."""
    if getattr(task,'_full_replay_consumed',False):raise FullCollectionRejected('Refuse duplicate replay of the full episode')
    task._full_replay_consumed=True
    for index,row in enumerate(rows):
        task._full_replay_index=index
        # Planning metrics are deliberately ignored; this call must observe
        # the new physical transition and recompute all outcomes.
        task.apply_sdk24(np.asarray(row['requested_action'],dtype=float),phase=row['phase'])
    return task.info

def close_unclosed_task(task):
    """Release exceptional runs without replacing the primary rollout error."""
    if task is None or getattr(task,'_full_env_closed',True):return None
    try:task.close_env()
    except BaseException as exc:
        return {'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
    return None

def verify_agent_evidence(path,receipt_path,parser_path,manifest):
    verify_candidate_ast(path);expert={'kind':'provided_candidate_provenance_unverified','source':str(path),'sha256':sha(path),'agent_generated':False}
    if receipt_path is None:return expert
    evidence=json.loads(Path(receipt_path).read_text());tested=Path(evidence['frozen_candidate'])
    if evidence['task_name']!=TASK or evidence['status']!='physical_test_complete'or not evidence['original_test_gen_code_run_called']or not evidence['official_threshold_0_5_met']:
        raise FullCollectionRejected('No successful official physical test for this complete task')
    if not evidence['source_unchanged']or not evidence['candidate_unchanged']or sha(tested)!=evidence['candidate_sha256']:
        raise FullCollectionRejected('Physically tested candidate/source changed')
    if evidence['candidate_sha256']!=sha(path):
        if parser_path is None:raise FullCollectionRejected('Byte-different expert requires original parser provenance')
        parsed=json.loads(Path(parser_path).read_text())
        if (parsed['generated_sha256']!=sha(path)or parsed['input_candidate_sha256']!=sha(tested)
                or not parsed['method_ast_equal']or not parsed['original_generate_code_post_response_executed']or parsed['new_physical_trials']!=0):
            raise FullCollectionRejected('Wrong original Agent parser receipt')
        verify_candidate_ast(path,tested)
    # Existing empty-task session receipts bind only two old sources. They
    # cannot silently certify this new full implementation/dependency closure.
    bound=evidence.get('full_task_source_sha256',{})
    required={p:v for p,v in manifest.items()if p!=str(Path(path).resolve())and p not in{str(Path(x).resolve())for x in(receipt_path,parser_path)if x}}
    if not required or any(bound.get(p)!=value for p,value in required.items()):
        raise FullCollectionRejected('Official full physical receipt lacks the complete current full-task source manifest')
    expert.update(kind='current_session_full_candidate_officially_tested',agent_generated=True,
        official_test_receipt=str(receipt_path),official_test_success_rate=evidence['success_rate'])
    return expert

def make_collection_task(base,cfg,output,expert,manifest,cli,exporter,attempts,validations):
    class FullCollectionTask(base):
        def setup_demo(self,**kwargs):
            self._full_frozen_success=None;self._full_replay_consumed=False
            self._full_env_closed=False
            self.collection_receipt={'success':False,'error':{'type':'not_yet_completed'}}
            self.collection_initial_state=None
            self.collection_need_plan=bool(kwargs.get('need_plan',True));self.capture_counter=0
            seed=int(kwargs.get('seed',0));ep=int(kwargs.get('now_ep_num',0))
            if self.collection_need_plan and len(attempts)>=cli.max_seed_attempts:raise SeedBudgetExceeded('Finite full-task seed budget exhausted')
            label=('plan'if self.collection_need_plan else'capture')+f'_ep{ep:07d}_seed{seed:05d}'
            self.collection_run=output/'runs'/label
            if self.collection_run.exists():raise FileExistsError(self.collection_run)
            if self.collection_need_plan:attempts.append({'seed':seed,'episode':ep,'run':str(self.collection_run)})
            self.collection_cache=self.collection_run/'transitions'
            kwargs.update(runtime_path=str(self.collection_run),capture_rgb=not self.collection_need_plan,save_data=False,on_transition=self.capture_transition)
            result=super().setup_demo(**kwargs)
            if abs(float(self.scene.get_timestep())-.004)>1e-9:raise FullCollectionRejected('Actual simulator clock differs from4ms')
            self.collection_initial_state=actual_initial_state(self)
            # Complete implementation must audit its ACTUAL setup, including
            # dish against initial bottle/cap/tablets/robot/table, not pour goal.
            if not getattr(self,'initial_layout_receipt',{}).get('full_world_runtime_verified'):
                raise FullCollectionRejected('Missing actual full initial-world geometry validation')
            return result
        def capture_transition(self,before,command,after,row):
            actual=bind_actual_transition(self,before,command,after,row)
            if not self.collection_need_plan:
                expected=self.loaded_trajectory['rows'][self._full_replay_index]['action']
                if not np.array_equal(actual.astype(np.float32),np.asarray(expected,dtype=np.float32)):
                    raise FullCollectionRejected('Replay actual drive differs from the actual source planning target')
                exporter.save_transition(self.collection_cache,self.capture_counter,before,actual,after,row)
                self.capture_counter+=1
        def check_success(self):
            frozen=getattr(self,'_full_frozen_success',None)
            if frozen is not None:return bool(frozen)
            return bool(final_gate(getattr(self,'metrics',{}),None,cfg)['passed']and super().check_success())
        def play_once(self):
            try:
                if self.collection_need_plan:result=super().play_once()
                else:
                    same_initial_state(self.collection_initial_state,self.loaded_trajectory['initial_state'])
                    result=central_replay(self,self.loaded_trajectory['rows'])
                self.persist_run(None);return result
            except Exception as exc:
                self.persist_run({'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()});raise
        def persist_run(self,error):
            gate=final_gate(copy.deepcopy(self.metrics),error,cfg)
            self.collection_receipt={'task_name':TASK,'success':bool(gate['passed']and self.check_success()),'final_gate':gate,
                'error':error,'metrics':copy.deepcopy(self.metrics),'seed':int(self.seed),'expert':expert,
                'phase':'planning'if self.collection_need_plan else'capture','actions':len(self.transition_log),
                'control_dt_seconds':.06,'physics_dt_seconds':.004,'actual_robot_driven':True,
                'source_sha256':manifest,'initial_state':self.collection_initial_state}
            dump(self.collection_run/'rollout_receipt.json',self.collection_receipt)
            dump(self.collection_run/'actual_transitions.json',self.transition_log)
            dump(self.collection_run/'physics_trace.json',self.physics_trace)
        def planned_joints_legal(self):
            values=np.asarray([r['action']for r in self.transition_log],dtype=float)
            if values.ndim!=2 or values.shape[1]!=24 or not np.isfinite(values).all():return False,float('inf')
            self.left_joint_path=[{'position':values[:,:6]}];self.right_joint_path=[{'position':values[:,12:18]}]
            legal,excess=super().planned_joints_legal()
            hands=values[:,[6,7,8,9,10,11,18,19,20,21,22,23]];sdk_excess=max(0.,-float(hands.min()),float(hands.max())-1000.)
            return bool(legal and sdk_excess<=1e-4),max(float(excess),sdk_excess)
        def save_traj_data(self,idx):
            if not self.collection_receipt['success']or self.collection_receipt['error']is not None:raise FullCollectionRejected('Refuse stage/failed planning trajectory')
            dump(output/'_traj_data'/f'episode{idx}.json',{'task_name':TASK,'seed':int(self.seed),'schema':'actual_full_sdk24_commands_v1',
                'source_sha256':manifest,'initial_state':self.collection_initial_state,'rows':self.transition_log,'success_receipt':self.collection_receipt})
        def load_tran_data(self,idx):
            self.loaded_trajectory=json.loads((output/'_traj_data'/f'episode{idx}.json').read_text());d=self.loaded_trajectory
            if d['task_name']!=TASK or d['seed']!=int(self.seed)or d['source_sha256']!=manifest or not d['success_receipt']['success']or d['success_receipt']['error']is not None:
                raise FullCollectionRejected('Wrong source task/seed/implementation/failed trajectory')
            values=np.asarray([r['action']for r in d['rows']]);return {'left_joint_path':[{'position':values[:,:6]}],'right_joint_path':[{'position':values[:,12:18]}]}
        def set_path_lst(self,config):
            self.need_plan=False;self.left_joint_path=config['left_joint_path'];self.right_joint_path=config['right_joint_path']
        def close_env(self,*args,**kwargs):
            if getattr(self,'_full_env_closed',False):return None
            # Original run closes before merge/check_success; preserve immutable
            # final-only verdict and metrics, not references to released actors.
            self._full_frozen_success=bool(getattr(self,'collection_receipt',{}).get('success',False))
            self._full_frozen_metrics=copy.deepcopy(getattr(self,'metrics',{}))
            self._full_frozen_info=copy.deepcopy(getattr(self,'info',{}))
            result=super().close_env(*args,**kwargs)
            self._full_env_closed=True
            return result
        def merge_pkl_to_hdf5_video(self,instructions=None):
            receipt=self.collection_receipt
            if not self.check_success()or not receipt['success']or receipt['error']is not None:raise FullCollectionRejected('Refuse failed or stage-only replay export')
            validate_language(self._full_frozen_info,instructions,cfg)
            if source_manifest(cfg,cli.extra_sources)!=manifest:raise FullCollectionRejected('Source changed during full physical collection')
            episode=f'episode_{self.ep_num:07d}'
            result=exporter.export_episode(self.collection_cache,output/'data'/f'{episode}.hdf5',instructions,receipt,output/'video'/f'{episode}.mp4')
            validations.append(result)
        def remove_data_cache(self):
            if not cli.keep_cache and validations and validations[-1]['passed']:shutil.rmtree(self.collection_cache)
    return FullCollectionTask

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);mode=p.add_mutually_exclusive_group()
    mode.add_argument('--preflight',action='store_true');mode.add_argument('--execute',action='store_true')
    p.add_argument('--config',type=Path,default=CONFIG);p.add_argument('--output',type=Path)
    p.add_argument('--episodes',type=int);p.add_argument('--max-seed-attempts',type=int)
    p.add_argument('--expert-code',type=Path);p.add_argument('--agent-receipt',type=Path);p.add_argument('--parser-receipt',type=Path);p.add_argument('--keep-cache',action='store_true')
    cli=p.parse_args(argv);cfg=load_config(cli.config)
    report=preflight(cfg)
    if not cli.execute:print(json.dumps(report,ensure_ascii=False,indent=2));return 0 if report['ready_to_attempt_full_collection']else 2
    if not report['ready_to_attempt_full_collection']:raise FullCollectionRejected('; '.join(report['reasons']))
    if cli.output is None:p.error('--execute requires a new --output')
    output=cli.output.resolve()
    if not output.is_relative_to(ROOT)or output.exists():raise FullCollectionRejected('Use a fresh task-local output; never overwrite evidence')
    cli.episodes=cli.episodes or cfg['episode_num'];cli.max_seed_attempts=cli.max_seed_attempts or cfg['max_seed_attempts']
    if not 1<=cli.episodes<=cli.max_seed_attempts:raise FullCollectionRejected('Require1<=episodes<=max_seed_attempts')
    cli.extra_sources=[p.resolve()for p in(cli.expert_code,cli.agent_receipt,cli.parser_receipt)if p is not None]
    manifest=source_manifest(cfg,cli.extra_sources);base=load_task_class(cfg)
    expert={'kind':'native_full_reference_expert','agent_generated':False,'source':cfg['task_module'],'sha256':sha(cfg['task_module'])}
    if cli.expert_code:
        expert=verify_agent_evidence(cli.expert_code.resolve(),cli.agent_receipt,cli.parser_receipt,manifest)
        generated=load_file('envs_gen.gpt_'+TASK,cli.expert_code.resolve());candidate=getattr(generated,'gpt_'+TASK)
        if not issubclass(candidate,base):raise FullCollectionRejected('Wrong generated full class inheritance')
        base=candidate
    elif cli.agent_receipt or cli.parser_receipt:raise FullCollectionRejected('Agent receipts require the actual expert file')
    sys.path[:0]=[str(ROOT),cfg['source_root']]
    import sdk24_export as exporter
    os.chdir(cfg['source_root']);official=load_file('full_cap_original_collect_data',Path(cfg['source_root'])/'scripts/collect_data.py')
    import description.utils.generate_episode_instructions as language
    language.parent_directory=str(Path(cfg['metadata_root'])/'description/utils')
    args=copy.deepcopy(cfg['runtime_task_args']);args.update(task_name=TASK,task_config='labware_cap115_realman_sdk24',save_path=str(output),episode_num=cli.episodes,
        use_seed=False,collect_data=True,clear_cache_freq=1,render_freq=0,save_data=False,language_num=cfg['language_num'])
    output.mkdir(parents=True);attempts=[];validations=[];status='running'
    snapshots={};(output/'source_snapshot').mkdir()
    for index,(name,digest)in enumerate(manifest.items()):
        path=Path(name)
        if path.suffix in('.py','.json','.yaml','.yml','.urdf','.srdf')and path.stat().st_size<2_000_000:
            target=output/'source_snapshot'/f'{index:04d}_{path.name}';shutil.copyfile(path,target)
            if sha(target)!=digest:raise FullCollectionRejected('Snapshot byte mismatch')
            snapshots[name]=str(target)
    dump(output/'launch.json',{'task_name':TASK,'args':args,'resolved_config':cfg,'expert':expert,'source_sha256':manifest,'source_snapshots':snapshots,
        'unchanged_official_entry':'scripts.collect_data.run','physical_success_claimed':False,'max_seed_attempts':cli.max_seed_attempts})
    task=None;primary_error=None;cleanup_error=None
    try:
        task=make_collection_task(base,cfg,output,expert,manifest,cli,exporter,attempts,validations)()
        official.run(task,args)
        if len(validations)!=cli.episodes:raise FullCollectionRejected('Missing full HDF5 readback for requested episode')
        status='passed'
    except BaseException as exc:
        status='failed';primary_error={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
        dump(output/'failure.json',primary_error);raise
    finally:
        cleanup_error=close_unclosed_task(task)
        if cleanup_error is not None:
            status='failed';dump(output/'cleanup_failure.json',cleanup_error)
        dump(output/'collection_receipt.json',{'task_name':TASK,'status':status,'attempts':attempts,'requested_episodes':cli.episodes,'validated_episodes':len(validations),
            'validations':validations,'expert':expert,'cleanup_error':cleanup_error,'official_source_unchanged':sha(Path(cfg['source_root'])/'scripts/collect_data.py')==manifest[str((Path(cfg['source_root'])/'scripts/collect_data.py').resolve())]})
    if cleanup_error is not None and primary_error is None:raise FullCollectionRejected('Full collection cleanup failed; see cleanup_failure.json')
    return 0

if __name__=='__main__':raise SystemExit(main())
