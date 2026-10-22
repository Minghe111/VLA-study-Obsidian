"""Independent Agent metadata for a future full115 task; no robot success.

Only this new directory is written. Original profiles, empty metadata, tasks,
experts and datasets are input-only. This is not a generic mesh-loader binding:
the115 body/cap must be built by their physical assembly factory.
"""
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import os
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
PIPELINE=HERE.parents[1]
PROFILE=HERE.parent/'profiles/bottle115'
OLD=PIPELINE/'metadata'
ROBOTWIN=Path('/home/admin123/liminghe/vla-benchmark/RoboTwin')
TASK='labware_cap115_unscrew_pour'
OBJECTS=HERE/'assets/objects'
DESCRIPTIONS=HERE/'description/objects_description'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,j):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(j,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def input_paths():
    p=[PROFILE/n for n in ['physics_spec115.json','profile_manifest.json','body115.glb',
        'cap14.glb','body_model_data0.json','cap_model_data0.json']]
    p += [OLD/'registry.json']
    for asset in ['reagent_bottle_body','reagent_bottle_cap','petri_dish']:
        p += [OLD/'assets/objects'/asset/'points_info.json']
    p += [OLD/'assets/objects/petri_dish/model_data0.json',
          OLD/'description/objects_description/petri_dish/base0.json']
    return p

def semantics(kind,data):
    source='reagent_bottle_body' if kind=='body' else 'reagent_bottle_cap'
    result=copy.deepcopy(read(OLD/'assets/objects'/source/'points_info.json'))
    matrix_keys={'contact_points':'contact_points_pose','functional_points':'functional_matrix',
                 'target_points':'target_pose','orientation_points':'orientation_point'}
    for points,key in matrix_keys.items():
        for p in result[points]:
            m=np.asarray(data[key][p['id']],dtype=float)
            p['coordinate_in_this_local_mesh_m']=m[:3,3].tolist()
            p['usage'] += ' This115 profile is a separate asset; robot grasp remains unverified.'
    result['metadata_provenance']={'date':'2026-10-22','authoring':'local metadata from actual115 profile matrices',
        'not_official_original':True,'all_existing_ids_and_meanings_preserved':True,
        'frame':'local mesh coordinates; cap assembly origin .118m is NOT a local cap point',
        'source_model_data':str(PROFILE/f'{kind}_model_data0.json'),
        'source_model_data_sha256':sha(PROFILE/f'{kind}_model_data0.json'),
        'source_spec_sha256':sha(PROFILE/'physics_spec115.json'),
        'grasp_execution_verified':False,'full_task_verified':False}
    return result

def prepare():
    before={str(p):sha(p) for p in input_paths()}
    old=read(OLD/'registry.json');manifest=read(PROFILE/'profile_manifest.json')
    registry={'date':'2026-10-22','task_name_proposed':TASK,'status':'metadata_only; full task not implemented here',
        'OBJECTS_PATH':str(OBJECTS),'source_profile_manifest':str(PROFILE/'profile_manifest.json'),
        'source_profile_manifest_sha256':sha(PROFILE/'profile_manifest.json'),
        'input_sha256':before,'assets':{},'full_robot_task_verified':False,
        'binding':'Use absolute_modelname for original enrich_actors.115 physical assembly factory loads actual shapes; no90 collision fallback.'}
    for kind,visual in [('body','body115.glb'),('cap','cap14.glb')]:
        alias=f'reagent_bottle_{kind}115';dest=OBJECTS/alias
        dest.mkdir(parents=True,exist_ok=True)
        data=read(PROFILE/f'{kind}_model_data0.json')
        assert data['annotation_provenance']['grasp_execution_verified'] is False
        (dest/'model_data0.json').write_bytes((PROFILE/f'{kind}_model_data0.json').read_bytes())
        write(dest/'points_info.json',semantics(kind,data))
        (dest/'visual').mkdir(exist_ok=True);link=dest/'visual/base0.glb'
        if link.is_symlink():assert link.resolve()==(PROFILE/visual).resolve()
        else:
            if link.exists():raise FileExistsError(link)
            link.symlink_to(PROFILE/visual)
        noun='115-millimetre reagent bottle body' if kind=='body' else 'matching detachable reagent bottle screw-cap'
        desc={'raw_description':noun,'seen':[noun,'laboratory '+noun,'cylindrical '+noun],
            'unseen':['the115-profile vessel body' if kind=='body' else 'the separate lid of the115-profile bottle',
                      'tall reagent container body' if kind=='body' else 'detachable laboratory bottle closure'],
            'provenance':{'local_authoring':True,'phrase_split_only':True,'robot_grasp_verified':False,
                          'profile':manifest['profile'],'geometry_source_sha256':sha(PROFILE/visual)}}
        write(DESCRIPTIONS/alias/'base0.json',desc)
        registry['assets'][alias]={'asset_id':alias,'absolute_modelname':str(dest),'model_id':0,
            'model_data':str(dest/'model_data0.json'),'model_data_sha256':sha(dest/'model_data0.json'),
            'points_info':str(dest/'points_info.json'),'visual':str(PROFILE/visual),'visual_sha256':sha(PROFILE/visual),
            'episode_description_value':str(DESCRIPTIONS/alias/'base0'),
            'functional_ids':({'opening':0,'inner_floor':1,'rim':2,'bottom':3} if kind=='body'
                              else {'underside_opening':0,'twist_axis':1,'bottom':2}),
            'geometry_loader':'Bottle115Assembly factory; generic create_actor not registered',
            'generic_create_actor_ready':False,'grasp_execution_verified':False,
            'mass_kg':manifest['mass_kg'][kind]}
    petri=copy.deepcopy(old['assets']['petri_dish'])
    petri.update(asset_id='petri_dish',source_registry=str(OLD/'registry.json'),
                 reuse_existing_verified_asset=True,full_robot_task_verified=False)
    write(DESCRIPTIONS/'petri_dish/base0.json',read(OLD/'description/objects_description/petri_dish/base0.json'))
    petri['episode_description_value']=str(DESCRIPTIONS/'petri_dish/base0')
    registry['assets']['petri_dish']=petri
    actors={k:{'name':n,'description':d,'modelname':registry['assets'][a]['absolute_modelname']} for k,n,d,a in [
        ('self.bottle','115mm bottle body','Right arm body support; actual local opening .115m. Candidate grasp not yet verified for full task.','reagent_bottle_body115'),
        ('self.cap','matching detachable cap','Left arm cap operation; underside/opening and twist-axis are cap LOCAL points, not body assembly origin .118m.','reagent_bottle_cap115'),
        ('self.dish','existing Petri dish','Receiver for12 original dynamic capsules. Reuse verified asset ID; full transfer not yet verified.','petri_dish')]}
    bindings={'{A}':registry['assets']['reagent_bottle_body115']['episode_description_value'],
              '{B}':registry['assets']['reagent_bottle_cap115']['episode_description_value'],
              '{C}':registry['assets']['petri_dish']['episode_description_value'],'{a}':'right','{b}':'left'}
    starts=['Hold {A} with {a}','Support {A} using {a}','Keep {A} controlled with {a}',
            'Use {a} to secure {A}','Stabilize {A} with {a}','Maintain control of {A} using {a}']
    seen=[s+', use {b} to unscrew and clear {B}, then pour all twelve tablets into {C} while keeping the cap safe.' for s in starts]
    seen += [s+'; with {b}, remove {B} by unscrewing, keep it clear, and transfer all twelve tablets into {C}.' for s in starts]
    unseen=[s+', have {b} unthread and move {B} clear of the mouth, then empty all twelve capsules into {C} safely.' for s in starts]
    unseen += [s+' while {b} twists {B} free and keeps it away from the opening; pour the full twelve-tablet contents into {C}.' for s in starts]
    write(HERE/'description/task_instruction'/f'{TASK}.json',{'seen':seen,'unseen':unseen,
        'provenance':{'authoring':'current Codex session, no API call','split':'phrasing only',
        'status':'future physical task specification, not a success result'}})
    (HERE/'description/utils').mkdir(parents=True,exist_ok=True)
    write(HERE/'registry.json',registry);write(HERE/'actor_list.json',actors);write(HERE/'episode_language_bindings.json',bindings)
    write(HERE/'task_info_template.json',{'task_name':TASK,
        'task_description':'PENDING FULL TASK IMPLEMENTATION. Intended task: right arm supports the115 bottle; left arm physically grasps, unscrews and clears its cap; right arm pours all12 original dynamic capsules into the Petri dish and the transfer settles. No pose/velocity writes, fixture, copied success flag or intermediate body-hold success. Implement and physically verify the full task and actual API before generating/running an expert.',
        'current_code':f'class gpt_{TASK}({TASK}):\n    def play_once(self):\n        pass\n',
        'actor_list':actors,'task_module_implemented_here':False,'expert_generated_here':False,
        'full_physical_task_verified':False,'planned_language_bindings':bindings})
    assert before=={str(p):sha(p) for p in input_paths()}
    return registry,actors,bindings

def validate(registry,actors,bindings):
    oldcwd=Path.cwd();sys.path[:0]=[str(ROBOTWIN/'code_gen'),str(ROBOTWIN)]
    try:
        os.chdir(ROBOTWIN)
        import test_gen_code as original
        enriched=original.enrich_actors(actors)
        rows=[]
        for key,info in actors.items():
            path=Path(info['modelname']);points=read(path/'points_info.json');data=read(path/'model_data0.json')
            assert enriched[key]['functional_points']==points['functional_points']
            assert enriched[key]['contact_points']==points['contact_points']
            for group,mkey in [('functional_points','functional_matrix'),('contact_points','contact_points_pose'),('target_points','target_pose'),('orientation_points','orientation_point')]:
                assert [p['id'] for p in points[group]]==list(range(len(data[mkey])))
                for matrix in data[mkey]:
                    m=np.array(matrix,dtype=float);assert m.shape==(4,4) and np.isfinite(m).all()
                    assert np.allclose(m[3],[0,0,0,1]) and np.allclose(m[:3,:3].T@m[:3,:3],np.eye(3),atol=1e-7)
                    assert abs(np.linalg.det(m[:3,:3])-1)<1e-7
            rows.append({'actor':key,'modelname':str(path),'original_enrich_actors_passed':True,
                         'all_point_ids_and_rigid_matrices_passed':True})
        spec=importlib.util.spec_from_file_location('full_labware_original_language',ROBOTWIN/'description/utils/generate_episode_instructions.py')
        language=importlib.util.module_from_spec(spec);spec.loader.exec_module(language)
        language.parent_directory=str(HERE/'description/utils')
        text=language.generate_episode_descriptions(TASK,[bindings],12,seed=42)[0]
        for split in ['seen','unseen']:
            assert len(text[split])==12 and len(set(text[split]))==12
            assert all('{' not in t and '}' not in t and 'the right arm' in t and 'the left arm' in t for t in text[split])
        assert registry['input_sha256']=={str(p):sha(p) for p in input_paths()}
        write(HERE/'enriched_actors.json',enriched);write(HERE/'language_validation.json',{'passed':True,'seen_count':12,'unseen_count':12,'outputs':text,'all_placeholders_resolved':True,'actual_function':'original generate_episode_descriptions'})
        write(HERE/'component_validation.json',{'date':'2026-10-22','passed':True,'scope':'metadata/Agent enrichment/rigid-matrix/official language components only',
            'rows':rows,'language_seen_unseen_counts':[12,12],'source_inputs_unchanged':True,
            'geometry_loader_run':False,'robot_full_task_run':False,'expert_generated':False,'dataset_collected':False,
            'source_sha256':{'original_enrich_actors':sha(ROBOTWIN/'code_gen/test_gen_code.py'),
                             'original_language':sha(ROBOTWIN/'description/utils/generate_episode_instructions.py')}})
    finally:os.chdir(oldcwd)

if __name__=='__main__':
    registry,actors,bindings=prepare();validate(registry,actors,bindings)
    print(json.dumps({'metadata_component_passed':True,'path':str(HERE),'robot_full_task_verified':False},ensure_ascii=False))
