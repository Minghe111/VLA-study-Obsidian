#!/usr/bin/env python3
"""Independent final-dataset readback; no simulator or collector is invoked.

Reads actual HDF5 camera bytes using RoboTwin's original decoder, verifies
command/state/clock/source provenance, then exports only stored RGB keyframes.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import traceback
import numpy as np

ROOT=Path(__file__).resolve().parent
ROBOTWIN=Path('/home/admin123/liminghe/vla-benchmark/RoboTwin')
CAMERAS=('cam_head','cam_left_wrist','cam_right_wrist')
LABELS=('Head RGB','Left wrist RGB','Right wrist RGB')
HAND_COLUMNS=np.array([6,7,8,9,10,11,18,19,20,21,22,23])


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def read_json(path):return json.loads(Path(path).read_text())


def text_value(value):
    if isinstance(value,bytes):return value.decode('utf-8')
    return str(value)


def require(condition,message):
    if not condition:raise AssertionError(message)


def dump(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def load_decoder():
    path=ROBOTWIN/'data/decode_image_bit.py'
    spec=importlib.util.spec_from_file_location('independent_official_image_decoder',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.decode_image_bit,{'path':str(path),'sha256':sha(path)}


def official_wrapper_equivalence(actual_path,tested_path,task_name='labware_pick_place'):
    """Derive wrapper AST from the real generate_code res=prefix+response node."""
    actual=ast.parse(Path(actual_path).read_text())
    tested=ast.parse(Path(tested_path).read_text())
    source=ast.parse((ROBOTWIN/'code_gen/task_generation.py').read_text())
    function=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='generate_code')
    prefixes=[]
    for node in function.body:
        if not isinstance(node,ast.Assign) or not any(isinstance(t,ast.Name) and t.id=='res' for t in node.targets):continue
        if isinstance(node.value,ast.BinOp) and isinstance(node.value.op,ast.Add) and isinstance(node.value.left,ast.JoinedStr):
            text=[]
            for part in node.value.left.values:
                if isinstance(part,ast.Constant) and isinstance(part.value,str):text.append(part.value)
                elif isinstance(part,ast.FormattedValue) and isinstance(part.value,ast.Name) and part.value.id=='task_name':text.append(task_name)
                else:raise AssertionError('Unexpected official wrapper format expression')
            prefixes.append(''.join(text))
    require(len(prefixes)==1,'Could not uniquely identify original response-parser wrapper')
    wrapper=ast.parse(prefixes[0]+'def play_once(self):\n        pass\n')
    require(len(actual.body)==len(wrapper.body),'Extra top-level generated statements')
    cls=actual.body[-1]
    require(isinstance(cls,ast.ClassDef) and cls.name=='gpt_'+task_name,'Unexpected generated class')
    require(len(cls.body)==1 and isinstance(cls.body[0],ast.FunctionDef) and cls.body[0].name=='play_once','Generated class contains extra executable content')
    expected=wrapper
    expected.body[-1].body=[cls.body[0]]
    require(ast.dump(actual,include_attributes=False)==ast.dump(expected,include_attributes=False),'Generated imports/class differ from actual original wrapper')
    old_class=next(n for n in tested.body if isinstance(n,ast.ClassDef) and n.name==cls.name)
    old_method=next(n for n in old_class.body if isinstance(n,ast.FunctionDef) and n.name=='play_once')
    method=ast.dump(cls.body[0],include_attributes=False)
    require(method==ast.dump(old_method,include_attributes=False),'Parsed method differs from physically tested candidate')
    return hashlib.sha256(method.encode()).hexdigest()


def verify_provenance(collection,launch):
    hashes=launch['source_sha256']; checked={}
    used_sources={}
    for path,digest in hashes.items():
        snapshot=launch.get('source_snapshots',{}).get(path)
        if snapshot is None:
            legacy_name='original_collect_data.py' if path.endswith('/RoboTwin/scripts/collect_data.py') else Path(path).name
            legacy=collection/'source_snapshot'/legacy_name
            if legacy.is_file():snapshot=str(legacy)
        target=Path(snapshot) if snapshot else Path(path)
        require(target.is_file(),'Recorded source/snapshot is missing: '+str(target))
        actual=sha(target)
        require(actual==digest,'Recorded source snapshot hash mismatch: '+path)
        checked[path]=actual
        used_sources[path]={'verified_file':str(target),'snapshot_used':snapshot is not None,
            'live_file_still_matches':Path(path).is_file() and sha(path)==digest}
    expert=launch['expert']
    candidate=Path(expert['source']); digest=sha(candidate)
    require(digest==expert['sha256'],'Candidate differs from collection launch')
    require(expert.get('agent_generated') is True,'Final episode lacks a verified Agent candidate')
    require(expert['kind']=='current_session_candidate_officially_tested','Unexpected candidate provenance')
    test_path=Path(expert['official_test_receipt']); test=read_json(test_path)
    tested_digest=expert.get('tested_candidate_sha256',digest)
    require(test['candidate_sha256']==tested_digest,'Official test receipt candidate mismatch')
    for field in ('original_test_gen_code_run_called','source_unchanged','candidate_unchanged','official_threshold_0_5_met'):
        require(test[field] is True,'Official Agent test field failed: '+field)
    require(test['status']=='physical_test_complete','Official Agent test incomplete')
    require(test['check_num']==10,'Expected original ten-trial official test')
    require(float(test['success_rate'])>=.5,'Official success threshold failed')
    frozen=Path(test['frozen_candidate'])
    require(sha(frozen)==tested_digest,'Tested frozen candidate differs from physical-test receipt')
    parser_proof=None
    if digest!=tested_digest:
        parser_path=Path(expert['parser_receipt']); parsed=read_json(parser_path)
        require(parsed['generated_sha256']==digest and parsed['input_candidate_sha256']==tested_digest,'Parser receipt breaks tested/generated SHA linkage')
        require(Path(parsed['generated_file']).resolve()==candidate.resolve(),'Parser output path differs from actual collection class')
        require(Path(parsed['linked_physical_test_receipt']).resolve()==test_path.resolve(),'Parser links a different physical-test receipt')
        for field in ('method_ast_equal','generated_template_structure_passed','generated_class_import_passed',
            'original_generate_code_post_response_executed','task_matches_linked_physics','source_unchanged'):
            require(parsed[field] is True,'Parser contract failed: '+field)
        require(parsed['new_physical_trials']==0 and parsed['score'] is None,'Parser-only check misstates physical evaluation')
        require(parsed['task_implementation_sha256']==test['task_implementation_sha256'],'Parser task implementation differs from tested implementation')
        method_sha=official_wrapper_equivalence(candidate,frozen)
        require(method_sha==expert['method_ast_sha256'],'Method AST digest differs from collector verification')
        parser_proof={'receipt':str(parser_path),'sha256':sha(parser_path),
            'actual_official_wrapper_AST_checked':True,'method_ast_sha256':method_sha,
            'method_matches_physically_tested_candidate':True,'new_physical_trials':0}
    for name,expected in test['task_implementation_sha256'].items():
        require(hashes[str(ROOT/name)]==expected,'Test/collection implementation mismatch: '+name)
    for name,record in test.get('actual_imported_runtime_modules',{}).items():
        path=Path(record['file'])
        require(sha(path)==record['sha256'],'Imported Agent/runtime module changed: '+name)
        checked[str(path)]=record['sha256']
    for name,expected in test['source']['files'].items():
        require(sha(ROBOTWIN/name)==expected,'Original Agent source changed: '+name)
    return {'verified':True,'source_sha256':checked,'source_verification_files':used_sources,'actual_collected_candidate_sha256':digest,
        'physically_tested_candidate_sha256':tested_digest,'byte_identity_to_tested_candidate':digest==tested_digest,
        'parser_proof':parser_proof,'official_test_receipt':str(test_path),'official_test_receipt_sha256':sha(test_path),
        'official_test_trials':test['check_num'],'official_test_success_rate':test['success_rate'],
        'generation_transport':test['generation_transport'],'remote_api_auto_loop':test['remote_api_auto_loop'],
        'actual_llm_request_sent':test['actual_llm_request_sent']}


def camera_source_contract(provenance):
    import yaml
    source_paths=provenance['source_sha256']
    base_paths=[Path(p) for p in source_paths if p.endswith('/envs/_base_task.py')]
    require(len(base_paths)==1,'Missing unambiguous actual Base_Task source')
    base=base_paths[0];native=base.parents[1]
    adapter=native/'integrations/realman_rm65b_gen4/three_camera_rgb.py'
    profile_path=native/'integrations/realman_rm65b_gen4/camera_profiles/three_d435_rgb_default.yml'
    profile=yaml.safe_load(profile_path.read_text())
    require(profile['hardware_io'] is False,'Camera profile unexpectedly enables hardware')
    expected={'head_camera':'camera_link','left_camera':'l_link6','right_camera':'r_link6'}
    require({k:v['link'] for k,v in profile['mounts'].items()}==expected,'Unexpected camera parent-link mapping')
    tree=ast.parse(adapter.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MountedD435RGBCamera')
    methods={n.name:n for n in cls.body if isinstance(n,ast.FunctionDef)}
    require('update_mount' in methods and 'update_wrist_camera' in methods,'Missing authoritative robot-mount adapter')
    base_tree=ast.parse(base.read_text())
    base_cls=next(n for n in base_tree.body if isinstance(n,ast.ClassDef) and n.name=='Base_Task')
    update=next(n for n in base_cls.body if isinstance(n,ast.FunctionDef) and n.name=='_update_render')
    require(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='update_mount' for n in ast.walk(update)),
        'Actual Base_Task does not call authoritative update_mount')
    return {'profile':profile,'profile_path':str(profile_path),'profile_sha256':sha(profile_path),
        'camera_adapter_path':str(adapter),'camera_adapter_sha256':sha(adapter),
        'Base_Task_update_render_calls_update_mount':True,'configured_parent_links':expected,
        'generic_camera_link_warning_scope':'Generic fallback links are not the authoritative custom optical mounts',
        'evidence_scope':'Current source/profile plus independently reviewed saved camera images; no runtime extrinsic readback or physical calibration claim'}


def get_phase_frame(phases,token,fraction=.5):
    indices=np.array([i for i,p in enumerate(phases) if token in p],dtype=int)
    require(len(indices)>0,'Missing actual recorded phase: '+token)
    return int(indices[min(len(indices)-1,int(fraction*(len(indices)-1)))])


def frame_selection(phases,metrics):
    # Image k is the observation BEFORE action k; metrics[k] are AFTER that
    # action. Use metrics[k-1] when labelling a stored image, never one-step lead.
    previous=[None]+metrics[:-1]
    gaps=np.array([m['bottom_gap_m'] if m is not None else -np.inf for m in previous])
    candidates=np.array([i for i,m in enumerate(previous) if m is not None and
        m['bottom_gap_m']>=.02 and m['opposed_contact'] and m['table_contact_N']<1e-5])
    require(len(candidates)>0,'No stored observation has measured airborne opposing contact')
    airborne=int(candidates[np.argmax(gaps[candidates])])
    return [('Prepare',get_phase_frame(phases,'prepare_three_finger_grasp')),
        ('Approach',get_phase_frame(phases,'approach_grasp')),
        ('Airborne opposing grasp',airborne),('Release',get_phase_frame(phases,'release')),
        ('Final stored observation',len(phases)-1)],previous


def make_montage(out_dir,episode_name,selected,images,times,phases,before_metrics):
    from PIL import Image,ImageDraw,ImageFont
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font=ImageFont.truetype(font_path,21);small=ImageFont.truetype(font_path,17)
    row_height=550
    canvas=Image.new('RGB',(1920,len(selected)*row_height+70),(242,245,247))
    draw=ImageDraw.Draw(canvas)
    draw.text((18,10),'Actual stored RGB only | RM65-B simulation | '+episode_name,font=font,fill=(20,35,50))
    draw.text((18,40),'Images are pre-action observations; metric labels use the preceding transition end.',font=small,fill=(60,75,88))
    records=[]
    for row,(title,index) in enumerate(selected):
        yy=70+row*row_height
        metric=before_metrics[index]
        suffix=('gap=%.2f mm | opposed=%s | table=%.3f N'%(metric['bottom_gap_m']*1000,
            metric['opposed_contact'],metric['table_contact_N'])) if metric else 'Initial observation'
        draw.text((18,yy+4),f'{title} | t={times[index]:.2f} s | frame={index} | {suffix}',font=font,fill=(20,35,50))
        draw.text((18,yy+30),'Next recorded action phase: '+phases[index],font=small,fill=(70,85,95))
        row_img=Image.new('RGB',(1920,520),(242,245,247));rd=ImageDraw.Draw(row_img)
        for col,(camera,label) in enumerate(zip(CAMERAS,LABELS)):
            rgb=images[camera][index]
            image=Image.fromarray(rgb)
            canvas.paste(image,(col*640,yy+70))
            row_img.paste(image,(col*640,40))
            draw.text((col*640+12,yy+51),label,font=small,fill=(50,65,80))
            rd.text((col*640+12,9),f'{label} | t={times[index]:.2f}s',font=small,fill=(20,35,50))
        stem=f'{episode_name}_{row+1:02d}_{title.lower().replace(" ","_")}'
        row_file=out_dir/(stem+'.png');row_img.save(row_file)
        records.append({'title':title,'frame':index,'sim_time_seconds':float(times[index]),
            'image_path':str(row_file),'metrics_alignment':'preceding transition end / current pre-action observation',
            'metrics':metric})
    montage=out_dir/(episode_name+'_keyframes.png');canvas.save(montage)
    return {'montage':str(montage),'frames':records,'image_source':'decoded original HDF5 bytes only; no new render or state writes'}


def head_framing_geometry_audit(collection,capture_dir,launch,receipt,ticks):
    if not launch.get('camera_framing'):return {'applied':False,'scope':'Original nominal framing; no extra full-target-crop guarantee'}
    import trimesh
    framing=read_json(capture_dir/'camera_framing_receipt.json')
    require(framing==receipt['camera_framing'],'Saved and embedded camera preset receipts differ')
    require(launch['source_sha256'][framing['source']]==framing['source_sha256'],'Applied camera framing source differs from launch snapshot')
    require(framing['wrist_local_transforms_unchanged'] is True and framing['physical_d435_calibration'] is False,'Unexpected camera calibration/scope')
    require(framing['persistent_on_update_mount'] is True and framing['physics_steps_taken']==0 and framing['robot_joint_or_object_pose_modified'] is False,'Framing changed physical state')
    preset_path=launch.get('source_snapshots',{}).get(launch['camera_framing'],launch['camera_framing'])
    preset=read_json(preset_path)
    require(framing['config']==preset,'Applied camera config differs from launch preset')
    registry=read_json(ROOT/'metadata/registry.json')
    entry=registry['assets'][launch['args']['asset_kind']]
    mesh_path=Path(entry['absolute_modelname'])/'collision/base0.glb'
    model=read_json(Path(entry['absolute_modelname'])/'model_data0.json')
    mesh=trimesh.load(mesh_path,force='scene').dump(concatenate=True)
    vertices=np.asarray(mesh.vertices)*np.asarray(model['scale'])
    camera=np.asarray(framing['runtime_head_world_pose']);k=np.asarray(framing['runtime_intrinsic_matrix'])
    trace=read_json(capture_dir/'physics_trace.json');poses={int(row['tick']):np.asarray(row['pose']) for row in trace}
    projections=[]
    for tick in ticks[:,0]:
        require(int(tick) in poses,'Missing actual target pose at stored image timestamp')
        matrix=poses[int(tick)];world=vertices@matrix[:3,:3].T+matrix[:3,3]
        local=(world-camera[:3,3])@camera[:3,:3]
        require(np.all(local[:,0]>.005),'Target partly behind the head camera')
        xy=np.column_stack([k[0,2]-k[0,0]*local[:,1]/local[:,0],k[1,2]-k[1,1]*local[:,2]/local[:,0]])
        low=xy.min(0);high=xy.max(0)
        margins=np.array([low[0],low[1],639-high[0],479-high[1]])
        require(float(margins.min())>=40.,'New head framing still crops or lacks 40px target margin')
        projections.append({'tick':int(tick),'bbox_xyxy_pixels':[float(v) for v in [*low,*high]],'minimum_margin_pixels':float(margins.min())})
    return {'applied':True,'preset':preset,'source_sha256':framing['source_sha256'],
        'actual_runtime_fixed_head_world_pose':camera.tolist(),'actual_intrinsic_matrix':k.tolist(),
        'target_collision_mesh':str(mesh_path),'target_collision_mesh_sha256':sha(mesh_path),
        'stored_observations_checked':len(projections),'all_target_geometry_inside_with_at_least_40px_margin':True,
        'minimum_margin_pixels':min(p['minimum_margin_pixels'] for p in projections),
        'projections':projections,
        'scope':'Geometric frustum cropping check on actual stored-observation times and fixed-head source; finger occlusion and physical calibration are separate'}


def audit_episode(collection,hdf_path,decode,visual_dir):
    import h5py
    index=int(hdf_path.stem.split('_')[-1])
    planned_path=collection/'_traj_data'/f'episode{index}.json'
    plan=read_json(planned_path);seed=int(plan['seed'])
    launch=read_json(collection/'launch.json');replay_source=launch.get('replay_source')
    planning_collection=collection
    replay_source_proof=None
    if replay_source:
        planning_collection=Path(replay_source['source_collection'])
        source_summary=planning_collection/'collection_receipt.json'
        require(sha(source_summary)==replay_source['source_collection_sha256'],'Replay source receipt changed')
        require(read_json(source_summary)['status']=='passed','Replay source was not successfully collected')
        require(replay_source['new_planning_performed'] is False,'Unexpected new-planning claim')
        expected=[digest for name,digest in replay_source['trajectory_sha256'].items() if Path(name).name==planned_path.name]
        require(len(expected)==1 and sha(planned_path)==expected[0],'Copied trajectory hash differs from replay provenance')
        original_plan=planning_collection/'_traj_data'/planned_path.name
        require(sha(original_plan)==sha(planned_path),'Copied and source trajectories differ')
        for name in ('labware_task.py','robot_expert.py'):
            source_launch=read_json(planning_collection/'launch.json')
            require(launch['source_sha256'][str(ROOT/name)]==source_launch['source_sha256'][str(ROOT/name)],'Replay changed task/expert implementation')
        require(launch['args']['asset_kind']==source_launch['args']['asset_kind'] and launch['args']['arm']==source_launch['args']['arm'],'Replay asset/arm differs')
        replay_source_proof={'source_collection':str(planning_collection),'source_collection_receipt_sha256':sha(source_summary),
            'copied_trajectory_sha256':sha(planned_path),'new_planning_performed':False}
    planning_receipt_path=planning_collection/'runs'/f'plan_ep{index:07d}_seed{seed:05d}'/'rollout_receipt.json'
    plan_receipt=read_json(planning_receipt_path)
    capture_dir=collection/'runs'/f'capture_ep{index:07d}_seed{seed:05d}'
    replay_receipt=read_json(capture_dir/'rollout_receipt.json')
    for name,receipt in [('plan',plan_receipt),('replay',replay_receipt),('saved_trajectory',plan['success_receipt'])]:
        require(receipt['success'] is True and receipt['error'] is None,name+' physical rollout failed')
        require(receipt['actual_robot_driven'] is True and receipt['object_state_writes_after_reset']==0,name+' is not contact-only robot driven')
        require(receipt['metrics']['lift_achieved'] is True and receipt['metrics']['max_hold_seconds']>=.5,name+' lacks actual lift/hold')
        require(receipt['metrics']['stable_place_seconds']>=.5,name+' lacks stable placement')
        require(receipt['metrics']['thumb_contact_N']+receipt['metrics']['finger_contact_N']<.02,name+' still holding bottle at end')
    require(plan_receipt['expert']==replay_receipt['expert']==plan['success_receipt']['expert'],'Plan/replay candidate provenance differs')
    native_path=planning_collection/'runs'/f'plan_ep{index:07d}_seed{seed:05d}'/'expert_source_receipt.json'
    native=read_json(native_path)
    require(sha(native['native_expert_source'])==native['native_expert_sha256'],'Frozen native expert source changed')
    require(sha(native['full_mesh_source'])==native['full_mesh_sha256'],'Frozen native mesh auditor changed')
    require(native['shared_source_modified'] is False and native['hardware_io'] is False,'Native expert scope changed')
    language=read_json(collection/'instruction'/f'episode_{index:07d}.json')
    require(language['seed']==seed,'Instruction seed does not match physical replay')
    for split in ('seen','unseen'):
        words=language[split]
        require(len(words)==12,split+' instruction count is not 12')
        require(all(isinstance(w,str) and w.strip() and '{' not in w and '}' not in w for w in words),'Missing or unsubstituted task instruction')
        require(len(set(words))>1,split+' instructions collapsed to a fallback')
    with h5py.File(hdf_path,'r') as f:
        state=f['state/vector'][:];action=f['action/vector'][:];nxt=f['next_state/vector'][:]
        n=len(state);ticks=f['additional_info/ticks'][:];times=f['timestamp'][:]
        require(n>1 and state.shape==action.shape==nxt.shape==(n,24),'Wrong state/action dimensionality')
        require(all(np.isfinite(a).all() for a in (state,action,nxt)),'Nonfinite state/action')
        require(float(action[:,HAND_COLUMNS].min())>=-1e-3 and float(action[:,HAND_COLUMNS].max())<=1000+1e-3,'Requested hand SDK commands are outside 0..1000')
        require(np.array_equal(state[1:],nxt[:-1]),'Discontinuous measured state chain')
        require(np.all(ticks[:,1]-ticks[:,0]==15) and np.all(ticks[1:,0]==ticks[:-1,1]),'Clock is not contiguous 15 ticks')
        require(np.allclose(times,ticks[:,0]*.004,rtol=0,atol=1e-10),'Timestamp/physics clock mismatch')
        require(np.allclose(np.diff(times),.06,rtol=0,atol=1e-10),'Action cadence is not 60 ms')
        require(float(f['additional_info/control_dt_seconds'][()])==.06 and float(f['additional_info/physics_dt_seconds'][()])==.004,'Wrong declared clocks')
        require(str(f.attrs['order'])=='left_arm6,left_hand6,right_arm6,right_hand6','Wrong SDK24 ordering')
        require(bool(f.attrs['success']) is True,'HDF declares failure')
        require(bool(f.attrs['sdk_mapping_hardware_calibrated']) is False and bool(f.attrs['camera_mount_hardware_calibrated']) is False,'Unexpected hardware-calibration claim')
        for group,array in [('state',state),('action',action)]:
            for name,sl in [('left_arm_joint_states',slice(0,6)),('left_ee_joint_states',slice(6,12)),
                ('right_arm_joint_states',slice(12,18)),('right_ee_joint_states',slice(18,24))]:
                require(np.array_equal(f[group+'/'+name][:],array[:,sl]),'SDK24 component mapping differs: '+group+'/'+name)
        planned=np.asarray([r['action'] for r in plan['rows']],dtype=float)
        require(planned.shape==action.shape,'Plan/replay target count differs')
        target_error=float(np.max(np.abs(planned-action)))
        require(target_error<=1e-3,'HDF drive target differs from saved requested trajectory')
        drives=[json.loads(v) for v in f['diagnostics/actual_drive_records'].asstr()[:]]
        require(len(drives)==n and all(isinstance(d,dict) for d in drives),'Missing actual executor drive readbacks')
        for i,d in enumerate(drives):
            require(d['executed_physics_steps']==15 and d['endpoint_command_written'],'Incomplete physical command interval')
            require(d['physics_step_start']==int(ticks[i,0]) and d['physics_step_end']==int(ticks[i,1]),'Drive/timestamp mismatch')
            require(np.array_equal(np.asarray(d['actual_final_drive_command24'],dtype=np.float32),action[i]),'HDF action is not actual drive readback')
            require(np.max(np.abs(np.asarray(d['raw_model_action24'])-planned[i]))<=1e-3,'Executor requested target differs from plan')
        stored_language=json.loads(text_value(f['language_instructions'][()]))
        require(stored_language==language['seen'],'HDF language differs from official seen instruction file')
        metrics=[json.loads(v) for v in f['diagnostics/physical_metrics'].asstr()[:]]
        phases=list(f['diagnostics/phase'].asstr()[:])
        receipt=json.loads(text_value(f['diagnostics/task_receipt'][()]))
        require(receipt==replay_receipt,'Embedded task receipt differs from captured run receipt')
        framing_quality=head_framing_geometry_audit(collection,capture_dir,launch,replay_receipt,ticks)
        require(metrics[-1]['success'] is True and metrics[-1]['stable_place_seconds']>=.5,'Stored final physical success failed')
        selected,before_metrics=frame_selection(phases,metrics)
        wanted={i for _,i in selected};images={camera:{} for camera in CAMERAS};stats={}
        for camera in CAMERAS:
            ds=f['vision/'+camera+'/colors'];require(len(ds)==n,'Camera stream length mismatch')
            hashes=[];means=[];stds=[]
            for i,encoded in enumerate(ds):
                rgb=decode(encoded)
                require(rgb.dtype==np.uint8 and rgb.shape==(480,640,3),'Decoded camera type/shape mismatch')
                deviation=float(rgb.std());require(deviation>1.,'Blank camera image: '+camera+' frame '+str(i))
                hashes.append(hashlib.sha256(rgb.tobytes()).hexdigest());means.append(float(rgb.mean()));stds.append(deviation)
                if i in wanted:images[camera][i]=rgb.copy()
            stats[camera]={'decoded_frames':n,'unique_decoded_frames':len(set(hashes)),
                'shape':[480,640,3],'dtype':'uint8','mean_range':[min(means),max(means)],'minimum_std':min(stds),
                'decoded_frame_hash_chain_sha256':hashlib.sha256(''.join(hashes).encode()).hexdigest()}
        cross_camera=[]
        for title,frame_index in selected:
            hashes={c:hashlib.sha256(images[c][frame_index].tobytes()).hexdigest() for c in CAMERAS}
            require(len(set(hashes.values()))==3,'Camera views are exact duplicates at frame '+str(frame_index))
            pairs={}
            for i,a in enumerate(CAMERAS):
                for b in CAMERAS[i+1:]:
                    pairs[a+' vs '+b]=float(np.abs(images[a][frame_index].astype(float)-images[b][frame_index].astype(float)).mean())
            cross_camera.append({'title':title,'frame':frame_index,'decoded_sha256':hashes,'pair_mean_absolute_pixel_difference':pairs})
        source_data_comparison=None
        if replay_source:
            with h5py.File(planning_collection/'data'/hdf_path.name,'r') as original:
                require(original['action/vector'].shape==action.shape,'Source data has different action count')
                differences={key:float(np.max(np.abs(value-original[key][:]))) for key,value in
                    [('state/vector',state),('action/vector',action),('next_state/vector',nxt)]}
                pixel_changes={}
                for camera in CAMERAS:
                    changes=[]
                    for _,frame_index in selected:
                        source_rgb=decode(original['vision/'+camera+'/colors'][frame_index])
                        changes.append(float(np.abs(images[camera][frame_index].astype(float)-source_rgb.astype(float)).mean()))
                    pixel_changes[camera]={'selected_frame_mean_absolute_pixel_differences':changes,
                        'all_selected_frames_exactly_unchanged':all(v==0 for v in changes)}
                if framing_quality['applied']:
                    require(any(v>1. for v in pixel_changes['cam_head']['selected_frame_mean_absolute_pixel_differences']),
                        'New head framing did not produce changed stored RGB')
                source_data_comparison={'numeric_trajectory_max_absolute_differences':differences,
                    'camera_pixel_changes':pixel_changes,'scope':'Observed new-versus-source camera/numeric comparison; no relabelling source RGB as new frames'}
        visuals=make_montage(visual_dir,hdf_path.stem,selected,images,times,phases,before_metrics)
        # Images were BEFORE each action; final metrics refer to its measured end.
        return {'passed':True,'hdf5':str(hdf_path),'sha256':sha(hdf_path),'seed':seed,'frames_per_camera':n,
            'state_shape':list(state.shape),'action_shape':list(action.shape),'duration_seconds':float(ticks[-1,1]*.004),
            'state_chain_exact':True,'actual_drive_readbacks_match':True,'saved_requested_target_max_error':target_error,
            'maximum_bottom_lift_m':max(m['bottom_gap_m'] for m in metrics),
            'airborne_opposing_observation_count':sum(bool(m and m['bottom_gap_m']>=.02 and m['opposed_contact'] and m['table_contact_N']<1e-5) for m in before_metrics),
            'target_next_measured_state_max_difference':float(np.max(abs(action-nxt))),
            'plan_physical_success':True,'replay_physical_success':True,'physical_final':metrics[-1],
            'language_counts':{s:len(language[s]) for s in ('seen','unseen')},'all_camera_frames_decoded':stats,
            'cross_camera_keyframe_identity_check':{'all_views_distinct':True,'keyframes':cross_camera},
            'head_framing_geometry_check':framing_quality,'source_data_comparison':source_data_comparison,
            'visualization':visuals,'plan_receipt_sha256':sha(planning_receipt_path),
            'planning_source_collection':str(planning_collection),'replay_source_proof':replay_source_proof,
            'replay_receipt_sha256':sha(capture_dir/'rollout_receipt.json'),'frozen_native_source_receipt_sha256':sha(native_path)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection',type=Path,required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--visual-dir',type=Path)
    args=p.parse_args();collection=args.collection.resolve(strict=True)
    output=(args.output or collection/'independent_dataset_audit.json').resolve()
    visual=(args.visual_dir or ROOT/'dataset_visualization'/collection.name).resolve()
    if output.exists():raise FileExistsError('Refusing to overwrite prior audit: '+str(output))
    if visual.exists():raise FileExistsError('Use a fresh visualization directory: '+str(visual))
    report={'passed':False,'collection':str(collection),'auditor':str(Path(__file__).resolve()),
        'auditor_sha256':sha(__file__),'gpu_or_simulator_used':False,'episodes':[]}
    try:
        completed=read_json(collection/'collection_receipt.json')
        require(completed['status']=='passed','Collection did not finish successfully')
        launch=read_json(collection/'launch.json')
        report['provenance']=verify_provenance(collection,launch)
        report['camera_mount_source_contract']=camera_source_contract(report['provenance'])
        decode,decoder=load_decoder();report['decoder']=decoder
        paths=sorted((collection/'data').glob('episode_*.hdf5'))
        require(len(paths)>0 and len(paths)==completed['requested_episodes']==completed['validated_episodes'],'Requested/exported episode counts differ')
        require(completed['official_source_unchanged'] is True,'Original collection source changed')
        for path in paths:report['episodes'].append(audit_episode(collection,path,decode,visual))
        report['passed']=True
        report['scope']='Final saved HDF5, all camera buffers, physical plan/replay receipts and fixed source provenance; no SR or physical-robot claim'
    except BaseException as exc:
        report['error']={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
        dump(output,report)
        raise
    dump(output,report)
    print(json.dumps({'passed':True,'audit':str(output),'visualization':str(visual),'episodes':len(report['episodes'])}))


if __name__=='__main__':main()
