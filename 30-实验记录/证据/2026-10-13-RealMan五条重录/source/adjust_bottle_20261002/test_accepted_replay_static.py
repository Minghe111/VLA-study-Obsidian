"""Pure CPU replay guard/artifact tests; never call replay, scene, GPU or exporter.
Pass --candidate/--capture snapshots if the accepted source is not on server yet.
"""
from pathlib import Path
import argparse,ast,copy,json,hashlib,tempfile,types,sys
import numpy as np


def audit(base,candidate,capture):
    source=base/'accepted_reset_replay.py';tree=ast.parse(source.read_text());compile(tree,str(source),'exec')
    if not (isinstance(tree.body[-1],ast.Expr) and ast.unparse(tree.body[-1].value)=='_run_accepted_reset_replay(globals())'):
        raise ValueError('Replay entrypoint changed; review static extraction')
    ctx={};exec(compile(ast.Module(body=tree.body[:-1],type_ignores=[]),str(source),'exec'),ctx)
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();checks=[]
    def expect_fail(name,fn):
        try:fn()
        except (RuntimeError,ValueError) as e:
            checks.append({'name':name,'rejected':True,'reason':str(e)});return
        raise AssertionError('Guard failed '+name)
    def worker():
        t=types.SimpleNamespace(robot=types.SimpleNamespace(communication_flag=False),planned_joints_legal=lambda:(True,0),plan_success=True,check_success=lambda:True)
        return dict(accepted=True,accepted_replay_version='v12',report={'accepted':True,'result':'physics_success','canonical_sources_unchanged':True},task=t,args={},BASE=base,OUT=base,protected=[],sha=sha,accepted_grasp_setting={},accepted_seed=7,RealManRobot=object,Base_Task=object)
    for label,changes in [('failed',{'accepted':False}),('missingversion',{'accepted_replay_version':None}),('v10',{'accepted_replay_version':'v10'}),('v9',{'accepted_replay_version':'v9'}),('wrongreport',{'report':{'accepted':True,'result':'physics_failed','canonical_sources_unchanged':True}}),('sourcesmissing',{'report':{'accepted':True,'result':'physics_success'}})]:
        w=worker();w.update(changes);expect_fail(label,lambda w=w:ctx['_accepted_gate'](w))
    assert ctx['_accepted_gate'](worker())['result']=='physics_success'
    functions={'side_entry_grasp','realman_place_pose','realman_lift_displacement','grip_measure','strict_official_episode'}
    fresh={'np':np};ctx['_extract_function_defs'](candidate,functions,fresh)
    for name in functions:assert fresh[name].__globals__ is fresh
    exec(compile(capture.read_text(),str(capture),'exec'),fresh)
    assert fresh['install_contact_capture'].__globals__ is fresh
    ctx['_extract_native_exporter'](base/'run_adjust_bottle.py',fresh);assert '_export_native_collector_block' in fresh
    signatures={name:ctx['_function_code_signature'](fresh[name]) for name in functions|{'install_contact_capture'}}
    a={};exec(compile('def f(x):\n return x+1\n','a.py','exec'),a);b={};exec(compile('\n\ndef f(x):\n return x+1\n','b.py','exec'),b)
    assert ctx['_function_code_signature'](a['f'])==ctx['_function_code_signature'](b['f'])
    exec(compile('def f(x):\n return x+2\n','c.py','exec'),b);assert ctx['_function_code_signature'](a['f'])!=ctx['_function_code_signature'](b['f'])
    with tempfile.TemporaryDirectory(prefix='v11_replay_static_') as td:
        out=Path(td);left=[]
        for i in range(5):
            n=i+2;q=np.ones((n,6))*(i+.01);v=np.ones_like(q)*.02
            left.append({'status':'Success','position':q,'velocity':v})
        paths={'left_joint_path':left,'right_joint_path':[]}
        refs={str(i+2):np.eye(4).tolist() for i in (2,3,4)}
        for i in (2,3,4):
            num=i+2;native=left[i];n=len(native['position'])
            np.savez(out/f'planned_carry_path_{num}.npz',position=native['position'],velocity=native['velocity'],planning_reference=np.eye(4))
            f=np.zeros((n,3));f[:,0]=np.arange(n)*.123;f[:,2]=1.17;t=np.zeros_like(f);t[:,1]=.013
            np.savez(out/f'planned_wrench_{num}.npz',force_target_world_N=f,torque_target_world_Nm=t)
            (out/f'carry_mesh_path_{num}.json').write_text(json.dumps({'passed':True,'metadata':{'rigid_carry_reference':np.eye(4).tolist()}}));(out/f'carry_wrench_preflight_{num}.json').write_text(json.dumps({'passed':True}))
        schedules,manifest=ctx['_load_candidate_wrenches'](out,paths,sha,refs);assert sorted(schedules)==[5,6,7]
        for dense in (5,6,7):
            item=schedules[dense];fb={'wrench_schedule':None,'wrench_index':99,'last_lp_phase':'stale'};seq={'left_arm':left[item['path_index']]}
            ctx['_install_saved_wrench_for_action'](fb,seq,dense_call=dense,native_left_consumed=item['path_index']+1,schedules=schedules,actual_relative=np.eye(4))
            assert fb['wrench_index']==0 and fb['last_lp_phase'] is None
            assert np.array_equal(fb['wrench_schedule']['force_target_world_N'],item['schedule']['force_target_world_N'])
            assert np.array_equal(fb['wrench_schedule']['torque_target_world_Nm'],item['schedule']['torque_target_world_Nm'])
            fb['wrench_schedule']['force_target_world_N'][0,0]=999;assert item['schedule']['force_target_world_N'][0,0]!=999
        expect_fail('consumptionindex',lambda:ctx['_install_saved_wrench_for_action']({}, {'left_arm':left[2]},dense_call=5,native_left_consumed=2,schedules=schedules,actual_relative=np.eye(4)))
        expect_fail('unscheduledaction',lambda:ctx['_install_saved_wrench_for_action']({}, {'left_arm':left[2]},dense_call=8,native_left_consumed=5,schedules=schedules,actual_relative=np.eye(4)))
        expect_fail('missingarmsequence',lambda:ctx['_install_saved_wrench_for_action']({}, {},dense_call=5,native_left_consumed=3,schedules=schedules,actual_relative=np.eye(4)))
        bad=copy.deepcopy(left[2]);bad['position'][0,0]+=.001
        expect_fail('changedactualpath',lambda:ctx['_install_saved_wrench_for_action']({}, {'left_arm':bad},dense_call=5,native_left_consumed=3,schedules=schedules,actual_relative=np.eye(4)))
        altered=copy.deepcopy(paths);altered['left_joint_path'][2]['position'][0,0]+=.001
        expect_fail('nativepathnotcandidate',lambda:ctx['_load_candidate_wrenches'](out,altered,sha,refs))
        expect_fail('unexpectedpathcount',lambda:ctx['_load_candidate_wrenches'](out,{'left_joint_path':left[:-1],'right_joint_path':[]},sha,refs))
        p=out/'planned_wrench_4.npz';orig=p.read_bytes();np.savez(p,force_target_world_N=np.zeros((1,3)),torque_target_world_Nm=np.zeros((1,3)))
        expect_fail('wrenchlength',lambda:ctx['_load_candidate_wrenches'](out,paths,sha,refs));p.write_bytes(orig)
        f=np.zeros((len(left[2]['position']),3));f[0,0]=np.nan;np.savez(p,force_target_world_N=f,torque_target_world_Nm=np.zeros_like(f))
        expect_fail('nonfinitewrench',lambda:ctx['_load_candidate_wrenches'](out,paths,sha,refs));p.write_bytes(orig)
        proof=out/'carry_wrench_preflight_4.json';proof.write_text(json.dumps({'passed':False}));expect_fail('failedwrenchpreflight',lambda:ctx['_load_candidate_wrenches'](out,paths,sha,refs));proof.write_text(json.dumps({'passed':True}))
        calls={}
        def changing_sha(file):
            key=str(file);calls[key]=calls.get(key,0)+1
            return sha(file) if calls[key]==1 else 'sourcechanged'
        expect_fail('sourcechangedwhilereading',lambda:ctx['_load_candidate_wrenches'](out,paths,changing_sha,refs))
        badrefs=copy.deepcopy(refs);badrefs['4'][0][3]=.01
        expect_fail('source_report_reference_mismatch',lambda:ctx['_load_candidate_wrenches'](out,paths,sha,badrefs))
        proof=out/'carry_mesh_path_4.json';original=proof.read_bytes();proof.write_text(json.dumps({'passed':True,'metadata':{'rigid_carry_reference':np.zeros((4,4)).tolist()}}))
        expect_fail('mesh_reference_mismatch',lambda:ctx['_load_candidate_wrenches'](out,paths,sha,refs));proof.write_bytes(original)
        slipped=np.eye(4);slipped[0,3]=.003
        expect_fail('replay_segment_grip_reference_mismatch',lambda:ctx['_install_saved_wrench_for_action']({}, {'left_arm':left[2]},dense_call=5,native_left_consumed=3,schedules=schedules,actual_relative=slipped))
        p.unlink();expect_fail('missingwrenchartifact',lambda:ctx['_load_candidate_wrenches'](out,paths,sha,refs))
    return {'scope':'Pure Conda CPU AST/helper/guard tests only; no warm-worker execution, queue, SAPIEN scene, GPU, planner or HDF export','sha256':sha(source),'ast_compiled':True,'version':'v12','rejection_tests':checks,'fresh_v12_helper_globals_verified':True,'native_exporter_ast_anchor_verified':True,'v11_capture_ast_verified':True,'wrench_mapping':{'path_index':[2,3,4],'candidate_planner_dense_call':[4,5,6],'replay_action_dense_call':[5,6,7]},'inertial_arrays_loaded_exactly_and_fresh':True,'source_v12_control_signatures':signatures,'static_signature_contract_verified':True,'candidate_snapshot':str(candidate),'candidate_snapshot_sha256':sha(candidate),'capture_snapshot':str(capture),'capture_snapshot_sha256':sha(capture),'actual_reset_replay_executed':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,default=Path(__file__).resolve().parent);parser.add_argument('--candidate',type=Path);parser.add_argument('--capture',type=Path);args=parser.parse_args()
    sys.path.insert(0,str(args.base));candidate=args.candidate or args.base/'try_coupled_side_entry_v12.py';capture=args.capture or args.base/'contact_capture_control_v11.py'
    result=audit(args.base,candidate,capture);(args.base/'accepted_reset_replay_v12_static_audit_20261003.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
