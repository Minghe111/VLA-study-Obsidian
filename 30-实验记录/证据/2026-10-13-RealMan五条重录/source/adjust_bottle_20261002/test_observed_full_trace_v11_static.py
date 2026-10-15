"""Conda CPU contact/fake-FCL control-flow regression. No SAPIEN/GPU/scene steps.
Synthetic FCL tests check classifications/gates, not real-model geometry accuracy.
"""
from pathlib import Path
import ast,copy,hashlib,json,tempfile
from types import SimpleNamespace as S
import numpy as np
import observed_full_trace_audit_v11 as full
import full_mesh_path_audit as fma
import preflight_place_goals as ppg


def run():
    checks=[]
    def point(impulse):return S(impulse=np.array(impulse,dtype=float),normal=np.array([0.,0.,1.]),position=np.zeros(3),separation=-.0001)
    def contact(a,b,impulse):return S(bodies=[S(entity=S(name=a)),S(entity=S(name=b))],points=[point(impulse)])
    contacts=[contact('a','b',[0,0,2e-7]),contact('a','c',[.01,0,0]),contact('a','d',[0,0,.01]),contact('a','external',[0,0,.01])]
    task=S(scene=S(get_contacts=lambda:contacts));record=full.collect_actual_self_contacts(task,{'a','b','c','d'},{frozenset(('a','d'))})
    assert len(record)==2 and record[0]['active'] and not record[1]['active'];checks.append('true_normal_not_tangential; native_only; configured_exemptions')
    row={'phase':{'name':'after_grasp'},'bottle_actual_contacts':[{'other_link':'l_finger','normal_impulse_Ns':1e-6},{'other_link':'r_finger','normal_impulse_Ns':1},{'other_link':'l_near','normal_impulse_Ns':0}]}
    assert full._touch_links(row,{'l_finger','l_near'},1e-10)=={'l_finger'}
    row={'phase':{'name':'after_grasp'},'contact_forces':[{'link':'l_finger','normal_force_N':.01},{'link':'l_near','normal_force_N':0}]};assert full._touch_links(row,{'l_finger','l_near'},1e-10)=={'l_finger'}
    row['phase']['name']='approach';assert not full._touch_links(row,{'l_finger'},1e-10);checks.append('exact_selected_touch_only; normal_force_dt_fallback; no_approach_exemption')
    class Pose:
        def __init__(self,value=None):self.t=np.eye(4) if value is None else np.array(value)
        def to_transformation_matrix(self):return self.t.copy()
    class Obj:
        def __init__(self,name,pose=None,shapes=None,shape_poses=None):self.name=name;self.pose=Pose() if pose is None else pose;self.shapes=[];self.shape_poses=[];self.radius=.02 if name=='bottle' else .01
    def bounds(obj):
        p=obj.pose.to_transformation_matrix()[:3,3];return np.array([p-obj.radius,p+obj.radius])
    def collides(a,b,_):
        if b.name=='table':return S(is_collision=lambda:bool(bounds(a)[0,2]<=0))
        x,y=bounds(a),bounds(b);return S(is_collision=lambda:bool(np.all((x[0]<=y[1])&(y[0]<=x[1]))))
    signs=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
    flags={'self':False,'table':False}
    class Auditor:
        def __init__(self,task,check_table=True):
            self.sim_joints=['j'+str(i) for i in range(36)];self.art=task.robot.left_entity;self.old_model_q=np.zeros(36)
            self.objects={n:Obj(n) for n in ('a','b','l_inspire_base_link')};self.vertices={n:signs*.01 for n in self.objects};self.self_pairs=[('a','b')]
            self.static={'bottle':Obj('bottle'),'table':Obj('table')};self.static_boxes={'table':np.array([[-100,-100,-1],[100,100,0]])}
            self.metadata={'synthetic_control_flow_test':True};self.Pose=Pose;self.collision_request=None
            self.fcl=S(FCLObject=Obj,collide=collides)
            def setq(q,full=True):
                t=np.eye(4);t[:3,3]=[2,0,-.001 if flags['table'] else 2];self.objects['a'].pose=Pose(t)
                v=t.copy();v[0,3]=2 if flags['self'] else 3;self.objects['b'].pose=Pose(v)
                p=np.eye(4);p[2,3]=.02+q[1];self.objects['l_inspire_base_link'].pose=Pose(p)
            self.model=S(set_qpos=setq);setq(np.zeros(36))
        def _distance(self,first,second):
            assert second.name=='table';z=bounds(first)[0,2];return max(0.,float(z)),bool(z<=0)
    art=S(active_joints=[S(limits=np.array([[-1.,1.]])) for _ in range(36)])
    bottle=S(get_name=lambda:'001_bottle',actor=S(get_components=lambda:[S(collision_shapes=[S(half_size=np.array([.02,.02,.02]),local_pose=Pose())])]))
    fake_task=S(model_id=13,qpose_tag=0,bottle=bottle,scene=S(get_timestep=lambda:.004),robot=S(communication_flag=False,left_entity=art,left_arm_joints_name=['j'+str(i) for i in range(6)],right_arm_joints_name=['j'+str(i) for i in range(6,12)]))
    old_audit=fma.FullMeshPathAuditor;old_snapshot=ppg._snapshot;old_assert=ppg._assert_snapshot
    fma.FullMeshPathAuditor=Auditor;ppg._snapshot=lambda task:'physical_unchanged';ppg._assert_snapshot=lambda task,before:None
    try:
        with tempfile.TemporaryDirectory(prefix='actual36_static_') as td:
            directory=Path(td);tr=[]
            for i in range(3):
                q=np.zeros(36);q[1]=0. if i==0 else .05
                pose=np.eye(4);pose[2,3]=.02+q[1]
                tr.append({'step':i+1,'phase':{'name':'close_hold' if i==0 else 'after_grasp'},'arm_qpos':{'left':q[:6].tolist(),'right':q[6:12].tolist()},'full_qpos_actual_rad':q.tolist(),'bottle':{'pose':pose.tolist(),'model_id':13,'qpose_tag':0},'actual_self_contacts':[],'bottle_actual_contacts':[{'other_link':'l_inspire_base_link','normal_impulse_Ns':.001}]})
            rp={'accepted':True,'result':'physics_success','canonical_sources_unchanged':True,'source_hashes':{},'full_qpos_joint_names':['j'+str(i) for i in range(36)],'physical_steps':3,'pre_retreat_grip':{'passed':True,'actual':{'relative_bottle_palm':np.eye(4).tolist()}}}
            def evaluate(trace=None,report=None):
                t=directory/'physics_trace.json';r=directory/'attempt_report.json';t.write_text(json.dumps(trace or tr));r.write_text(json.dumps(report or rp))
                return full.audit_accepted_actual_trace(fake_task,t,report_path=r)
            result=evaluate();assert result['passed'] and result['full_temporal_coverage'] and result['bottle_table_first_release']['step']==2;checks.append('full_fake_pipeline_pass; initial_table_support_allowed; later_release_required')
            flags['self']=True;assert not evaluate()['passed'];flags['self']=False;checks.append('nonallowed_geometric_self_collision_rejected')
            flags['table']=True;assert not evaluate()['passed'];flags['table']=False;checks.append('robot_table_collision_rejected')
            wrong=copy.deepcopy(tr);wrong[1]['bottle_actual_contacts'][0]['other_link']='r_inspire_base_link';assert not evaluate(trace=wrong)['passed'];checks.append('wrong_side_grasp_touch_not_exempted')
            historical=copy.deepcopy(tr);historical[1]['actual_self_contacts']=[{'names':['a','b'],'normal_impulse_Ns':.001}];assert not evaluate(trace=historical)['passed'];checks.append('historical_true_normal_self_collision_rejected')
            for name,trace,report in [('failed_acceptance',tr,dict(rp,accepted=False)),('wrong_q_order',tr,dict(rp,full_qpos_joint_names=list(reversed(rp['full_qpos_joint_names'])))),('missing_physical_step',tr[1:],rp)]:
                try:evaluate(trace=trace,report=report)
                except ValueError:checks.append(name+'_rejected')
                else:raise AssertionError(name)
            missing=copy.deepcopy(tr);missing[1]['full_qpos_actual_rad'][0]=float('nan')
            try:evaluate(trace=missing)
            except ValueError:checks.append('nonfinite_fullq_rejected')
            else:raise AssertionError('nonfinite')
    finally:fma.FullMeshPathAuditor=old_audit;ppg._snapshot=old_snapshot;ppg._assert_snapshot=old_assert
    base=Path(__file__).resolve().parent;collector=base/'accepted_reset_replay.py';source=collector.read_text();ast.parse(source)
    assert source.index('full_actual_audit=audit_accepted_actual_trace')<source.index("dataset = context['_export_native_collector_block']()")
    assert "full_actual_audit.get('passed') is not True" in source and "context['report'].update(accepted=False,result='accepted_replay_actual36_failed_no_export')" in source
    assert 'full36_qpos_before_rad' in source and 'full36_qpos_after_rad' in source and "'actual_self_contacts':actual_self_contacts" in source
    checks.append('mandatory_full_audit_guard_precedes_export; full36_hdf_npz_fields; failure_sets_accepted_false')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    result={'scope':'Pure CPU readers, synthetic independent FK/FCL control-flow and AST gates only; no SAPIEN import/scene step, GPU, planner, worker queue, actual-model experiment or HDF export','checks':checks,'passed':True,'collector_sha256':sha(collector),'full_audit_sha256':sha(base/'observed_full_trace_audit_v11.py'),'v9_helper_preserved_sha256':sha(base/'observed_trace_audit.py')}
    (base/'observed_full_trace_v11_static_audit_20261003.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':run()
