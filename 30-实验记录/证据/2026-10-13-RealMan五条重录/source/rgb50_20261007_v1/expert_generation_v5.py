"""Native screw approach and exact functional-point carry targets.

All motion is still official arm servo plus the unchanged contact controller.
Candidate paths execute only after the inherited full mesh, contact-wrench and
real-physics guards pass. The object is never positioned or attached by code.
"""
from pathlib import Path
import copy
import json
import math
import numpy as np
from expert_generation_v4 import (ExpertSession as _BaseSession,
                                 install_initializer, command24, measured24,
                                 _sha)
from expert_generation import _json


def align_unit_vectors(source, target):
    a=np.asarray(source,float);a/=np.linalg.norm(a)
    b=np.asarray(target,float);b/=np.linalg.norm(b)
    dot=float(np.clip(a@b,-1.,1.))
    if dot>1.-1e-12:return np.eye(3)
    if dot<-1.+1e-12:
        axis=np.array([1.,0.,0.]) if abs(a[0])<.9 else np.array([0.,1.,0.])
        axis-=a*(a@axis);axis/=np.linalg.norm(axis)
        return 2*np.outer(axis,axis)-np.eye(3)
    v=np.cross(a,b)
    skew=np.array([[0.,-v[2],v[1]],[v[2],0.,-v[0]],[-v[1],v[0],0.]])
    return np.eye(3)+skew+skew@skew/(1.+dot)


def rigid_functional_point_target(live_ee_pose, bottle_matrix, current_fp,
                                  target_fp, yaw_delta_deg=0.):
    """Read-only task-EE goal for minimal upright rotation then world-Z yaw."""
    from transforms3d.quaternions import quat2mat, mat2quat
    live=np.asarray(live_ee_pose,float);current=np.asarray(current_fp,float)[:3]
    desired=np.asarray(target_fp,float)[:3]
    rotation=align_unit_vectors(np.asarray(bottle_matrix)[:3,1],[0.,0.,1.])
    yaw=math.radians(float(yaw_delta_deg))
    rotation=np.array([[math.cos(yaw),-math.sin(yaw),0.],
                       [math.sin(yaw),math.cos(yaw),0.],[0.,0.,1.]])@rotation
    ee_position=desired+rotation@(live[:3]-current)
    ee_rotation=rotation@quat2mat(live[-4:])
    if np.max(abs(ee_rotation.T@ee_rotation-np.eye(3)))>1e-5 or np.linalg.det(ee_rotation)<.99999:
        raise RuntimeError('Invalid rigid upright target rotation')
    return [*ee_position,*mat2quat(ee_rotation)],rotation


class ExpertSession(_BaseSession):
    def __init__(self,task,sample_spec,output_dir,legacy_dir):
        super().__init__(task,sample_spec,output_dir,legacy_dir)
        source=Path(__file__).resolve();self.sources[str(source)]=_sha(source)
        self.placement_candidates=[]
        self.report.update(expert_version='v5',source_hashes=self.sources)
        self.write()

    def place_pose(self,actor,arm_tag,target_pose,**kwargs):
        if actor is not self.task.bottle or str(arm_tag)!=self.side:
            raise RuntimeError('Unexpected bottle/arm in placement')
        if int(kwargs.get('functional_point_id',0))!=0 or float(kwargs.get('pre_dis',0.))!=0.:
            raise RuntimeError('This hardware adapter requires official adjust_bottle functional point 0, pre_dis 0')
        live=getattr(self.task.robot,'get_'+self.side+'_ee_pose')()
        matrix=actor.get_pose().to_transformation_matrix()
        current=actor.get_functional_point(0)
        yaws=list(self.spec.get('placement_yaw_candidates_deg',[0.,30.,-30.,60.,-60.,90.,-90.,180.]))
        preferred=float(self.spec.get('placement_yaw_delta_deg',0.))
        yaws=[preferred]+[float(v) for v in yaws if abs(float(v)-preferred)>1e-9]
        y_offsets=[float(v) for v in self.spec.get('placement_corridor_y_offsets_m',[0.])]
        if not y_offsets or not yaws or not all(np.isfinite(v) for v in y_offsets+yaws):
            raise ValueError('Finite nonempty placement candidate lists required')
        self.placement_candidates=[]
        for offset in y_offsets:
            fp=np.asarray(target_pose[:3],float).copy();fp[1]+=offset
            for yaw in yaws:
                pose,rotation=rigid_functional_point_target(live,matrix,current,fp,yaw)
                self.placement_candidates.append({'target_pose':pose,'target_functional_point0_world':fp.tolist(),
                    'yaw_delta_deg':yaw,'corridor_y_offset_m':offset,
                    'desired_bottle_axis_world':(rotation@matrix[:3,1]).tolist()})
        self.report['placement_geometry']={'strategy':'minimum upright rotation; preserve upright yaw before optional world-Z candidates',
            'actual_live_ee_pose':np.asarray(live).tolist(),'actual_functional_point0':np.asarray(current).tolist(),
            'official_target_functional_point0':np.asarray(target_pose[:3]).tolist(),
            'candidates':copy.deepcopy(self.placement_candidates),
            'physical_pose_writes':False,'official_success_criteria_unchanged':True}
        self.write()
        return self.placement_candidates[0]['target_pose']

    def _carry_candidate(self,robot,target_pose,constraint_pose,number,**kwargs):
        index=self.dense_calls
        preflight=self.out/f'carry_wrench_preflight_{index}.json'
        previous_preflight_stamp=preflight.stat().st_mtime_ns if preflight.exists() else None
        result=super().plan_path(robot,target_pose,constraint_pose=constraint_pose,**kwargs)
        evidence={k:copy.deepcopy(v) for k,v in result.items() if k not in ('position','velocity','time')}
        evidence.update(index=index,candidate=number,side=self.side,
                        target_pose=np.asarray(target_pose).tolist(),samples=len(result.get('position',[])))
        fresh_preflight=preflight.exists() and preflight.stat().st_mtime_ns!=previous_preflight_stamp
        evidence['wrench_preflight_generated_for_this_candidate']=fresh_preflight
        if fresh_preflight:
            data=json.loads(preflight.read_text())
            snapshot=self.out/f'carry_candidate_{index}_{number:02d}_wrench.json'
            snapshot.write_bytes(preflight.read_bytes())
            evidence['wrench_preflight']={k:data.get(k) for k in
                ('passed','failure_count','failure_reason_counts','checked_samples','sample_count')}
            evidence['wrench_preflight_evidence']=str(snapshot)
        if result.get('expected_wrench_preflight_failed'):
            evidence['failure_stage']='full native path found; exact contact wrench preflight rejected'
        if 'position' in result:
            np.savez(self.out/f'carry_candidate_{index}_{number:02d}_path.npz',
                     position=result['position'],velocity=result['velocity'])
        (self.out/f'carry_candidate_{index}_{number:02d}_result.json').write_text(json.dumps(evidence,default=_json))
        self.report.setdefault('carry_candidate_results',[]).append(evidence)
        self.write()
        return result

    def plan_path(self,robot,target_pose,constraint_pose=None,**kwargs):
        index=self.dense_calls
        if index>=3:
            candidates=self.placement_candidates if index>=4 and self.placement_candidates else [{'target_pose':target_pose}]
            result=None
            for number,candidate in enumerate(candidates):
                result=self._carry_candidate(robot,candidate['target_pose'],constraint_pose,number,**kwargs)
                if result.get('status')=='Success':
                    self.report.setdefault('selected_carry_candidates',{})[str(index)]=dict(candidate,candidate=number)
                    self.write();return result
            return result
        if self.spec.get('planning_backend')=='full_native' and constraint_pose is not None:
            record={'index':index,'side':self.side,'backend':'official MPLib screw plus complete native mesh gate',
                    'target_pose':np.asarray(target_pose).tolist()}
            try:
                palm=robot._trans_from_gripper_to_endlink(target_pose,arm_tag=self.side)
                result=getattr(robot,self.side+'_mplib_planner').planner.plan_screw(
                    goal_pose=palm,current_qpos=robot.left_entity.get_qpos(),time_step=.004,qpos_step=.005)
                record.update(status=result.get('status'),samples=len(result.get('position',[])),reason=result.get('reason'))
                if result.get('status')=='Success':
                    audit=self.fma.audit_path(self.task,result,self.side,min_bottle_clearance_m=.0005,
                        min_table_clearance_m=0.,min_self_clearance_m=0.)
                    record['full_mesh_audit_passed']=bool(audit['passed'])
                    (self.out/f'mesh_path_{index}_screw.json').write_text(json.dumps(audit,default=_json))
                    if audit['passed']:
                        (self.out/f'mesh_path_{index}.json').write_text(json.dumps(audit,default=_json))
                        np.savez(self.out/f'planned_path_{index}.npz',position=result['position'],velocity=result['velocity'])
                        self.report.setdefault('planning_calls',[]).append(record);self.write();return result
                    record['status']='Fail';record['reason']='Complete native mesh gate rejected screw path'
            except Exception as error:
                record.update(status='Fail',reason=str(error),planner_exception=type(error).__name__)
            self.report.setdefault('planning_calls',[]).append(record);self.write()
        return super().plan_path(robot,target_pose,constraint_pose=constraint_pose,**kwargs)


def install_expert(task,sample_spec,output_dir,*,legacy_dir=None):
    if legacy_dir is None:legacy_dir=Path(__file__).resolve().parent.parent/'adjust_bottle_20261002'
    return ExpertSession(task,sample_spec,output_dir,legacy_dir)


def run_episode(task,sample_spec,output_dir,*,legacy_dir=None,record=False,
                capture_callback=None,validate_actual_geometry=True):
    session=install_expert(task,sample_spec,output_dir,legacy_dir=legacy_dir)
    return session.run(record=record,capture_callback=capture_callback,
                       validate_actual_geometry=validate_actual_geometry)
