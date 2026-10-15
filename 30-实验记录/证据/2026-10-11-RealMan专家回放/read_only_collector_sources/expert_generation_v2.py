"""Isolated RGB50 extension: approach azimuth, palm roll and explicit hand skin.

V1 remains immutable for active trials. Arm/hand control, contact LP, collision
audits and recording are inherited without changing their feasible sets.
"""
from pathlib import Path
import copy
import math
import numpy as np
from expert_generation import (ExpertSession as _BaseSession, DEFAULT_SETTING,
                               install_initializer, command24, measured24,
                               _sha, _extract, _json)


SPRITE_SETTING = dict(DEFAULT_SETTING, hand_y=.075, hand_z=.177,
                      palm_roll_deg=5., hand_contact_offset_m=.001,
                      close_sdk=[900, 900, 608, 610, 610, 0])


class ExpertSession(_BaseSession):
    def __init__(self, task, sample_spec, output_dir, legacy_dir):
        spec=copy.deepcopy(sample_spec)
        setting=copy.deepcopy(SPRITE_SETTING if int(spec['model_id'])==16 else DEFAULT_SETTING)
        setting.update(copy.deepcopy(spec.get('grasp_setting',{})))
        spec['grasp_setting']=setting
        super().__init__(task,spec,output_dir,legacy_dir)
        self.sources[str(Path(__file__).resolve())]=_sha(Path(__file__).resolve())
        self.report.update(expert_version='v2', source_profile_boundary='Sprite starts from original accepted horizontal-left profile; each new orientation/arm must independently pass physics',
                           source_hashes=self.sources)
        self.context={"np":np}
        _extract(self.legacy/'try_coupled_side_entry_v29.py',{'verify_hand_drive_parameters'},self.context)
        self.report['hardware_hand_drive']={'before_motion':self.context['verify_hand_drive_parameters'](task,self.setting)}
        self.report['hand_material_actual']=copy.deepcopy(getattr(task.robot,'hand_material_audit',None))
        self.report['hand_joint_friction_actual']=copy.deepcopy(getattr(task.robot,'hand_friction_audit',None))
        if 'hand_contact_offset_m' in self.setting:
            self.configure_hand_skin(float(self.setting['hand_contact_offset_m']))
        fma_receipt=getattr(self.fma,'rgb50_shape_count_overlay_receipt',None)
        if fma_receipt is not None:self.report['native_shape_count_overlay']=copy.deepcopy(fma_receipt)
        self.write()

    def configure_hand_skin(self, requested):
        from integrations.realman_rm65b_gen4.hand_coupling import material_snapshot
        if not np.isfinite(requested) or requested<=0:raise ValueError('Hand skin must be positive finite metres')
        robot=self.task.robot;q_before=robot.left_entity.get_qpos().copy()
        material_before=material_snapshot(robot.left_entity,robot._manifest)
        rows=[]
        for link in robot.left_entity.get_links():
            for component_index,component in enumerate(link.entity.get_components()):
                for shape_index,shape in enumerate(getattr(component,'collision_shapes',[])):
                    if not link.name.startswith(('l_inspire','r_inspire')):continue
                    old=float(shape.contact_offset);rest=float(shape.rest_offset)
                    if requested<=rest:raise ValueError('Contact offset must exceed unchanged rest offset')
                    shape.contact_offset=requested
                    after=float(shape.contact_offset)
                    if abs(after-requested)>1e-8 or float(shape.rest_offset)!=rest:raise RuntimeError('Hand contact/rest readback differs')
                    rows.append({'link':link.name,'component_index':component_index,'shape_index':shape_index,
                                 'before_contact_offset_m':old,'actual_contact_offset_m':after,'unchanged_rest_offset_m':rest})
        if not all(any(r['link'].startswith(p) for r in rows) for p in ('l_inspire','r_inspire')):raise RuntimeError('Both hand collision groups must exist')
        material_after=material_snapshot(robot.left_entity,robot._manifest)
        if material_before!=material_after or not np.array_equal(q_before,robot.left_entity.get_qpos()):raise RuntimeError('Numerical skin changed material or physical qpos')
        self.report['hand_contact_offset_actual']={'passed':True,'requested_m':requested,'rows':rows,
            'mesh_geometry_changed':False,'materials_unchanged':True,'rest_offsets_unchanged':True,'qpos_unchanged':True,
            'hardware_calibrated':False}

    def grasp_pose(self,actor,arm_tag,pre_dis=.1,target_dis=0.,contact_point_id=None):
        from transforms3d.quaternions import mat2quat
        matrix=actor.get_pose().to_transformation_matrix()
        reflection=np.diag([-1.,1.,1.]) if self.side=='right' else np.eye(3)
        axis=reflection@matrix[:3,1]
        if abs(axis[0])>.1:
            if axis[0]<0:axis=-axis
        elif axis[2]<0:axis=-axis
        axis/=np.linalg.norm(axis)
        pitch=math.radians(float(self.setting['pitch_deg']))
        azimuth=math.radians(float(self.setting.get('approach_azimuth_deg',0.)))
        radial=np.array([math.sin(azimuth)*math.cos(pitch),math.cos(azimuth)*math.cos(pitch),-math.sin(pitch)])
        radial-=axis*(axis@radial)
        if np.linalg.norm(radial)<.1:
            radial=np.array([0.,0.,-1.]);radial-=axis*(axis@radial)
        if np.linalg.norm(radial)<.1:raise ValueError('Degenerate bottle/approach geometry')
        radial/=np.linalg.norm(radial);lateral=np.cross(radial,axis)
        canonical_rotation=np.stack([radial,lateral,-axis],axis=1)
        roll=math.radians(float(self.setting.get('palm_roll_deg',0.)))
        canonical_rotation=canonical_rotation@np.array([[1.,0.,0.],[0.,math.cos(roll),-math.sin(roll)],[0.,math.sin(roll),math.cos(roll)]])
        delta=np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
        local_reflection=np.diag([-1.,1.,1.]) if self.side=='right' else np.eye(3)
        rotation=reflection@(canonical_rotation@delta)@local_reflection@delta.T
        center=reflection@actor.get_contact_point(0,'matrix')[:3,3]
        goal=reflection@(center-(self.setting['hand_z']-.06)*radial-self.setting['hand_y']*lateral-self.setting['hand_x']*axis)
        world_lateral=reflection@lateral
        pre=goal-(pre_dis+.095-self.setting['hand_y'])*world_lateral
        self.report['grasp_geometry']={'bottle_axis_world':matrix[:3,1].tolist(),'selected_axis_world':axis.tolist(),
            'approach_azimuth_deg_canonical':float(self.setting.get('approach_azimuth_deg',0.)),
            'palm_roll_deg_canonical':float(self.setting.get('palm_roll_deg',0.)),
            'approach_vector_world':world_lateral.tolist(),'pre_pose':[*pre,*mat2quat(rotation)],'grasp_pose':[*goal,*mat2quat(rotation)]}
        return [*pre,*mat2quat(rotation)],[*(goal-target_dis*world_lateral),*mat2quat(rotation)]

    def save_evidence(self):
        self.report['hardware_hand_drive']['after_motion']=self.context['verify_hand_drive_parameters'](self.task,self.setting)
        return super().save_evidence()

    def plan_path(self,robot,target_pose,constraint_pose=None,**kwargs):
        result=super().plan_path(robot,target_pose,constraint_pose=constraint_pose,**kwargs)
        # A legal CuRobo pre-grasp branch can be unreachable by local screw IK.
        # Try the official global arm planner only after an explicit screw
        # failure. The same full-mesh and real no-touch guards still gate it.
        if self.dense_calls>=3 or constraint_pose is None or result.get('status')=='Success':return result
        retry=getattr(self.Robot,self.side+'_plan_path')(robot,target_pose,constraint_pose=None,**kwargs)
        record={'index':self.dense_calls,'side':self.side,'scope':'official CuRobo fallback after local MPLib screw failure',
                'original_screw_status':result.get('status'),'status':retry.get('status'),
                'target_pose':np.asarray(target_pose).tolist(),'samples':len(retry.get('position',[]))}
        if retry.get('status')=='Success':
            audit=self.fma.audit_path(self.task,retry,self.side,min_bottle_clearance_m=.0005,min_table_clearance_m=0.,min_self_clearance_m=0.)
            (self.out/f'mesh_path_{self.dense_calls}_native_fallback.json').write_text(__import__('json').dumps(audit,default=_json))
            record['full_mesh_audit_passed']=bool(audit['passed'])
            if not audit['passed']:retry={'status':'Fail','full_mesh_audit_failed':True}
            else:np.savez(self.out/f'planned_path_{self.dense_calls}.npz',position=retry['position'],velocity=retry['velocity'])
        record['status']=retry.get('status')
        self.report.setdefault('planning_calls',[]).append(record);self.write()
        return retry


def install_expert(task,sample_spec,output_dir,*,legacy_dir=None):
    if legacy_dir is None:legacy_dir=Path(__file__).resolve().parent.parent/'adjust_bottle_20261002'
    return ExpertSession(task,sample_spec,output_dir,legacy_dir)


def run_episode(task,sample_spec,output_dir,*,legacy_dir=None,record=False,capture_callback=None,validate_actual_geometry=True):
    session=install_expert(task,sample_spec,output_dir,legacy_dir=legacy_dir)
    return session.run(record=record,capture_callback=capture_callback,validate_actual_geometry=validate_actual_geometry)
