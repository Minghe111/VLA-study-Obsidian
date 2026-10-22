"""Independent 115-mm profile; frozen 90-mm mechanism and task stay unchanged."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import sapien
from transforms3d.quaternions import quat2mat

from labware_cap_task import labware_closed_cap_pick,install_complete_assembly_obstacle
from robot_expert import adapted_native
from robot_cap_assembly import RobotCapAssembly,rigid,FROZEN

ROOT=Path(__file__).resolve().parent
PROFILE=ROOT/'profiles/bottle115'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class _VisualBuilderProxy:
    def __init__(self,builder,replacements,receipt):
        object.__setattr__(self,'_builder',builder)
        object.__setattr__(self,'_replacements',replacements)
        object.__setattr__(self,'_receipt',receipt)
    def __getattr__(self,key):return getattr(self._builder,key)
    def __setattr__(self,key,value):setattr(self._builder,key,value)
    def add_visual_from_file(self,filename,*args,**kwargs):
        replacement=self._replacements.get(str(filename),str(filename))
        self._receipt.append({'requested_file':str(filename),'actual_file':replacement,'sha256':sha(replacement)})
        return self._builder.add_visual_from_file(replacement,*args,**kwargs)

class _VisualSceneProxy:
    def __init__(self,scene,replacements,receipt):self.scene=scene;self.replacements=replacements;self.receipt=receipt
    def __getattr__(self,key):return getattr(self.scene,key)
    def create_actor_builder(self):return _VisualBuilderProxy(self.scene.create_actor_builder(),self.replacements,self.receipt)

class Bottle115Assembly(RobotCapAssembly):
    """Reuse original build/constraints, replacing only profile-bound visual inputs."""
    def __init__(self,scene,**kwargs):
        self.profile=PROFILE
        self.visual_binding_receipt=[]
        self.kinematics_trace=[]
        self.step_context=None
        super().__init__(scene,spec_path=self.profile/'physics_spec115.json',**kwargs)
    def _frozen_inputs(self):
        result=super()._frozen_inputs()
        paths=[FROZEN/'physics_spec.json',Path(__file__),self.profile/'profile_manifest.json',
            self.profile/'body115.glb',self.profile/'cap14.glb',self.profile/'body_model_data0.json',self.profile/'cap_model_data0.json']
        result.update({str(path):sha(path) for path in paths})
        return result
    def _build(self,visual,seed):
        original=self.scene
        replacements={str(FROZEN/'prepared/reagent_bottle_body/visual/base0.glb'):str(self.profile/'body115.glb'),
                      str(FROZEN/'prepared/reagent_bottle_cap/visual/base0.glb'):str(self.profile/'cap14.glb')}
        self.scene=_VisualSceneProxy(original,replacements,self.visual_binding_receipt)
        try:super()._build(visual,seed)
        finally:self.scene=original
        if visual and {row['actual_file'] for row in self.visual_binding_receipt}!=set(replacements.values()):
            raise RuntimeError('Both profile-specific visuals must be loaded by the original builder')
    def make_actor_wrappers(self):
        from envs.utils.actor_utils import Actor
        return {key:Actor(entity,json.loads((self.profile/f'{key}_model_data0.json').read_text()),
                    mass=.08 if key=='body' else .012) for key,entity in [('body',self.body),('cap',self.cap)]}
    def _kinematics(self):
        result={}
        for key,entity in [('body',self.body),('cap',self.cap)]:
            c=rigid(entity)
            result[key]={'world_pose_pq':np.r_[entity.pose.p,entity.pose.q].tolist(),
                'world_linear_velocity_m_s':np.asarray(c.linear_velocity).tolist(),
                'world_angular_velocity_rad_s':np.asarray(c.angular_velocity).tolist(),
                'cmass_local_pose_pq':np.r_[c.cmass_local_pose.p,c.cmass_local_pose.q].tolist(),
                'cmass_world_pose_pq':np.r_[(entity.pose*c.cmass_local_pose).p,(entity.pose*c.cmass_local_pose).q].tolist()}
        axis=quat2mat(self.body.pose.q)[:,2]
        result['body_z_world']=axis.tolist()
        result['relative_omega_body_z_rad_s']=float((np.asarray(result['cap']['world_angular_velocity_rad_s'])-np.asarray(result['body']['world_angular_velocity_rad_s']))@axis)
        result['tablet_world_pose_pq']=[np.r_[entity.pose.p,entity.pose.q].tolist() for entity in self.tablets]
        return result
    def before_step(self):
        self._pre_sample=self._kinematics()
        self._step_context=copy.deepcopy(self.step_context() if self.step_context else {'phase':'construction'})
        super().before_step()
    def after_step(self):
        super().after_step()
        contacts=[]
        for contact in self.scene.get_contacts():
            names=[body.entity.name for body in contact.bodies]
            if not ({self.body.name,self.cap.name}&set(names)):continue
            points=[]
            for point in contact.points:
                impulse=np.asarray(point.impulse,dtype=float);normal=np.asarray(point.normal,dtype=float)
                points.append({'position_world_m':np.asarray(point.position).tolist(),'normal_raw_world':normal.tolist(),
                    'impulse_raw_world_Ns':impulse.tolist(),'normal_impulse_abs_Ns':abs(float(impulse@normal)),
                    'tangent_impulse_world_Ns':(impulse-(impulse@normal)*normal).tolist()})
            contacts.append({'body_names':names,'body_actor_ids':[int(body.entity.per_scene_id) for body in contact.bodies],
                'points':points,'raw_impulse_convention':'retained PhysX contact point; participant ordering saved, no inferred torque applied'})
        self.kinematics_trace.append({'physics_step':self.physics_steps,'dt_s':self.dt,'context':self._step_context,
            'pre':self._pre_sample,'post':self._kinematics(),'post_assembly_state':self.state(),'contacts_post':contacts})
    def contract(self):
        result=super().contract()
        result.update(component='Bottle115Assembly: frozen passive mechanism plus independent visual/metadata profile',
            profile_manifest=json.loads((self.profile/'profile_manifest.json').read_text()),
            visual_binding_receipt=copy.deepcopy(self.visual_binding_receipt),
            actual_mass_inertia={key:{'mass_kg':float(rigid(entity).mass),'principal_inertia_kg_m2':np.asarray(rigid(entity).inertia).tolist(),
                'cmass_local_pose_pq':np.r_[rigid(entity).cmass_local_pose.p,rigid(entity).cmass_local_pose.q].tolist()}
                for key,entity in [('body',self.body),('cap',self.cap)]},
            complete_world_kinematics_pre_post_each_actual_step=True,
            physical_success_claimed_by_profile=False)
        return result

class labware_closed_cap115_pick(labware_closed_cap_pick):
    def setup_demo(self,**kwargs):
        self.current_phase='initialization';self.current_command24=None;self.cap_contact_limit_N=5.
        self.early_stop=None;self.full_state_trace=[];self._extra_settle_used=False
        self._strict_body_hold_ticks=0;self._strict_closed_hold_ticks=0
        self._max_strict_body_hold_seconds=0.;self._max_strict_closed_hold_seconds=0.
        return super().setup_demo(**kwargs)
    def check_stable(self):
        stable,bad=super().check_stable()
        if stable or self._extra_settle_used:return stable,bad
        self._extra_settle_used=True
        first=json.loads((self.runtime_path/'initialization_stability.json').read_text())
        (self.runtime_path/'initialization_stability_first_window.json').write_text(json.dumps(first,indent=2)+'\n')
        # One explicitly authorized additional10s, identical tolerances and
        # ordinary physics. No qpos/velocity write, damping change or seed search.
        print(json.dumps({'initialization':'one additional bounded10s natural settle',
            'first_window_bad_entities':bad,'thresholds_changed':False}),flush=True)
        stable,bad=super().check_stable()
        receipt={'extra_settle_windows':1,'extra_physics_steps':2500,'dt_s':float(self.scene.get_timestep()),
            'first_window_passed':False,'second_window_passed':stable,'second_window_bad_entities':bad,
            'thresholds_unchanged':True,'velocities_or_poses_reset':False,'seed_unchanged':True}
        (self.runtime_path/'bounded_extra_settle_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        return stable,bad
    def load_actors(self):
        self.qpose_tag=1 if self.arm=='right' else 0;self.model_id=0
        if self.center_xy is None:self.center_xy=[.10,-.11]
        self.assembly=Bottle115Assembly(self.scene,pose=sapien.Pose([*self.center_xy,.742],
            [np.cos(self.yaw/2),0,0,np.sin(self.yaw/2)]),seed=int(self.seed),visual=True,name_prefix='derived115_closed_tablet_bottle')
        self.assembly.step_context=lambda:{'phase':self.current_phase,'active_tick_before_step':self.physics_step_count,
            'action_index':len(self.transition_log),'requested_command24':copy.deepcopy(self.current_command24),
            'actual_drive_command24':adapted_native().command24(self.robot).tolist() if self._active else None,
            'support_actor_identity':{'name':self.table.name,'per_scene_id':int(self.table.per_scene_id)}}
        wrapped=self.assembly.make_actor_wrappers();self.bottle=wrapped['body'];self.cap=wrapped['cap']
        self.vertices=adapted_native().fma._link_vertices(rigid(self.assembly.body))[0]
        self.asset_data=copy.deepcopy(self.bottle.config)
        self.delay(1);self.add_prohibit_area(self.bottle,padding=.1)
    def apply_sdk24(self,action,phase='unspecified'):
        self.current_phase=phase;self.current_command24=np.asarray(action).tolist()
        result=super().apply_sdk24(action,phase)
        art=self.robot.left_entity
        self.full_state_trace.append({'physics_tick':self.physics_step_count,'phase':phase,
            'joint_names':[joint.name for joint in art.get_active_joints()],
            'q36':np.asarray(art.get_qpos()).tolist(),'qvel36':np.asarray(art.get_qvel()).tolist(),
            'root_matrix':np.asarray(art.root_pose.to_transformation_matrix()).tolist(),
            'right_palm_matrix':np.asarray(art.find_link_by_name('r_inspire_base_link').entity.pose.to_transformation_matrix()).tolist(),
            'body_matrix':np.asarray(self.assembly.body.pose.to_transformation_matrix()).tolist(),
            'cap_matrix':np.asarray(self.assembly.cap.pose.to_transformation_matrix()).tolist(),
            'tablet_world_pose_pq':[np.r_[entity.pose.p,entity.pose.q].tolist() for entity in self.assembly.tablets]})
        return result
    def _update_metrics(self):
        super()._update_metrics()
        if not getattr(self,'_active',False) or self.assembly is None:return
        # Correct only this new observer: the frozen empty observer's substring
        # test mistakes bottle-content '*tablet*' contacts for the actual table.
        # Require the exact support entity ID, keeping all physical tolerances.
        table_id=int(self.table.per_scene_id);body_id=int(self.bottle.actor.per_scene_id)
        table_force=0.
        for contact in self.scene.get_contacts():
            ids=[int(part.entity.per_scene_id) for part in contact.bodies]
            if set(ids)!={table_id,body_id}:continue
            table_force+=sum(abs(float(np.asarray(point.impulse)@np.asarray(point.normal)))/.004 for point in contact.points)
        legacy_table=self.metrics['table_contact_N']
        held=(self.metrics['opposed_contact'] and self.metrics['bottom_gap_m']>=.02
            and table_force<1e-5 and self.metrics['upright_angle_deg']<15.)
        closed_held=held and self.metrics['cap_closed_observed'] and self.metrics['tablet_centers_inside']==12
        self._strict_body_hold_ticks=self._strict_body_hold_ticks+1 if held else 0
        self._strict_closed_hold_ticks=self._strict_closed_hold_ticks+1 if closed_held else 0
        self._max_strict_body_hold_seconds=max(self._max_strict_body_hold_seconds,self._strict_body_hold_ticks*.004)
        self._max_strict_closed_hold_seconds=max(self._max_strict_closed_hold_seconds,self._strict_closed_hold_ticks*.004)
        self.hold_ticks=self._strict_body_hold_ticks;self.max_hold_seconds=self._max_strict_body_hold_seconds
        self.closed_fill_hold_ticks=self._strict_closed_hold_ticks;self.max_closed_fill_hold_seconds=self._max_strict_closed_hold_seconds
        self.lift_achieved=self._max_strict_body_hold_seconds>=.5
        self.metrics.update(table_contact_N=table_force,legacy_substring_table_contact_N=legacy_table,
            support_actor_identity={'name':self.table.name,'per_scene_id':table_id},
            hold_seconds=self._strict_body_hold_ticks*.004,max_hold_seconds=self.max_hold_seconds,lift_achieved=self.lift_achieved,
            closed_fill_hold_seconds=self._strict_closed_hold_ticks*.004,
            max_closed_fill_hold_seconds=self.max_closed_fill_hold_seconds,
            stage_body_hold_success=self.max_closed_fill_hold_seconds>=.5,
            success=self.max_closed_fill_hold_seconds>=.5)
        if self.physics_step_count%15==0:
            self.physics_trace[-1].update(self.metrics)
            self.assembly_observation_trace[-1]['body_hold_metrics']=copy.deepcopy(self.metrics)
        force=self.metrics['cap_robot_contact_N']
        if force>self.cap_contact_limit_N:
            self.early_stop={'reason':'Unintended cap contact in body-only grasp','threshold_N':self.cap_contact_limit_N,
                'actual_N':force,'physics_tick':self.physics_step_count,'phase':self.current_phase,
                'actual_drive_target24':adapted_native().command24(self.robot).tolist()}
            raise RuntimeError(f'Body-only grasp stopped: cap contact {force:.6f}N > {self.cap_contact_limit_N}N')
    def run_closed_cap_pick(self,max_targets=700):
        native=adapted_native();install_complete_assembly_obstacle(native)
        prepare=native.command24(self.robot);final=prepare.copy();final[18:24]=[0.,0.,800.,800.,800.,0.]
        for fraction in np.linspace(0.,1.,45)[1:]:self.apply_sdk24(prepare+(final-prepare)*fraction,'prepare_prebent_three_finger_grasp')
        setting={'open_sdk':[0.,0.,800.,800.,800.,0.],'close_sdk':[0.,0.,608.,610.,610.,0.],'hand_x':-.015}
        self.teacher=native.RealManSDK24Teacher(self,setting=setting,audit_approach=True,max_close_boundaries=150)
        self.teacher.planner.wall_budget_s=180.
        while not self.body_hold_stage_success():
            if self.take_action_cnt>=max_targets:raise RuntimeError('Finite115mm pickup target budget exceeded')
            proposal=self.teacher.next_action();self.apply_sdk24(proposal['action'],proposal['phase'])
            if self.take_action_cnt%25==0:print(json.dumps({'targets':self.take_action_cnt,'phase':self.teacher.phase,
                'lift_m':self.metrics['bottom_gap_m'],'contained':self.metrics['tablet_centers_inside'],
                'cap_force_N':self.metrics['cap_robot_contact_N'],'closed':self.metrics['cap_closed_observed'],
                'success':self.check_success()}),flush=True)
        self.teacher.write_report();return self.info
    def body_hold_stage_success(self):
        """Reusable stage gate; a future full task must own final check_success."""
        return bool(self.metrics.get('stage_body_hold_success',False))
