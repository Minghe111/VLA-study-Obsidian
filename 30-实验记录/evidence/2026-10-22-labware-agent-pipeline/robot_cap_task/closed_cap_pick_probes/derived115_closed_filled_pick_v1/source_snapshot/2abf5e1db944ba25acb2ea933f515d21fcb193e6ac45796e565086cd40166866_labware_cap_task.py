"""First contact-driven closed/filled bottle stage for the next cap/pour goal.

This stage only grasps and holds a CLOSED dynamic bottle with twelve tablets.
No full opening/pouring success or dataset label is manufactured here.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from labware_task import labware_pick_place
from robot_expert import adapted_native
from robot_cap_assembly import RobotCapAssembly,rigid

ASSEMBLY_SHA='0785eb4e2a487f4ec4a7c637011e163ceabcce18bbc990fc17980e86b95708d5'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class labware_closed_cap_pick(labware_pick_place):
    """Right-hand body grasp with passive cap constraints maintained each 4ms."""
    def setup_demo(self,**kwargs):
        if sha(ROOT/'robot_cap_assembly.py')!=ASSEMBLY_SHA:
            raise RuntimeError('Passive assembly source changed; stop before simulating')
        self.assembly=None
        self.closed_fill_hold_ticks=0
        self.max_closed_fill_hold_seconds=0.
        self.assembly_observation_trace=[]
        result=super().setup_demo(**kwargs)
        self.set_instruction('Grasp the body of the closed tablet bottle with the right hand and hold it upright without opening or spilling.')
        self.info['next_goal_scope']='Closed-cap/filled-bottle robot hold probe only; not unscrewing or pouring'
        self.assembly_contract_at_start=self.assembly.contract()
        return result

    def load_actors(self):
        import sapien
        self.qpose_tag=1 if self.arm=='right' else 0
        self.model_id=0
        if self.center_xy is None:
            sign=1 if self.arm=='right' else -1
            self.center_xy=[sign*np.random.uniform(.095,.105),np.random.uniform(-.115,-.105)]
        self.assembly=RobotCapAssembly(self.scene,pose=sapien.Pose([*self.center_xy,.742],
            [np.cos(self.yaw/2),0,0,np.sin(self.yaw/2)]),seed=int(self.seed),visual=True,
            name_prefix='contact_closed_tablet_bottle')
        wrapped=self.assembly.make_actor_wrappers()
        self.bottle=wrapped['body'];self.cap=wrapped['cap']
        native=adapted_native()
        self.vertices=native.fma._link_vertices(rigid(self.assembly.body))[0]
        self.asset_data=copy.deepcopy(self.bottle.config)
        self.delay(1)
        self.add_prohibit_area(self.bottle,padding=.1)

    def _step_simulation(self):
        if hasattr(self,'robot') and self.robot is not None:self.robot.step_controller()
        # During the normal action clock, advance() owns the paired hook.
        if getattr(self,'_active',False):
            self.scene.step()
            return
        assembly=getattr(self,'assembly',None)
        if assembly is not None:assembly.before_step()
        self.scene.step()
        if assembly is not None:assembly.after_step()

    def check_stable(self):
        """Explicit capsule-shape stability, without suppressing motion/escape.

        Capsule geometry is invariant to spin around local X and axis reversal;
        the generic task quaternion criterion incorrectly treats that as a
        geometric scene change. Keep its 2000+500 step bounded settling window,
        check all centre positions and capsule long-axis directions instead,
        and retain full quaternion/velocity diagnostics for audit.
        """
        from transforms3d.quaternions import quat2mat
        assembly=self.assembly
        if assembly is None:return super().check_stable()
        entities=[assembly.body,assembly.cap,*assembly.tablets]
        history={actor.name:[] for actor in entities}
        for _ in range(2000):self._step_simulation()
        for _ in range(500):
            self._step_simulation()
            for actor in entities:
                history[actor.name].append((np.asarray(actor.pose.p).copy(),np.asarray(actor.pose.q).copy()))
        records=[];bad=[]
        for actor in entities:
            samples=history[actor.name][-200:]
            points=np.array([r[0] for r in samples]);quats=np.array([r[1] for r in samples])
            drift=float(np.linalg.norm(points-points[-1],axis=1).max())
            quat_degrees=np.degrees(2*np.arccos(np.clip(np.abs(quats@quats[-1]),0,1)))
            component=rigid(actor)
            axis=np.array([quat2mat(q)[:,0] for q in quats])
            axis_degrees=np.degrees(np.arccos(np.clip(np.abs(axis@axis[-1]),0,1)))
            capsule=actor in assembly.tablets
            orientation=float(axis_degrees.max() if capsule else quat_degrees.max())
            speed=float(np.linalg.norm(component.linear_velocity))
            transverse=float(np.linalg.norm(np.cross(np.asarray(component.angular_velocity),axis[-1])))
            limit_ok=drift<=.001 and orientation<=3. and speed<=.01
            if capsule:limit_ok=limit_ok and transverse<=.1
            if not limit_ok:bad.append(actor.name)
            records.append({'actor':actor.name,'capsule_shape_symmetry_used':capsule,
                'center_drift_last_200_steps_m':drift,'full_quaternion_drift_degrees':float(quat_degrees.max()),
                'geometric_orientation_drift_degrees':orientation,'final_linear_speed_m_s':speed,
                'final_angular_velocity_rad_s':np.asarray(component.angular_velocity).tolist(),
                'capsule_transverse_angular_speed_rad_s':transverse,'passed':bool(limit_ok)})
        local=assembly.tablet_local_positions()
        contained=(np.linalg.norm(local[:,:2],axis=1)<=assembly.spec['bottle']['inner_radius_m']+.001)&(local[:,2]>=-.001)&(local[:,2]<=assembly.closed_z+.001)
        if not contained.all():bad.append('tablet_center_outside_closed_container')
        receipt={'passed':not bad,'bad_entities':bad,'post_settle_checked_steps':500,
            'warmup_steps':2000,'dt_s':float(self.scene.get_timestep()),'records':records,
            'tablet_centers_inside':int(contained.sum()),
            'scope':'Capsule collision geometry is invariant to local-X spin; positional motion, transverse axis motion and escape remain checked; no pose/dynamics writes'}
        (self.runtime_path/'initialization_stability.json').write_text(json.dumps(receipt,indent=2)+'\n')
        return not bad,bad

    def advance(self):
        self.assembly.before_step()
        self._raw_step()
        self.assembly.after_step()
        self.physics_step_count+=1
        self._update_metrics()

    def _update_metrics(self):
        super()._update_metrics()
        if not getattr(self,'_active',False) or self.assembly is None:return
        state=self.assembly.state()
        points=self.assembly.tablet_local_positions()
        radius=np.linalg.norm(points[:,:2],axis=1)
        center_inside=(radius<=self.assembly.spec['bottle']['inner_radius_m']+.001)&(points[:,2]>=-.001)&(points[:,2]<=self.assembly.closed_z+.001)
        count=int(center_inside.sum())
        closed=(not state['released'] and abs(state['rotation_measured_rad'])<.15 and abs(state['lift_measured_m'])<.001
            and state['radial_error_m']<.001 and state['axis_tilt_rad']<.02)
        held=(self.metrics['opposed_contact'] and self.metrics['bottom_gap_m']>=.02
            and self.metrics['table_contact_N']<1e-5 and self.metrics['upright_angle_deg']<15)
        valid=closed and count==12 and held
        self.closed_fill_hold_ticks=self.closed_fill_hold_ticks+1 if valid else 0
        self.max_closed_fill_hold_seconds=max(self.max_closed_fill_hold_seconds,self.closed_fill_hold_ticks*.004)
        cap_hand_force=0.;cap_hand_links=set()
        for contact in self.scene.get_contacts():
            names=[b.entity.name for b in contact.bodies]
            if self.assembly.cap.name not in names:continue
            other=names[1] if names[0]==self.assembly.cap.name else names[0]
            if not other.startswith(('l_inspire_','r_inspire_')):continue
            for point in contact.points:
                force=abs(float(np.asarray(point.impulse)@np.asarray(point.normal)))/.004
                if force>1e-6:cap_hand_links.add(other);cap_hand_force+=force
        self.metrics.update(success=self.max_closed_fill_hold_seconds>=.5,
            stage='closed_filled_bottle_hold',tablet_centers_inside=count,tablet_total=12,
            closed_fill_hold_seconds=self.closed_fill_hold_ticks*.004,
            max_closed_fill_hold_seconds=self.max_closed_fill_hold_seconds,
            cap_closed_observed=bool(closed),cap_state=state,
            cap_robot_contact_N=cap_hand_force,cap_robot_contact_links=sorted(cap_hand_links),
            unscrew_verified=False,pour_verified=False)
        if self.physics_step_count%15==0:
            self.assembly_observation_trace.append({'tick':self.physics_step_count,'tablet_local_centers_m':points.tolist(),
                'tablet_centers_inside':count,'closed_observed':closed,'cap_state':state,
                'cap_robot_contact_N':cap_hand_force,'cap_robot_contact_links':sorted(cap_hand_links),
                'body_hold_metrics':copy.deepcopy(self.metrics)})

    def play_once(self):
        return self.run_closed_cap_pick()

    def run_closed_cap_pick(self,max_targets=700):
        native=adapted_native()
        install_complete_assembly_obstacle(native)
        prepare=native.command24(self.robot)
        hand=slice(18,24) if self.arm=='right' else slice(6,12)
        final=prepare.copy();final[hand]=[0.,0.,1000.,1000.,1000.,0.]
        for fraction in np.linspace(0.,1.,45)[1:]:
            self.apply_sdk24(prepare+(final-prepare)*fraction,'prepare_three_finger_grasp')
        setting={'open_sdk':[0.,0.,1000.,1000.,1000.,0.],
            'close_sdk':[0.,0.,608.,610.,610.,0.],'hand_x':-.015}
        self.teacher=native.RealManSDK24Teacher(self,setting=setting,audit_approach=True,max_close_boundaries=150)
        self.teacher.planner.wall_budget_s=180.
        while not self.check_success():
            if self.take_action_cnt>=max_targets:raise RuntimeError('Finite closed-filled pickup target budget exceeded')
            proposal=self.teacher.next_action()
            self.apply_sdk24(proposal['action'],proposal['phase'])
            if self.take_action_cnt%25==0:
                print(json.dumps({'targets':self.take_action_cnt,'phase':self.teacher.phase,
                    'lift_m':self.metrics['bottom_gap_m'],'contained':self.metrics['tablet_centers_inside'],
                    'closed':self.metrics['cap_closed_observed'],'success':self.check_success()}),flush=True)
        self.teacher.write_report()
        return self.info

    def close(self):
        super().close()
        self.cap=None
        self.assembly=None


def install_complete_assembly_obstacle(native):
    """Private CPU obstacle union includes real cap and all twelve pills.

    The old expert's static key 'bottle' becomes the actual closed assembly
    compound, preserving every physical body/cap shape. Capsule contents are
    overapproximated by containing convex boxes ONLY in the planning model.
    Nothing in SAPIEN is changed and no extra collision exclusion is added.
    """
    if getattr(native.fma.FullMeshPathAuditor,'_closed_assembly_extension',False):return
    original=native.fma.FullMeshPathAuditor
    class AssemblyMeshAuditor(original):
        _closed_assembly_extension=True
        def __init__(self,task,*args,**kwargs):
            super().__init__(task,*args,**kwargs)
            if not hasattr(task,'assembly') or task.assembly is None:return
            from mplib.sapien_utils.conversion import SapienPlanningWorld
            import trimesh
            assembly=task.assembly
            body_matrix=np.asarray(assembly.body.pose.to_transformation_matrix())
            source=self.static['bottle'];shapes=[];locals_=[];all_world=[]
            for shape,pose in zip(source.shapes,source.shape_poses):
                shapes.append(self.fcl.CollisionObject(shape.get_collision_geometry()))
                locals_.append(self.Pose(pose))
            all_world.append(native.fma._link_vertices(rigid(assembly.body))[0]@body_matrix[:3,:3].T+body_matrix[:3,3])
            for entity in [assembly.cap,*assembly.tablets]:
                component=rigid(entity)
                entity_matrix=np.asarray(entity.pose.to_transformation_matrix())
                vertices,_=native.fma._link_vertices(component)
                all_world.append(vertices@entity_matrix[:3,:3].T+entity_matrix[:3,3])
                if entity is assembly.cap:
                    converted=native.fma._convert_component_scaled(component,SapienPlanningWorld.convert_physx_component,self.fcl,self.Pose)
                    for shape,pose in zip(converted.shapes,converted.shape_poses):
                        local=np.linalg.inv(body_matrix)@entity_matrix@np.asarray(pose.to_transformation_matrix())
                        shapes.append(self.fcl.CollisionObject(shape.get_collision_geometry()));locals_.append(self.Pose(local))
                else:
                    # Exact capsule axis is local X; its AABB is a safe outer
                    # planning proxy. No physical shape is replaced or enlarged.
                    low=vertices.min(0);high=vertices.max(0)
                    box=trimesh.creation.box(extents=high-low)
                    box.apply_translation((low+high)/2)
                    hull=self.fcl.Convex(vertices=np.asarray(box.vertices),faces=np.asarray(box.faces,dtype=np.int32))
                    shapes.append(self.fcl.CollisionObject(hull))
                    locals_.append(self.Pose(np.linalg.inv(body_matrix)@entity_matrix))
            self.static['bottle']=self.fcl.FCLObject('closed_filled_bottle_assembly',self.Pose(body_matrix),shapes,locals_)
            world=np.concatenate(all_world)
            self.static_boxes['bottle']=np.stack([world.min(0),world.max(0)])
            self.metadata['closed_assembly_obstacle']={'body_original_shape_count':len(source.shapes),
                'total_shape_count':len(shapes),'cap_original_shapes_preserved':True,
                'tablets':12,'tablet_proxy':'containing convex box, CPU planning only',
                'physical_shapes_modified':False}
    native.fma.FullMeshPathAuditor=AssemblyMeshAuditor
