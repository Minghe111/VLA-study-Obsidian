"""Finite left-cap contact experiment, with phase-specific physical guards."""
import copy
import numpy as np
from labware_cap_workflow import labware_cap_workflow
from cap_contact_controller import read_hand_contacts

class FirstContactStop(RuntimeError):
    """Diagnostic stop at the first measured 4-ms contact, not a transition."""

class left_cap_contact_task(labware_cap_workflow):
    def setup_demo(self,**kwargs):
        self.left_probe_active=False;self.first_left_cap_contact=None
        self.left_contact_guard_failure=None;self.left_cap_separation_trace=[];self.stop_at_first_contact=True
        return super().setup_demo(**kwargs)
    def activate_left_probe(self):
        # The body-only guard is replaced by separate expected-left/right guards.
        self.cap_contact_limit_N=float('inf');self.left_probe_active=True
        self.workflow_phase='left_prepare'
    def _update_metrics(self):
        super()._update_metrics()
        if not getattr(self,'_active',False) or not self.left_probe_active:return
        left=read_hand_contacts(self.scene,self.cap,'left')
        right=read_hand_contacts(self.scene,self.cap,'right')
        lcap=sum(left['forces_N']);rcap=sum(right['forces_N'])
        body_id=int(self.bottle.actor.per_scene_id);table_id=int(self.table.per_scene_id)
        cap_id=int(self.cap.actor.per_scene_id);cap_geometry=[]
        forbidden={'left_body':0.,'left_table':0.,'left_right':0.}
        for contact in self.scene.get_contacts():
            entities=[part.entity for part in contact.bodies]
            if cap_id in [int(entity.per_scene_id) for entity in entities] and any(entity.name.startswith('l_inspire_') for entity in entities):
                cap_geometry.append({'actor_names':[entity.name for entity in entities],
                    'actor_ids':[int(entity.per_scene_id) for entity in entities],
                    'points':[{'separation_m':float(point.separation),
                        'position_world_m':np.asarray(point.position).tolist(),
                        'normal_raw_world':np.asarray(point.normal).tolist(),
                        'impulse_raw_world_Ns':np.asarray(point.impulse).tolist()} for point in contact.points]})
            force=sum(abs(float(np.asarray(point.impulse)@np.asarray(point.normal)))/.004 for point in contact.points)
            for entity,other in [entities,entities[::-1]]:
                if not entity.name.startswith('l_'):continue
                if int(other.per_scene_id)==body_id:forbidden['left_body']+=force
                if int(other.per_scene_id)==table_id:forbidden['left_table']+=force
                if other.name.startswith('r_'):forbidden['left_right']+=force
        self.left_cap_separation_trace.append({'physics_tick':self.physics_step_count,
            'phase':self.current_phase,'cap_actor_id':cap_id,'contacts':cap_geometry})
        self.metrics.update(left_cap_contact=left,right_cap_contact=right,
            left_cap_force_N=lcap,right_cap_force_N=rcap,forbidden_left_contact_N=forbidden,
            success=False,full_unscrew_pour_success=False)
        if self.first_left_cap_contact is None and lcap>.001:
            self.first_left_cap_contact={'physics_tick':self.physics_step_count,'phase':self.current_phase,
                'left_contact':copy.deepcopy(left),'cap_state':copy.deepcopy(self.metrics['cap_state']),
                'normal_force_threshold_N':.001,'threshold_impulse_Ns':.000004,
                'scope':'First actual left-cap normal load only; no grasp, opposition or turn inferred'}
        if self.physics_step_count%15==0:
            self.physics_trace[-1].update(self.metrics)
            self.assembly_observation_trace[-1]['body_hold_metrics']=copy.deepcopy(self.metrics)
        reason=None
        if rcap>5.:reason=f'Unintended right-cap load {rcap:.6f}N exceeds5N'
        if lcap>2.:reason=f'Left first-contact load {lcap:.6f}N exceeds2N'
        if max(forbidden.values())>.1:reason=f'Forbidden left contact {forbidden}'
        if self.metrics['cap_state']['released']:reason='Unexpected cap release before commanded unscrewing'
        if reason:
            self.left_contact_guard_failure={'reason':reason,'physics_tick':self.physics_step_count,
                'phase':self.current_phase,'left_cap_force_N':lcap,'right_cap_force_N':rcap,
                'forbidden_left_contact_N':forbidden}
            raise RuntimeError(reason)
        if self.first_left_cap_contact is not None and self.stop_at_first_contact:
            raise FirstContactStop('First actual left-cap normal impulse measured; diagnostic stop')
