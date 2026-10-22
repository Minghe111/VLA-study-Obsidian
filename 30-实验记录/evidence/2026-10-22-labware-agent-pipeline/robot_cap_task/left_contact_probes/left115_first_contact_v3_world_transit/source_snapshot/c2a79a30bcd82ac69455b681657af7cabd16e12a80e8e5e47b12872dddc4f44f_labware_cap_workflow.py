"""Workflow skeleton: body-hold is a stage, never full cap/pour success."""
import copy
from labware_cap115_task import labware_closed_cap115_pick

class labware_cap_workflow(labware_closed_cap115_pick):
    def setup_demo(self,**kwargs):
        self.workflow_phase='replay_body_pickup'
        result=super().setup_demo(**kwargs)
        self.info['workflow_scope']='Post-lift right-hand static-reference probe only; cap removal and dish pouring not implemented'
        return result
    def _update_metrics(self):
        super()._update_metrics()
        if not getattr(self,'_active',False) or self.assembly is None:return
        self.metrics.update(stage=self.workflow_phase,stage_success=self.body_hold_stage_success(),
            success=False,full_unscrew_pour_success=False)
        if self.physics_step_count%15==0:
            self.physics_trace[-1].update(self.metrics)
            self.assembly_observation_trace[-1]['body_hold_metrics']=copy.deepcopy(self.metrics)
    def check_success(self):
        # No final12-in-dish gate exists in this finite reference probe.
        return False
