"""v3 control unchanged; retain exact LP/COM diagnostics on failed trials too."""
from pathlib import Path
import copy
import numpy as np
from expert_generation_v3 import (ExpertSession as _BaseSession,
                                 install_initializer, command24, measured24,
                                 _sha)


class ExpertSession(_BaseSession):
    def __init__(self, task, sample_spec, output_dir, legacy_dir):
        super().__init__(task, sample_spec, output_dir, legacy_dir)
        source = Path(__file__).resolve()
        self.sources[str(source)] = _sha(source)
        component = next(c for c in task.bottle.actor.get_components()
                         if hasattr(c, 'cmass_local_pose'))
        self.report.update(expert_version='v4', source_hashes=self.sources,
            actual_bottle_mass_properties={
                'mass_kg': float(component.mass),
                'principal_inertia_kg_m2': np.asarray(component.inertia).tolist(),
                'cmass_local_matrix': component.cmass_local_pose.to_transformation_matrix().tolist(),
                'scope': 'actual PhysX component readback; no writes'})
        self.write()

    def save_evidence(self):
        # The immutable baseline only copies this history in its success branch.
        # Save identical diagnostic records for rejected trajectories as well.
        self.report['adaptive_force_allocations'] = copy.deepcopy(self.feedback['lp_history'])
        keys = ('active', 'scalar', 'index', 'latched', 'latched_q',
                'filtered_N', 'normal_N', 'actual_pair_streak',
                'lost_pair_steps', 'had_pair', 'control_steps',
                'lp_failure_steps', 'force_target_N',
                'passive_contact_force_limit_N', 'lp_min_main_force_N',
                'feedback_simulation_parameters')
        self.report['capture_controller_final'] = {
            key: copy.deepcopy(self.feedback[key]) for key in keys}
        return super().save_evidence()


def install_expert(task, sample_spec, output_dir, *, legacy_dir=None):
    if legacy_dir is None:
        legacy_dir = Path(__file__).resolve().parent.parent / 'adjust_bottle_20261002'
    return ExpertSession(task, sample_spec, output_dir, legacy_dir)


def run_episode(task, sample_spec, output_dir, *, legacy_dir=None,
                record=False, capture_callback=None,
                validate_actual_geometry=True):
    session = install_expert(task, sample_spec, output_dir, legacy_dir=legacy_dir)
    return session.run(record=record, capture_callback=capture_callback,
                       validate_actual_geometry=validate_actual_geometry)
