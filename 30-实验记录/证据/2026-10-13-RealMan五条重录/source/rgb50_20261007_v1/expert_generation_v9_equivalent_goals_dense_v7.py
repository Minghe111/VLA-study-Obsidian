"""v9 physics with finite native-FK-validated free-approach goal declarations."""
from pathlib import Path
from expert_generation_v9_pathdiag import ExpertSession as _BaseSession, install_initializer, command24, measured24, _sha


class ExpertSession(_BaseSession):
    def __init__(self, task, sample_spec, output_dir, legacy_dir):
        super().__init__(task, sample_spec, output_dir, legacy_dir)
        import full_native_approach_equivalent_goals_dense_v7
        self.free_approach = full_native_approach_equivalent_goals_dense_v7
        import full_native_approach_waypoint_v4
        for source in (Path(__file__).resolve(), Path(self.free_approach.__file__).resolve(),
                       Path(self.free_approach.equivalent.__file__).resolve(),
                       Path(self.free_approach.dense_policy.__file__).resolve(),
                       Path(full_native_approach_waypoint_v4.__file__).resolve()):
            self.sources[str(source)] = _sha(source)
        self.report.update(expert_version='v9_free_approach_explicit_bounded_equivalent_goals_dense_v7',
            source_hashes=self.sources,
            bounded_goal_adapter={'finite_actual_native_limits_only': True,
                'actual_native_revolute_type_required': True,
                'every_declared_goal_native_FK_checked_for_all_links': True,
                'every_declared_goal_full_native_collision_checked': True,
                'RRT_endpoint_requires_predeclared_numeric_goal': True,
                'raw_start_and_end_numeric_tolerances_unchanged': True,
                'returned_paths_never_wrapped_or_clipped': True,
                'full_raw_limits_official_TOPP_full_FMA_and_actual36_retained': True,
                'physical_control_LP_profile_and_acceptance_inherited_from_v9': True,
                'waypoint_detour_not_used': True, 'LP_200ms_default_preserved': True,
                'historical_failure_cause_not_inferred': True, 'CPU_path_is_not_physical_success': True, 'dense_only_after_original_complete_insufficient_certificate': True, 'dense_vertex_step_m': .0001, 'dense_finite_sample_capacity': 250000, 'every_branch_actual_TOPP_vertices_and_retiming_retained': True})
        self.write()


def install_expert(task, sample_spec, output_dir, *, legacy_dir=None):
    if legacy_dir is None:
        legacy_dir = Path(__file__).resolve().parent.parent / 'adjust_bottle_20261002'
    return ExpertSession(task, sample_spec, output_dir, legacy_dir)


def run_episode(task, sample_spec, output_dir, *, legacy_dir=None, record=False, capture_callback=None, validate_actual_geometry=True):
    return install_expert(task, sample_spec, output_dir, legacy_dir=legacy_dir).run(
        record=record, capture_callback=capture_callback, validate_actual_geometry=validate_actual_geometry)
