"""v9/gain .3/LP 200ms physics; extra contact-free CPU path candidate only."""
from pathlib import Path
from expert_generation_v9_pathdiag import ExpertSession as _BaseSession,install_initializer,command24,measured24,_sha

class ExpertSession(_BaseSession):
    def __init__(self,task,sample_spec,output_dir,legacy_dir):
        super().__init__(task,sample_spec,output_dir,legacy_dir)
        import full_native_approach_waypoint_v4
        self.free_approach=full_native_approach_waypoint_v4
        for source in (Path(__file__).resolve(),Path(self.free_approach.__file__).resolve(),
                       Path(self.free_approach.original.__file__).resolve()):
            self.sources[str(source)]=_sha(source)
        self.report.update(expert_version='v9_contact_free_goal_height_waypoint_v4',source_hashes=self.sources,
            waypoint_path_candidate={'world_z_offset_m':.08,
                'only_after_original_direct_CPU_path_failure':True,
                'via_is_original_hardware_palm_target_plus_world_Z':True,
                'original_goal_and_orientation_retained':True,
                'original_native_TOPP_and_full_mesh_limit_gates_retained':True,
                'physical_control_LP_profile_and_acceptance_inherited_from_v9':True,
                'CPU_path_is_not_physical_success':True})
        self.write()

def install_expert(task,sample_spec,output_dir,*,legacy_dir=None):
    if legacy_dir is None:legacy_dir=Path(__file__).resolve().parent.parent/'adjust_bottle_20261002'
    return ExpertSession(task,sample_spec,output_dir,legacy_dir)

def run_episode(task,sample_spec,output_dir,*,legacy_dir=None,record=False,capture_callback=None,validate_actual_geometry=True):
    return install_expert(task,sample_spec,output_dir,legacy_dir=legacy_dir).run(
        record=record,capture_callback=capture_callback,validate_actual_geometry=validate_actual_geometry)
