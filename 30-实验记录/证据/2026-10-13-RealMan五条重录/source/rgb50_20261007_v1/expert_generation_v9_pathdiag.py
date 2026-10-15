"""v9 physics expert with metadata-only free-path diagnostics v2."""
from pathlib import Path
from expert_generation_v9 import ExpertSession as _BaseSession,install_initializer,command24,measured24,_sha

class ExpertSession(_BaseSession):
    def __init__(self,task,sample_spec,output_dir,legacy_dir):
        super().__init__(task,sample_spec,output_dir,legacy_dir)
        import full_native_approach_diagnostics_v2
        self.free_approach=full_native_approach_diagnostics_v2
        for path in (Path(__file__).resolve(),Path(self.free_approach.__file__).resolve()):
            self.sources[str(path)]=_sha(path)
        self.report.update(expert_version='v9_path_diagnostics_v2',
            free_approach_diagnostics='Metadata only: original endpoint, retiming, native-clearance and physics criteria retained',
            source_hashes=self.sources)
        self.write()

def install_expert(task,sample_spec,output_dir,*,legacy_dir=None):
    if legacy_dir is None:legacy_dir=Path(__file__).resolve().parent.parent/'adjust_bottle_20261002'
    return ExpertSession(task,sample_spec,output_dir,legacy_dir)

def run_episode(task,sample_spec,output_dir,*,legacy_dir=None,record=False,capture_callback=None,validate_actual_geometry=True):
    return install_expert(task,sample_spec,output_dir,legacy_dir=legacy_dir).run(
        record=record,capture_callback=capture_callback,validate_actual_geometry=validate_actual_geometry)
