"""Explicit conservative native TOPP request; all wake-v4/v7 control inherited."""
from pathlib import Path
import native_topp_request_margin_v1 as overlay
from expert_v9_settle_wake_v4 import ExpertSession as _BaseSession, install_initializer, command24, measured24, _sha


class ExpertSession(_BaseSession):
    def __init__(self,task,sample_spec,output_dir,legacy_dir):
        margin=overlay.request_margin(sample_spec)
        super().__init__(task,sample_spec,output_dir,legacy_dir)
        self.ncp,binding=overlay.private_carry_module(self.ncp,margin)
        task._rgb50_native_topp_request_receipts=[]
        source=Path(__file__).resolve()
        for p in [source,Path(overlay.__file__).resolve(),source.parent/'worker_wake_native_topp_request_v1.py']:
            if not p.is_file():raise RuntimeError('Missing complete conservative TOPP request producer source')
            self.sources[str(p)]=_sha(p)
        self.report.update(expert_version='v9_dense_v7_wake_v4_explicit_native_topp_request_v1',
            source_hashes=self.sources,native_topp_request_binding=binding,
            native_topp_request_receipts=task._rgb50_native_topp_request_receipts,
            native_topp_request_registered_before_run_and_all_final_audits=True)
        self.write()


def install_expert(task,sample_spec,output_dir,*,legacy_dir=None):
    if legacy_dir is None:legacy_dir=Path(__file__).resolve().parent.parent/'adjust_bottle_20261002'
    return ExpertSession(task,sample_spec,output_dir,legacy_dir)


def run_episode(task,sample_spec,output_dir,*,legacy_dir=None,record=False,capture_callback=None,validate_actual_geometry=True):
    return install_expert(task,sample_spec,output_dir,legacy_dir=legacy_dir).run(
        record=record,capture_callback=capture_callback,validate_actual_geometry=validate_actual_geometry)
