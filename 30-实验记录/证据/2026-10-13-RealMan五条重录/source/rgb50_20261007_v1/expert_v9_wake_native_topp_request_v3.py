"""Explicit conservative native TOPP request; all wake-v5/v7 control inherited."""
from pathlib import Path
from types import ModuleType
import inspect
import native_topp_request_margin_v1 as overlay
from expert_v9_settle_wake_v5 import ExpertSession as _BaseSession, install_initializer, command24, measured24, _sha
from expert_generation_v6 import _NativeCarryReadback

READBACK_SOURCE_SHA256='d1401783c2b6d90d791127e32e8cbc726f66b30ede212eec2868e0a3ef3135e9'


class ExpertSession(_BaseSession):
    def __init__(self,task,sample_spec,output_dir,legacy_dir):
        margin=overlay.request_margin(sample_spec)
        super().__init__(task,sample_spec,output_dir,legacy_dir)
        readback_path=Path(inspect.getsourcefile(_NativeCarryReadback)).resolve()
        if _sha(readback_path)!=READBACK_SOURCE_SHA256:
            raise RuntimeError('Frozen original carry readback source changed')
        delegate=self.ncp
        if (type(delegate) is not _NativeCarryReadback or delegate.session is not self
                or getattr(delegate.plan_carry_to_task_pose,'__func__',None) is not _NativeCarryReadback.plan_carry_to_task_pose):
            raise RuntimeError('Expected current session-owned immutable native carry readback delegate')
        source=delegate.source
        if not isinstance(source,ModuleType) or getattr(source.plan_carry_to_task_pose,'__self__',None) is not None:
            raise RuntimeError('Native carry delegate must reference original module function, not a bound method')
        private,binding=overlay.private_carry_module(source,margin)
        if margin!=1:
            delegate.source=private
        if self.ncp is not delegate or delegate.session is not self or delegate.source is not private:
            raise RuntimeError('Original carry readback delegate/owner/source binding differs')
        binding={**binding,'session_local_original_v6_delegate_preserved':True,
            'default1_original_delegate_and_source_identity_preserved':margin==1,
            'private_module_applied_only_to_delegate_source':margin!=1,
            'bound_wrapper_method_not_recompiled_as_module_function':True}
        task._rgb50_native_topp_request_receipts=[]
        source=Path(__file__).resolve()
        for p in [source,Path(overlay.__file__).resolve(),source.parent/'worker_wake_native_topp_request_v3.py']:
            if not p.is_file():raise RuntimeError('Missing complete conservative TOPP request producer source')
            self.sources[str(p)]=_sha(p)
        self.sources[str(readback_path)]=READBACK_SOURCE_SHA256
        self.report.update(expert_version='v9_dense_v7_wake_v5_explicit_native_topp_request_v3',
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
