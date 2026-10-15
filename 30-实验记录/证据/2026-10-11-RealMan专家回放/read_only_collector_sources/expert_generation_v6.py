"""v5 control, exact FMA broadphase vectorization and raw carry diagnostics.

Bootstrap only: install the independently verified equivalent AABB filter
before importing any geometry-dependent planner. Scalar/FCL bodies, geometry,
thresholds, physics/contact control and final v13 actual audit are unchanged.
"""
from pathlib import Path
import copy
import json
import numpy as np

from geometry_vector_overlay import install_vectorized_aabb_overlay

_LEGACY=Path(__file__).resolve().parent.parent/'adjust_bottle_20261002'
_VECTOR_RECEIPT=install_vectorized_aabb_overlay(_LEGACY)

from expert_generation_v5 import (ExpertSession as _BaseSession,
                                 install_initializer, command24, measured24,
                                 _sha, _json)


class _NativeCarryReadback:
    """Session-local delegate; the immutable helper and its global API stay intact."""
    def __init__(self,session,source):self.session=session;self.source=source
    def __getattr__(self,name):return getattr(self.source,name)
    def plan_carry_to_task_pose(self,*args,**kwargs):
        result=self.source.plan_carry_to_task_pose(*args,**kwargs)
        session=self.session;index=session.dense_calls;number=session.native_carry_candidate
        evidence={k:copy.deepcopy(v) for k,v in result.items() if k not in ('position','velocity','time')}
        evidence.update(index=index,candidate=number,
                        scope='raw immutable native carry planner return before contact-wrench preflight; not physical success',
                        physical_success_checked=False,
                        samples=len(result.get('position',[])))
        if 'position' in result:
            np.savez(session.out/f'native_carry_{index}_{number:02d}_path.npz',
                     position=result['position'],velocity=result['velocity'],time=result.get('time',[]))
        (session.out/f'native_carry_{index}_{number:02d}_result.json').write_text(json.dumps(evidence,default=_json))
        session.report.setdefault('raw_native_carry_results',[]).append(evidence)
        session.write()
        return result


class ExpertSession(_BaseSession):
    def __init__(self,task,sample_spec,output_dir,legacy_dir):
        super().__init__(task,sample_spec,output_dir,legacy_dir)
        import geometry_vector_overlay,geometry_overlay
        for path in (Path(__file__).resolve(),Path(geometry_vector_overlay.__file__).resolve(),
                     Path(geometry_overlay.__file__).resolve()):self.sources[str(path)]=_sha(path)
        self.native_carry_candidate=0
        self.ncp=_NativeCarryReadback(self,self.ncp)
        self.report.update(expert_version='v6',source_hashes=self.sources,
            geometry_vector_overlay=copy.deepcopy(_VECTOR_RECEIPT),
            geometry_vector_scope='unchanged scalar/FCL, thresholds, meshes, sample grid and negative collision checks; verified equivalent broadphase only',
            actual_geometry_auditor='immutable observed_full_trace_audit_v13')
        self.write()

    def _carry_candidate(self,robot,target_pose,constraint_pose,number,**kwargs):
        self.native_carry_candidate=number
        return super()._carry_candidate(robot,target_pose,constraint_pose,number,**kwargs)


def install_expert(task,sample_spec,output_dir,*,legacy_dir=None):
    if legacy_dir is None:legacy_dir=_LEGACY
    return ExpertSession(task,sample_spec,output_dir,legacy_dir)


def run_episode(task,sample_spec,output_dir,*,legacy_dir=None,record=False,
                capture_callback=None,validate_actual_geometry=True):
    session=install_expert(task,sample_spec,output_dir,legacy_dir=legacy_dir)
    return session.run(record=record,capture_callback=capture_callback,
                       validate_actual_geometry=validate_actual_geometry)
