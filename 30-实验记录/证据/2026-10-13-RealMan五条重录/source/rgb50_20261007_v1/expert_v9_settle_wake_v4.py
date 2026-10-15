"""Metadata-only v9 subclass binds pre-recording extra-settle source evidence.

All motion, LP, actual geometry, success and recording methods are inherited.
"""
from pathlib import Path
import copy
import collector_settle_wake_v4 as collector
from expert_generation_v9_equivalent_goals_dense_v7 import ExpertSession as _BaseSession, install_initializer, command24, measured24, _sha


class ExpertSession(_BaseSession):
    def __init__(self, task, sample_spec, output_dir, legacy_dir):
        pins = collector.producer_pins()
        receipt = copy.deepcopy(getattr(task, '_rgb50_initial_extra_settle_receipt', None))
        if receipt is None or receipt['source_hashes'] != pins:
            raise RuntimeError('Expert requires unchanged initialization producer receipt')
        plan = collector.settle_plan(sample_spec, task.scene.get_timestep())
        if any(receipt[k] != v for k, v in plan.items()) or receipt['completed_extra_physics_steps'] != plan['extra_physics_steps']:
            raise RuntimeError('Initialization extra-settle clock/spec differs')
        wake = receipt.get('native_wake')
        if not isinstance(wake, dict) or wake.get('requested') is not plan['initial_wake_before_extra_settle']:
            raise RuntimeError('Native wake receipt differs from explicit spec')
        if plan['initial_wake_before_extra_settle']:
            if wake.get('wake_calls') != 1:
                raise RuntimeError('Exactly one requested native wake required')
            verification = collector._verify_immediate_native_wake(wake['native_state_before'], wake['native_state_immediately_after'])
            if verification != wake['immediate_verification']:
                raise RuntimeError('Native wake immediate numeric receipt differs')
        elif wake.get('wake_calls') != 0 or any(wake.get(k) is not None for k in ('native_state_before', 'native_state_immediately_after', 'immediate_verification', 'native_state_after_extra')):
            raise RuntimeError('Unrequested native wake or unexpected native state evidence')
        super().__init__(task, sample_spec, output_dir, legacy_dir)
        if collector.producer_pins() != pins:
            raise RuntimeError('Initialization sources changed before expert execution')
        self.sources.update(pins)
        self.report.update(source_hashes=self.sources,
            expert_version='v9_dense_v7_pre_recording_optional_native_wake_v4',
            initial_extra_settle=receipt,
            producer_provenance_registered_before_run_and_all_final_audits=True)
        self.write()


def install_expert(task, sample_spec, output_dir, *, legacy_dir=None):
    if legacy_dir is None:
        legacy_dir = Path(__file__).resolve().parent.parent / 'adjust_bottle_20261002'
    return ExpertSession(task, sample_spec, output_dir, legacy_dir)


def run_episode(task, sample_spec, output_dir, *, legacy_dir=None, record=False, capture_callback=None, validate_actual_geometry=True):
    return install_expert(task, sample_spec, output_dir, legacy_dir=legacy_dir).run(
        record=record, capture_callback=capture_callback, validate_actual_geometry=validate_actual_geometry)