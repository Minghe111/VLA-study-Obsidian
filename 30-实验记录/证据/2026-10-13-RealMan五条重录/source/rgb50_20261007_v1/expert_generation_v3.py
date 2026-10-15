"""Additive expert v3: audited native free-hand approach, unchanged v2 grip/carry.

The optional native backend changes planning before contact only. Contact force
allocation, physical servo, held-object planning, source pins and RGB clock are
inherited unchanged. No native arm state is copied into the physical scene.
"""
from pathlib import Path
import importlib
import json
import numpy as np

from expert_generation_v2 import (ExpertSession as _BaseSession,
                                 install_initializer, command24, measured24,
                                 _sha, _json)


class ExpertSession(_BaseSession):
    def __init__(self, task, sample_spec, output_dir, legacy_dir):
        super().__init__(task, sample_spec, output_dir, legacy_dir)
        self.free_approach = importlib.import_module('full_native_approach')
        for path in (Path(__file__).resolve(),
                     Path(self.free_approach.__file__).resolve()):
            self.sources[str(path)] = _sha(path)
        self.report.update(expert_version='v3',
                           free_approach_backend=self.spec.get('planning_backend', 'curobo'),
                           source_hashes=self.sources)
        self.write()

    def plan_path(self, robot, target_pose, constraint_pose=None, **kwargs):
        if self.dense_calls >= 3 or self.spec.get('planning_backend') != 'full_native':
            return super().plan_path(robot, target_pose,
                                     constraint_pose=constraint_pose, **kwargs)
        index = self.dense_calls
        result = self.free_approach.plan_free_approach(
            self.task, target_pose, self.side, time_step=.004,
            random_seed=int(self.spec.get('seed', 0)) + 1009 * index,
            ik_attempts=int(self.spec.get('native_ik_attempts', 24)))
        metadata = result.get('planning_metadata', {})
        evidence = {'index': index, 'side': self.side,
                    'backend': 'full_native_free_hand_MPLib',
                    'constraint_request': constraint_pose,
                    'status': result.get('status'),
                    'target_pose': np.asarray(target_pose).tolist(),
                    'samples': len(result.get('position', [])),
                    'reason': result.get('reason'),
                    'planning_metadata': metadata}
        if not metadata.get('physical_state_unchanged') or not metadata.get('no_attached_objects'):
            raise RuntimeError('Native free-hand planner lacked scene-preservation evidence')
        if result.get('status') == 'Success':
            geometry = result.get('audit')
            if not geometry or not geometry.get('passed'):
                raise RuntimeError('Native free-hand path lacked successful full-mesh audit')
            (self.out / f'mesh_path_{index}.json').write_text(
                json.dumps(geometry, default=_json))
            np.savez(self.out / f'planned_path_{index}.npz',
                     position=result['position'], velocity=result['velocity'])
            evidence['full_mesh_audit_passed'] = True
        (self.out / f'free_approach_{index}.json').write_text(
            json.dumps(evidence, default=_json))
        self.report.setdefault('planning_calls', []).append(evidence)
        self.write()
        return result


def install_expert(task, sample_spec, output_dir, *, legacy_dir=None):
    if legacy_dir is None:
        legacy_dir = Path(__file__).resolve().parent.parent / 'adjust_bottle_20261002'
    return ExpertSession(task, sample_spec, output_dir, legacy_dir)


def run_episode(task, sample_spec, output_dir, *, legacy_dir=None,
                record=False, capture_callback=None,
                validate_actual_geometry=True):
    session = install_expert(task, sample_spec, output_dir,
                             legacy_dir=legacy_dir)
    return session.run(record=record, capture_callback=capture_callback,
                       validate_actual_geometry=validate_actual_geometry)
