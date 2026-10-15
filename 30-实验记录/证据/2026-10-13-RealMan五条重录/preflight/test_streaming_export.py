"""Small bounded fixtures: exact native compatibility plus admission failures."""
from __future__ import annotations
import copy
import importlib.util
import json
from pathlib import Path
import pickle
import sys
import tempfile
import types
import unittest
from unittest import mock

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('streaming_export', HERE / 'streaming_export.py')
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def original_exporter():
    # Avoid importing envs.__init__ or a task: equivalence test is CPU-only.
    pkg = types.ModuleType('_tiny_original_export')
    pkg.__path__ = [str(exporter.NATIVE_EXPORTER.parent)]
    sys.modules[pkg.__name__] = pkg
    spec = importlib.util.spec_from_file_location(pkg.__name__ + '.pkl2hdf5', exporter.NATIVE_EXPORTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_capture(root, count=4):
    """Explicit synthetic data; never claimed to be a physical success."""
    candidate = root / 'raw'
    cache = candidate / '.cache/episode0'
    cache.mkdir(parents=True)
    names = [f'{side}_joint{i}' for i in range(1, 7) for side in ('r', 'l')]
    hands = [f'{side}_synthetic_hand{i}' for side in ('l', 'r') for i in range(12)]
    names += hands
    steps = 15 * (count - 1)
    full = np.asarray([[j * .01 + s * .0001 for j in range(36)] for s in range(steps + 1)], dtype=np.float32)
    state24 = np.asarray([[j + s * .01 for j in range(24)] for s in range(steps + 1)], dtype=np.float64)
    command24 = state24 + .5
    telemetry = {
        'physics_step': np.arange(1, steps + 1, dtype=np.int64),
        'sim_time_before_s': np.arange(steps) * .004,
        'sim_time_after_s': np.arange(1, steps + 1) * .004,
        'measured_state24_before': state24[:-1], 'applied_command24': command24[1:],
        'measured_state24_after': state24[1:],
        'hand24_qpos_before_rad': full[:-1, 12:], 'hand24_qpos_after_rad': full[1:, 12:],
        'hand24_drive_target_rad': full[1:, 12:] + np.float32(.002),
        'hand24_actual_fd_rad_s': np.ones((steps, 24), np.float32) * .025,
        'hand24_engine_qvel_rad_s': np.ones((steps, 24), np.float32) * .024,
        'full36_qpos_before_rad': full[:-1], 'full36_qpos_after_rad': full[1:]}
    samples, frames = [], []
    rng = np.random.default_rng(127)
    for index in range(count):
        step = index * 15
        sample = {'physics_step': step, 'sim_time_s': step * .004,
                  'measured_state24': state24[step].tolist(), 'command24': command24[step].tolist()}
        samples.append(sample)
        observation = {}
        for ci, camera in enumerate(exporter.REQUIRED_CAMERAS):
            # Last high-entropy image must NOT enlarge the native S-width.
            rgb = rng.integers(0, 256, (12, 16, 3), dtype=np.uint8) if index == count - 1 else np.full((12, 16, 3), 10 + ci * 40 + index, np.uint8)
            intrinsic = np.asarray([[32, 0, 8], [0, 32, 6], [0, 0, 1]], dtype=np.float32)
            ext = np.eye(4, dtype=np.float64)
            ext[0, 3] = index + ci * .1
            observation[camera] = {'rgb': rgb, 'intrinsic_cv': intrinsic,
                'cam2world_gl': ext, 'extrinsic_cv': ext[:3].astype(np.float32),
                'depth': np.full((12, 16), ci + index / 10, np.float32)}
        frame = {'observation': observation,
            'joint_action': {'left_arm': full[step, [1, 3, 5, 7, 9, 11]].tolist(),
                'right_arm': full[step, [0, 2, 4, 6, 8, 10]].tolist(),
                'left_gripper': 1 - index / 10, 'right_gripper': 1 - index / 11},
            'endpose': {'left_endpose': [index + k / 10 for k in range(7)],
                'right_endpose': [index + k / 11 for k in range(7)]},
            'dexterous_state': {'joint_position_rad': dict(zip(names, full[step].tolist())),
                'state24': state24[step].tolist(), 'command24': command24[step].tolist()},
            'recording_clock': {'physics_step': step, 'sim_time_s': step * .004}}
        frames.append(frame)
        with (cache / f'{index}.pkl').open('wb') as f:
            pickle.dump(frame, f)
    report = {'result': 'synthetic_test_fixture', 'accepted': False, 'hardware_io': False,
        'full_qpos_joint_names': names, 'physical_steps': steps, 'recorded_frames': count,
        'action_clock': {'physics_dt_s': .004, 'record_every_physics_steps': 15}}
    exporter.write_json(candidate / 'expert_report.json', report)
    exporter.write_json(candidate / 'recording_samples.json', samples)
    exporter.write_json(candidate / 'inventory_snapshot.json', {'full36_joint_names': names, 'hand24_joint_names': hands})
    np.savez_compressed(candidate / 'physics_telemetry.npz', **telemetry)
    return candidate, frames, samples, telemetry


class StreamingExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.candidate, self.frames, self.samples, self.telemetry = synthetic_capture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def run_export(self):
        return exporter.export_candidate(self.candidate, self.root / 'export', benchmark_only=True,
            hdf_only=True, expected_rgb_shape=None)

    def rewrite_frame(self, index, function):
        frame = copy.deepcopy(self.frames[index])
        function(frame)
        with (self.candidate / f'.cache/episode0/{index}.pkl').open('wb') as f:
            pickle.dump(frame, f)

    def test_exact_original_native_numeric_jpeg_shapes_and_attributes(self):
        original = original_exporter()
        data = original.parse_dict_structure(self.frames[0])
        for frame in self.frames:
            original.append_data_to_structure(data, frame)
        old = self.root / 'original.hdf5'
        count = original.create_xpolicylab_hdf5(data, old, [exporter.INSTRUCTION], 15)
        receipt = self.run_export()
        self.assertEqual(count, 3)
        self.assertEqual(receipt['native_state_action_pairs'], 3)
        self.assertFalse(receipt['accepted'])
        self.assertTrue(receipt['component_only'])
        new = self.root / 'export/data/episode_0000000.hdf5'
        with h5py.File(old) as before, h5py.File(new) as after:
            self.assertEqual(dict(before.attrs), {k: after.attrs[k] for k in before.attrs})
            paths = []
            before.visititems(lambda n, v: paths.append(n) if isinstance(v, h5py.Dataset) else None)
            for path in paths:
                self.assertEqual(before[path].shape, after[path].shape, path)
                self.assertEqual(before[path].dtype, after[path].dtype, path)
                np.testing.assert_array_equal(before[path][()], after[path][()], err_msg=path)
            # Separate exact original dexterous append semantics.
            np.testing.assert_array_equal(after['realman_dexterous/measured_state24'], [s['measured_state24'] for s in self.samples[:-1]])
            np.testing.assert_array_equal(after['realman_dexterous/next_command24'], [s['command24'] for s in self.samples[1:]])
            for name, array in self.telemetry.items():
                np.testing.assert_array_equal(after['realman_dexterous_physics/' + name], array)
                self.assertEqual(after['realman_dexterous_physics/' + name].dtype, array.dtype)
            self.assertEqual(after['realman_dexterous_physics'].attrs['hand24_joint_names'],
                json.dumps(json.loads((self.candidate / 'inventory_snapshot.json').read_text())['hand24_joint_names']))

    def test_npz_only_loaded_from_disk_mmap_not_full_decompression(self):
        native_load = np.load
        seen = []
        def strict_load(path, *a, **kw):
            self.assertTrue(str(path).endswith('.npy'))
            self.assertEqual(kw.get('mmap_mode'), 'r')
            self.assertFalse(kw.get('allow_pickle'))
            seen.append(str(path))
            return native_load(path, *a, **kw)
        with mock.patch.object(exporter.np, 'load', strict_load):
            self.run_export()
        self.assertEqual(len(seen), len(exporter.PHYSICS_FIELDS))

    def test_missing_middle_cache_refused(self):
        (self.candidate / '.cache/episode0/1.pkl').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing or duplicate'):
            self.run_export()

    def test_single_observation_is_not_state_action_pair(self):
        for index in range(1, 4):
            (self.candidate / f'.cache/episode0/{index}.pkl').unlink()
        with self.assertRaisesRegex(ValueError, 'Incomplete N\\+1'):
            self.run_export()

    def test_last_terminal_observation_schema_is_verified(self):
        self.rewrite_frame(3, lambda f: f['observation'].pop('right_camera'))
        with self.assertRaisesRegex(ValueError, 'three-camera'):
            self.run_export()

    def test_per_frame_intrinsic_dtype_drift_refused(self):
        self.rewrite_frame(1, lambda f: f['observation']['head_camera'].update(
            intrinsic_cv=f['observation']['head_camera']['intrinsic_cv'].astype(np.float64)))
        with self.assertRaisesRegex(ValueError, 'schema/dtype/shape changed'):
            self.run_export()

    def test_rgb_uint16_refused(self):
        self.rewrite_frame(1, lambda f: f['observation']['head_camera'].update(rgb=f['observation']['head_camera']['rgb'].astype(np.uint16)))
        with self.assertRaisesRegex(ValueError, 'RGB shape/dtype'):
            self.run_export()

    def test_sdk_next_command_mismatch_refused(self):
        self.rewrite_frame(2, lambda f: f['dexterous_state']['command24'].__setitem__(6, 999))
        with self.assertRaisesRegex(ValueError, 'SDK24 cache/recorded sample differ'):
            self.run_export()

    def test_real_joint_name_order_mismatch_refused(self):
        snapshot = json.loads((self.candidate / 'inventory_snapshot.json').read_text())
        snapshot['hand24_joint_names'][0:2] = snapshot['hand24_joint_names'][1::-1]
        exporter.write_json(self.candidate / 'inventory_snapshot.json', snapshot)
        with self.assertRaisesRegex(ValueError, 'hand24 observation differs'):
            self.run_export()

    def test_actual_clock_mismatch_refused(self):
        self.rewrite_frame(1, lambda f: f['recording_clock'].update(sim_time_s=.061))
        with self.assertRaisesRegex(ValueError, 'observation clock differs'):
            self.run_export()

    def test_physical_clock_gap_refused(self):
        self.telemetry['physics_step'][12] += 1
        np.savez_compressed(self.candidate / 'physics_telemetry.npz', **self.telemetry)
        with self.assertRaisesRegex(ValueError, 'Physical step gap'):
            self.run_export()

    def test_npz_extra_field_refused(self):
        np.savez_compressed(self.candidate / 'physics_telemetry.npz', **self.telemetry, bogus=np.zeros(4))
        with self.assertRaisesRegex(ValueError, 'telemetry schema'):
            self.run_export()

    def test_strict_mode_never_admits_component_fixture(self):
        with self.assertRaisesRegex(ValueError, 'physical terminal report'):
            exporter.export_candidate(self.candidate, self.root / 'export', hdf_only=True, expected_rgb_shape=None)
        self.assertFalse((self.root / 'export').exists())

    def test_atomic_output_refuses_overwrite_and_preserves_inputs(self):
        pins = {p.name: exporter.sha(p) for p in exporter.cache_paths(self.candidate)}
        receipt = self.run_export()
        self.assertFalse(any(self.root.glob('export.partial.*')))
        self.assertFalse((self.root / 'export/_telemetry_mmap').exists())
        with self.assertRaisesRegex(ValueError, 'overwrite'):
            self.run_export()
        self.assertEqual(pins, {p.name: exporter.sha(p) for p in exporter.cache_paths(self.candidate)})
        self.assertEqual(receipt['bounded_strategy']['resident_raw_observations_max'], 1)

    def test_changed_cache_between_passes_refused(self):
        original_write = exporter.write_native_hdf
        def mutate(*args, **kwargs):
            p = self.candidate / '.cache/episode0/1.pkl'
            p.write_bytes(p.read_bytes() + b'x')
            return original_write(*args, **kwargs)
        with mock.patch.object(exporter, 'write_native_hdf', mutate):
            with self.assertRaisesRegex(ValueError, 'changed between export passes'):
                self.run_export()
        self.assertFalse((self.root / 'export').exists())
        partial = list(self.root.glob('export.partial.*'))
        self.assertEqual(len(partial), 1)
        self.assertTrue((partial[0] / 'export_failure.json').is_file())

    def test_symlink_candidate_refused(self):
        alias = self.root / 'alias'
        alias.symlink_to(self.candidate, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'Symlinked physical candidate'):
            exporter.export_candidate(alias, self.root / 'export', benchmark_only=True, hdf_only=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
