"""CPU-only wrapper admission tests; media descriptors are synthetic stubs."""
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('render_staged_video', HERE / 'render_staged_video.py')
render = importlib.util.module_from_spec(spec)
spec.loader.exec_module(render)


class VideoPhaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.candidate = self.root / 'raw'
        self.candidate.mkdir()
        self.output = self.root / 'rendered'
        self.calls = []
        self.mutate = lambda receipt: None
        self.producer = self.root / 'synthetic_producer.py'
        self.producer.write_text('# CPU-only synthetic stub, no actual media verification\n')
        self.module = types.SimpleNamespace(__file__=str(self.producer), export_candidate=self.stub_export)

    def tearDown(self):
        self.temp.cleanup()

    def stub_export(self, candidate, output, **kwargs):
        self.calls.append((candidate, output, kwargs))
        output.mkdir()
        (output / 'video').mkdir()
        outputs = {}
        for name in render.VIDEO_NAMES:
            path = output / 'video' / name
            path.write_bytes(b'explicit synthetic fixture; not an encoded video')
            native = name == 'episode_0000000.mp4'
            count = 4 if native else 6
            outputs[name] = {'path': str(path), 'sha256': render.sha(path), 'bytes': path.stat().st_size,
                'frames': count, 'playback_fps': 30, 'duration_s': count / 30,
                'full_ffmpeg_decode_passed': True, 'max_observation_age_s': None if native else .055,
                'method': 'native N+1 preview at30fps' if native else 'zero-order hold actual sample clock'}
        receipt = {'export_complete': True, 'accepted': False, 'component_only': kwargs['benchmark_only'],
            'mode': 'video_only', 'hardware_io': False, 'task_or_scene_constructed': False,
            'raw_input_or_physical_source_writes': 0, 'candidate': str(candidate), 'output': str(output),
            'original_physical_terminal_and_actual36_admission_passed': not kwargs['benchmark_only'],
            'observations': 4, 'actual_duration_s': .180,
            'bounded_strategy': {'resident_raw_observations_max': 1, 'ffmpeg_workers_max': 1, 'ffmpeg_threads': 2},
            'outputs': outputs}
        self.mutate(receipt)
        (output / 'export_receipt.json').write_text(json.dumps(receipt))
        if not kwargs['benchmark_only']:
            (output / 'staged_export_binding.json').write_text(json.dumps({'synthetic_stub': True, 'accepted': False}))
        return receipt

    def call_wrapper(self, benchmark_only=True):
        with mock.patch.object(render, 'load_implementation', return_value=self.module):
            return render.render_candidate(self.candidate, self.output, benchmark_only=benchmark_only,
                ffmpeg=Path('/synthetic_ffmpeg_not_executed'))

    def test_forces_isolated_video_only_and_binds_producer_source(self):
        binding = self.call_wrapper()
        self.assertEqual(len(self.calls), 1)
        kwargs = self.calls[0][2]
        self.assertTrue(kwargs['video_only'])
        self.assertFalse(kwargs['hdf_only'])
        self.assertTrue(kwargs['benchmark_only'])
        self.assertFalse(binding['accepted'])
        self.assertTrue(binding['component_only'])
        self.assertEqual(binding['wrapper_and_export_producer_sha256'][str(self.producer)], render.sha(self.producer))
        self.assertEqual(binding['frozen_export_receipt_sha256'], render.sha(self.output / 'export_receipt.json'))

    def test_strict_staged_binding_is_retained(self):
        binding = self.call_wrapper(benchmark_only=False)
        self.assertFalse(binding['component_only'])
        self.assertIn('staged_export_binding_sha256', binding)

    def test_missing_wrist_video_refused(self):
        self.mutate = lambda receipt: receipt['outputs'].pop('right_wrist_realtime.mp4')
        with self.assertRaisesRegex(ValueError, 'three-camera video is missing'):
            self.call_wrapper()
        self.assertFalse((self.output / 'render_stage_receipt.json').exists())

    def test_zero_order_hold_age_violation_refused(self):
        self.mutate = lambda receipt: receipt['outputs']['head_realtime.mp4'].update(max_observation_age_s=.061)
        with self.assertRaisesRegex(ValueError, 'ZOH observation semantics'):
            self.call_wrapper()

    def test_unbounded_encoder_strategy_refused(self):
        self.mutate = lambda receipt: receipt['bounded_strategy'].update(ffmpeg_workers_max=3)
        with self.assertRaisesRegex(ValueError, 'bounded single-frame'):
            self.call_wrapper()

    def test_media_cannot_claim_data_acceptance(self):
        self.mutate = lambda receipt: receipt.update(accepted=True)
        with self.assertRaisesRegex(ValueError, 'cannot accept data'):
            self.call_wrapper()

    def test_frame_count_source_clock_violation_refused(self):
        self.mutate = lambda receipt: receipt['outputs']['head_realtime.mp4'].update(frames=7)
        with self.assertRaisesRegex(ValueError, 'original capture clock'):
            self.call_wrapper()

    def test_output_hash_mismatch_refused(self):
        self.mutate = lambda receipt: receipt['outputs']['left_wrist_realtime.mp4'].update(sha256='0' * 64)
        with self.assertRaisesRegex(ValueError, 'path/bytes/SHA differs'):
            self.call_wrapper()

    def test_refuses_existing_output_before_export_call(self):
        self.output.mkdir()
        with self.assertRaisesRegex(ValueError, 'overwrite'):
            self.call_wrapper()
        self.assertEqual(len(self.calls), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
