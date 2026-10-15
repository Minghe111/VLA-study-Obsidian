"""Separate CPU video phase over an immutable staged RealMan capture.

Run this under run_with_resource_guard.py --phase render. It constructs no
simulation, invokes the frozen streaming exporter in video-only mode, retains
native preview plus three actual-clock ZOH videos and binds its producer and
every fully-decoded output SHA. A benchmark remains component-only.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE / 'streaming_export.py'
BASE_SHA = '5688b0f084fb215283c498bbefd9b04670479a93e2d4f403134abe59a3676ce3'
STAGED = HERE / 'streaming_export_staged.py'
VIDEO_NAMES = ('episode_0000000.mp4', 'head_realtime.mp4',
               'left_wrist_realtime.mp4', 'right_wrist_realtime.mp4')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def load_implementation(benchmark_only):
    require(BASE.is_file() and not BASE.is_symlink() and sha(BASE) == BASE_SHA,
            'Frozen bounded exporter changed')
    path = BASE if benchmark_only else STAGED
    require(path.is_file() and not path.is_symlink(), 'Staged bounded export binding missing')
    spec = importlib.util.spec_from_file_location('_render_phase_export_implementation', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def render_candidate(candidate, output, *, benchmark_only=False, ffmpeg=None):
    candidate, output = Path(candidate), Path(output)
    require(not output.exists() and not output.is_symlink(), 'Refusing to overwrite video phase output')
    implementation = load_implementation(benchmark_only)
    source_paths = [Path(__file__).resolve(), BASE, Path(implementation.__file__).resolve()]
    producer_pins = {str(path): sha(path) for path in source_paths}
    receipt = implementation.export_candidate(candidate, output, benchmark_only=benchmark_only,
        video_only=True, hdf_only=False, ffmpeg=ffmpeg)
    require(receipt.get('export_complete') is True and receipt.get('accepted') is False and
        receipt.get('component_only') is bool(benchmark_only) and receipt.get('mode') == 'video_only',
        'Video phase cannot accept data or change its benchmark scope')
    require(receipt.get('hardware_io') is False and receipt.get('task_or_scene_constructed') is False and
        receipt.get('raw_input_or_physical_source_writes') == 0,
        'Independent video phase attempted physical IO or input modification')
    strategy = receipt.get('bounded_strategy', {})
    require(strategy.get('resident_raw_observations_max') == 1 and
        strategy.get('ffmpeg_workers_max') == 1 and strategy.get('ffmpeg_threads') == 2,
        'Video phase did not preserve bounded single-frame/single-encoder strategy')
    require(receipt.get('candidate') == str(candidate.resolve()) and receipt.get('output') == str(output.resolve()),
        'Video receipt belongs to a different input/output')
    if not benchmark_only:
        require(receipt.get('original_physical_terminal_and_actual36_admission_passed') is True,
            'Original physical collection admission was not preserved')
    videos = receipt.get('outputs', {})
    require(set(videos) == set(VIDEO_NAMES), 'Native head preview or one actual three-camera video is missing')
    expected_realtime_frames = int(__import__('math').ceil(receipt['actual_duration_s'] * 30))
    for filename in VIDEO_NAMES:
        row = videos[filename]
        path = output.resolve() / 'video' / filename
        require(not path.is_symlink() and path.is_file() and row.get('path') == str(path) and
            row.get('sha256') == sha(path) and row.get('bytes') == path.stat().st_size,
            'Video output path/bytes/SHA differs: ' + filename)
        require(row.get('full_ffmpeg_decode_passed') is True and row.get('playback_fps') == 30,
            'Video did not pass full-frame bounded decode: ' + filename)
        expected = receipt['observations'] if filename == 'episode_0000000.mp4' else expected_realtime_frames
        require(row.get('frames') == expected and row.get('duration_s') == expected / 30,
            'Video frame count/duration differs from its original capture clock')
        if filename != 'episode_0000000.mp4':
            age = row.get('max_observation_age_s')
            require(isinstance(age, (int, float)) and 0 <= age <= .060 + 1e-7 and
                row.get('method') == 'zero-order hold actual sample clock', 'Video changed original ZOH observation semantics')
    require(all(sha(path) == digest for path, digest in producer_pins.items()), 'Video phase producer changed while rendering')
    export_receipt = output / 'export_receipt.json'
    require(export_receipt.is_file() and not export_receipt.is_symlink(), 'Frozen exporter completion receipt missing')
    with export_receipt.open(encoding='utf-8') as stream:
        require(json.load(stream) == receipt, 'Returned video evidence differs from saved frozen export receipt')
    binding = {'version': 'realman_staged_video_phase_v1', 'render_complete': True,
        'accepted': False, 'component_only': bool(benchmark_only), 'hardware_io': False,
        'candidate': str(candidate.resolve()), 'output': str(output.resolve()),
        'wrapper_and_export_producer_sha256': producer_pins,
        'frozen_export_receipt_sha256': sha(export_receipt),
        'actual_three_camera_video_outputs': {name: videos[name] for name in VIDEO_NAMES},
        'source_clock_unchanged': True,
        'resource_contract': {'external_guard_required': True, 'phase': 'render',
            'memory_max_GiB': 3, 'CPU_quota_cores': 2, 'swap_max_bytes': 0,
            'one_ffmpeg_process_at_a_time': True, 'ffmpeg_threads': 2},
        'scope': 'CPU media component; original source/physics/motion/DR acceptance is separate'}
    staged_binding = output / 'staged_export_binding.json'
    if not benchmark_only:
        require(staged_binding.is_file(), 'Strict staged full-audit export binding missing')
    if staged_binding.is_file():
        binding['staged_export_binding_sha256'] = sha(staged_binding)
    path = output / 'render_stage_receipt.json'
    require(not path.exists(), 'Video phase binding already exists')
    path.write_text(json.dumps(binding, ensure_ascii=False, indent=2), encoding='utf-8')
    return binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--benchmark-only', action='store_true')
    parser.add_argument('--ffmpeg', type=Path)
    args = parser.parse_args()
    receipt = render_candidate(args.candidate, args.output, benchmark_only=args.benchmark_only, ffmpeg=args.ffmpeg)
    print(json.dumps({'render_complete': receipt['render_complete'], 'accepted': False,
        'component_only': receipt['component_only'], 'output': receipt['output']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
