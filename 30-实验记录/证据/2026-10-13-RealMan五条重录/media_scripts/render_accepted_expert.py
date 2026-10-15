"""Composite three native videos of one actually accepted new DR expert.

This tool only reads captured files. It never loads a simulator or a policy.
Run with the existing robotwin Python; no environment changes are required.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys


CAMERAS = ('head', 'left_wrist', 'right_wrist')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def ffmpeg_binary(requested):
    candidate = requested or shutil.which('ffmpeg')
    if not candidate:
        candidate = str(Path(sys.executable).resolve().parent / 'ffmpeg')
    check(Path(candidate).is_file(), 'ffmpeg unavailable; use --ffmpeg with existing binary')
    return str(Path(candidate).resolve())


def decode_and_probe(path, ffmpeg, logs):
    """Read metadata and fully decode; never infer validity from container alone."""
    import cv2
    capture = cv2.VideoCapture(str(path))
    check(capture.isOpened(), 'Cannot open source video: ' + str(path))
    try:
        metadata = {'width': int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    'height': int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    'fps': float(capture.get(cv2.CAP_PROP_FPS)),
                    'container_frames': int(capture.get(cv2.CAP_PROP_FRAME_COUNT))}
        readable, frame = capture.read()
        check(readable and frame is not None, 'First frame could not be decoded')
    finally:
        capture.release()
    check(metadata['fps'] > 0, 'Video FPS missing')
    command = [ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error',
               '-i', str(path), '-map', '0:v:0', '-an', '-f', 'null', '-',
               '-nostats', '-progress', 'pipe:1']
    result = subprocess.run(command, capture_output=True, text=True)
    progress = {}
    for line in result.stdout.splitlines():
        if '=' in line:
            key, value = line.split('=', 1)
            progress[key] = value
    (logs / (path.stem + '_decode.txt')).write_text(result.stdout + result.stderr)
    check(result.returncode == 0 and progress.get('progress') == 'end',
          'Full ffmpeg decode failed: ' + str(path))
    check(not result.stderr.strip(), 'Full ffmpeg decode reported corruption: ' + str(path))
    frames = int(progress.get('frame', '0'))
    check(frames > 0 and frames == metadata['container_frames'], 'Decoded frame count differs from metadata')
    metadata.update({'decoded_frames': frames, 'duration_s': frames / metadata['fps'],
                     'decoder_exit_code': result.returncode, 'full_decode_passed': True,
                     'decoder_out_time_us': int(progress.get('out_time_us', '0')),
                     'sha256': sha(path), 'path': str(path)})
    return metadata


def ass_time(seconds):
    # ASS resolution is centiseconds. Floor each shared boundary so every
    # 30fps frame starts inside its own interval, rather than inheriting the
    # previous label when e.g. 0.066667 rounds up to 0.07.
    centiseconds = int(math.floor(seconds * 100 + 1e-8))
    hours, rest = divmod(centiseconds, 360000)
    minutes, rest = divmod(rest, 6000)
    seconds, hundredths = divmod(rest, 100)
    return f'{hours}:{minutes:02d}:{seconds:02d}.{hundredths:02d}'


def ass_text(value):
    return str(value).replace('\\', '/').replace('{', '(').replace('}', ')').replace('\n', ' ')


def load_timeline(hdf):
    import h5py
    import numpy as np
    with h5py.File(hdf, 'r') as source:
        names = ('realman_dexterous/sample_time_s', 'realman_dexterous/next_sample_time_s',
                 'realman_dexterous/measured_state24',
                 'realman_dexterous_physics/sim_time_before_s',
                 'realman_dexterous_physics/sim_time_after_s',
                 'realman_dexterous_physics/measured_state24_before',
                 'realman_dexterous_physics/measured_state24_after')
        arrays = {name: source[name][...] for name in names}
        check(bool(source.attrs.get('new_physics_episode', False)), 'HDF is not marked as a new physics episode')
    observation = np.asarray(arrays[names[0]], dtype=float)
    following = np.asarray(arrays[names[1]], dtype=float)
    states = np.asarray(arrays[names[2]], dtype=float)
    before_time, after_time = [np.asarray(arrays[name], dtype=float) for name in names[3:5]]
    before, after = [np.asarray(arrays[name], dtype=float) for name in names[5:7]]
    check(len(observation) >= 2 and observation.shape == following.shape, 'Incomplete native observation clock')
    check(states.shape == (len(observation), 24), 'Observation SDK24 shape differs')
    check(before.shape == after.shape == (len(before_time), 24), 'Dense actual SDK24 shapes differ')
    check(before_time.shape == after_time.shape and len(before_time) >= 2, 'Incomplete physical clock')
    check(all(np.isfinite(a).all() for a in arrays.values()), 'Nonfinite actual timeline data')
    check(abs(observation[0]) < 1e-6 and abs(before_time[0]) < 1e-6, 'Capture prefix must start at physical zero')
    check(np.allclose(np.diff(observation), .060, atol=1e-6, rtol=0)
          and np.allclose(following - observation, .060, atol=1e-6, rtol=0), 'Native 60ms clock differs')
    check(np.allclose(np.diff(before_time), .004, atol=1e-6, rtol=0)
          and np.allclose(after_time - before_time, .004, atol=1e-6, rtol=0), 'Physical 4ms clock differs')
    check(float(np.max(np.abs(before[1:] - after[:-1]))) <= 1e-6, 'Measured physical state chain differs')
    dense_time = np.r_[before_time[:1], after_time]
    dense_states = np.vstack((before[:1], after))
    # The original video exporter includes a terminal observation at following[-1].
    # HDF state/action pairs omit that observation, so only interior RGB timestamps
    # are bound to HDF sample_time_s; the duration is separately read from following.
    duration = float(following[-1])
    check(abs(duration - float(after_time[-1])) <= 1e-6, 'Terminal native and physics clocks differ')
    indices = np.round(observation / .004).astype(int)
    check(np.all(indices < len(dense_time)) and np.max(np.abs(dense_time[indices] - observation)) <= 1e-6,
          'Native observation timestamp does not exist in physical clock')
    check(float(np.max(np.abs(dense_states[indices] - states))) <= 1e-6, 'Native state does not match physical actual measurement')
    proof = {'hdf_fields_sha256': {name: hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
                                  for name, values in arrays.items()},
             'native_observation_pairs': len(observation), 'physical_steps': len(before_time),
             'native_observation_dt_s': .060, 'physics_dt_s': .004,
             'actual_duration_s': duration, 'state_alignment_passed': True,
             'j6_measurement_source': 'physical measured_state24 before/after; left column5,right column17',
             'terminal_rgb_observation_is_not_a_state_action_pair': True}
    return observation, dense_time, dense_states, duration, proof


def create_overlay(out, spec, observation, dense_time, dense_states, frames, duration):
    import numpy as np
    header = '''[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 600
WrapStyle: 2
ScaledBorderAndShadow: yes
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,DejaVu Sans,22,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,1,0,7,0,0,0,1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
'''
    events = []
    def event(start, end, x, y, text):
        events.append(f'Dialogue: 0,{ass_time(start)},{ass_time(end)},Default,,0,0,0,,{{\\pos({x},{y})}}{ass_text(text)}')
    end = frames / 30
    event(0, end, 20, 9, 'NEW EXPERT | REAL PHYSICS SIM | OFFICIAL DOMAIN RANDOMIZATION | not policy inference')
    event(0, end, 20, 563, 'Camera RGB: captured 60ms observations, 30fps zero-order hold. J6: actual physical measurement, no angle wrapping.')
    for x, name in zip((20, 660, 1300), ('HEAD CAMERA', 'LEFT WRIST CAMERA', 'RIGHT WRIST CAMERA')):
        event(0, end, x, 87, name)
    timeline = []
    for tick in range(frames):
        t = tick / 30
        # Match the original exporter exactly, including floating-point
        # boundary comparisons; do not relabel a held older RGB sample.
        obs_row = max(int(np.searchsorted(observation, t, side='right') - 1), 0)
        physical_row = max(int(np.searchsorted(dense_time, t, side='right') - 1), 0)
        measured = dense_states[physical_row]
        record = {'video_frame': tick, 'playback_time_s': t, 'held_rgb_observation_index': obs_row,
                  'held_rgb_observation_time_s': float(observation[obs_row]),
                  'actual_physics_step': physical_row, 'actual_physics_time_s': float(dense_time[physical_row]),
                  'actual_left_J6_deg': float(measured[5] * 180 / np.pi),
                  'actual_right_J6_deg': float(measured[17] * 180 / np.pi)}
        timeline.append(record)
        text = (f"slot {int(spec['index'])} | {spec['brand']} | {spec['arm']} | domain seed {int(spec['actual_domain_seed'])} | "
                f"video {t:.3f}s | RGB #{obs_row} at {record['held_rgb_observation_time_s']:.3f}s")
        event(t, (tick + 1) / 30, 20, 36, text)
        event(t, (tick + 1) / 30, 20, 61,
              f"actual physics step {physical_row} at {record['actual_physics_time_s']:.3f}s | "
              f"measured J6: left {record['actual_left_J6_deg']:+.2f} deg; right {record['actual_right_J6_deg']:+.2f} deg")
    (out / 'overlay.ass').write_text(header + '\n'.join(events) + '\n')
    with (out / 'actual_frame_clock_j6.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(timeline[0]))
        writer.writeheader(); writer.writerows(timeline)
    return {'frames': frames, 'display_fps': 30, 'native_rgb_zoh_dt_s': .060,
            'physics_values_zoh_dt_s': .004, 'maximum_rgb_observation_age_s':
            max(row['playback_time_s'] - row['held_rgb_observation_time_s'] for row in timeline),
            'maximum_physics_measurement_age_s':
            max(row['playback_time_s'] - row['actual_physics_time_s'] for row in timeline),
            'duration_quantization_s': frames / 30 - duration,
            'video_is_rendered_from_captured_rgb_not_new_simulation': True}


def render(candidate, output, ffmpeg):
    candidate, output = Path(candidate).resolve(), Path(output).resolve()
    check(candidate.is_dir(), 'Candidate directory missing')
    check(candidate != output and candidate not in output.parents, 'Output must be separate from candidate source directory')
    check(not output.exists() or (output.is_dir() and not any(output.iterdir())), 'Refusing to overwrite output')
    output.mkdir(parents=True, exist_ok=True)
    logs = output / 'decode_logs'; logs.mkdir()
    protected = {}
    try:
        reporting = candidate / 'recording_report.json'
        quality_path = candidate / 'independent_motion_quality.json'
        spec_path = candidate / 'sample_spec.json'
        report, quality, spec = [read_json(p) for p in (reporting, quality_path, spec_path)]
        check(report.get('accepted') is True and report.get('new_physics_episode') is True
              and report.get('hardware_io') is False, 'Candidate is not an accepted new simulated expert')
        check(quality.get('overall_publish_gate_passed') is True, 'Complete independent quality gate has not passed')
        check(report.get('additional_quality_report_sha256') == sha(quality_path), 'Quality report hash differs')
        check(spec.get('index') in (17, 18, 20, 29, 34)
              and spec.get('actual_domain_seed') == spec.get('seed'), 'New randomized slot/seed metadata invalid')
        hdf = candidate / 'data/episode_0000000.hdf5'
        source_sha = sha(hdf)
        check(report.get('hdf5_sha256') == source_sha and quality.get('numeric_source_sha256') == source_sha,
              'Accepted recording/quality does not bind this exact HDF')
        inputs = [reporting, quality_path, spec_path, hdf] + [candidate / f'video/{name}_realtime.mp4' for name in CAMERAS]
        protected = {str(path): sha(path) for path in inputs}
        observation, dense_time, dense_states, duration, timeline_proof = load_timeline(hdf)
        check(abs(float(report['actual_duration_s']) - duration) <= 1e-6, 'Recording duration differs from actual HDF clock')
        count = int(math.ceil(duration * 30))
        sources = {}
        for camera in CAMERAS:
            path = candidate / f'video/{camera}_realtime.mp4'
            meta = decode_and_probe(path, ffmpeg, logs)
            official = report['realtime_videos'][camera]
            check(meta['width'] == 640 and meta['height'] == 480 and abs(meta['fps'] - 30) < 1e-6,
                  'Native source video dimensions/FPS differ')
            check(meta['decoded_frames'] == count == int(official['frames'])
                  and abs(float(official['duration_s']) - meta['duration_s']) < 1e-6
                  and official['method'] == 'zero-order hold actual sample clock', 'Native export frame/clock report differs')
            sources[camera] = meta
        overlay = create_overlay(output, spec, observation, dense_time, dense_states, count, duration)
        dest = output / f"expert_slot_{int(spec['index']):07d}_dr_seed_{int(spec['actual_domain_seed'])}.mp4"
        command = [ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error']
        for camera in CAMERAS:
            command += ['-i', str(candidate / f'video/{camera}_realtime.mp4')]
        command += ['-filter_complex',
                    '[0:v][1:v][2:v]hstack=inputs=3:shortest=1,pad=1920:600:0:80:black,subtitles=filename=overlay.ass[v]',
                    '-map', '[v]', '-an', '-frames:v', str(count), '-r', '30', '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p', '-crf', '18', '-movflags', '+faststart', str(dest)]
        process = subprocess.run(command, capture_output=True, text=True, cwd=output)
        (output / 'encoding_log.txt').write_text(process.stdout + process.stderr)
        check(process.returncode == 0, 'Composite encoding failed')
        composite = decode_and_probe(dest, ffmpeg, logs)
        check((composite['width'], composite['height'], composite['decoded_frames']) == (1920, 600, count)
              and abs(composite['fps'] - 30) < 1e-6, 'Composite frame geometry or timing changed')
        unchanged = {path: sha(path) == digest for path, digest in protected.items()}
        check(all(unchanged.values()), 'Source input bytes changed during media validation')
        receipt = {'schema': 'new-randomized-expert-three-camera-composite-v1', 'passed': True,
                   'candidate': str(candidate), 'slot_index': int(spec['index']),
                   'actual_domain_seed': int(spec['actual_domain_seed']), 'brand': spec['brand'], 'arm': spec['arm'],
                   'accepted_quality_sha256': sha(quality_path), 'hdf5_sha256': source_sha,
                   'source_video_full_decode': sources, 'hdf_clock_actual_j6_proof': timeline_proof,
                   'display_timeline': overlay, 'composite': composite,
                   'overlay_sha256': sha(output / 'overlay.ass'),
                   'display_clock_csv_sha256': sha(output / 'actual_frame_clock_j6.csv'),
                   'protected_inputs_sha256': protected, 'input_bytes_unchanged': unchanged,
                   'new_expert_physical_simulation_official_DR': True, 'policy_inference': False,
                   'simulator_started_by_this_tool': False, 'hardware_io': False,
                   'video_generation_does_not_establish_quality_or_physical_success': True}
        (output / 'video_validation.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps({'passed': True, 'video': str(dest), 'receipt': str(output / 'video_validation.json')}, ensure_ascii=False))
    except BaseException as exc:
        failure = {'passed': False, 'error': str(exc), 'candidate': str(candidate),
                   'protected_inputs_sha256': protected,
                   'input_bytes_unchanged': {path: sha(path) == digest for path, digest in protected.items() if Path(path).is_file()},
                   'candidate_acceptance_was_not_changed': True}
        (output / 'video_validation_failed.json').write_text(json.dumps(failure, ensure_ascii=False, indent=2) + '\n')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True, help='Fresh candidate that already passed complete quality gate')
    parser.add_argument('--output', type=Path, required=True, help='New separate output directory; never inside candidate')
    parser.add_argument('--ffmpeg', help='Existing ffmpeg binary; defaults to PATH or this Python runtime sibling')
    args = parser.parse_args()
    render(args.candidate, args.output, ffmpeg_binary(args.ffmpeg))


if __name__ == '__main__':
    main()
