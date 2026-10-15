"""Bounded 161 NVIDIA/SAPIEN loader diagnostic; never imports a RoboTwin task.

The default check constructs and destroys only SapienRenderer. The optional
--offscreen-example mode runs SAPIEN's official 128x128 offscreen example in
the same guarded system unit. Neither mode loads a RoboTwin task or episode.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


GPUCOMP = Path('/lib/x86_64-linux-gnu/libnvidia-gpucomp.so.570.133.07')
GPUCOMP_REAL = Path('/usr/lib/x86_64-linux-gnu/libnvidia-gpucomp.so.570.133.07')
GPUCOMP_SHA256 = '66178cf6eef724402b77193665ab1defede1b36a6333d81423b48ee32f0742dc'
NVIDIA_ICD = Path('/usr/share/vulkan/icd.d/nvidia_icd.json')
NVIDIA_ICD_SHA256 = '46cee75b42b43649839bc7f29820d36ac6e85782b792a97d5b91d9adbca8ca09'
NVIDIA_DRIVER_VERSION = '570.133.07'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def gpu_snapshot():
    helper_env = dict(os.environ)
    helper_env.pop('LD_PRELOAD', None)
    result = subprocess.run(['nvidia-smi',
        '--query-gpu=index,uuid,pci.bus_id,memory.used,driver_version',
        '--format=csv,noheader,nounits'], capture_output=True, text=True,
        env=helper_env, timeout=5)
    if result.returncode:
        raise RuntimeError('Cannot read eight GPU identities and memory: ' + result.stderr.strip())
    rows = []
    for line in result.stdout.splitlines():
        cells = [cell.strip() for cell in line.split(',')]
        if len(cells) != 5 or cells[4] != NVIDIA_DRIVER_VERSION:
            raise RuntimeError('Unparseable nvidia-smi GPU row')
        rows.append({'index': int(cells[0]), 'uuid': cells[1], 'pci_bus_id': cells[2],
                     'memory_used_mib': int(cells[3]), 'driver_version': cells[4]})
    if {row['index'] for row in rows} != set(range(8)):
        raise RuntimeError('Expected all eight server GPUs in renderer diagnostic')
    return sorted(rows, key=lambda row: row['index'])


def loader_proof():
    if (os.environ.get('LD_PRELOAD') != str(GPUCOMP)
        or os.environ.get('VK_ICD_FILENAMES') != str(NVIDIA_ICD)):
        raise RuntimeError('Renderer child did not receive the exact pinned loader environment')
    if (GPUCOMP.resolve(strict=True) != GPUCOMP_REAL
        or sha(GPUCOMP_REAL) != GPUCOMP_SHA256
        or sha(NVIDIA_ICD) != NVIDIA_ICD_SHA256):
        raise RuntimeError('Pinned NVIDIA loader input changed inside renderer child')
    kernel_line = Path('/proc/driver/nvidia/version').read_text().splitlines()[0]
    kernel_versions = re.findall(r'\b\d{3}\.\d{1,3}\.\d{1,3}\b', kernel_line)
    if kernel_versions != [NVIDIA_DRIVER_VERSION]:
        raise RuntimeError('Running NVIDIA kernel module differs in renderer child')
    mapped = any(line.split(None, 5)[-1].strip() == str(GPUCOMP_REAL)
                 for line in Path('/proc/self/maps').read_text().splitlines()
                 if len(line.split(None, 5)) == 6)
    if not mapped:
        raise RuntimeError('Pinned NVIDIA GPU compiler is absent from this process maps')
    return {'ld_preload': str(GPUCOMP), 'gpucomp_realpath': str(GPUCOMP_REAL),
            'gpucomp_sha256': GPUCOMP_SHA256, 'vk_icd_filenames': str(NVIDIA_ICD),
            'vk_icd_sha256': NVIDIA_ICD_SHA256,
            'kernel_module_version': NVIDIA_DRIVER_VERSION,
            'gpucomp_mapped_before_sapien_import': True}


def write_result(row):
    directory = Path(os.environ['REALMAN_RENDERER_SMOKE_DIR']).resolve(strict=True)
    expected = Path(__file__).resolve().parent / 'resource_runs'
    if directory.parent != expected or not directory.name.startswith('renderer_smoke_'):
        raise RuntimeError('Renderer smoke result directory is outside guarded stage')
    target = directory / 'renderer_smoke_result.json'
    if target.exists() or target.is_symlink():
        raise RuntimeError('Refusing renderer diagnostic overwrite')
    temp = target.with_name(target.name + '.tmp.' + str(os.getpid()))
    temp.write_text(json.dumps(row, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temp, target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--offscreen-example', action='store_true')
    args = parser.parse_args()
    loader = loader_proof()
    before = gpu_snapshot()
    import sapien.core as sapien
    device_summary = str(sapien.render.get_device_summary())
    renderer = sapien.SapienRenderer()
    during = gpu_snapshot()
    del renderer
    gc.collect()
    frame = None
    if args.offscreen_example:
        directory = Path(os.environ['REALMAN_RENDERER_SMOKE_DIR']).resolve(strict=True)
        example = directory / 'official_offscreen_example'
        example.mkdir(exist_ok=False)
        result = subprocess.run([sys.executable, '-m', 'sapien.example.offscreen'],
                                cwd=example, capture_output=True, text=True, timeout=60)
        if result.returncode:
            raise RuntimeError('Official SAPIEN offscreen module failed: ' + result.stderr[-1000:])
        png = example / 'sapien_offscreen.png'
        if not png.is_file() or png.is_symlink() or len(list(example.iterdir())) != 1:
            raise RuntimeError('Official SAPIEN offscreen output missing or mixed with other files')
        from PIL import Image
        with Image.open(png) as image:
            rgb = image.convert('RGB')
            colors = rgb.getcolors(maxcolors=128 * 128)
            if rgb.size != (128, 128) or colors is None or len(colors) < 3:
                raise RuntimeError('Official SAPIEN offscreen image is empty or malformed')
            frame = {'path': str(png), 'sha256': sha(png), 'width': 128,
                     'height': 128, 'distinct_colors': len(colors)}
    after = gpu_snapshot()
    result = {'schema': 'realman_renderer_only_smoke_v7', 'passed': True,
              'scope': 'official SAPIEN 128x128 offscreen example' if args.offscreen_example
                       else 'constructor/destructor only',
              'robot_or_task_scene_loaded': False, 'physical_episode_started': False,
              'loader_proof': loader, 'sapien_device_summary': device_summary,
              'gpu_before': before, 'gpu_during': during, 'gpu_after': after,
              'gpu_memory_deltas_are_not_exclusive_device_affinity_proof': True,
              'frame': frame}
    write_result(result)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
