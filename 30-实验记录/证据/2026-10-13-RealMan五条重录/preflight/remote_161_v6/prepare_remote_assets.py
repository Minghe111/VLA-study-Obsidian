#!/usr/bin/env python3
"""Bind only reviewed official scene assets into a fresh RealMan runtime.

The independent runtime contains its own code and robot hardware files. Its
official object tree and seen-background tree stay at the configured existing
path. Only seen backgrounds are exposed in the runtime; the server's extra
unseen images are not offered to this recording's random sampler. This tool
checks relative-name/size inventories before making two links inside the new
runtime. No source or existing repository is written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat

NAMES = ('background_texture', 'objects')
SCHEMA = 'realman-161-portability-bundle-v1'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def real_owned_dir(path):
    path = Path(path)
    require(path.is_absolute() and path.is_dir() and not path.is_symlink()
            and path == path.resolve(strict=True)
            and path.stat().st_uid == os.getuid(),
            'Expected an absolute real directory owned by the current user: ' + str(path))
    return path


def inventory_tree(base, *, prefix=None):
    base = Path(base)
    require(base.is_dir() and not base.is_symlink(), 'Scene asset tree missing or linked: ' + str(base))
    rows = []
    for path in base.rglob('*'):
        require(not path.is_symlink(), 'Scene asset tree contains a link: ' + str(path))
        if path.is_file():
            mode = path.stat().st_mode
            require(stat.S_ISREG(mode), 'Nonregular scene asset: ' + str(path))
            relative = path.relative_to(base).as_posix()
            rows.append(((prefix + '/' + relative) if prefix else relative, path.stat().st_size))
    digest = hashlib.sha256()
    for relative, size in sorted(rows):
        digest.update(relative.encode('utf-8') + b'\0' + str(size).encode('ascii') + b'\n')
    return {'file_count': len(rows), 'bytes': sum(size for _, size in rows),
            'relative_path_and_size_sha256': digest.hexdigest(),
            'content_sha256_verified': False,
            'selection_scope': 'seen_only' if prefix == 'seen' else 'full_directory'}


def prepare(manifest_path, *, install=False):
    manifest_path = Path(manifest_path)
    require(manifest_path.is_absolute() and manifest_path.is_file()
            and not manifest_path.is_symlink(), 'Use the absolute extracted bundle manifest')
    manifest = json.loads(manifest_path.read_text())
    require(manifest.get('schema') == SCHEMA, 'Unknown bundle manifest')
    cfg = manifest['complete_config']
    root = real_owned_dir(cfg['run_root'])
    require(manifest_path == root / 'bundle_manifest.json',
            'Manifest is not at this independent run root')
    require(root.is_relative_to(Path('/bigdata/liminghe'))
            or root.is_relative_to(Path('/bigdata2/liminghe')),
            'Independent run root is outside the task-owned area')
    repo = real_owned_dir(root / 'runtime_repo/RoboTwin')
    assets = real_owned_dir(repo / 'assets')
    source = real_owned_dir(cfg['existing_asset_root'])
    require(source.is_relative_to(Path('/bigdata/liminghe'))
            or source.is_relative_to(Path('/bigdata2/liminghe')),
            'Existing official asset root is outside the task-owned area')
    require(not source.is_relative_to(root) and not root.is_relative_to(source),
            'Official asset root and independent run root must be disjoint')
    expected = manifest['external_asset_inventory']
    require(set(expected) == set(NAMES), 'Only the two reviewed scene-asset trees may be bound')
    tool_name = 'bundle_tools/prepare_remote_assets.py'
    tool_pin = manifest['package_files'][tool_name]['sha256']
    require(sha(__file__) == tool_pin, 'Asset-binding helper changed since bundle construction')
    seen_source = source / 'background_texture/seen'
    objects_source = source / 'objects'
    observed = {'background_texture': inventory_tree(seen_source, prefix='seen'),
                'objects': inventory_tree(objects_source)}
    for name in NAMES:
        require(observed[name] == expected[name],
                'Official selected scene-asset inventory differs from local reviewed source: ' + name)
    unseen_source = source / 'background_texture/unseen'
    excluded_unseen = inventory_tree(unseen_source) if unseen_source.is_dir() else None
    objects_dest = assets / 'objects'
    require(not objects_dest.exists() or
            (objects_dest.is_symlink() and objects_dest.resolve(strict=True) == objects_source),
            'Refusing to overwrite a non-matching runtime objects path')
    background_dest = assets / 'background_texture'
    if background_dest.exists():
        real_owned_dir(background_dest)
        require({p.name for p in background_dest.iterdir()} <= {'seen'},
                'Runtime background directory already exposes an unreviewed entry')
        seen_dest = background_dest / 'seen'
        if seen_dest.exists() or seen_dest.is_symlink():
            require(seen_dest.is_symlink() and seen_dest.resolve(strict=True) == seen_source,
                    'Refusing to replace a non-matching runtime seen-background path')
    created = 0
    directories_created = 0
    if install:
        if not objects_dest.is_symlink():
            os.symlink(str(objects_source), str(objects_dest), target_is_directory=True)
            created += 1
        if not background_dest.exists():
            background_dest.mkdir()
            directories_created += 1
        seen_dest = background_dest / 'seen'
        if not seen_dest.is_symlink():
            os.symlink(str(seen_source), str(seen_dest), target_is_directory=True)
            created += 1
    seen_dest = background_dest / 'seen'
    require(objects_dest.is_symlink() and objects_dest.resolve(strict=True) == objects_source,
            'Required runtime object binding absent or different')
    require(background_dest.is_dir() and not background_dest.is_symlink()
            and {p.name for p in background_dest.iterdir()} == {'seen'}
            and seen_dest.is_symlink() and seen_dest.resolve(strict=True) == seen_source
            and not (background_dest / 'unseen').exists(),
            'Runtime must expose exactly the reviewed seen-background subtree')
    result = {'schema': 'realman-161-external-asset-binding-v1', 'passed': True,
              'run_root': str(root), 'remote_repo': str(repo), 'official_asset_root': str(source),
              'bundle_manifest_sha256': sha(manifest_path), 'asset_helper_sha256': tool_pin,
              'structural_inventory': observed,
              'content_sha256_verified': False,
              'excluded_unseen_inventory': excluded_unseen,
              'unseen_exposed_to_runtime': False,
              'links': {'background_texture/seen': {'runtime': str(seen_dest), 'source': str(seen_source)},
                        'objects': {'runtime': str(objects_dest), 'source': str(objects_source)}},
              'source_or_existing_repository_writes': 0,
              'link_writes_inside_independent_run_root': created,
              'directory_writes_inside_independent_run_root': directories_created}
    if install:
        receipt = root / 'asset_binding_receipt.json'
        require(not receipt.exists() and not receipt.is_symlink(), 'Refuse asset-binding receipt overwrite')
        receipt.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--install', action='store_true', help='Create exactly two links inside the new run root')
    args = parser.parse_args()
    print(json.dumps(prepare(args.manifest, install=args.install), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
