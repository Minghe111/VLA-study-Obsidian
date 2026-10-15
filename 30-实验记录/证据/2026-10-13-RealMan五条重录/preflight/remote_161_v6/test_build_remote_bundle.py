"""CPU checks of relocation and source binding; constructs no task or scene."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('bundle_builder_under_test', HERE / 'build_remote_bundle.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
asset_spec = importlib.util.spec_from_file_location('asset_preparer_under_test', HERE / 'prepare_remote_assets.py')
assets = importlib.util.module_from_spec(asset_spec)
asset_spec.loader.exec_module(assets)


def config():
    return {'schema': 'realman-161-bundle-config-v1',
        'run_root': '/bigdata/liminghe/VLA-benchmark/deploy/realman_gen4/portability_cpu_fixture',
        'variant': 'rgb50_motion_quality_20261014_v6_161',
        'python': '/bigdata/liminghe/miniconda3/envs/robotwin/bin/python',
        'gpu_index': 3, 'collect_cpu_cores': 8,
        'existing_asset_root': '/bigdata/liminghe/VLA-benchmark/RoboTwin/assets'}


class StaticPortability(unittest.TestCase):
    def test_config_rejects_placeholder_and_wrong_scope(self):
        for key, value in [('run_root', '/bigdata/liminghe/<pending>'),
                           ('run_root', '/home/another_user/run'),
                           ('python', '/bigdata/liminghe/TODO/bin/python'),
                           ('existing_asset_root', '/bigdata/other/assets')]:
            cfg = config(); cfg[key] = value
            with self.assertRaises(ValueError): m.validate_config(cfg)

    def test_config_rejects_cpu_gpu_type_and_old_variant(self):
        for key, value in [('gpu_index', True), ('collect_cpu_cores', 0),
                           ('variant', m.LOCAL_VARIANT)]:
            cfg = config(); cfg[key] = value
            with self.assertRaises(ValueError): m.validate_config(cfg)

    def test_ast_token_rewrite_preserves_other_bytes_and_unicode(self):
        source = '# 原有注释\na = "前缀"; p = "/old/root/child"\nq = "/old/root2"\n'
        result, rows = m.rewrite_literals(source, {'/old/root': '/new/target'})
        self.assertEqual(result, '# 原有注释\na = "前缀"; p = \'/new/target/child\'\nq = "/old/root2"\n')
        self.assertEqual(len(rows), 1)

    def test_ast_rewrite_changes_only_constant_values(self):
        before = ast.parse('p = "/a/b"\nf(p, 0.004, 15, 24)\n')
        result, _ = m.rewrite_literals(ast.unparse(before), {'/a/b': '/c/d'})
        after = ast.parse(result)
        after.body[0].value.value = '/a/b'
        self.assertEqual(ast.dump(before), ast.dump(after))

    def test_current_producers_adapt_and_compile(self):
        cfg = config(); paths = m.binding_paths(cfg)
        for name in m.CORE:
            original = m.E / 'preflight' / name
            before = m.sha(original)
            text, _ = m.adapt_core(name, original.read_text(), cfg, paths)
            compile(text, name, 'exec')
            self.assertEqual(m.sha(original), before)
            self.assertNotIn('/home/admin123/liminghe/miniconda3/envs/robotwin/bin/python', text)
            if name == 'run_guarded_stage.py':
                self.assertIn("'800%' if stage == 'collect' else '200%'", text)
                self.assertIn("'CUDA_VISIBLE_DEVICES': '3'", text)
            if name == 'staged_quality.py':
                self.assertIn("limits.get('CPUQuota') == '800%'", text)
            if name == 'publish_staged_patch.py':
                self.assertIn("'800000 100000' if stage == 'collect'", text)

    def test_cpu_quota_kernel_readback_corresponds(self):
        for cpu in [2, 8, 16]:
            cfg = config(); cfg['collect_cpu_cores'] = cpu
            text, _ = m.adapt_core('run_guarded_stage.py',
                (m.E / 'preflight/run_guarded_stage.py').read_text(), cfg, m.binding_paths(cfg))
            tree = ast.parse(text)
            function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'limits_for')
            env = {'STAGES': {'collect': (10, 12, 2100), 'render': (2, 3, 1200)},
                'GIB': 1024**3, 'math': __import__('math')}
            exec(compile(ast.Module(body=[function], type_ignores=[]), 'bounded_limits', 'exec'), env)
            self.assertEqual(env['limits_for']('collect')['CPUQuota'], str(cpu * 100) + '%')
            self.assertEqual(env['limits_for']('render')['CPUQuota'], '200%')
            self.assertEqual(env['limits_for']('collect')['MemorySwapMax'], '0')

    def test_real_urdf_relocation_only_mesh_paths(self):
        original = m.LOCAL_ARCHIVE / 'shared_frozen_runtime/robot_head_down_fixed.urdf'
        before = m.sha(original)
        relocated, rows = m.relocate_urdf(original, m.binding_paths(config())['repo'])
        self.assertEqual(len(rows), 186)
        self.assertEqual(m.sha(original), before)
        self.assertNotIn(str(m.LOCAL_REPO).encode(), relocated)
        original_tree = ET.parse(original).getroot(); new_tree = ET.fromstring(relocated)
        for old, new in zip(original_tree.iter('mesh'), new_tree.iter('mesh')):
            old.attrib['filename'] = new.attrib['filename']
        self.assertEqual(m.semantic_xml(original_tree), m.semantic_xml(new_tree))

    def test_current_601_and_116_source_sha_readback(self):
        receipt = json.loads((m.E / 'source/download_receipt.json').read_text())
        self.assertEqual(len(receipt['files']), 601)
        for rel, row in receipt['files'].items():
            self.assertEqual(m.sha(m.E / 'source' / rel), row['sha256'])
        guard = json.loads((m.LOCAL_ARCHIVE / 'hardware_guard.json').read_text())
        self.assertEqual(len(guard['expected_sha256']), 116)
        aliases = {m.FROZEN_TMP: m.LOCAL_ARCHIVE / 'shared_frozen_runtime/robot_head_down_fixed.urdf',
                   m.BASE_TMP: m.LOCAL_ARCHIVE / 'shared_frozen_runtime/remote_base_task_snapshot.py'}
        for path, digest in guard['expected_sha256'].items():
            self.assertEqual(m.sha(aliases.get(path, Path(path))), digest)

    def test_export_and_publisher_sha_pins_update(self):
        text = (m.E / 'preflight/render_staged_video.py').read_text()
        digest = hashlib.sha256(b'relocated exporter').hexdigest()
        changed = m.patch_assignment(text, 'BASE_SHA', digest)
        self.assertIn(digest, changed)
        text = (m.E / 'preflight/staged_quality.py').read_text()
        changed = m.patch_assignment(text, 'PUBLISH_HELPER_SHA256', digest)
        self.assertIn(digest, changed)

    def test_server_native_speed_overlay_has_independent_frozen_proof(self):
        before = m.sha(m.E / 'planning/conservative_ompl_overlay.py')
        proof = m.verify_speed_variant()
        self.assertEqual(proof['original_sha256'], before)
        self.assertNotEqual(proof['server_sha256'], before)
        self.assertEqual(proof['effective_native_policy'],
                         {'time': 30.0, 'range': .01, 'simplify': True})
        self.assertTrue(proof['component_only'])
        self.assertFalse(proof['physical_speedup_verified'])
        self.assertEqual(m.sha(m.E / 'planning/conservative_ompl_overlay.py'), before)

    def test_scene_asset_inventory_rejects_links_and_records_names_sizes(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            (p / 'one').write_bytes(b'abc')
            result = assets.inventory_tree(p)
            expected = hashlib.sha256(b'one\0' + b'3\n').hexdigest()
            self.assertEqual(result['relative_path_and_size_sha256'], expected)
            self.assertEqual(result['file_count'], 1)
            self.assertFalse(result['content_sha256_verified'])
            (p / 'linked').symlink_to(p / 'one')
            with self.assertRaises(ValueError):
                assets.inventory_tree(p)

    def test_seen_only_background_selection_matches_remote_bind_view(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            seen = root / 'background_texture/seen'
            seen.mkdir(parents=True)
            (seen / 'wall.jpg').write_bytes(b'seen')
            objects = root / 'objects'
            objects.mkdir()
            (objects / 'bottle.obj').write_bytes(b'object')
            expected = m.external_asset_inventory(root)
            self.assertEqual(expected['background_texture'],
                             assets.inventory_tree(seen, prefix='seen'))
            self.assertEqual(expected['objects'], assets.inventory_tree(objects))
            unseen = root / 'background_texture/unseen'
            unseen.mkdir()
            (unseen / 'extra.jpg').write_bytes(b'not for this run')
            with self.assertRaises(ValueError):
                m.external_asset_inventory(root)

    def test_copy_refuses_conflict_and_preserves_input(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp); (p / 'a').write_bytes(b'original'); (p / 'b').write_bytes(b'conflict')
            with self.assertRaises(ValueError): m.copy_exact(p / 'a', p / 'b')
            self.assertEqual((p / 'a').read_bytes(), b'original')


if __name__ == '__main__':
    unittest.main()
