"""Local-only package binding and system Python 3.8 syntax checks."""

from pathlib import Path
from unittest import TestCase
import ast
import json
import unittest

import build_remote_bundle as builder


HERE = Path(__file__).resolve().parent


class HybridBundleBindingTest(TestCase):
    def test_v2_variant_namespaces_system_units(self):
        cfg = builder.validate_config(json.loads((HERE / 'server_config_hybrid_v2.json').read_text()))
        self.assertEqual(builder.unit_prefix(cfg),
                         'realman-rerecord-20261014-v6-hybrid-v2')
        paths = builder.binding_paths(cfg)
        template = (HERE / 'run_guarded_stage_system_v6.py').read_text()
        rendered, sites = builder.adapt_system_template(template, cfg, paths)
        self.assertIn("UNIT_PREFIX = 'realman-rerecord-20261014-v6-hybrid-v2'", rendered)
        self.assertIn(str(paths['root']), rendered)
        self.assertTrue(sites)
        self.assertNotIn('@UNIT_PREFIX@', rendered)

    def test_release_contract_uses_same_v2_unit_prefix(self):
        cfg = builder.validate_config(json.loads((HERE / 'server_config_hybrid_v2.json').read_text()))
        source = (HERE / 'hybrid_release_contract.py').read_text()
        rendered = builder.replace_once(source,
            "UNIT_PREFIX = 'realman-rerecord-20261014-v6-hybrid'",
            'UNIT_PREFIX = ' + repr(builder.unit_prefix(cfg)))
        self.assertIn("UNIT_PREFIX = 'realman-rerecord-20261014-v6-hybrid-v2'", rendered)
        compile(rendered, 'hybrid_release_contract.py', 'exec')

    def test_system_python38_templates_parse(self):
        for name in ('run_guarded_stage_system_v6.py',
                     'guarded_stage_cpu_probe_system_v6.py',
                     'test_guarded_stage_system_v6.py',
                     'hybrid_cgroup_adapter.py'):
            with self.subTest(name=name):
                ast.parse((HERE / name).read_text(), filename=name,
                          feature_version=(3, 8))


if __name__ == '__main__':
    unittest.main()
