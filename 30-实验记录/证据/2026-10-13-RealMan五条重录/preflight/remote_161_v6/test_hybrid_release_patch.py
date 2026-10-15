"""Ensure server-only V6 release and render metadata match the v1 guard."""

from pathlib import Path
from unittest import TestCase
import unittest

from hybrid_release_patch import patch_publisher, patch_quality, patch_render


ROOT = Path(__file__).resolve().parent.parent.parent


class ReleasePatchTest(TestCase):
    def test_quality_requires_independent_hybrid_contract(self):
        source = (ROOT / 'preflight/staged_quality.py').read_text()
        changed = patch_quality(source)
        self.assertEqual(changed.count("hybrid_release.validate_stage(receipt, path, 'collect', HERE)"), 1)
        self.assertEqual(changed.count('import hybrid_release_contract as hybrid_release'), 1)

    def test_publisher_replaces_unified_kernel_keys(self):
        source = (ROOT / 'preflight/publish_staged_patch.py').read_text()
        changed = patch_publisher(source)
        self.assertEqual(changed.count('hybrid_release.validate_stage(receipt, path, stage, HERE)'), 1)
        self.assertNotIn("actual.get('memory.high')", changed)
        self.assertNotIn("actual.get('cpu.max')", changed)

    def test_multiple_or_missing_insertion_sites_fail(self):
        source = (ROOT / 'preflight/staged_quality.py').read_text()
        with self.assertRaises(ValueError):
            patch_quality(source + '\nimport streaming_export_staged as exporter\n')

    def test_server_render_receipt_does_not_claim_hard_swap_disable(self):
        path = ROOT / 'preflight/render_staged_video.py'
        original = path.read_text()
        changed = patch_render(original)
        self.assertIn('run_guarded_stage.py --stage render --run', changed)
        self.assertNotIn('run_with_resource_guard.py --phase render', changed)
        self.assertNotIn("'swap_max_bytes': 0", changed)
        self.assertIn("'swap_hard_disabled': False", changed)
        self.assertIn("'zero_swap_evidence_required_from_external_guard': True", changed)
        self.assertEqual(path.read_text(), original)
        with self.assertRaises(ValueError):
            patch_render(changed)


if __name__ == '__main__':
    unittest.main()
