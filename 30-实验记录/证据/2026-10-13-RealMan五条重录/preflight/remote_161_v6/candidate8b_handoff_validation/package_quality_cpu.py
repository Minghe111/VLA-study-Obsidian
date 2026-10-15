"""Run unchanged source quality boundary fixtures against the frozen package.

Only test-process import and TASK data-root bindings are relocated. Production
files are never modified. Desktop v2 ResourceFixture is intentionally excluded;
the package guard suite tests the server v1 gate separately.
"""
from pathlib import Path
import importlib.util
import sys
import unittest

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
E = BASE.parent.parent
PACKAGE = BASE / 'server_bundle_20261015_v8b_handoff/payload/evidence'
sys.path.insert(0, str(PACKAGE / 'preflight'))
spec = importlib.util.spec_from_file_location('package_quality_fixtures', E / 'preflight/test_staged_quality.py')
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
assert Path(fixtures.quality.__file__).resolve() == PACKAGE / 'preflight/staged_quality.py'
fixtures.quality.TASK = PACKAGE
suite = unittest.TestSuite()
for cls in (fixtures.BoundedIO, fixtures.ComponentBoundary, fixtures.PublisherDispatch):
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(cls))
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
