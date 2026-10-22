"""CPU-only error-path review of unchanged AST-extracted production methods.

Controlled fixtures below are not physical episodes or Agent generation proof.
"""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import tempfile
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expect_rejection(function):
    try:
        function()
    except (AssertionError, RuntimeError, ValueError):
        return True
    raise AssertionError("Negative fixture was accepted")


def main():
    collector = ROOT / "collect_official.py"
    tree = ast.parse(collector.read_text())
    cls = next(x for x in ast.walk(tree) if isinstance(x, ast.ClassDef) and x.name == "CollectionTask")
    methods = {x.name: x for x in cls.body if isinstance(x, ast.FunctionDef)}
    checks = {}
    saved = []
    exported = []
    namespace = {
        "dump": lambda p, v: saved.append((str(p), v)), "output": Path("/tmp/not-written-review-fixture"),
        "expert": {"kind": "synthetic_CPU_fixture"}, "Path": Path,
        "args": {"language_num": 12}, "validations": [],
        "export_episode": lambda *args: exported.append(args) or {"passed": True, "fixture_only": True},
    }
    for name in ("persist_run", "save_traj_data", "merge_pkl_to_hdf5_video"):
        exec(compile(ast.Module(body=[methods[name]], type_ignores=[]), str(collector), "exec"), namespace)
    fake = types.SimpleNamespace(
        check_success=lambda: True, seed=0, collection_need_plan=True, metrics={"success": True},
        transition_log=[], physics_trace=[], collection_run=Path("/tmp/not-written-review-fixture/plan"),
        collection_cache=Path("/tmp/not-written-review-fixture/cache"), ep_num=0, arm="right")
    for name in ("persist_run", "save_traj_data", "merge_pkl_to_hdf5_video"):
        setattr(fake, name, types.MethodType(namespace[name], fake))
    fake.persist_run({"type": "SyntheticProbeError", "message": "CPU fixture, not a rollout"})
    assert fake.collection_receipt["success"] is False
    checks["error_receipt_success_false"] = True
    checks["save_rejects_error_receipt"] = expect_rejection(lambda: fake.save_traj_data(0))
    checks["merge_rejects_error_receipt"] = expect_rejection(lambda: fake.merge_pkl_to_hdf5_video(["test"] * 12))
    fake.persist_run(None)
    fake.info = {"info": {"{A}": str(ROOT / "metadata/description/objects_description/reagent_bottle_body/base0"), "{a}": "right"}}
    checks["merge_rejects_fallback_language"] = expect_rejection(lambda: fake.merge_pkl_to_hdf5_video(["labware pick place"]))
    fake.info["info"]["{a}"] = "left"
    checks["merge_rejects_wrong_arm_language"] = expect_rejection(lambda: fake.merge_pkl_to_hdf5_video(["test"] * 12))
    fake.info["info"]["{a}"] = "right"
    fake.info["info"]["{A}"] = "/tmp/not-existing-review-description/base0"
    checks["merge_rejects_missing_object_description"] = expect_rejection(lambda: fake.merge_pkl_to_hdf5_video(["test"] * 12))
    assert not exported

    # Execute the exact candidate-provenance branch with an intercepted loader.
    body = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "main")
    branch = next(x for x in body.body if isinstance(x, ast.If) and ast.unparse(x.test) == "cli.expert_code")
    with tempfile.TemporaryDirectory(prefix="labware-review-negative-") as temporary:
        folder = Path(temporary)
        candidate = folder / "synthetic_candidate.py"
        candidate.write_text("# Synthetic CPU review fixture. Never loaded or physically executed.\n")
        class FakeBase:
            pass
        class FakeCandidate(FakeBase):
            pass
        ns = {"cli": types.SimpleNamespace(expert_code=candidate, agent_receipt=None),
              "load_file": lambda *args: types.SimpleNamespace(gpt_labware_pick_place=FakeCandidate),
              "labware_pick_place": FakeBase, "hashlib": hashlib, "json": json, "ROOT": ROOT}
        code = compile(ast.Module(body=[branch], type_ignores=[]), str(collector), "exec")
        exec(code, ns)
        assert ns["expert"]["agent_generated"] is False and ns["expert"]["kind"] == "provided_candidate_provenance_unverified"
        checks["provided_candidate_not_mislabeled_agent_generated"] = True
        receipt = folder / "synthetic_receipt.json"
        evidence = {"candidate_sha256": "wrong", "task_name": "labware_pick_place",
                    "original_test_gen_code_run_called": True, "status": "physical_test_complete",
                    "official_threshold_0_5_met": True, "source_unchanged": True, "candidate_unchanged": True,
                    "generation_transport": "SYNTHETIC_CPU_FIXTURE", "success_rate": 1.0}
        receipt.write_text(json.dumps(evidence))
        ns["cli"].agent_receipt = receipt
        checks["candidate_sha_mismatch_rejected"] = expect_rejection(lambda: exec(code, ns))
        evidence["candidate_sha256"] = sha(candidate)
        evidence["official_threshold_0_5_met"] = False
        receipt.write_text(json.dumps(evidence))
        checks["candidate_failed_threshold_rejected"] = expect_rejection(lambda: exec(code, ns))

        # Real export function, with invalid receipt: rejection precedes HDF5 creation.
        import sys
        sys.path.insert(0, "/home/admin123/liminghe/vla-benchmark/RoboTwin")
        spec = importlib.util.spec_from_file_location("review_original_sdk_export", ROOT / "sdk24_export.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cache = folder / "synthetic_cache"
        cache.mkdir()
        for i in range(2):
            np.savez(cache / f"{i:07d}.npz", state=np.zeros(24, dtype=np.float32), action=np.ones(24, dtype=np.float32),
                     next_state=np.zeros(24, dtype=np.float32), tick_start=i * 15, tick_end=(i + 1) * 15,
                     row_json=json.dumps({"phase": "synthetic", "metrics": {"success": True}}),
                     **{name: np.array([1], dtype=np.uint8) for name in module.CAMERAS})
        output = folder / "must_not_exist.hdf5"
        checks["export_rejects_interrupted_success_flag"] = expect_rejection(lambda: module.export_episode(
            cache, output, ["synthetic"] * 12, {"success": True, "error": {"type": "SyntheticError"}}))
        assert not output.exists()
        checks["error_export_creates_no_false_hdf5"] = True

    task_tree = ast.parse((ROOT / "labware_task.py").read_text())
    task_cls = next(x for x in task_tree.body if isinstance(x, ast.ClassDef) and x.name == "labware_pick_place")
    close_method = next(x for x in task_cls.body if isinstance(x, ast.FunctionDef) and x.name == "close")
    close_ns = {}
    exec(compile(ast.Module(body=[close_method], type_ignores=[]), str(ROOT / "labware_task.py"), "exec"), close_ns)
    close_fixture = types.SimpleNamespace(scene=types.SimpleNamespace(step=lambda: None), _raw_step=lambda: None,
                                          _active=True, metrics={"success": False}, collection_receipt={"success": False},
                                          collection_cache=Path("/tmp/not-written-review-fixture/cache"))
    close_ns["close"](close_fixture)
    close_ns["close"](close_fixture)
    assert close_fixture.metrics == {"success": False} and close_fixture.collection_receipt == {"success": False}
    assert close_fixture.collection_cache == Path("/tmp/not-written-review-fixture/cache")
    checks["close_twice_is_safe_and_preserves_merge_receipts"] = True

    result = {"date": "2026-10-22", "passed": all(checks.values()),
              "scope": "CPU-only controlled negative fixtures; not physical task, camera, or Agent-generation validation",
              "source_sha256": {p.name: sha(p) for p in (collector, ROOT / "sdk24_export.py", ROOT / "labware_task.py")},
              "checks": checks, "root_files_modified": False}
    (ROOT / "metadata/review_regression_checks.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
