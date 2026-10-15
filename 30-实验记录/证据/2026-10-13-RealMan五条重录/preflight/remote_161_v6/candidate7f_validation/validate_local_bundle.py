"""Local CPU-only validation of the independent candidate7f package.

Writes only beside this validator. No SSH, scene, model, GPU or systemd calls.
The derived unit tests relocate two test input paths only; runtime payload files
are read-only and retain the builder's published hashes.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
E = BASE.parent.parent
BUNDLE = BASE / "server_bundle_20261014_v6_hybrid_candidate7f"
OLD = BASE / "server_bundle_20261014_v6_hybrid_candidate7e"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(value, message):
    if not value:
        raise RuntimeError(message)


def verify_payload(bundle):
    manifest = json.loads((bundle / "bundle_manifest.json").read_text())
    require(sha(bundle / "bundle_manifest.json") == sha(bundle / "payload/bundle_manifest.json"),
            "Top-level and payload manifest differs")
    for relative, row in manifest["package_files"].items():
        path = bundle / "payload" / relative
        require(not path.is_symlink() and path.is_file(), "Payload file missing or linked: " + relative)
        require(path.stat().st_size == row["bytes"] and sha(path) == row["sha256"],
                "Payload SHA or length differs: " + relative)
    return manifest


def test_count(path):
    text = path.read_text()
    count = re.search(r"Ran (\d+) tests? in", text)
    require(count is not None and text.rstrip().endswith("OK"), "Incomplete or failed test log: " + str(path))
    return int(count.group(1))


def run_cpu_test(path, name):
    log = HERE / (name + ".log")
    require(not log.exists(), "Refuse test log overwrite")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="2",
               OMP_NUM_THREADS="2", TMPDIR=str(HERE / "tmp"), CUDA_VISIBLE_DEVICES="")
    with log.open("w") as stream:
        subprocess.run([sys.executable, "-B", str(path)], stdout=stream,
                       stderr=subprocess.STDOUT, env=env, timeout=90, check=True)
    return {"test_path": str(path), "test_source_sha256": sha(path), "log": str(log),
            "log_sha256": sha(log), "tests_run": test_count(log), "passed": True,
            "synthetic_CPU_fixtures_only": True}


def derived_grasp_test(label, runtime):
    source = E / "planning/test_grasp_acquisition_guard.py"
    original = ast.parse(source.read_text())
    tree = copy.deepcopy(original)
    bindings = {"DIRECTORY": runtime / "planning",
                "LEGACY": runtime / "source/adjust_bottle_20261002"}
    backups = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            key = node.targets[0].id
            if key in bindings:
                require(key not in backups, "Repeated test binding")
                backups[key] = copy.deepcopy(node.value)
                node.value = ast.parse("Path(" + repr(str(bindings[key])) + ")", mode="eval").body
    require(set(backups) == set(bindings), "Expected exactly two test input bindings")
    restored = copy.deepcopy(tree)
    for node in restored.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            key = node.targets[0].id
            if key in backups:
                node.value = backups[key]
    require(ast.dump(restored, include_attributes=False) == ast.dump(original, include_attributes=False),
            "Test derivation changes more than DIRECTORY/LEGACY path bindings")
    ast.fix_missing_locations(tree)
    dest = HERE / (label + "_grasp_tests.py")
    require(not dest.exists(), "Refuse derived test overwrite")
    dest.write_text(ast.unparse(tree) + "\n")
    result = run_cpu_test(dest, label + "_grasp_tests")
    result.update(original_test_path=str(source), original_test_sha256=sha(source),
                  input_path_bindings={key: str(value) for key, value in bindings.items()},
                  all_other_test_AST_semantics_unchanged=True,
                  runtime_overlay_sha256=sha(runtime / "planning/grasp_acquisition_guard.py"))
    return result


def main():
    require(HERE.is_relative_to(Path("/home/admin123/liminghe/VLA-WAM-RL")), "Validation write scope differs")
    result_path = HERE / "validation_receipt.json"
    require(not result_path.exists(), "Refuse final receipt overwrite")
    before = json.loads((HERE / "before.json").read_text())
    manifest = verify_payload(BUNDLE)
    old_manifest = verify_payload(OLD)
    deployment = BUNDLE / "payload/deploy/realman_gen4" / manifest["complete_config"]["variant"]
    runtimes = [("evidence", BUNDLE / "payload/evidence")]
    runtimes += [("slot" + str(slot), deployment / ("jobs/slot" + str(slot) + "_runtime"))
                 for slot in (17, 18, 20, 29, 34)]
    overlays = {}
    for label, runtime in runtimes:
        rows = {path.name: sha(path) for path in (runtime / "planning").glob("*.py")}
        require(len(rows) == 13, "Expected 13 planning helpers at " + label)
        overlays[label] = rows
    require(all(rows == overlays["evidence"] for rows in overlays.values()),
            "Six planning overlay sets differ")
    require(overlays["evidence"]["grasp_acquisition_guard.py"] == sha(E / "planning/grasp_acquisition_guard.py"),
            "Narrow repair not copied byte-exactly")
    tests = [derived_grasp_test(label, runtime) for label, runtime in runtimes]
    tests.append(run_cpu_test(BUNDLE / "payload/evidence/preflight/test_guarded_stage.py", "package_system_guard_tests"))
    archive = BUNDLE / "realman_161_bundle.tar.gz"
    receipt = json.loads((BUNDLE / "build_receipt.json").read_text())
    require(sha(archive) == receipt["archive_sha256"] and archive.stat().st_size == receipt["archive_bytes"],
            "Builder archive SHA or size differs")
    expected = dict(manifest["package_files"])
    expected["bundle_manifest.json"] = {"sha256": sha(BUNDLE / "bundle_manifest.json"),
                                        "bytes": (BUNDLE / "bundle_manifest.json").stat().st_size}
    seen = set()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            require(not member.issym() and not member.islnk(), "Archive contains a link")
            if member.isdir():
                continue
            require(member.isfile(), "Unexpected archive member type")
            relative = member.name.removeprefix("./")
            require(relative in expected and relative not in seen and ".." not in Path(relative).parts,
                    "Unexpected or duplicate archive file")
            h = hashlib.sha256()
            with tar.extractfile(member) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(block)
            require(h.hexdigest() == expected[relative]["sha256"] and member.size == expected[relative]["bytes"],
                    "Archive file SHA or size differs")
            seen.add(relative)
    require(seen == set(expected), "Archive file membership differs")
    verify_payload(BUNDLE)
    verify_payload(OLD)
    for value, row in before.items():
        require(sha(value) == row["sha256"] and Path(value).stat().st_size == row["bytes"],
                "Original sealed evidence or active source changed: " + value)
    source_logs = {name: {"tests_run": test_count(HERE / name), "sha256": sha(HERE / name), "passed": True}
                   for name in ("source_grasp_tests.log", "source_builder_tests.log", "source_system_guard_tests.log")}
    report = {"passed": True, "static_offline_local_only": True, "hardware_io": False,
              "remote_upload_or_connection_performed": False, "scene_or_episode_executed": False,
              "bundle": str(BUNDLE), "build_receipt_sha256": sha(BUNDLE / "build_receipt.json"),
              "archive_sha256": sha(archive), "archive_bytes": archive.stat().st_size,
              "manifest_sha256": sha(BUNDLE / "bundle_manifest.json"),
              "all_manifest_files_verified_count": len(manifest["package_files"]),
              "all_archive_file_SHA_verified_count_including_manifest": len(seen),
              "original601_source_count": manifest["protected_source_count"],
              "protected_hardware_count": manifest["protected_hardware_count"],
              "original7e_manifest_files_verified_count": len(old_manifest["package_files"]),
              "candidate7e_archive_manifest_build_receipt_unchanged": True,
              "active_repaired_source_unchanged_during_build_tests": True,
              "six_overlay_sets_byte_exact_equal": True, "planning_overlays_each_runtime": 13,
              "grasp_repair_sha256": overlays["evidence"]["grasp_acquisition_guard.py"],
              "overlay_sets": overlays, "source_tests": source_logs, "package_tests": tests,
              "tests_run_total": sum(row["tests_run"] for row in source_logs.values()) + sum(row["tests_run"] for row in tests),
              "limitations": ["CPU fixtures and package hashes do not establish physical grasp success.",
                              "No 161 environment, remote resource controller or renderer was executed.",
                              "External large scene assets are not bundled or freshly content-hashed here."]}
    result_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("passed", "archive_sha256", "archive_bytes", "manifest_sha256",
                     "all_manifest_files_verified_count", "tests_run_total", "grasp_repair_sha256")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
