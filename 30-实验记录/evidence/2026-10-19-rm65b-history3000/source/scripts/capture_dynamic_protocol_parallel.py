"""Bounded orchestration of the frozen v2 capture; parent alone writes manifest.

Stop the serial collector and confirm its simulator child has closed before
using --execute --resume --serial-stopped-confirmed. Default is plan-only.
Each independent episode is run by the exact frozen capture_episode function;
only its private runtime root changes. No policy or training is loaded.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

EXPECTED_V2_SHA = "4af8bdfe824a00e41b35408e27b2bb89f5f45977fa16928d0227329f168cf30b"
MAX_WORKERS = 3


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_v2(path):
    path = Path(path).resolve()
    if sha(path) != EXPECTED_V2_SHA:
        raise ValueError("Frozen v2 SHA differs; refusing to change the episode implementation")
    spec = importlib.util.spec_from_file_location("frozen_dynamic_capture_v2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def proc_members(pgid):
    members = []
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == pgid and fields[0] != "Z":
                members.append(int(path.parent.name))
        except (OSError, ValueError, IndexError):
            pass
    return members


def serial_collectors(output):
    """Extra protection beyond the explicit human stop/shutdown attestation."""
    found = []
    output = str(Path(output).resolve())
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            pid = int(path.parent.name)
            if pid == os.getpid():
                continue
            argv = [a.decode(errors="replace") for a in path.read_bytes().split(b"\0") if a]
            try:
                cwd = str((path.parent / "cwd").resolve(strict=True))
            except OSError:
                cwd = ""
            if cwd.startswith(output + "/runtime/") or cwd.startswith(output + "/parallel_runtime/"):
                found.append({"pid": pid, "role": "live_native_runtime_owner", "cwd": cwd})
                continue
            if not any(Path(a).name in ("capture_dynamic_protocol.py", "capture_dynamic_protocol_v2.py") for a in argv):
                continue
            for i, item in enumerate(argv[:-1]):
                if item == "--output" and str(Path(argv[i + 1]).resolve()) == output:
                    found.append({"pid": pid, "argv": argv})
        except (OSError, ValueError):
            pass
    return found


def verify_completed(entry, planned):
    if not (entry.get("completed") and entry.get("worker_shutdown_confirmed") and entry.get("status") == "completed"):
        return False
    for key in ("episode_id", "family_id", "split", "seed", "target_side", "layout"):
        if entry.get(key) != planned.get(key):
            raise ValueError(f"Completed receipt changed planned {key}")
    path = Path(entry["episode_json_path"])
    if not path.is_file() or sha(path) != entry["episode_json_sha256"]:
        raise ValueError("Completed episode JSON receipt is missing or altered")
    on_disk = json.loads(path.read_text())
    if not (on_disk.get("completed") and on_disk.get("worker_shutdown_confirmed") and on_disk.get("status") == "completed"):
        raise ValueError("Completed manifest receipt disagrees with episode JSON")
    for observation in entry["observations"]:
        p = Path(observation["npz_path"])
        if not p.is_file() or sha(p) != observation["sha256"]:
            raise ValueError("Completed actual observation was altered")
    for name, expected in entry["artifact_sha256"].items():
        p = path.parent / name
        if not p.is_file() or sha(p) != expected:
            raise ValueError("Completed command/history/monitor/forensic trace was altered")
    return True


def next_attempt(output, episode_id):
    base = Path(output) / "episodes" / episode_id
    folder, index = base, 0
    runtime_root = Path(output) / "parallel_runtime" / f"{episode_id}_attempt{index:03d}"
    while folder.exists() or runtime_root.exists():
        index += 1
        folder = base.with_name(base.name + f"_attempt{index:03d}")
        runtime_root = Path(output) / "parallel_runtime" / f"{episode_id}_attempt{index:03d}"
    # Parent reserves the *distinct worker runtime* instead of episode folder:
    # frozen v2 must create its episode folder itself with exist_ok=False.
    runtime_root.mkdir(parents=True, exist_ok=False)
    return folder, index, runtime_root


def fake_worker(payload):
    """CPU-only self-test backend; never selectable in a production payload."""
    time.sleep(payload.get("test_delay", 0.08))
    if payload.get("test_failure"):
        raise RuntimeError("intentional CPU worker failure")
    folder = Path(payload["folder"])
    folder.mkdir(parents=True, exist_ok=False)
    observation = folder / "obs.npz"
    observation.write_bytes(b"CPU fake receipt, never real capture data")
    trace = folder / "trace.jsonl"
    trace.write_text("{}\n")
    entry = {**payload["planned"], "completed": True, "status": "completed", "worker_shutdown_confirmed": True,
             "worker_pid": os.getpid(), "runtime_root": payload["runtime_root"],
             "observations": [{"npz_path": str(observation), "sha256": sha(observation)}],
             "artifact_sha256": {"trace.jsonl": sha(trace)}}
    atomic_json(folder / "episode.json", entry)
    return entry


def worker_main(payload_path):
    payload = json.loads(Path(payload_path).read_text())
    result_path = Path(payload["result_path"])
    started = time.monotonic()
    try:
        if payload.get("cpu_selftest"):
            entry = fake_worker(payload)
        else:
            v2 = load_v2(payload["v2_script"])
            import yaml
            from types import SimpleNamespace
            robotwin_root = Path(payload["robotwin_root"])
            sys.path.insert(0, str(robotwin_root))
            os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
            from integrations.realman_rm65b_gen4.rl_env import RealManVectorEnv
            from integrations.realman_rm65b_gen4.sample_verification import VerificationSpec
            base = yaml.safe_load(Path(payload["config"]).read_text())["environment"]["task_config"]
            if base.get("task_name") != "sample_verify_return" or base.get("profile") != "full50_sdk24":
                raise ValueError("Parallel worker requires the original dual-bottle SDK24 environment")
            reference, receipt = v2.load_reference(payload["source_actions"])
            if receipt["sha256"] != payload["source_sha256"]:
                raise ValueError("Source commands changed before worker launch")
            if sha(payload["config"]) != payload["config_sha256"]:
                raise ValueError("Configuration changed before worker launch")
            command_spec = v2.load_command_spec(base["asset_root"])
            if command_spec["urdf_sha256"] != payload["urdf_sha256"] or command_spec["manifest_sha256"] != payload["command_manifest_sha256"]:
                raise ValueError("Robot assets changed before worker launch")
            for name, expected in payload["runtime_source_sha256"].items():
                if sha(robotwin_root / "integrations/realman_rm65b_gen4" / name) != expected:
                    raise ValueError("Native simulator source changed before worker launch")
            args = SimpleNamespace(output=Path(payload["runtime_root"]), controls=payload["controls"], command_spec=command_spec)
            entry = v2.capture_episode(args, base, payload["planned"], reference,
                                       RealManVectorEnv, VerificationSpec, Path(payload["folder"]))
            if sha(payload["source_actions"]) != payload["source_sha256"] or sha(payload["config"]) != payload["config_sha256"]:
                raise ValueError("Source or configuration changed during episode")
            if v2.load_command_spec(base["asset_root"])["urdf_sha256"] != payload["urdf_sha256"]:
                raise ValueError("Original URDF changed during episode")
        episode_file = Path(payload["folder"]) / "episode.json"
        entry.update(episode_json_path=str(episode_file), episode_json_sha256=sha(episode_file))
        atomic_json(result_path, {"status": "completed", "episode": entry, "worker_pid": os.getpid(),
                                  "elapsed_s": time.monotonic() - started})
        return 0
    except BaseException as exc:
        atomic_json(result_path, {"status": "failed", "worker_pid": os.getpid(), "elapsed_s": time.monotonic() - started,
                                  "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
                                  "episode_json_path": str(Path(payload["folder"]) / "episode.json")})
        return 130 if isinstance(exc, KeyboardInterrupt) else 1


def run_parent(output, manifest, pending, workers, payload_base, max_episodes=None, cpu_selftest=False):
    """The only master-manifest writer. Workers own only individual attempts."""
    output = Path(output)
    path = output / "manifest.json"
    active, errors, launched = {}, [], 0
    interrupted = False
    stop_dispatch = False
    cancel_deadline = None
    cancel_phase = 0
    last_progress = 0.0
    parent_pid = os.getpid()
    orchestration = {"schema": "rm65b_parallel_capture_orchestration_v1", "parallel_orchestrator_sha256": sha(__file__),
                     "frozen_v2_sha256": EXPECTED_V2_SHA,
                     "parent_pid": parent_pid, "worker_limit": workers, "max_workers_observed": 0,
                     "implementation": "subprocess_per_episode_calling_exact_frozen_v2_capture_episode",
                     "private_runtime_per_episode": True, "episode_algorithm_changed": False,
                     "started_at": datetime.now(UTC).isoformat(), "jobs": [], "manifest_writers": [parent_pid],
                     "cpu_selftest": cpu_selftest}
    manifest.setdefault("parallel_orchestrations", []).append(orchestration)

    def write_master():
        manifest["last_manifest_writer_pid"] = parent_pid
        atomic_json(path, manifest)

    def request_interrupt(signum, frame):
        nonlocal interrupted, stop_dispatch, cancel_deadline
        interrupted = stop_dispatch = True
        if cancel_deadline is None:
            cancel_deadline = time.monotonic() + 300
            for process, *_ in active.values():
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGINT)

    previous_handler = signal.signal(signal.SIGINT, request_interrupt)
    manifest.update(status="in_progress", completed=False)
    write_master()
    try:
        while active or (pending and not stop_dispatch and (max_episodes is None or launched < max_episodes)):
            while pending and not stop_dispatch and len(active) < workers and (max_episodes is None or launched < max_episodes):
                planned = pending.pop(0)
                folder, attempt, runtime_root = next_attempt(output, planned["episode_id"])
                result_path, payload_path, log_path = runtime_root / "worker_result.json", runtime_root / "worker_payload.json", runtime_root / "worker.log"
                payload = {**payload_base, "planned": planned, "folder": str(folder), "runtime_root": str(runtime_root),
                           "result_path": str(result_path), "cpu_selftest": cpu_selftest}
                if cpu_selftest:
                    payload.update(test_delay=planned.get("test_delay", 0.08), test_failure=planned.get("test_failure", False))
                atomic_json(payload_path, payload)
                receipt = {"episode_id": planned["episode_id"], "folder": str(folder), "attempt": attempt,
                           "runtime_root": str(runtime_root), "result_path": str(result_path), "log_path": str(log_path), "status": "dispatched"}
                manifest.setdefault("attempts", []).append({"episode_id": planned["episode_id"], "folder": str(folder), "attempt": attempt})
                orchestration["jobs"].append(receipt)
                write_master()
                log = log_path.open("wb")
                process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker-payload", str(payload_path)],
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                receipt.update(worker_pid=process.pid, process_group_id=process.pid, status="running")
                active[process.pid] = (process, log, receipt, planned)
                launched += 1
                orchestration["max_workers_observed"] = max(orchestration["max_workers_observed"], len(active))
                write_master()
            changed = False
            for pid, (process, log, receipt, planned) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                log.close()
                receipt["exit_code"] = code
                result_path = Path(receipt["result_path"])
                result = json.loads(result_path.read_text()) if result_path.is_file() else {"status": "failed", "error": "Worker exited without a result envelope"}
                remaining = proc_members(pid)
                receipt["remaining_process_group_members_before_cleanup"] = remaining
                if remaining:
                    # Only this worker's private session/process group is in
                    # scope. A normal v2 close leaves no descendants here.
                    os.killpg(pid, signal.SIGTERM)
                    deadline = time.monotonic() + 10
                    while proc_members(pid) and time.monotonic() < deadline:
                        time.sleep(0.1)
                    if proc_members(pid):
                        os.killpg(pid, signal.SIGKILL)
                        deadline = time.monotonic() + 10
                        while proc_members(pid) and time.monotonic() < deadline:
                            time.sleep(0.1)
                    result = {**result, "status": "failed", "error": "Worker left simulator descendants; private group cleaned"}
                receipt["process_group_empty_after_cleanup"] = not proc_members(pid)
                try:
                    if code != 0 or result["status"] != "completed":
                        raise RuntimeError(result.get("error", f"Worker exit {code}"))
                    entry = result["episode"]
                    if not verify_completed(entry, planned):
                        raise ValueError("Worker completion lacks strict episode/shutdown receipts")
                    manifest["episodes"] = [e for e in manifest["episodes"] if e["episode_id"] != planned["episode_id"]] + [entry]
                    receipt["status"] = "completed"
                except Exception as exc:
                    receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                    errors.append({"episode_id": planned["episode_id"], "error": receipt["error"], "result_path": str(result_path)})
                    stop_dispatch = True
                del active[pid]
                changed = True
                write_master()
            if cancel_deadline is not None and time.monotonic() > cancel_deadline:
                for process, *_ in active.values():
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM if cancel_phase == 0 else signal.SIGKILL)
                cancel_phase += 1
                cancel_deadline = time.monotonic() + 15
            if changed or time.monotonic() - last_progress >= 10:
                print(json.dumps({"completed_episodes": len(manifest["episodes"]), "active_workers": len(active),
                                  "new_launched": launched, "pending": len(pending), "stopped_dispatch": stop_dispatch}), flush=True)
                last_progress = time.monotonic()
            if active:
                time.sleep(0.2)
    except BaseException as exc:
        errors.append({"parent_error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        raise
    finally:
        # Unexpected parent exceptions must not leave untracked simulators.
        for process, log, receipt, _ in active.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=300)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=15)
            if proc_members(process.pid):
                os.killpg(process.pid, signal.SIGKILL)
                deadline = time.monotonic() + 10
                while proc_members(process.pid) and time.monotonic() < deadline:
                    time.sleep(0.1)
            log.close()
            receipt.update(status="aborted_by_parent", exit_code=process.returncode,
                           process_group_empty_after_cleanup=not proc_members(process.pid))
        signal.signal(signal.SIGINT, previous_handler)
        orchestration.update(finished_at=datetime.now(UTC).isoformat(), errors=errors, interrupted=interrupted,
                             launched=launched, all_worker_processes_reaped=not active or all(p.poll() is not None for p, *_ in active.values()))
        manifest["completed"] = len(manifest["episodes"]) == manifest["expected_full_episodes"]
        manifest["status"] = "interrupted" if interrupted else ("failed" if errors else ("completed" if manifest["completed"] else "pilot_complete"))
        write_master()
    return 130 if interrupted else (1 if errors else 0)


def self_test():
    """Real CPU subprocesses test central writes, resume and failed-worker drain."""
    with tempfile.TemporaryDirectory(prefix="parallel-capture-cpu-") as temp:
        output = Path(temp)
        planned = [{"episode_id": f"cpu_{i}", "family_id": f"family_{i // 2}", "split": "train", "seed": i,
                    "target_side": "left" if i % 2 == 0 else "right", "layout": {"cpu_test": i}} for i in range(6)]
        manifest = {"episodes": [], "expected_full_episodes": len(planned), "planned_episodes": planned}
        assert run_parent(output, manifest, planned.copy(), 3, {}, max_episodes=3, cpu_selftest=True) == 0
        assert len(manifest["episodes"]) == 3 and manifest["status"] == "pilot_complete"
        completed_ids = {e["episode_id"] for e in manifest["episodes"]}
        remaining = [s for s in planned if not any(e["episode_id"] == s["episode_id"] and verify_completed(e, s) for e in manifest["episodes"])]
        assert len(remaining) == 3
        assert run_parent(output, manifest, remaining, 3, {}, cpu_selftest=True) == 0
        assert manifest["completed"] and len(manifest["episodes"]) == 6
        assert len({e["episode_id"] for e in manifest["episodes"]}) == 6
        all_jobs = [j for o in manifest["parallel_orchestrations"] for j in o["jobs"]]
        assert len({j["runtime_root"] for j in all_jobs}) == 6
        assert all(j["process_group_empty_after_cleanup"] for j in all_jobs)
        assert all(o["manifest_writers"] == [os.getpid()] and o["max_workers_observed"] <= 3 for o in manifest["parallel_orchestrations"])
        assert all(e["worker_pid"] != os.getpid() for e in manifest["episodes"])
        assert completed_ids.issubset({e["episode_id"] for e in manifest["episodes"]})
        # Altered completed receipts cannot silently pass a resume.
        entry = manifest["episodes"][0]
        Path(entry["observations"][0]["npz_path"]).write_bytes(b"tampered")
        try:
            verify_completed(entry, next(s for s in planned if s["episode_id"] == entry["episode_id"]))
        except ValueError:
            pass
        else:
            raise AssertionError("Tampered completed observation was accepted")
    with tempfile.TemporaryDirectory(prefix="parallel-capture-cpu-fail-") as temp:
        planned = [{"episode_id": f"fail_{i}", "family_id": str(i), "split": "train", "seed": i,
                    "target_side": "right", "layout": {}, "test_failure": i == 0,
                    "test_delay": 0.02 if i == 0 else 0.2} for i in range(5)]
        manifest = {"episodes": [], "expected_full_episodes": 5}
        assert run_parent(Path(temp), manifest, planned.copy(), 2, {}, cpu_selftest=True) == 1
        jobs = manifest["parallel_orchestrations"][0]["jobs"]
        assert len(jobs) == 2 and len(manifest["episodes"]) == 1
        assert all(j["process_group_empty_after_cleanup"] for j in jobs)
        assert manifest["status"] == "failed"
    with tempfile.TemporaryDirectory(prefix="parallel-capture-cpu-attempt-") as temp:
        first, index, runtime = next_attempt(Path(temp), "failed_before_episode_mkdir")
        assert not first.exists() and runtime.exists() and index == 0
        second, index, runtime2 = next_attempt(Path(temp), "failed_before_episode_mkdir")
        assert second != first and runtime2 != runtime and index == 1
    print(json.dumps({"cpu_selftest_passed": True, "concurrent_limit": 3,
                      "checks": ["parent_only_manifest_writes", "3_CPU_subprocesses", "skip_3_completed_on_resume",
                                 "unique_episode_runtime", "tamper_rejected", "failure_stops_dispatch_and_drains"]}))
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--worker-payload", type=Path)
    pre.add_argument("--self-test", action="store_true")
    pre.add_argument("--v2-script", type=Path, default=Path(__file__).with_name("capture_dynamic_protocol_v2.py"))
    known, _ = pre.parse_known_args(argv)
    if known.worker_payload:
        return worker_main(known.worker_payload)
    if known.self_test:
        return self_test()
    v2 = load_v2(known.v2_script)
    parser = v2.parser()
    parser.description = __doc__
    parser.add_argument("--v2-script", type=Path, default=known.v2_script)
    parser.add_argument("--workers", type=int, choices=range(1, MAX_WORKERS + 1), default=3)
    parser.add_argument("--serial-stopped-confirmed", action="store_true")
    parser.add_argument("--serial-parent-pid", type=int, help="Optional prior serial launcher PID, which must no longer exist")
    args = parser.parse_args(argv)
    args.config = (args.config or args.experiment_root / "configs/rm65b_mem_protocol.yaml").resolve()
    args.robotwin_root = (args.robotwin_root or args.experiment_root / "code/RoboTwin").resolve()
    args.output = args.output.resolve()
    if args.controls <= 0 or args.controls > 256 or args.controls % 32:
        raise ValueError("Controls must match v2's positive multiple of32 up to256")
    if args.max_episodes is not None and args.max_episodes < 1:
        raise ValueError("--max-episodes must be positive")
    import yaml
    base = yaml.safe_load(args.config.read_text())["environment"]["task_config"]
    args.asset_root = args.asset_root or Path(base["asset_root"])
    if args.asset_root.resolve() != Path(base["asset_root"]).resolve():
        raise ValueError("Asset override must match the original task config")
    args.command_spec = v2.load_command_spec(args.asset_root)
    _, source = v2.load_reference(args.source_actions)
    plan = v2.plan_for(args, source)
    path = args.output / "manifest.json"
    if not path.is_file():
        raise ValueError("This wrapper resumes an existing v2 directory; run/close the serial pilot first")
    manifest = json.loads(path.read_text())
    if manifest.get("plan_identity_sha256") != plan["plan_identity_sha256"]:
        raise ValueError("Master manifest is not the exact same v2 code/source/config/split/physics plan")
    pending = []
    for spec in plan["planned_episodes"]:
        entry = next((e for e in manifest["episodes"] if e["episode_id"] == spec["episode_id"]), None)
        if entry is None or not verify_completed(entry, spec):
            pending.append(spec)
    if not args.execute:
        print(json.dumps({"mode": "plan_only", "parallel_orchestrator_sha256": sha(__file__),
                          "frozen_v2_sha256": EXPECTED_V2_SHA, "same_plan_identity_sha256": plan["plan_identity_sha256"],
                          "workers": args.workers, "already_complete": len(manifest["episodes"]), "pending": len(pending),
                          "maximum_new_episodes": args.max_episodes, "serial_collectors_still_alive": serial_collectors(args.output),
                          "serial_parent_pid": args.serial_parent_pid,
                          "proc_scan_limitation": "visibility-limited supplementary check; explicit prior launcher and SAPIEN-child shutdown remains required",
                          "execute_requires_serial_stopped_confirmed": True}, indent=2))
        return 0
    if not args.resume or not args.serial_stopped_confirmed:
        raise ValueError("First stop serial and confirm all simulator children closed; then use --resume --serial-stopped-confirmed")
    if args.serial_parent_pid is not None and Path(f"/proc/{args.serial_parent_pid}").exists():
        raise RuntimeError("Prior serial launcher PID still exists; parallel start refused")
    serial = serial_collectors(args.output)
    if serial:
        raise RuntimeError(f"Serial capture is still alive; parallel start refused: {[x['pid'] for x in serial]}")
    payload_base = {"v2_script": str(args.v2_script.resolve()), "robotwin_root": str(args.robotwin_root),
                    "config": str(args.config), "source_actions": str(args.source_actions.resolve()),
                    "source_sha256": source["sha256"], "config_sha256": plan["identities"]["config_sha256"],
                    "urdf_sha256": args.command_spec["urdf_sha256"], "command_manifest_sha256": args.command_spec["manifest_sha256"],
                    "runtime_source_sha256": plan["identities"]["runtime_source_sha256"], "controls": args.controls}
    with (args.output / ".parallel_master.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Recheck after lock; only parent owns master mutations from here.
        if serial_collectors(args.output):
            raise RuntimeError("Serial capture appeared during preflight; parallel start refused")
        manifest = json.loads(path.read_text())
        if manifest.get("plan_identity_sha256") != plan["plan_identity_sha256"]:
            raise ValueError("Plan identity changed while acquiring parent lock")
        pending = []
        for spec in plan["planned_episodes"]:
            entry = next((e for e in manifest["episodes"] if e["episode_id"] == spec["episode_id"]), None)
            if entry is None or not verify_completed(entry, spec):
                pending.append(spec)
        result = run_parent(args.output, manifest, pending, args.workers, payload_base, args.max_episodes)
        manifest["vendor_urdf_unchanged"] = v2.load_command_spec(args.asset_root)["urdf_sha256"] == args.command_spec["urdf_sha256"]
        manifest["source_actions_unchanged"] = sha(args.source_actions) == source["sha256"]
        manifest["command_manifest_unchanged"] = v2.load_command_spec(args.asset_root)["manifest_sha256"] == args.command_spec["manifest_sha256"]
        manifest["config_unchanged"] = sha(args.config) == plan["identities"]["config_sha256"]
        manifest["native_runtime_source_unchanged"] = all(sha(args.robotwin_root / "integrations/realman_rm65b_gen4" / name) == expected for name, expected in plan["identities"]["runtime_source_sha256"].items())
        if not all(manifest[k] for k in ("vendor_urdf_unchanged", "source_actions_unchanged", "command_manifest_unchanged", "config_unchanged", "native_runtime_source_unchanged")):
            manifest.update(status="failed", completed=False, integrity_error="Reference source or URDF changed")
            result = 1
        atomic_json(path, manifest)
        return result


if __name__ == "__main__":
    raise SystemExit(main())
