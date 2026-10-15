"""Publish actual simulator RGB captures to the task-local live viewer.

Only call update(images=...) after an actual capture. A metadata-only update
changes stage information without incrementing the image sequence.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
from pathlib import Path


class LivePublisher:
    CAMERA_ALIASES = {
        "head": ("head", "cam_head", "head_camera"),
        "left_wrist": ("left_wrist", "cam_left_wrist", "left_camera"),
        "right_wrist": ("right_wrist", "cam_right_wrist", "right_camera"),
        "external": ("external", "observer", "external_camera"),
    }

    def __init__(self, directory="/tmp/realman-eval80k-20261012/live", *, checkpoint=80000):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "frames").mkdir(exist_ok=True)
        self.lock = threading.RLock()
        try:
            previous = json.loads((self.directory / "status.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            previous = {}
        known_seqs = [int(p.name) for p in (self.directory / "frames").iterdir()
                      if p.name.isdigit()]
        self.seq = max([int(previous.get("frame_seq", 0)), *known_seqs])
        self.revision = int(previous.get("status_revision", 0))
        self.status = {
            "checkpoint": checkpoint,
            "phase": "preparing",
            "frame_seq": self.seq,
            "model_query_count": 0,
            "physics_step": 0,
            "sim_time_s": 0.0,
            "camera_paths": {},
            "frame_is_current_episode": False,
            "frame_source": "actual_simulator_capture",
            "run_started_wall_time": time.time(),
            "results": previous.get("results", []),
            "results_title": previous.get("results_title"),
            "formal_six_completed": previous.get("formal_six_completed", False),
            "formal_six_summary_source": previous.get("formal_six_summary_source"),
            "overview_video_path": previous.get("overview_video_path"),
            "manual_run": previous.get("manual_run"),
        }
        self._write_status()

    @staticmethod
    def _serializable(value):
        if hasattr(value, "tolist"):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Unsupported metadata value: {type(value).__name__}")

    def _write_status(self):
        self.revision += 1
        self.status["status_revision"] = self.revision
        self.status["status_wall_time"] = time.time()
        raw = json.dumps(self.status, ensure_ascii=False, allow_nan=False,
                         default=self._serializable).encode("utf-8")
        tmp = self.directory / ".status.tmp"
        tmp.write_bytes(raw)
        os.replace(tmp, self.directory / "status.json")

    def begin_episode(self, index, label=None, **metadata):
        with self.lock:
            self.status.update(metadata)
            self.status.update({
                "episode_index": int(index),
                "episode_label": label or f"示范场景 {index}",
                "phase": "restoring_scene",
                "camera_paths": {},
                "frame_is_current_episode": False,
                "model_query_count": 0,
                "physics_step": 0,
                "sim_time_s": 0.0,
                "native_success": False,
            })
            self.status.pop("error", None)
            self.status.pop("result", None)
            self._write_status()

    def update(self, images=None, metadata=None, **extra_metadata):
        """Atomically publish one real RGB capture, or update stage only."""
        with self.lock:
            self.status.update(metadata or {})
            self.status.update(extra_metadata)
            if images is not None:
                capture_wall_time = float(extra_metadata.get(
                    "capture_wall_time", (metadata or {}).get("capture_wall_time", time.time())))
                # Imports are deferred so metadata-only producers need no image runtime.
                import cv2
                import numpy as np

                canonical = {}
                for name, aliases in self.CAMERA_ALIASES.items():
                    for key in aliases:
                        if key in images:
                            im = np.asarray(images[key])
                            if im.dtype != np.uint8 or im.ndim != 3 or im.shape[2] != 3:
                                raise ValueError(f"{key} must be actual RGB uint8 H x W x 3")
                            canonical[name] = np.ascontiguousarray(im)
                            break
                missing = {"head", "left_wrist", "right_wrist"} - canonical.keys()
                if missing:
                    raise ValueError(f"Missing actual policy-camera captures: {sorted(missing)}")
                seq = self.seq + 1
                pending = self.directory / "frames" / f".{seq:09d}.tmp"
                committed = self.directory / "frames" / f"{seq:09d}"
                pending.mkdir()
                paths = {}
                dimensions = {}
                try:
                    for name, rgb in canonical.items():
                        good, encoded = cv2.imencode(
                            ".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 85])
                        if not good:
                            raise RuntimeError(f"JPEG encoding failed for {name}")
                        (pending / f"{name}.jpg").write_bytes(encoded.tobytes())
                        paths[name] = f"frames/{seq:09d}/{name}.jpg"
                        dimensions[name] = [int(rgb.shape[1]), int(rgb.shape[0])]
                    os.replace(pending, committed)
                except BaseException:
                    shutil.rmtree(pending, ignore_errors=True)
                    raise
                self.seq = seq
                self.status.update({
                    "frame_seq": seq,
                    "camera_paths": paths,
                    "camera_dimensions": dimensions,
                    "frame_wall_time": capture_wall_time,
                    "frame_sim_time_s": float(self.status.get("sim_time_s", 0.0)),
                    "frame_physics_step": int(self.status.get("physics_step", 0)),
                    "frame_episode_index": self.status.get("episode_index"),
                    "frame_is_current_episode": True,
                    "frame_source": "actual_simulator_capture",
                })
            self._write_status()
            if images is not None:
                # Retain recent committed captures long enough for concurrent clients.
                cutoff = self.seq - 12
                for frame in (self.directory / "frames").iterdir():
                    if frame.name.isdigit() and int(frame.name) <= cutoff:
                        shutil.rmtree(frame, ignore_errors=True)
            return self.seq

    def finish(self, results=None, *, error=None):
        self.update(metadata={"phase": "error" if error else "complete",
                              "results": results or [], "error": error})


__all__ = ["LivePublisher"]
