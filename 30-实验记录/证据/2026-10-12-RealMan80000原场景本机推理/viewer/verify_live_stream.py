"""Read advancing real MJPEG frames and save a compact verification receipt."""
import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", default="left_wrist")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).parent / "actual_stream_receipt.json")
    args = parser.parse_args()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    base = "http://127.0.0.1:8768"
    response = opener.open(base + "/mjpeg/" + args.camera, timeout=45)
    frames = []
    for _ in range(args.samples):
        boundary = response.readline().strip()
        if boundary != b"--actual_rgb_frame":
            raise RuntimeError("Expected MJPEG frame boundary")
        headers = {}
        while True:
            line = response.readline().strip()
            if not line:
                break
            name, value = line.decode().split(":", 1)
            headers[name] = value.strip()
        jpeg = response.read(int(headers["Content-Length"]))
        if response.read(2) != b"\r\n":
            raise RuntimeError("Incomplete MJPEG frame")
        frames.append({"frame_seq": int(headers["X-Frame-Seq"]),
                       "physics_step": int(headers["X-Physics-Step"]),
                       "sim_time_s": float(headers["X-Sim-Time"]),
                       "jpeg_sha256": hashlib.sha256(jpeg).hexdigest(),
                       "jpeg_bytes": len(jpeg), "received_wall_time": time.time()})
    response.close()
    with opener.open(base + "/api/status", timeout=5) as status_response:
        state = json.loads(status_response.read())
    advancing = all(b["frame_seq"] > a["frame_seq"]
                    and b["physics_step"] > a["physics_step"]
                    and b["sim_time_s"] > a["sim_time_s"]
                    for a, b in zip(frames, frames[1:]))
    changed_image = len({f["jpeg_sha256"] for f in frames}) > 1
    source_actual = (state.get("frame_source") == "actual_simulator_capture"
                     and not state.get("ui_test") and state.get("checkpoint") == 80000)
    inference_seen = state.get("model_query_count", 0) > 0
    receipt = {"passed": bool(advancing and changed_image and source_actual and inference_seen),
               "camera": args.camera, "advancing_physical_clock": advancing,
               "distinct_actual_camera_images": changed_image,
               "actual_capture_source": source_actual, "inference_queries_seen": inference_seen,
               "frames": frames,
               "status": {k: state.get(k) for k in (
                   "episode_index", "episode_position", "phase", "checkpoint", "model_query_count",
                   "frame_seq", "physics_step", "sim_time_s", "frame_source", "frame_wall_time")}}
    args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(receipt, ensure_ascii=False))
    if not receipt["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
