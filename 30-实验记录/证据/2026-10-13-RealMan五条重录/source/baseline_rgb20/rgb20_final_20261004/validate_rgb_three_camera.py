"""Independent readback for three-camera RGB-only RoboTwin HDF5.

The native exporter permits absent depth. This verifier does not call SAPIEN or
run any physical replay. With --source-hdf, every non-vision dataset and its
attributes must be identical to the accepted source episode. Camera matrices
describe simulation geometry; they are not measured real-camera calibration.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import traceback

import h5py
import numpy as np

CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
FORBIDDEN = {"depth", "depths", "depth_image", "depth_images", "pointcloud",
             "point_cloud", "pointclouds", "segmentation", "mesh_segmentation",
             "actor_segmentation", "segmentation_mesh", "segmentation_actor"}
JOINT_FIELDS = {"left_arm_joint_states": 6, "left_ee_joint_states": 1,
                "right_arm_joint_states": 6, "right_ee_joint_states": 1}
PHYSICS_FIELDS = {
    "physics_step": (), "sim_time_before_s": (), "sim_time_after_s": (),
    "measured_state24_before": (24,), "applied_command24": (24,),
    "measured_state24_after": (24,), "hand24_qpos_before_rad": (24,),
    "hand24_qpos_after_rad": (24,), "hand24_drive_target_rad": (24,),
    "hand24_actual_fd_rad_s": (24,), "hand24_engine_qvel_rad_s": (24,),
    "full36_qpos_before_rad": (36,), "full36_qpos_after_rad": (36,),
}
SOURCE_REFS = {
    "native_exporter": "envs/utils/pkl2hdf5.py:79-168; CAMERA_MAP:19-24; optional depth:145-148",
    "native_observation": "envs/_base_task.py:449-513; RGB:461-464; depth:480-483",
    "native_three_camera_rgb_profile": "env_cfg/task_config/demo_clean.yml:18-32",
    "native_loader": "XPolicyLab/utils/data_loader.py:8-54",
    "dexterous_extension": "deployment/realman_adjust_bottle_20261002/accepted_reset_replay_v32.py:983-1003",
}


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def text(value):
    return value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)


def finite(value, message):
    require(np.asarray(value).dtype.kind in "bifu" and np.isfinite(value).all(), message)


def codec_module(root):
    path = (Path(root) / "data/decode_image_bit.py").resolve()
    require(path.is_file(), "Official CPU codec not found: " + str(path))
    spec = importlib.util.spec_from_file_location("rgb_readback_official_codec", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def dataset_paths(handle, outside_vision=False):
    result = []
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset) and (not outside_vision or not name.startswith("vision/")):
            result.append(name)
    handle.visititems(visitor)
    return sorted(result)


def exact_values(left, right, label):
    a, b = np.asarray(left), np.asarray(right)
    require(a.shape == b.shape and a.dtype == b.dtype, "Shape/dtype changed: " + label)
    # Object-backed scalar UTF-8 uses Python object pointers, not numeric bytes.
    if a.dtype.hasobject:
        require(np.array_equal(a, b), "String/object value changed: " + label)
    else:
        require(a.tobytes(order="C") == b.tobytes(order="C"),
                "Dataset value bytes changed: " + label)


def exact_attrs(left, right, label):
    require(set(left.attrs) == set(right.attrs), "Attributes changed: " + label)
    for name in left.attrs:
        exact_values(left.attrs[name], right.attrs[name], label + "@" + name)


def compare_nonvision(output, source):
    paths = dataset_paths(output, True)
    require(paths == dataset_paths(source, True), "Non-vision dataset path set differs from source")
    count, value_bytes = 0, 0
    for name in paths:
        a, b = output[name], source[name]
        require(a.shape == b.shape and a.dtype == b.dtype, "Source shape/dtype changed: " + name)
        exact_attrs(a, b, name)
        if a.ndim == 0:
            exact_values(a[()], b[()], name)
            value_bytes += a.dtype.itemsize
        else:
            row_size = max(1, int(np.prod(a.shape[1:], dtype=np.int64)) * a.dtype.itemsize)
            chunk = max(1, (8 * 1024 * 1024) // row_size)
            for start in range(0, a.shape[0], chunk):
                exact_values(a[start:start + chunk], b[start:start + chunk], name)
            value_bytes += int(np.prod(a.shape, dtype=np.int64)) * a.dtype.itemsize
        count += 1
    groups = []
    def visitor(name, obj):
        if isinstance(obj, h5py.Group) and name != "vision" and not name.startswith("vision/"):
            groups.append(name)
    source.visititems(visitor)
    for name in groups:
        require(name in output and isinstance(output[name], h5py.Group), "Missing source group: " + name)
        exact_attrs(output[name], source[name], name)
    return {"passed": True, "compared_dataset_count": count,
            "numeric_and_fixed_string_value_bytes_compared": value_bytes,
            "comparison": "Exact path, shape, dtype, value bytes and dataset/group attributes; variable UTF-8 strings compared by value",
            "hdf_container_bytes_expected_to_differ": True}


def validate(hdf, root, source_hdf=None):
    hdf = Path(hdf).resolve()
    source_hdf = Path(source_hdf).resolve() if source_hdf is not None else None
    require(hdf.is_file(), "HDF5 missing")
    require(source_hdf != hdf, "Source and output HDF must be distinct")
    codec, codec_path = codec_module(root)
    before = {str(hdf): sha(hdf), str(codec_path): sha(codec_path)}
    if source_hdf is not None:
        require(source_hdf.is_file(), "Source HDF5 missing")
        before[str(source_hdf)] = sha(source_hdf)
    result = {"passed": False, "readback_only": True, "hardware_io": False,
              "hdf5": str(hdf), "source_hdf5": str(source_hdf) if source_hdf else None,
              "validator_source_sha256": sha(Path(__file__)),
              "codec_path": str(codec_path), "codec_sha256": before[str(codec_path)],
              "source_refs": SOURCE_REFS, "checks": {}}
    checks = result["checks"]
    with h5py.File(hdf, "r") as f:
        require(text(f["data_format_version"][()]) == "v1.0", "Native v1.0 missing")
        require(text(f.attrs.get("source_format")) == "RoboTwin", "Native source_format differs")
        instructions = json.loads(text(f["instructions"][()]))
        require(isinstance(instructions, list) and bool(instructions)
                and all(isinstance(v, str) for v in instructions), "Instructions invalid")
        require(set(f["vision"]) == set(CAMERAS), "Exactly head and left/right wrist cameras required")
        paths = dataset_paths(f)
        forbidden = [p for p in paths if any(s.lower() in FORBIDDEN for s in p.split("/"))]
        require(not forbidden, "Forbidden non-RGB dataset(s): " + repr(forbidden))
        n = int(f["state/left_arm_joint_states"].shape[0])
        require(n > 0, "Empty episode")
        for group in ("state", "action"):
            for name, dim in JOINT_FIELDS.items():
                ds = f[group + "/" + name]
                require(ds.shape == (n, dim), "Native 14D dimensions differ: " + ds.name)
                finite(ds[:], "Nonfinite native 14D: " + ds.name)
            for name in ("left_ee_poses", "right_ee_poses"):
                values = f[group + "/" + name][:]
                require(values.shape == (n, 7), "Endpose dimensions differ")
                finite(values, "Endpose nonfinite")
                require(np.allclose(np.linalg.norm(values[:, 3:], axis=1), 1., atol=2e-4),
                        "Endpose quaternion not unit")
        checks["native14d_and_endposes"] = {"passed": True, "pairs": n}
        dex = f["realman_dexterous"]
        for name in ("measured_state24", "next_command24"):
            ds = dex[name]
            require(ds.shape == (n, 24), "24D dimensions differ: " + name)
            finite(ds[:], "24D nonfinite: " + name)
        require(not bool(dex.attrs.get("sdk_mapping_calibrated", True)), "SDK calibration incorrectly claimed")
        times = np.asarray(dex["sample_time_s"][:])
        next_times = np.asarray(dex["next_sample_time_s"][:])
        require(times.shape == next_times.shape == (n,), "Sample clock dimensions differ")
        finite(times, "Sample times nonfinite"); finite(next_times, "Next sample times nonfinite")
        require(times[0] == 0 and np.all(np.diff(times) >= 0)
                and np.all(np.diff(next_times) >= 0) and np.all(next_times >= times),
                "Sample times must begin at zero and be nondecreasing")
        require(np.array_equal(next_times[:-1], times[1:]), "Adjacent NEXT observation clocks differ")
        checks["measured24d_commands_and_clock"] = {
            "passed": True, "duplicate_boundary_sample_count": int(np.count_nonzero(np.diff(times) == 0)),
            "sample_first_s": float(times[0]), "sample_last_s": float(times[-1]),
            "next_sample_last_s": float(next_times[-1]),
            "native_frequency_metadata": int(f["additional_info/frequency"][()]),
            "frequency_metadata_is_not_assumed_rgb_fps": True,
        }
        cameras = {}
        for name in CAMERAS:
            cg = f["vision/" + name]
            require(cg["colors"].shape == (n,) and cg["colors"].dtype.kind == "S",
                    "Native compressed fixed-byte RGB column invalid: " + name)
            require(np.array_equal(cg["shape"][:], [480, 640, 3]), "RGB shape differs: " + name)
            K, C = cg["intrinsic_matrix"][:], cg["extrinsics_matrix"][:]
            require(K.shape == (n, 3, 3) and C.shape == (n, 4, 4), "Matrix/count schema invalid: " + name)
            finite(K, "Camera K nonfinite: " + name); finite(C, "Camera extrinsics nonfinite: " + name)
            require(np.all(K[:, 0, 0] > 0) and np.all(K[:, 1, 1] > 0)
                    and np.all((K[:, 0, 2] >= 0) & (K[:, 0, 2] <= 640))
                    and np.all((K[:, 1, 2] >= 0) & (K[:, 1, 2] <= 480)),
                    "Camera focal/principal point invalid: " + name)
            require(np.allclose(K[:, 2, :], [0, 0, 1], atol=1e-6), "K bottom row invalid: " + name)
            require(np.allclose(C[:, 3, :], [0, 0, 0, 1], atol=2e-5), "Camera transform bottom row invalid")
            rotations = C[:, :3, :3]
            require(np.allclose(np.swapaxes(rotations, 1, 2) @ rotations, np.eye(3), atol=2e-5)
                    and np.allclose(np.linalg.det(rotations), 1, atol=2e-5),
                    "Camera rotations not proper rigid transforms: " + name)
            stream_hash = hashlib.sha256()
            low, high = 255, 0
            for idx in range(n):
                encoded = bytes(cg["colors"][idx]).rstrip(b"\0")
                require(codec._has_rgb_marker(encoded), "Official RGB JPEG marker missing: " + name)
                image = codec.decode_image_bit(encoded)
                require(image.dtype == np.uint8 and image.shape == (480, 640, 3),
                        "Decoded frame shape/dtype differs: " + name + "/" + str(idx))
                stream_hash.update(image.tobytes(order="C"))
                low, high = min(low, int(image.min())), max(high, int(image.max()))
            cameras[name] = {"passed": True, "decoded_rgb_frames": n, "shape_HWC": [480, 640, 3],
                             "dtype": "uint8", "decoded_rgb_stream_sha256": stream_hash.hexdigest(),
                             "pixel_min": low, "pixel_max": high, "intrinsic_first": K[0].tolist(),
                             "extrinsics_convention": "OpenGL camera-to-world as native exporter",
                             "depth_dataset_absent": True, "real_camera_calibrated": False}
        checks["three_camera_rgb_and_render_geometry"] = {"passed": True, "cameras": cameras}
        if "realman_dexterous_physics" in f:
            g = f["realman_dexterous_physics"]
            steps = int(g["physics_step"].shape[0])
            require(steps > 0, "Empty physical telemetry")
            for name, trailing in PHYSICS_FIELDS.items():
                ds = g[name]
                require(ds.shape == (steps,) + trailing, "Physical field shape differs: " + name)
                finite(ds[:], "Physical field nonfinite: " + name)
            require(np.array_equal(g["physics_step"][:], np.arange(1, steps + 1)), "Physical step sequence incomplete")
            tb, ta = g["sim_time_before_s"][:], g["sim_time_after_s"][:]
            dt = float(ta[0] - tb[0])
            require(dt > 0 and tb[0] == 0 and np.all(ta > tb)
                    and np.allclose(tb[1:], ta[:-1], atol=1e-10, rtol=0)
                    and np.allclose(ta - tb, dt, atol=1e-10, rtol=0), "Physical timeline discontinuity")
            require(next_times[-1] <= ta[-1] + 1e-8
                    and np.max(next_times - times) <= 15 * dt + 1e-8,
                    "Observation times outside source physical timeline/save interval")
            for attr, count in (("full36_joint_names", 36), ("hand24_joint_names", 24)):
                names = json.loads(text(g.attrs[attr]))
                require(len(names) == count and len(set(names)) == count, "Physical joint names invalid: " + attr)
            checks["full36_and_per_step24d"] = {"passed": True, "physics_steps": steps,
                                               "physics_dt_s": dt, "physical_duration_s": float(ta[-1]),
                                               "field_names": list(PHYSICS_FIELDS)}
        else:
            checks["full36_and_per_step24d"] = {"present": False, "scope": "Required if accepted source contains these fields"}
        if source_hdf is not None:
            with h5py.File(source_hdf, "r") as source:
                checks["all_nonvision_fields_exact_vs_accepted_source"] = compare_nonvision(f, source)
        else:
            checks["all_nonvision_fields_exact_vs_accepted_source"] = {"verified": False,
                                                                      "reason": "No --source-hdf supplied"}
    after = {p: sha(p) for p in before}
    require(before == after, "Input HDF or codec changed during readback")
    checks["readback_inputs_unchanged"] = {"passed": True, "source_sha256": after}
    result.update(passed=True, pairs=n, training_cameras=list(CAMERAS),
                  rgb_only=True, camera_calibration_available=False,
                  physical_trajectory_rerender_requires_source_provenance=True,
                  physical_success_not_inferred_from_images=True)
    return result


def self_test(root):
    """Small synthetic fixtures prove rejection branches, not task success."""
    import shutil
    codec, _ = codec_module(root)
    with tempfile.TemporaryDirectory(prefix="robotwin_rgb_validation_") as temp:
        temp = Path(temp)
        source, output = temp / "source.hdf5", temp / "rgb.hdf5"
        n, steps, dt = 3, 6, .004
        with h5py.File(source, "w") as f:
            f.attrs["source_format"] = "RoboTwin"
            f.create_dataset("data_format_version", data="v1.0")
            f.create_dataset("instructions", data=json.dumps(["adjust bottle"]))
            f.create_dataset("additional_info/frequency", data=15)
            for group in ("state", "action"):
                for name, dim in JOINT_FIELDS.items():
                    f.create_dataset(group + "/" + name, data=np.zeros((n, dim), np.float32))
                for name in ("left_ee_poses", "right_ee_poses"):
                    f.create_dataset(group + "/" + name, data=np.tile([0., 0., 0., 1., 0., 0., 0.], (n, 1)).astype(np.float32))
            dex = f.create_group("realman_dexterous")
            dex.attrs["sdk_mapping_calibrated"] = False
            dex.create_dataset("measured_state24", data=np.zeros((n, 24)))
            dex.create_dataset("next_command24", data=np.zeros((n, 24)))
            dex.create_dataset("sample_time_s", data=[0., 0., 3 * dt])
            dex.create_dataset("next_sample_time_s", data=[0., 3 * dt, steps * dt])
            g = f.create_group("realman_dexterous_physics")
            g.attrs["full36_joint_names"] = json.dumps(["joint" + str(i) for i in range(36)])
            g.attrs["hand24_joint_names"] = json.dumps(["hand" + str(i) for i in range(24)])
            for name, trailing in PHYSICS_FIELDS.items():
                if name == "physics_step":
                    values = np.arange(1, steps + 1)
                elif name == "sim_time_before_s":
                    values = np.arange(steps) * dt
                elif name == "sim_time_after_s":
                    values = np.arange(1, steps + 1) * dt
                else:
                    values = np.zeros((steps,) + trailing)
                g.create_dataset(name, data=values)
            f.create_dataset("vision/cam_head/depths", data=np.ones((n, 1, 1)))
        shutil.copyfile(source, output)
        rgb = np.zeros((480, 640, 3), np.uint8); rgb[:, :, 0] = 180; rgb[:, :, 1] = 40
        encoded = codec.encode_image_bit(rgb)
        with h5py.File(output, "a") as f:
            del f["vision"]
            for name in CAMERAS:
                cg = f.create_group("vision/" + name)
                cg.create_dataset("colors", data=[encoded] * n, dtype="S" + str(len(encoded)))
                cg.create_dataset("shape", data=[480, 640, 3])
                cg.create_dataset("intrinsic_matrix", data=np.tile([[400., 0., 320.], [0., 400., 240.], [0., 0., 1.]], (n, 1, 1)))
                cg.create_dataset("extrinsics_matrix", data=np.tile(np.eye(4), (n, 1, 1)))
        require(validate(output, root, source)["passed"], "Positive fixture failed")
        bad_cases = {
            "missing_camera": lambda f: f.__delitem__("vision/cam_right_wrist"),
            "unexpected_depth": lambda f: f.create_dataset("vision/cam_head/depths", data=np.zeros((n, 1, 1))),
            "mutated_nonvision24": lambda f: f["realman_dexterous/measured_state24"].__setitem__((0, 0), 1.),
            "missing_full36": lambda f: f.__delitem__("realman_dexterous_physics/full36_qpos_after_rad"),
            "reversed_clock": lambda f: f["realman_dexterous/sample_time_s"].__setitem__(2, -.01),
            "bad_camera_rotation": lambda f: f["vision/cam_left_wrist/extrinsics_matrix"].__setitem__((0, 0, 0), 2.),
            "wrong_rgb_shape": lambda f: f["vision/cam_head/shape"].__setitem__(0, 240),
            "missing_rgb_marker": lambda f: f["vision/cam_head/colors"].__setitem__(0, b"not-a-jpeg"),
        }
        rejected = []
        for name, mutate in bad_cases.items():
            bad = temp / (name + ".hdf5")
            shutil.copyfile(output, bad)
            with h5py.File(bad, "a") as f:
                mutate(f)
            try:
                validate(bad, root, source)
            except (ValueError, KeyError):
                rejected.append(name)
            else:
                raise AssertionError("Negative fixture unexpectedly passed: " + name)
        return {"passed": True, "positive_fixture_passed": True, "negative_fixtures_rejected": rejected,
                "duplicate_boundary_clock_positive_fixture": True,
                "scope": "Synthetic schema/rejection tests only; no real episode or physical success claimed"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("hdf5", nargs="?")
    ap.add_argument("--source-hdf")
    ap.add_argument("--report")
    ap.add_argument("--robotwin-root", default="/bigdata2/liminghe/VLA-benchmark/RoboTwin")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        print(json.dumps(self_test(args.robotwin_root), ensure_ascii=False, indent=2))
        return 0
    require(args.hdf5 is not None and args.report is not None, "hdf5 and --report are required")
    target = Path(args.report).resolve()
    require(not target.exists(), "Refusing to overwrite existing validation report")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = validate(args.hdf5, args.robotwin_root, args.source_hdf)
        code = 0
    except Exception as exc:
        report = {"passed": False, "validator_source_sha256": sha(Path(__file__)),
                  "hdf5": str(Path(args.hdf5).resolve()), "error": str(exc),
                  "traceback": traceback.format_exc(), "source_refs": SOURCE_REFS}
        code = 2
    report["checked_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"passed": report["passed"], "report": str(target)}, ensure_ascii=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
