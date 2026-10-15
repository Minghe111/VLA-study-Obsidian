"""Immutable RGB50 native-clock interface for official OpenPI pi05.

No simulator, pickle, hardware IO, RGB re-encoding, training, resampling or
extra policy control. The original 166 source and the 161 training-file mirror
share the exact source manifest bytes; only this derived index is new.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

MANIFEST_SHA256 = "2339d5749b650761011e3975cbeb8d42ceb31dbff4f1f2aba6d71cac9a3edf80"
OPENPI_COMMIT = "215abfb217dbac7d5f1273282331b9b1866c0479"
REPO_IDS = {"realman-rgb50-full50-native-v1/train": "train"}
ASSET_ID = "realman_rgb50_full50"
CAMERAS = {"cam_head": "base_0_rgb", "cam_left_wrist": "left_wrist_0_rgb",
           "cam_right_wrist": "right_wrist_0_rgb"}
ARM_MASK = np.array([True]*6 + [False]*6 + [True]*6 + [False]*6)
SDK_MASK = ~ARM_MASK
TRAIN_INDICES = list(range(50))
HORIZON = 16
PAIR_COUNT = 11098
ANCHOR_COUNT = 10348
STATE_FIELD = "realman_dexterous/measured_state24"
ACTION_FIELD = "realman_dexterous/next_command24"
TIME_FIELD = "realman_dexterous/sample_time_s"
NEXT_TIME_FIELD = "realman_dexterous/next_sample_time_s"


def require(value, message):
    if not bool(value):
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def safe_child(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    require(not relative.is_absolute() and ".." not in relative.parts,
            "Source path must be confined relative path")
    candidate = (root/relative).resolve()
    require(candidate.is_relative_to(root), "Source path escapes declared root")
    return candidate


def canonical(source, manifest_name="source_manifest.json", *, verify_hdfs=True):
    source = Path(source).resolve()
    path = safe_child(source, manifest_name)
    require(sha(path) == MANIFEST_SHA256, "Source manifest SHA differs from final published50")
    document = load(path)
    expected = {"arm": {"left":25,"right":25}, "model_id":{"13":20,"16":20,"4":10},
                "pose_class":{"upright":20,"horizontal":15,"diagonal":15}}
    require(document.get("complete") is True and document.get("count") == 50
            and document.get("actual_counts") == expected
            and document.get("rgb_only") is True
            and document.get("synthetic_fixture_only") is False
            and document.get("transfer_mode") == "independent_copy"
            and document.get("total_state_action_pairs") == PAIR_COUNT
            and document.get("total_raw_observations") == PAIR_COUNT+50
            and document.get("total_physical_steps") == PAIR_COUNT*15,
            "Published50 source contract differs")
    rows = sorted(document["episodes"], key=lambda row:int(row["index"]))
    require([int(row["index"]) for row in rows] == TRAIN_INDICES,
            "All50 exact unique episode indices required")
    require(len({int(row["seed"]) for row in rows}) == 50,
            "Actual successful sources must have unique seeds")
    for row in rows:
        i = int(row["index"])
        relative = "data/episode_%07d.hdf5" % i
        require(row["physical_success"] is True and row["boundary_used"] is False
                and int(row["state_action_pairs"]) >= HORIZON
                and int(row["raw_observations"]) == int(row["state_action_pairs"])+1
                and int(row["physical_steps"]) == int(row["state_action_pairs"])*15,
                "Incomplete/unsupported source episode")
        payload = row["published_files"][relative]
        hdf = safe_child(source, relative)
        if verify_hdfs:
            require(hdf.is_file() and hdf.stat().st_size == int(payload["bytes"])
                    and sha(hdf) == payload["sha256"], "Published HDF SHA/length differs: "+str(i))
    return document


def physical24(value):
    value = np.asarray(value, dtype=np.float32).copy()
    require(value.shape[-1] == 24 and np.isfinite(value).all(), "Expected finite physical24")
    return value


def scale_sdk(value):
    result = physical24(value)
    result[...,SDK_MASK] /= 1000.0
    return result


def delta_arms(actions, state):
    result = np.asarray(actions, dtype=np.float32).copy()
    result[...,ARM_MASK] -= np.asarray(state, dtype=np.float32)[ARM_MASK]
    return result


def restore_arms(actions, state):
    result = np.asarray(actions, dtype=np.float32).copy()
    result[...,:24][...,ARM_MASK] += np.asarray(state, dtype=np.float32)[:24][ARM_MASK]
    return result


@dataclass(frozen=True)
class RealmanInputs:
    def __call__(self, data):
        require(set(data["images"]) == set(CAMERAS), "Three actual RGB views required")
        images = {}
        for source, target in CAMERAS.items():
            pixels = np.asarray(data["images"][source])
            require(pixels.shape == (480,640,3) and pixels.dtype == np.uint8,
                    "Native RGB8 640x480 required")
            images[target] = pixels
        state = scale_sdk(data["state"])
        require(state.shape == (24,), "Unbatched measured state24 required")
        result = {"image":images, "image_mask":{k:np.bool_(True) for k in images},
                  "state":state, "prompt":str(data["prompt"])}
        if "actions" in data:
            result["actions"] = scale_sdk(data["actions"])
        return result


@dataclass(frozen=True)
class RealmanOutputs:
    def __call__(self, data):
        actions = np.asarray(data["actions"], dtype=np.float32)
        require(actions.ndim == 2 and actions.shape[-1] == 32
                and np.isfinite(actions).all(), "Finite official model chunk32 required")
        result = actions[...,:24].copy()
        result[...,SDK_MASK] *= 1000.0
        return {"actions":result}


def make_data_transforms():
    from openpi import transforms
    group = transforms.Group(inputs=[RealmanInputs()], outputs=[RealmanOutputs()])
    return group.push(inputs=[transforms.DeltaActions(tuple(ARM_MASK.tolist()))],
                      outputs=[transforms.AbsoluteActions(tuple(ARM_MASK.tolist()))])


def shared_codec(xpolicylab_parent):
    """Use the shared marked/legacy-aware RGB decoder without caller swaps."""
    parent = Path(xpolicylab_parent).resolve()
    require((parent/"XPolicyLab/utils/process_data.py").is_file(),
            "Pass RoboTwin root containing XPolicyLab")
    if str(parent) not in sys.path:
        sys.path.insert(0,str(parent))
    from XPolicyLab.utils.process_data import decode_image_bit
    import XPolicyLab.utils.process_data as module
    require(Path(module.__file__).resolve() == (parent/"XPolicyLab/utils/process_data.py").resolve(),
            "Unexpected shared RGB codec import source")
    return decode_image_bit


def build_anchors(entries, split, horizon):
    return [(int(e["index"]),offset) for e in entries if e["split"] == split
            for offset in range(len(e["positive_rows"])-horizon+1)]


class RealmanNativeDataset:
    def __init__(self, data_root, split="train", action_horizon=16, verify_hashes=True,
                 source_override=None, xpolicylab_parent=None):
        self.root = Path(data_root).resolve()
        self.document = load(self.root/"native_index.json")
        require(split == "train" and action_horizon == 16, "Train-only complete H16 required")
        require(self.document["source_manifest_sha256"] == MANIFEST_SHA256
                and self.document["adapter_sha256"] == sha(__file__)
                and self.document["split"]["train_indices"] == TRAIN_INDICES
                and self.document["split"]["validation_indices"] == []
                and self.document["counts"]["train_full_chunks"] == ANCHOR_COUNT,
                "Index identity/train membership differs")
        self.source = Path(source_override or self.document["source_dataset"]).resolve()
        self.xpolicylab_parent = str(Path(xpolicylab_parent or self.document["xpolicylab_parent"]).resolve())
        self.split,self.horizon = split,int(action_horizon)
        self.entries = {int(e["index"]):dict(e) for e in self.document["episodes"]}
        require(sorted(self.entries) == TRAIN_INDICES
                and all(e["split"] == "train" for e in self.entries.values()), "Train-only50 required")
        for e in self.entries.values():
            e["hdf5"] = str(safe_child(self.source,e["source_hdf_relative_path"]))
        self.anchors = build_anchors(self.entries.values(),split,self.horizon)
        require(len(self.anchors) == ANCHOR_COUNT, "Exact10348 H16 anchors required")
        if verify_hashes:
            canonical(self.source,self.document["manifest_name"],verify_hdfs=True)
        shared_codec(self.xpolicylab_parent)

    def __len__(self):
        return len(self.anchors)

    def _resolve(self,index):
        require(isinstance(index,(int,np.integer)) and 0 <= int(index) < len(self),
                "Dataset index outside train split")
        episode,offset = self.anchors[int(index)]
        entry = self.entries[episode]
        rows = np.asarray(entry["positive_rows"][offset:offset+self.horizon],dtype=np.int64)
        require(len(rows) == 16 and np.array_equal(rows,np.arange(rows[0],rows[0]+16)),
                "H16 repeats, skips, or crosses episode")
        return entry,offset,rows

    def provenance(self,index):
        entry,offset,rows = self._resolve(index)
        time = np.asarray(entry["sample_time_s"])[offset:offset+16]
        next_time = np.asarray(entry["next_sample_time_s"])[offset:offset+16]
        return {"episode_index":entry["index"], "spec_index":entry["spec_index"],
                "source_hdf5_sha256":entry["hdf5_sha256"], "source_rows":rows.tolist(),
                "sample_time_s":time.tolist(), "next_sample_time_s":next_time.tolist(),
                "dt_s":(next_time-time).tolist(), "temporal_mask":[True]*16,
                "action_dimension_mask":[True]*24+[False]*8,
                "dimension_mask_is_loss_mask":False, "terminal_padding":False,
                "metadata_not_model_input":True}

    def __getitem__(self,index):
        import h5py
        entry,offset,rows = self._resolve(index)
        decoder = shared_codec(self.xpolicylab_parent)
        with h5py.File(entry["hdf5"],"r") as f:
            state = physical24(f[STATE_FIELD][rows[0]])
            actions = physical24(f[ACTION_FIELD][rows])
            images = {}
            for camera in CAMERAS:
                decoded = np.asarray(decoder(f["vision/"+camera+"/colors"][rows[0]]))
                require(decoded.shape == (480,640,3) and decoded.dtype == np.uint8,
                        "Shared codec did not return native RGB8")
                images[camera] = decoded.copy()
        return {"images":images,"state":state,"actions":actions,"prompt":entry["prompt"]}


def norm_stats(vectors):
    value = np.asarray(vectors,dtype=np.float64).reshape(-1,24)
    require(len(value)>0 and np.isfinite(value).all(), "Finite normalization vectors required")
    mean,std = value.mean(0),value.std(0)
    q01,q99 = np.quantile(value,[.01,.99],axis=0)
    fallback = (q99-q01)<1e-4
    center = np.median(value,axis=0)
    q01[fallback],q99[fallback] = center[fallback]-.001,center[fallback]+.001
    std = np.maximum(std,.001)
    return {"mean":mean.tolist(),"std":std.tolist(),"q01":q01.tolist(),"q99":q99.tolist()},np.flatnonzero(fallback).tolist()


def read_checked_episode(path,row,decoder,*,decode_all=True):
    import h5py
    n = int(row["state_action_pairs"])
    p = int(row["physical_steps"])
    with h5py.File(path,"r") as f:
        state,action = f[STATE_FIELD][:],f[ACTION_FIELD][:]
        t,tn = f[TIME_FIELD][:],f[NEXT_TIME_FIELD][:]
        require(state.shape == action.shape == (n,24) and np.isfinite(state).all()
                and np.isfinite(action).all() and t.shape == tn.shape == (n,), "Physical24 schema differs")
        before = f["realman_dexterous_physics/sim_time_before_s"][:]
        after = f["realman_dexterous_physics/sim_time_after_s"][:]
        require(before.shape == after.shape == (p,), "Dense physical clock length differs")
        dt = float(after[0]-before[0])
        require(abs(dt-.004)<1e-8 and np.array_equal(t[1:],tn[:-1]) and t[0]==0
                and np.all(tn>t) and np.allclose(t,np.arange(n)*15*dt,atol=1e-10,rtol=0)
                and np.allclose(tn,np.arange(1,n+1)*15*dt,atol=1e-10,rtol=0),
                "Original positive60ms sample clock differs")
        bi,ai = np.searchsorted(before,t),np.searchsorted(after,tn)
        require(np.array_equal(before[bi],t) and np.array_equal(after[ai],tn),
                "Sample timestamps do not index actual physics")
        measured = f["realman_dexterous_physics/measured_state24_before"][:]
        commands = f["realman_dexterous_physics/applied_command24"][:]
        require(np.array_equal(state,measured[bi]), "Model input is not current actual measured state")
        require(np.array_equal(action,commands[ai]), "Supervision is not actually applied next endpoint command")
        raw = f["instructions"][()]
        if isinstance(raw,(bytes,np.bytes_)):
            raw = raw.decode("utf-8")
        prompts = json.loads(str(raw))
        require(isinstance(prompts,list) and len(prompts)==1 and isinstance(prompts[0],str)
                and bool(prompts[0].strip()), "HDF task instruction schema differs")
        frames = 0
        for camera in CAMERAS:
            group = f["vision/"+camera]
            require(len(group["colors"])==n and np.array_equal(group["shape"][:],[480,640,3]),
                    "Native current-time RGB count/shape differs")
            require(not any("depth" in key.lower() or "pointcloud" in key.lower() for key in group),
                    "RGB-only camera required")
            for k in range(n) if decode_all else (0,n-1):
                image = np.asarray(decoder(group["colors"][k]))
                require(image.shape==(480,640,3) and image.dtype==np.uint8,
                        "Shared RGB codec frame shape/type differs")
                frames += 1
    return state,action,t,tn,prompts[0],{"index":int(row["index"]),"pairs":n,
        "physics_steps":p,"physics_dt_s":dt,"all_positive_time_rows":True,
        "measured_state_exact_to_dense":True,"next_command_exact_to_dense":True,
        "RGB_frames_decoded":frames,"all_RGB_frames_decoded":decode_all,
        "current_RGB_and_next_endpoint_semantics":True,"last_RGB_is_not_terminal_raw_N":True}


def prepare(source,output,*,manifest_name="source_manifest.json",xpolicylab_parent):
    source,output = Path(source).resolve(),Path(output).resolve()
    require(not output.exists(), "Use a new derived data output directory")
    document = canonical(source,manifest_name)
    decoder = shared_codec(xpolicylab_parent)
    episodes,readbacks,states,targets = [],[],{},{}
    for row in sorted(document["episodes"],key=lambda e:int(e["index"])):
        index = int(row["index"])
        relative = "data/episode_%07d.hdf5"%index
        state,action,t,tn,prompt,readback = read_checked_episode(safe_child(source,relative),row,decoder)
        episodes.append({"index":index,"spec_index":int(row["spec_index"]),
            "seed":int(row["seed"]),"model_id":int(row["model_id"]),"brand":row["brand"],
            "arm":row["arm"],"pose_class":row["pose_class"],"split":"train",
            "source_trial_dir_provenance_only":row["source_trial_dir"],
            "source_hdf_relative_path":relative,"hdf5_sha256":row["published_files"][relative]["sha256"],
            "hdf5_bytes":int(row["published_files"][relative]["bytes"]),
            "prompt":prompt,"original_pairs":len(t),"zero_time_rows_removed":[],
            "positive_rows":list(range(len(t))),"sample_time_s":t.tolist(),
            "next_sample_time_s":tn.tolist(),"duration_s":float(tn[-1])})
        states[index],targets[index] = physical24(state),physical24(action)
        readbacks.append(readback)
    anchors = build_anchors(episodes,"train",HORIZON)
    require(len(anchors)==ANCHOR_COUNT, "10348 full chunks required")
    nstates,nactions,covered = [],[],set()
    for index,offset in anchors:
        anchor = scale_sdk(states[index][offset])
        nstates.append(anchor)
        nactions.append(delta_arms(scale_sdk(targets[index][offset:offset+16]),anchor))
        covered.update((index,k) for k in range(offset,offset+16))
    require(covered == {(e["index"],k) for e in episodes for k in e["positive_rows"]}
            and len(covered)==PAIR_COUNT,"All11098 terminal-tail labels must be covered")
    state_norm,sfallback = norm_stats(nstates)
    action_norm,afallback = norm_stats(nactions)
    canonical(source,manifest_name)
    output.mkdir(parents=True,exist_ok=False)
    counts = {"original_pairs":PAIR_COUNT,"positive_pairs":PAIR_COUNT,"zero_time_duplicates_removed":0,
              "train_positive_pairs":PAIR_COUNT,"validation_positive_pairs":0,
              "train_full_chunks":len(anchors),"validation_full_chunks":0,
              "all_unique_targets_covered":len(covered),"raw_observations":PAIR_COUNT+50}
    index_document = {"schema_version":1,"dataset_variant":"rgb50_full50_train_only_native_v1",
        "source_dataset":str(source),"manifest_name":manifest_name,"source_manifest_sha256":MANIFEST_SHA256,
        "adapter_sha256":sha(__file__),"openpi_commit":OPENPI_COMMIT,
        "xpolicylab_parent":str(Path(xpolicylab_parent).resolve()),"physical_dimension":24,
        "model_dimension":32,"action_horizon":16,"asset_id":ASSET_ID,
        "state_field":STATE_FIELD,"action_field":ACTION_FIELD,"state_order":"Larm6rad,Lhand6SDK,Rarm6rad,Rhand6SDK",
        "arm_delta_mask":ARM_MASK.tolist(),"sdk_divisor":1000,"sdk_mapping_calibrated":False,
        "camera_mapping":CAMERAS,"input_metadata_excludes_arm_brand_pose_seed_and_success":True,
        "split":{"train_indices":TRAIN_INDICES,"validation_indices":[]},"counts":counts,
        "time_resampled":False,"terminal_padding":False,"dimension_mask_is_loss_mask":False,
        "full_chunks_only":True,"source_dataset_modified":False,"episodes":episodes}
    write_new(output/"native_index.json",index_document)
    normpath = output/"assets"/ASSET_ID/"norm_stats.json"
    write_new(normpath,{"norm_stats":{"state":state_norm,"actions":action_norm}})
    provenance = {"train_only":True,"all50_train":True,"train_indices":TRAIN_INDICES,
        "validation_indices_excluded":[],"state_anchor_rows":len(nstates),
        "action_vectors_after_SDK_scale_and_anchor_delta":len(anchors)*16,
        "unique_targets_covered":len(covered),"state_fallback_dimensions":sfallback,
        "action_fallback_dimensions":afallback,"constant_dimension_rule":"q99-q01<1e-4 => median +/-0.001 scaled units; std floor0.001",
        "official_quantile_epsilon":1e-6,"padded_dimensions_in_statistics":False,
        "dimension_metadata_is_loss_mask":False,"norm_stats_sha256":sha(normpath)}
    write_new(output/"normalization_provenance.json",provenance)
    report = {"passed":True,"counts":counts,"source_manifest_sha256":MANIFEST_SHA256,
        "all50_source_indices_are_train":True,"all11098_targets_covered":True,
        "all50_HDF_actual_label_time_checks":True,"all_shared_codec_RGB_frames_decoded":3*PAIR_COUNT,
        "per_episode_readbacks":readbacks,"train_only_normalization":True,
        "source_dataset_modified":False,"pickle_deserialized":False,"hardware_io":False,
        "GPU_used":False,"training_started":False,"official_transform_roundtrip_pending":True,
        "adapter_sha256":sha(__file__),"native_index_sha256":sha(output/"native_index.json"),
        "norm_stats_sha256":sha(normpath)}
    write_new(output/"conversion_validation.json",report)
    return report
