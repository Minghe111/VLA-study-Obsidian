"""Bounded control smoke; no publication and no changes to old datasets."""
from pathlib import Path
import argparse
import json
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--arm", choices=["left", "right"], default="right")
ap.add_argument("--pose", choices=["horizontal", "upright", "diagonal"], default="horizontal")
ap.add_argument("--model", type=int, default=13)
ap.add_argument("--out", required=True)
opt = ap.parse_args()
BASE = Path("/bigdata2/liminghe/VLA-benchmark/deploy/realman_gen4/adjust_bottle_20261002")
ROOT = Path("/bigdata2/liminghe/VLA-benchmark/RoboTwin")
OUT = Path(opt.out)
OUT.mkdir(parents=True, exist_ok=False)
sys.path[:0] = [str(BASE), str(ROOT), str(Path(__file__).resolve().parent)]
os.chdir(ROOT)
os.environ.setdefault("ROBOTWIN_RT_DENOISER", "none")
from expert_generation import install_initializer, run_episode
from integrations.realman_rm65b_gen4.native_robot import RealManRobot
from envs.adjust_bottle import adjust_bottle
from envs.utils import create_actor
import numpy as np
import sapien
import transforms3d as t3d
import pickle

# Bootstrap only. No original experiment body is evaluated.
prelude = (BASE / "run_adjust_bottle.py").read_text().split("\ntry:\n Task=")[0]
context = {"__name__": "__main__", "__file__": str(BASE / "run_adjust_bottle.py"), "bootstrap_out": OUT / "bootstrap"}
prelude = prelude.replace("OUT=BASE/f'{opt.task}_{opt.label}_seed{opt.seed}'", "OUT=bootstrap_out")
prelude = prelude.replace("BASE/'ready_home.json'", "BASE/'ready_home_elevated.json'")
saved_argv = sys.argv
sys.argv = [str(BASE / "run_adjust_bottle.py"), "--seed", "7", "--label", "rgb50smoke", "--align-table"]
exec(compile(prelude, str(BASE / "run_adjust_bottle.py") + ":bootstrap", "exec"), context)
sys.argv = saved_argv
args, ready = context["args"], context["ready"]
for side in ("left", "right"):
    args[side + "_embodiment_config"].update(hand_synergy={"open": [1000] * 5 + [0], "close": [650, 670, 610, 610, 887, 0]},
                                             hand_joint_stiffness=20., hand_joint_damping=.2)
args.update(need_plan=True, save_data=False, save_freq=None, save_path=str(OUT))

class SmokeScene(adjust_bottle):
    def load_actors(self):
        self.qpose_tag = 0 if opt.arm == "left" else 1
        self.model_id = opt.model
        if opt.pose == "upright":
            rotation = t3d.axangles.axangle2mat([1, 0, 0], np.pi / 2)
        else:
            angle = -.45 if opt.pose == "diagonal" else 0.
            rotation = t3d.axangles.axangle2mat([0, 0, 1], angle - np.pi / 2)
        # Geometry center is x +/-0.14, not the actor's bottom-origin x.
        model = json.loads((ROOT / f"assets/objects/001_bottle/model_data{opt.model}.json").read_text())
        center = np.asarray(model["center"]) * np.asarray(model["scale"])
        height = .74 + (np.asarray(model["extents"]) * np.asarray(model["scale"]))[2] / 2
        if opt.pose == "upright": height = .74 + (np.asarray(model["extents"]) * np.asarray(model["scale"]))[1] / 2
        desired = np.array([-.14 if opt.arm == "left" else .14, -.105, height + .001])
        origin = desired - rotation @ center
        self.bottle = create_actor(self, pose=sapien.Pose(origin, t3d.quaternions.mat2quat(rotation)),
                                   modelname="001_bottle", model_id=self.model_id, convex=True)
        self.delay(4)
        self.add_prohibit_area(self.bottle, padding=.15)
        self.left_target_pose = [-.25, -.12, .95, 0, 1, 0, 0]
        self.right_target_pose = [.25, -.12, .95, 0, 1, 0, 0]

task = SmokeScene()
original = install_initializer(ready, BASE, task=task)
try:
    task.setup_demo(**args)
    spec = {"arm": opt.arm, "pose_mode": opt.pose, "model_id": opt.model, "smoke_only": True}
    # A few actual renders provide initial/final visual diagnostics while the
    # expert itself keeps fixed 60ms metadata and all-physics evidence.
    from PIL import Image
    def capture(obs, step):
        if step == 0 or step % 1500 == 0:
            Image.fromarray(obs["observation"]["head_camera"]["rgb"]).save(OUT / f"head_{step:06d}.png")
    report = run_episode(task, spec, OUT / "expert", legacy_dir=BASE, capture_callback=capture)
    print("EXPERT_SMOKE_RESULT", json.dumps({k: report[k] for k in ("accepted", "result", "active_arm", "physical_steps", "minimum_hold_bottle_gap_m")}))
finally:
    RealManRobot._initialize_hands = original
    task.close_env()
