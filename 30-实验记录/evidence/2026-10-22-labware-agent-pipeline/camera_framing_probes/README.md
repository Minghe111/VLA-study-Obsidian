# Head-camera framing correction

The original v1 data placed the bottle at the lower image edge. This was an
observation-quality limitation despite correct SDK24 state/action export.

A single real simulator reset was followed by four actual camera-render
candidates: additional head optical down pitch 0, 15, 20 and 25 degrees, keeping
vertical FOV 37 degrees, 640 x 480 output, original near/far clipping, and both
wrist camera transforms unchanged. After reset, no additional physics step,
robot joint/velocity write or object pose write was performed. The original
frozen task, expert, manufacturer URDF and original camera YAML were unchanged.

The optical local pose is stored in the private camera instance's `local` and
`mounts` dictionaries, so every subsequent `update_mount` preserves the preset.
This is an explicitly named simulation runtime preset, not physical D435
calibration and not a change to the manufacturer's mechanical head pose.

`reset_candidates_v1/candidate_receipts.json` records actual camera world poses,
intrinsic matrices, repeated-update checks, and projected vessel geometry.
`head_framing_candidates.png` uses actual newly rendered images with a red
projected geometry box; it is not an edited replacement for collected RGB.

Selected `selected_head_framing.json`: extra down pitch **15 degrees**, vertical
FOV **37 degrees**. The original whole-vessel projection extended 51.8 pixels
below the reset image; over the actual v1 target trajectory the worst margin
was -61.0 pixels. The selected preset keeps every projected vessel vertex inside
with a worst trajectory margin of **142.0 pixels**. Wrist RGB frames were also
byte-identical between candidate renders in this static reset.

Use `apply_camera_framing(task, config=None)` from `camera_framing.py` after task
setup and before the first sampled RGB observation. It returns a runtime receipt
for the collector. New final dataset RGB must be captured by actually replaying
the robot targets and rendering these optics; the old v1 images are not cropped,
filled, or relabelled as new observations. Frustum inclusion does not guarantee
an unoccluded vessel when the fingers surround it.
