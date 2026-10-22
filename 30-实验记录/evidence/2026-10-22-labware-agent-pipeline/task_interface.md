# RealMan labware task interface

This task is an independent `Base_Task` subclass named `labware_pick_place`.
It is simulation only and does not import a physical robot client.

The minimum task actor is `self.bottle`, metadata asset `reagent_bottle_body`,
model ID 0: an **empty open derived reagent bottle**, OD 50 mm, height 90 mm.
Screw-cap mechanics and tablets are not used in this first robot task.

`create_task(task_name='labware_pick_place', runtime_config=None, run_dir=None,
**kwargs)` returns an **uninitialized** `(task, args)` pair for the original
Agent / collector to call `setup_demo(**args)`. The factory selects the actual
`envs_gen.gpt_labware_pick_place.gpt_labware_pick_place` class when available;
otherwise it returns the reference task and does not claim generated provenance.

## Explicit embodiment helper

`self.grasp_lift_place(actor=self.bottle, arm_tag='right')` is a **local RealMan
SDK24 extension**, not an original RoboTwin general API. The official Agent
prompt must explicitly register this description. The helper uses actual arm
and hand drives, independent CPU IK / full-mesh approach checks, actual contact
feedback, actual lift and release. No fixture or object pose reset is used.

The native successful source is frozen under
`/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/acob_fullsim_5090_rm65b_v1/code/RoboTwin`.
Private adaptations generalize its seven-convex-part assumption to the
imported 49-part vessel and its old local +Y axis to imported +Z. The new small
vessel uses bounded direct measured-force control; the legacy LP remains an
advisory diagnostic and is not asserted feasible. No original
source or hand dynamics are changed. Every actual run writes a source receipt.

## Data interfaces

- `task.get_state24()` returns actual measured state, ordered left arm 6, left
  hand SDK 6, right arm 6, right hand SDK 6.
- `task.observe_rgb()` returns state, `physics_tick`, and actual `full_image`,
  `left_wrist_image`, `right_wrist_image` RGB8 640 x 480 arrays. Camera mounts are
  simulation presets, not physical calibration.
- `task.execute_command24(target24, phase='replay')` forwards the requested
  24-D command to the retained `ClockedTask24Adapter`. Each target occupies
  exactly 15 x 4 ms ticks; no command clipping and no action = next-state fiction.
- `on_transition(before_obs, command24, after_obs, row)` is called after each
  actual 60 ms interval. `capture_rgb=True` includes the three images in both
  observations. `row` includes phase, measured state / next_state, actual
  requested action, tick_start / tick_end, physical metrics, and the declared
  release / retreat stage flags.
- `task.transition_log` contains the same compact numeric rows.
- `replay_commands` accepts these saved rows and executes the exact targets
  without invoking the expert planner again. Release and retreat flags only
  inform the success observer; all contact, lift, position and stability checks
  remain measured from physics.

## Success boundaries

Success requires an actual >= 20 mm lift, opposed thumb/finger contact, no
supporting table contact and upright holding for >= 0.5 s, followed by release,
hand retreat, position within 25 mm of the original spot and stable table
support for >= 0.5 s. A reset-only or object-only fixture test cannot pass this
contract. Initial small XY randomization is +/- 5 mm around the right near-arm
position. Generalization, screw-cap manipulation, pouring and physical hardware
are separate later validations.

## Small-vessel posture probe (2026-10-22)

The first reference configuration was rejected by the independent collision
auditor: factory-open little finger met the table. A second candidate with
SDK300 for unused fingers cleared the table but met the bottle. A bounded
3-height x 8-finger-SDK CPU end-pose table then identified SDK0 (fully folded
unused little/ring fingers) and a 15 mm upward grasp-pose offset with actual
mesh distances of 12.0 mm to the vessel and 15.35 mm to the desk. This is a
kinematic candidate only until the corresponding actual drive probe completes.
`task_probes/grasp_geometry_probe.json` records all candidates.

The v5 actual contact trace reached all three main fingers and opposing normals,
but the legacy wrench LP was infeasible at the original 80 g engineering mass.
The old controller therefore retained only 0.12 / 0.12 / 0.24 N and never lifted.
For v6, a declared local controller uses measured force targets 1.2 / 1.2 / 2.4 N
(within the old main-finger 1.5 / 1.5 / 2.5 N caps), the retained SDK latch and
0.16 rad/s position slew cap, and five-boundary 80% observed-force readiness.
LP failure remains recorded and is not relabelled feasible. These force values
are targets for a position feedback servo, never external applied object forces.
Actual lift/hold/release/stability criteria are unchanged.

## Actual robot-drive acceptance (v6)

`task_probes/bottle_probe_v6/result.json` reports `error=null`, `success=true`.
The actual robot issued 390 SDK24 targets and advanced 5,850 physics ticks
(23.4 seconds). The collision bottom reached 25.42 mm above the desk; actual
opposed airborne holding lasted up to 0.844 s. After opening and hand retreat,
the bottle was stable on the desk for 0.54 s, with finger normal force 0 N, table
support 0.7848 N, final XY displacement 16.86 mm and upright tilt 0 degrees.
The old LP also became feasible after the new contact controller built pressure;
its earlier infeasible result is preserved as a failure, not rewritten.

`independent_trace_audit.json` independently reads the saved rows: state chains
match exactly, all targets occupy 15 contiguous ticks, and the target vector
differs from next measured state (maximum difference 31.46 in mixed SDK/rad
units). This validates a real command/state distinction, not a physical error
metric. `frozen_success_sources.json` records the tested task and expert SHA.

This probe does not save the RGB dataset or call the official Agent itself. The
parent workflow must run the official code-testing loop, then collect and
validate an actual three-RGB SDK24 HDF5 episode. No claim is made for other
vessels, autonomous screw-cap manipulation, tablet pouring, task SR or hardware.
