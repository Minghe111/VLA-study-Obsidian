---
title: 闭盖含片药瓶机器人任务接口与探针
date: 2026-10-22
updated: 2026-10-22
tags:
  - RoboTwin
  - RealMan
  - 仿真验证
status: ongoing
---

# Closed filled-bottle robot stage

This directory belongs to the **next ongoing goal**, after the independently
verified empty-bottle Agent / SDK24 pipeline. It does not alter that frozen task,
expert, dataset, metadata, or the passive component.

`labware_cap_task.labware_closed_cap_pick` subclasses the validated empty-bottle
Base_Task only to add the complete passive dynamic assembly and a new stage
observer. `closed_cap_pick_probe.py --name <fresh_name>` runs a finite real
simulation probe. Its current goal is **pick up the body of the closed filled
bottle and hold it upright**, not unscrewing or pouring and not dataset export.

## Component binding

The imported `robot_cap_assembly.py` is SHA-256
`0785eb4e2a487f4ec4a7c637011e163ceabcce18bbc990fc17980e86b95708d5`.
The body is dynamic at 80 g, the cap dynamic at 12 g, and twelve capsules are
independent dynamic rigid bodies at 0.35 g each. The total gravity load is
therefore 96.2 g. `make_actor_wrappers()` supplies the original named point API
without adding another overlapping body. Body is `self.bottle`, cap is
`self.cap`. Initial position and orientation are set only at creation.

Every bootstrap and actual action-clock physics step calls `before_step()`,
then exactly one actual PhysX step, then `after_step()`. The component applies
paired passive screw/seat/friction reaction forces; it has no rotation motor,
world fixture, cap holder, or prescribed object trajectory. These are a derived
thread surrogate, not calibrated real threaded contact. Tablet containment and
cap release are observed from actual state.

The right hand uses the already tested SDK24 posture, rate-limited measured-force
feedback and independent CPU approach planning. The planner's object obstacle
is enlarged locally to contain the actual body and cap collision parts and all
twelve tablets; capsules use containing convex boxes **only in the CPU planner**.
The physical SAPIEN shapes, force/torque limits, mass and friction are not changed.
Cap-hand contact forces and link names are recorded rather than ignored.

## Stage success

Actual body collision geometry must be >= 20 mm off the table, upright within
15 degrees, with opposing thumb/finger load and zero table support for >= 0.5 s.
During that interval all twelve tablet centres must remain within the declared
closed-container volume; the cap must remain attached, relative rotation within
0.15 rad, relative axial motion within 1 mm, radial error within 1 mm, and tilt
within 0.02 rad. These are explicit probe tolerances, not hardware calibration.
The report retains complete capsule dynamics; centre containment is not a claim
of full watertightness for every possible pill orientation.

No opening/pouring success is filled in: `unscrew_verified=false` and
`pour_verified=false`. Left-hand cap grasp, repeated wrist regrasp, release,
12-tablet pouring into the dish, and an official Agent / SDK24 collected episode
remain later gates of the ongoing goal.

## Initial failure and stability diagnostic

`closed_filled_pick_v1` stopped before any action: generic Base_Task judged only
tablet 11 unstable because its full quaternion changed by > 3 degrees. The
second task version keeps the same bounded 2000+500-step settling window and
records actual centre drift, linear/angular speed, full quaternion drift and
capsule **long-axis** drift. A capsule is invariant to spin around local X (and
axis reversal), so shape stability uses the long axis while retaining positional,
transverse angular-speed and escape checks. No extra damping, friction, pose
reset or silent rejection bypass is used. All initialization diagnostics are
saved in `initialization_stability.json`.

## Actual closed-grasp failure (2026-10-22)

> [!failure] Closed/filled pickup has not passed
> `closed_filled_pick_v2_symmetry_stability` executed 355 real SDK24 targets,
> each for 15 × 4 ms, and stopped at the finite 150-boundary close budget.
> No body lift/hold, cap-opening demonstration or dataset success was claimed.

The original right-hand posture first contacted the **cap**, rather than forming
an opposing body grasp. At the first sampled cap-contact boundary (active tick
3300), index-2 and thumb-4 carried 6.048 N in total. The cap force rose to
32.698 N by tick 3435 while relative cap rotation was 0.1698 rad. Across the
recorded boundaries the body thumb force was always zero, the remaining body
finger force peaked at 1.219 N, and no opposed body contact was observed.
The cap-hand force peaked at 314.274 N. The body ultimately tipped onto its side;
the cap released incidentally during collision. This is a failed body grasp,
not a valid unscrewing skill or evidence that the loaded mass is too large.

All twelve capsule centres remained in the declared volume in this run. That
centre test does not certify every point of every capsule or successful pouring.
The passive thread model stayed geometrically aligned before release, but the
strong collision interval also showed disagreement between orientation change
and integrated relative angular velocity. Independent analysis is required
before treating the large incidental rotation as physically credible unscrewing.

Initialization diagnostics support the capsule symmetry interpretation:
tablet 11's full quaternion drift was 3.789 degrees while its physical long-axis
drift was only 0.0287 degrees, centre drift 0.198 mm and final transverse angular
speed 0.0204 rad/s. The cap's static quaternion drift was 0.0396 degrees. No
initialization pose or dynamics repair was applied.

## Deterministic pre-close diagnostic

`closed_filled_preclose_q36_diagnostic` replayed exactly the first 205 recorded
targets and stopped **before** close. Each action and phase equals the source
entry; each boundary advances 15 ticks; measured state and next-state differ
from the original prefix by exactly zero. No new grasp parameters were tried.

`measured_preclose_q36.json` contains the actual 36 articulation coordinates and
velocities, joint names/limits, root transform, measured right-palm FK and actual
body/cap transforms. These are direct readbacks, not values reconstructed from
SDK mimic rules. `preclose_replay_equality_receipt.json` preserves the comparison
and source SHA. The CLI is:

```bash
python closed_cap_pick_probe.py --name <fresh_name> \
  --preclose-replay <closed_filled_pick_v2_symmetry_stability_absolute_directory>
```

The geometry agent uses this snapshot for CPU-only clearance checks. Lowering
the original open hand can exchange cap collision for table collision, so a
new physical grasp probe must first have an explicit clearance-based candidate.
The existing task, empty-bottle expert and final passive assembly remain frozen.

The subsequent bounded CPU checks found no accepted candidate in the tested
pre-bend SDK 750/800 and 36 roll/height combinations using the actual-q snapshot.
This is a finite-family negative result, not proof of global ungraspability.
The measured contact-link region is near 86–88 mm above the bottle bottom while
the present cap skirt begins at 81 mm; the original hand posture therefore
overlaps the cap region. A separate geometry-only bottle-height counterfactual
is being assessed by the geometry agent. No frozen asset/spec/component has
been changed and no new physical attempt has been launched on that basis.

## Independent115mm candidate

The next candidate is isolated in `profiles/bottle115/`. Its only dimensional
changes are body height90→115mm and closed cap centre93→118mm. The original
90mm spec, assets, empty-bottle task/expert and `robot_cap_assembly.py` remain
unchanged. The CPU grid first found clearance at this tested height increment;
it does not establish a global minimum or physical success.

`labware_cap115_task.Bottle115Assembly` passes the new spec into the frozen
mechanism. The original component already uses that spec for hollow collisions,
cap placement and bearing anchors. Since it hardcodes old visual paths and
metadata, a local actor-builder proxy substitutes newly generated115mm body
and14mm cap GLBs, while a wrapper loads independent metadata. It does not
change the mechanism source, shape dimensions, motor gains or object poses.
The capsule initial placement and passive thread coefficients are retained.

The body mass remains the explicit80g engineering assumption; cap12g and
twelve0.35g tablets are unchanged. Actual principal inertia and centre of mass
are read back in the component contract; no calibrated material claim is made.
The original45mm contact anchor is preserved to reuse the same palm pose.
The selected CPU candidate uses main-finger SDK800 pre-bending before approach,
then the original bounded contact-force feedback. Geometric closed targets
are not used as a physical grasp-success label.

The independent `profile115_construction_audit.json` reports16 passing CPU
construction/metadata checks, zero physics steps and no renderer. It confirms
new visual and collision Z bounds, cap118mm placement, masses and Actor point
transforms. Inherited90mm mesh/sphere-drop evidence is explicitly retained under
`inherited90_reference_not_validated_for_profile`; current115mm physical
validation flags remain pending. The first actual115mm probe is separate and
may still fail.

`closed_cap115_pick_probe.py` logs every4ms body/cap pre/post world pose,
angular/linear velocity, COM transforms, actual SDK24 targets, phase and contact
points/normal/impulse/tangent. An unintended cap-hand contact above5N aborts the
body-only attempt immediately and records the partial target. This early-stop
gate changes no physical success threshold. The probe's `success` is only the
closed-filled **hold substage**, never final unscrew/pour or dataset success.

## 115 mm actual replay and observer correction

The first115mm run failed initialization before any robot target because several
capsules were genuinely rolling. One authorized additional10s natural settling
window passed the same thresholds, with no pose/velocity reset, damping change
or seed search. The first window's readings exactly reproduce the retained
failure. Both attempts and their source snapshots remain available.

`derived115_closed_filled_pick_v2_extended_settle` issued427 targets and raised
the bottle86mm. Its original observer reported failure. Independent dense
contact audit found a naming bug: the frozen observer's substring test
`'table' in other.lower()` counted the **twelve tablet-to-body internal loads**
as table support. The earlier cap-name hypothesis was not the actual source of
the false support in this run. The true support actor was clear of the bottle.
This bug is corrected only in the new115mm observer by exact support/entity ID;
the original empty-bottle source and original failed result remain untouched.

The offline audit found a3.236s strict closed+body-grip window. Later unintended
index contact with the cap raised its axial motion beyond1mm, which is a real
criterion failure and remains recorded. That later state is not the reference
used for the next stage.

`derived115_closed_filled_pick_v3_observer_identity_replay` then **actually**
replayed the original first269 targets and stopped at the new observer's first
stage pass:4035 physical ticks,27.816mm off the table,0.508s opposing contact,
cap axial motion40.74µm, relative spin−0.00514rad and zero cap-hand contact.
Every state/action/next-state entry exactly matches the source prefix; no new
planning or physical parameter change occurred. Its source files are frozen in
`source_snapshot/`, and `independent_stage_geometry_audit.*` verifies source,
clock, driver targets, actual terminal state and the whole12-capsule geometry.

Full capsule containment holds for the same0.508s at the original physics spec's
**0.4mm contact tolerance**. It does not establish zero penetration: strict-zero
containment lasted only0.008s, with worst observed radial/floor margins about
−86.82/−23.31µm. This tolerance must remain explicit in later evidence.

The terminal bottle was still moving upward at29.812mm/s. It is an airborne
grasp reference, not a stationary or rigidly attached bottle. The independent
`labware_cap_workflow.py` skeleton therefore reserves `check_success` for a
future full task and keeps it false, while preserving a separate body-hold
stage gate. `held_static_probe.py` next replays exactly those269 targets and
continues25 ordinary feedback boundaries with the right-arm drive target fixed.
No object state is written. Its finite reference result is distinct from cap
opening, pouring and official full-episode collection.

## Actual finite hold reference and first-contact boundary

`static_hold_probes/held115_static_reference_v1` executed the exact269-target
prefix plus25 real60ms feedback targets. Independent audit verifies the fixed
right-arm drive, continued original grip feedback, actual table clearance,
attached cap and all12 complete capsules at the declared0.4mm contact tolerance.
The final right index-to-cap load is0.499N (peak0.579N), not zero. The true last
0.60s spans11 samples, ticks4260–4410: body-to-palm translation changes1.675438mm
and orientation0.643137°. The original ten-point result covers0.54s and is kept
as historical evidence. This is a mildly moving real reference, not a rigid
attachment, and does not establish cap grasp, opening or pouring.

The independent LEFT candidate uses SDK `[0,0,0,1000,1000,400]`, and the
geometry-screened first-contact limits index560.09765625 / thumb-opposition
348.443359375. These are geometric limits, not successful contact targets.
The original straight joint line intersects the table; the selected seven-node
shortcut has only discrete CPU collision evidence. The runtime reads actual cap
pose at every via, solves LEFT-only IK and checks moving LEFT links against
body, cap, twelve capsules, table and the complete unexempt robot geometry.
Frozen-only right contacts are outside that LEFT path check; no physical
collision masks or broad compound-object exemptions are introduced.

`left_cap_contact_probe.py` builds every new command from actual drive targets
and original right-hand feedback, with no metadata-derived right action. It
reserves the full task success flag as false. The first actual LEFT-cap normal
load exceeding0.001N (4×10⁻⁶Ns at4ms) is a diagnostic stop. The proposed
two-actuator controller uses only0.1N/0.1N targets and never treats its `ready`
diagnostic as cap grasp. A partial60ms window is recorded separately, without
fabricating a15-tick training transition.

The initial `left115_first_contact_v1` run was deliberately interrupted after
independent review identified a measured-versus-drive path-audit discrepancy.
The same exec handle73817 returned130;323 complete targets were saved, with no
LEFT approach or first cap contact. `external_interruption.json` clarifies that
the original `error:null` is not a pass: `KeyboardInterrupt` bypassed its former
`Exception` handler. The next source revision checks the measured→actual-drive
connector and the entire actual-drive→endpoint quintic line, fixes a private
dynamic helper indentation error and records `BaseException` partial windows.
It retains the same geometry, controller targets and physical parameters.

`left115_first_contact_v2` then reproduced the exact294-target prefix and
completed49 real preparation targets. Its first rebound via was rejected
before execution: full moving-left geometry found index-to-table collision
near91.9% of the drive→endpoint line. The actual prepared cap had rotated2.334°;
rebinding the distant table-clearance corridor lowered that via palm by5.947mm.
Independent finite CPU comparison with actual measured hand joints and all five
native table parts found the original world-frame edge clear at6.030mm sampled
gap, while the cap-follow rebound reproduced the collision. No LEFT approach
or closing command was executed in v2.

`left115_first_contact_v3_world_transit` preserves those transit nodes in world
coordinates, and reserves current cap-frame binding for the final approach.
It actually executed162 approach targets across five segments, reaching505
complete60ms targets/active tick7575. All five scoped full-geometry checks
passed, and actual4ms guards observed no unintended contact or LEFT-cap load.
The final whole-cap-bound palm target failed masked IK before that segment was
commanded. An independent finite18-seed CPU check of the same actual-state
target also found no acceptable IK result; returned residuals cannot be used
as commands. This does not establish global unreachability.

At that retained endpoint the cap centre had shifted3.24mm and the cap axis
3.271° relative to the earlier reference, while full orientation changed6.214°.
The bottle remained airborne23.784mm and attached, but measured cap axial motion
1.319mm exceeded the original1mm closed criterion; this is explicitly a
`closed=false` state. There was no first LEFT contact, opening or pouring, and
the full success flag remains false. A subsequent binding comparison treats
the nearly axisymmetric cap's axial yaw separately from its physically relevant
centre and axis; it is CPU planning evidence until a new physical run executes.

`left115_first_contact_v4_axisymmetric` actually replayed all505 v3 targets with
exact state/action equality, then executed19 final approach targets using the
centre/axis binding with world grasp azimuth retained. The pure target function
matches the independent geometry calculation to5.55×10⁻¹⁷. The real controller
constructor/proposals passed the original numerical readback validation.

All160 bounded closing targets executed at the unchanged rate/force limits,
reaching the original screened SDK endpoints. The684-target run ended with no
positive LEFT-cap normal impulse. Independent4ms audit finds32087 matching
LEFT-cap manifolds and72025 points, **all with exactly zero normal impulse**;
these are contact candidates, not physical pressure or grasp. The run remains
failed as a first-load diagnostic; it is not relabelled from those manifolds.

During the9.6s closing interval the actual cap centre moved only0.0395mm and
the axis0.0531°. This contradicts substantial cap drift as the principal cause
of this particular failure. Actual palm tracking residual decreased from0.720
to0.407mm, but independent final geometry also finds a4.793mm thumb-to-cap gap
(4.457mm in an explicitly CPU-only drive-arm counterfactual). Therefore neither
perfect arm tracking nor extra waiting alone is justified as a complete fix.
The compound FCL index collision flag conflicts with the zero-pressure physical
result and is being checked against individual collision pieces and cap-local
geometry. No closure range, friction, mass, success threshold or object pose has
been changed on the strength of that compound flag.

The subsequent independent cap-local vertex, ConvexHull separation and triangle
distance checks resolve this specific conflict: v4 thumb and index are each
disjoint from all33 actual cap pieces. Thumb separation is bounded about
4.443–4.769mm; index4.058–4.135mm. The old original-pose “first-touch” targets
also remained roughly4.1–4.2mm away. A specific FCL piece query returned collision
true while its distance was positive4.905mm. Thus the old binary first-touch
screen stopped early on a false flag in these queries. This is not evidence
that all FCL checks or all robot geometry are wrong.

The next isolated continuation will retain the v4 actual palm and physical
parameters, replay its684 targets exactly, and use an independently checked
finite closure interval derived from actual convex surfaces. It will stop on
the first real normal impulse and record original PhysX `separation`, normals,
impulses and contact positions every4ms. A read-only live collision snapshot
will preserve cap/left-index/left-thumb shapes, scales, poses, vertices,
triangles, filter groups and contact/rest offsets. This planned continuation
has not yet established pressure, cap grip or rotation.

The finite independent continuation candidate is
`true_surface_closure_v4_v2/candidate.json`: index SDK560.09765625→536.09993935
first, then thumb opposition348.44335938→190.54951239. Actual convex face/edge
separating axes, not FCL's collision flag, define the target surface. Its
110 interval samples keep all forbidden pairs clear, with conservative interval
separation at least1.6225mm. The small final negative projections are50µm for
index and16.949µm for thumb; they are CPU geometric margins, not physical load.

`left115_first_contact_v5_true_surface` replayed684 targets exactly, then the
controller rejected the first ordered stage before any continuation action:
its former API required both feedback actuators to have a nonzero interval.
This is an interface failure, not a failed geometric/physical closure. It also
saved actual live collision geometry: cap33 shapes, and13 LEFT index/thumb
parent/sensor components. Cap contact/rest offsets are0.4mm/0; finger offsets
are10mm/0, explaining the old positive-gap zero-impulse manifolds.

The controller's explicit `allow_fixed_feedback_actuator=True` now permits
exactly one fixed paired channel for these ordered stages. Default rejection
and both-fixed rejection remain. Ten independent CPU boundary tests, including
actual archived start q/drive and both stage interfaces, passed before the next
physical run. `left115_first_contact_v6_ordered_true_surface` is the isolated
physical continuation; its final outcome must be taken from its actual receipt,
not inferred from the CPU candidate or controller tests.

The actual v6 continuation **did establish its first normal load** at active
tick10981: LEFT thumb4→cap normal impulse1.88510419×10⁻⁴Ns, equivalent0.0471276N
over the actual4ms step. The raw PhysX separation is+3.7574µm. Contact position
is[0.1167485,−0.1450771,0.8776451]m. The oriented normal on the cap is almost
radially inward (cap-frame radial dot−0.999662), establishing a side contact.
Index load is zero and the pair is not yet opposing or grasping.

The source684 prefix replays exactly, and the receipt contains732 complete
60ms targets plus a separate **one-step4ms diagnostic partial**10980→10981.
The remaining14 steps were not fabricated. No forbidden LEFT-body/table/right
force occurred, and no cap turn/release/pour is claimed. A subsequent pressure
probe will actually reexecute that pending target for a full new15-step window,
checking the reproduced first step against the saved raw physical state.

Installed PhysX/SAPIEN exposes a normal-times-scalar contact impulse report;
its vector does not expose the actual tangential friction impulse. Accordingly
near-zero reported tangential residual is **not** evidence of absent friction.
`contact_wrench_observer.py` labels its fields `reported_*` and sets tangential
observability false. Actual cap/body angular motion and loaded robot contact
must establish any later physical turning result.

`load_turn_probes/pair_load_small_turn_v1` then actually reexecuted all732
complete v6 targets and a new full15-step window containing the old partial.
At10981, all saved q36/qvel36/state24/drive, root/palm/body/cap matrices and
12 tablet poses reproduced exactly. The new complete boundary is10995; the
old one-tick diagnostic was not relabelled as a complete transition.

At the next real60ms window (tick11010), index normal load was0.202153N,
thumb0.144456N and their measured normal dot product−0.959225. This establishes
one actual opposing loaded window, not sustained cap grasp or turning. The
thumb remained at its real completed-window drive190.549518SDK; the independent
index-only micro-preload candidate had been referenced at thumb191.675355SDK,
and the small actual difference is disclosed in `pressure_interval.json`.

The following controller proposal rejected a small observed solver excursion:
thumb2 actual position−3.459237µrad and thumb4−3.801981µrad below nominal zero.
The drive command remained legal. The run preserves734 complete targets,
`partial_command=null`, unchanged source snapshots, and an explicit error.
No turn was executed. A subsequent revision may separate finite raw observation
tolerance from strict command limits; it must not clamp archived observations,
modify physics, or reclassify this original failed receipt as success.

`pair_load_small_turn_v2_observed_tolerance` preserved the same physical
commands through736 complete targets, but the raw thumb2 subsequently reached
−11.7204µrad and exceeded its explicit10µrad observation budget. Its original
error remains. The next isolated v3 uses the controller API's finite100µrad
observation budget; strict drive/SDK/mimic command limits and physical parameters
remain unchanged, and no measured q is clamped.

The actual v3 achieved three consecutive complete opposing loaded boundaries
at11040–11070. It then executed nine robot palm orbit increments totalling a
requested+0.045rad. The tenth same-goal IK call failed before an action was
executed. Over the actual11070→12195 physical interval, world rotation vectors
projected on the initial body axis were approximately+0.0505504rad for LEFT
palm, +0.0188017rad for cap, +0.0391639rad for body, and+0.0006668rad for RIGHT
palm. The resulting measured cap-relative angle changed−0.0203621rad: the
body rolled farther than the cap within the right grasp. This is not evidence
of positive unscrewing or of a cap rotating backwards in world coordinates.

Before this turn, left index/thumb loads were0.108/0.185N while the right index
pushed the cap skirt upwards with0.832N. At the terminal snapshot, left loads
were0.110/0.528N and right cap load0.782N. These are reported normal loads;
the installed binding does not expose friction impulses. No forbidden
LEFT-body/table/right loaded contact, cap release, or tablet pour is claimed.
Future correction must be chosen from actual body-palm slip and load evidence,
and must still act only through robot joint-drive commands.

The isolated v4 body-palm feedback attempt preserved the same real setup and
loading prefix. It ended after its first complete turn-phase window at11085,
before finishing an increment. Although the geometric body error was exactly
zero, absolute IK q was incorrectly used as the right drive target: this
removed the existing drive-minus-measured-q load bias (notably8.0906mrad at
right joint6). Left thumb normal load rose to1.12169N and its primary thumb2
readback reached−128.925µrad, exceeding the unchanged100µrad observation
budget. All739 targets and this failed receipt remain preserved.

The next mapping correction applies `q_IK - actual_q_start` to the current
drive, preserving the holding bias. Exact zero body error keeps the drive
unchanged. Strict limits are checked on the resulting command, and right
drive velocity remains capped0.45rad/s. The body-origin geometric correction
is at most0.5mm/3mrad per boundary; the palm also travels around the body
origin and can translate farther than0.5mm. The probe does not invoke a
frozen-right collision exemption: both arms move through actual drives and
their observed4ms contacts remain guarded. No result of this unexecuted
mapping revision is inferred from its arithmetic alone.

The actual v5 incremental mapping preserved the zero-error drive exactly. It
completed one increment, then four complete60ms windows of the second before
the same100µrad observation budget rejected thumb2 at−103.391µrad. All749
targets are complete; `partial=null` does not mean the second increment had
no actions. RIGHT body yaw drift decreased versus the matching-time baseline,
but cap world rotation became negative and right cap normal load increased.
This did not establish positive unscrewing. Commanded right drive slew
(peak0.036515rad/s) must not be confused with measured joint velocity: the
independent raw audit found a transient right thumb3 qvel around8.35rad/s.

`right_index_unload_probes/right115_index_close_v1` subsequently reproduced the
same loaded reference exactly and ran a separate finite contact diagnostic.
LEFT12 and RIGHT arm6 drives were fixed at that actual reference; original
RIGHT middle/thumb body-contact feedback remained active. Each index override
synchronized native latch/free-search state to actual drive before proposal
and after a completed15-step window. No measured state or force was forged.

Three zero-index-change windows measured right cap normal loads0.833/0.866/
0.822N. Relative to the actual627.208819SDK baseline, −1SDK measured0.772/
0.793/0.854N, −2SDK0.913/0.929/0.954N, and−4SDK1.329/1.181/1.235N. RIGHT
index-body load increased to2.34–2.58N at−4SDK without reaching the explicit
3N diagnostic stop. The static geometric direction did not yield physical
unloading. This run exited normally as a completed diagnostic, with no turn,
no unloading success claim, and no full-task success. It preserves raw q/qvel,
contacts, capsules and a separate actual target/state chain; original failure
records and the empty-bottle pipeline remain unchanged.

`right_lower_grasp_probes/right115_lower10_initial_pick_v1` ran a separate
private teacher with `hand_x=-.005` instead of `-.015`. The native Z-up grasp
arithmetic lowers the palm by10mm with the same orientation. This consumes
the finite airborne geometry hypothesis; the real native planner separately
checked and executed the initial approach. The existing shared teacher,
115mm assembly, hand waypoints, contact feedback and physics were unchanged.
All setup q/qvel, drives, body/cap/tablet poses matched the prior initial
snapshot exactly. Source SHA1008e5474f5a5b90a439548b0ea15bd6f7066b3ac09c92253ff833717bb7a66f
is frozen in the run's source snapshot.

The actual closed/opposed airborne body stage was reached at tick4050 after
270complete targets. Ten further complete60ms windows held the actual arm
drive fixed and retained the same stateful body-contact feedback. The run
exited0 with280complete targets,4200ticks and no partial. This is diagnostic
completion, not successful cap removal, pouring or a stable unloading result.

RIGHT-cap boundary load was initially0 through tick4170, then0.1583N at4185
and0.4898N at4200. Across the actual0.60s interval, the body moved world
[-7.094,+2.768,-2.226]mm and rotated7.964deg, while the right palm moved only
[+.038,-.037,+.132]mm and rotated0.0319deg. The bottom clearance reduction
24.559→18.991mm therefore includes tilt and must not be described entirely
as vertical translation. The bottle remained closed and the complete capsule
boundary checks read12 inside at the declared0.4mm contact tolerance. Dense
geometry and contact timing are independently audited from original logs.

The native contact LP's mass/COM readback concerns the80g bottle body only;
the actual closed assembly also carries a12g cap and12×0.35g capsules. A
body-only LP feasible flag is not an equilibrium certificate for that96.2g
assembly. The current adapted servo uses direct1.2/1.2/2.4N main-finger
targets with the LP advisory. This limitation and the observed body/palm
rotation are retained as evidence for the next bounded grasp revision.

The independent dense audit subsequently confirmed all150static steps were
closed/opposed, with no table or forbidden loads and all300pre/post complete
capsule states inside the0.4mm tolerance. RIGHT-cap first positive load was
tick4182 (+0.528s),0.020414N at right index2; the first131static steps
(0.524s) had exactly zero cap normal load. The body/right-palm relation
changed8.05071mm/7.97366deg. At the4170boundary the current20mm airgap
condition was already false even though the historical stage-success flag
remained true. This is not a static reference or a sustained unloading pass.

Source-control-flow caveat: the frozen wrapper resets `force_targets` before
and after `super().target(carry=False)`, but native `target` also invokes the
LP when its profile has ended and contact is paired. A usable LP can therefore
replace the targets inside that call before the SDK feedback result is
returned. Resetting them afterwards does not undo the SDK command. Thus
"LP advisory" in the report describes acceptance intent, not strict isolation
of LP effects from every command. A future assembly-aware controller must
explicitly route the intended allocation into feedback; merely changing the
outer `direct_targets` is insufficient. The frozen sources and actual traces
are retained rather than silently repaired.

`right_lower_grasp_probes/right115_lower10_frozen_drive_v1` isolates the
post-stage feedback effect. It physically reexecuted the original270targets
and matched all4050dense pre/post assembly states and actual drives, plus
all270boundary states/q36/qvel and the complete4050stage snapshot exactly.
No teacher or contact-controller proposals were called in the comparison
interval: every subsequent requested24D target was the same actual stage
drive readback. Physical gains, materials, object state and guards were unchanged.

Four complete static windows were followed by a real ten-step partial window
ending at tick4120 (+0.280s). RIGHT-cap normal load10.923805N triggered the
original5N guard. At that point body/right-palm displacement was6.5516mm and
rotation6.6756deg; body-thumb load8.2021N and finger-body load0 meant current
opposition was lost. The cap remained closed and all12centres were still
inside, but this is not a hold success. The final incomplete action is kept
as ten actual4ms steps and is never filled to15 or exported as a complete
training transition. The source SHA730224c9b3ad5266378e832e3d8fe3651579f2cc20c8fa232fa8ad9dc67a6c63
and all original traces are retained. Removing feedback did not cure the
instability; it produced an earlier loss of opposition and cap impact.
