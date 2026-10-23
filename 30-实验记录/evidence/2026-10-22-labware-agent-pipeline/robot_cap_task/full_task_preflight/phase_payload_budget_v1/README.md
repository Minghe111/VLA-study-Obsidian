# Phase-aware payload budget: isolated candidate

This directory contains **synthetic CPU interface checks only**. No SAPIEN
scene, physical step, robot motion, native allocator, installed wrapper, or
successful manipulation was tested.

Candidate source: `../../phase_payload_budget_v1.py`.

Interfaces:

- `phase_payload_budget_readback(task)` reads the current public scene/entity
  state, actual mass/COM, collision shapes, capsule poses, dynamic receiver,
  current contact impulses and one unchanged physical tick. It performs no
  scene step or command/state write. Production integration must call this
  interface rather than inventing evidence dictionaries.
- `select_payload_budget(inventory, evidence, closed_readback=...)` is the pure
  selection/accounting function. Explicitly synthetic fixtures can exercise
  its decisions; they cannot establish physical support.
- `result['actual_inventory']` always retains all fourteen actual rigid bodies.
- `result['nominal_control_budget']` retains the original five-field allocator
  data shape. This is a nominal quasi-static control request, not measured
  RIGHT-hand support. `physical_support_proven` is always false.

Selection rules:

1. While the passive cap remains attached, call the original
   `assembly_mass_com_readback(task)` and retain its result field for field.
   Initial looseness or a geometric cap-clear observation does not override
   actual attachment. There is no full-tight / 2.5-turn / manual-phase gate.
2. After release, retain the whole fourteen-body nominal budget until the
   existing full-task whole-cap clearance geometry passes and measured total
   cap/body impulse force is below the existing `1e-5 N` no-load threshold.
   Total impulse magnitude also guards tangential loads; normal load is logged
   separately.
3. After that cap exclusion becomes eligible, exclude an individual capsule
   only when its entire actual capsule geometry fits the actual dynamic
   receiver's 48-wall polygon, floor and rim and it has zero current body-contact
   points. All identities are unique and mass accounting partitions the actual
   inventory. The selection is recalculated each time; reappearing body contact
   or failed containment puts a capsule back in the nominal budget.
4. Missing, stale, duplicate or contradictory required evidence raises. A
   legitimate but insufficient exclusion predicate conservatively retains the
   affected entity. No fixture, support force, cap motor or pose write is added.

The current scene reader checks the physical capsule primitive dimensions and
frame, and re-reads the receiver's actual 49 collision pieces, contact offsets,
floor/rim and polygon planes. It deliberately rejects unrecognized geometry.

Run the checks from any directory:

```bash
PYTHONDONTWRITEBYTECODE=1 \
  /home/admin123/liminghe/miniconda3/envs/robotwin/bin/python -B -u \
  /home/admin123/liminghe/VLA-WAM-RL/30-实验记录/evidence/2026-10-22-labware-agent-pipeline/robot_cap_task/full_task_preflight/phase_payload_budget_v1/check_interface.py
```

`readback.json` records 32 passed synthetic checks, the exact source hashes and
zero physical steps. The `synthetic_*_result.json` outputs are explicitly
labelled fixtures. Closed compatibility includes an attached initial loose cap
whose geometric clearance predicate already passes.

Remaining validation belongs to the root integration: native LP inputs and
actual bounded SDK targets, real contact/hold stability after release, complete
cap removal and all twelve capsules into the free dynamic receiver, official
planning/reset/replay/export and independent three-RGB/state24/driver24/60-ms
readback. Gravity-only COM budgets do not model accelerations, impacts or
left/right support sharing and cannot replace those physical gates.
