# Preloose payload allocator binding: CPU interface candidate

Candidate: `../../preloose_payload_allocator.py`.

Use one installation on a fresh private `adapted_native()` module, before
constructing the teacher:

```python
from preloose_payload_allocator import install_preloose_payload_allocator

receipt = install_preloose_payload_allocator(task, native, config)
# Construct the normal native teacher afterwards.
```

This call is used **instead of separately calling**
`assembly_contact_allocator.install_assembly_allocator`. It calls that original
installer internally, retaining its source identity, configuration and native
controller restoration checks. It then replaces only the already-private
native module's allocator binding. The original installer module's globals are
never rebound, and its source is not modified.

Attached/closed calls delegate to the original closed allocator wrapper and
check all original readback fields for equality. Initial looseness, including
an attached cap already passing the geometric clearance predicate, does not
change this branch. Released calls use the current
`phase_payload_budget_readback` nominal mass/COM and gravity-force fields to
call the captured original native allocator implementation. Original friction
cones, caps, minima, previous allocation and target method remain in use.

The existing `task.assembly_allocator_trace` remains the main allocation log.
Each row additionally contains `phase_payload_budget`; the compact
`task.phase_payload_budget_trace` records allocation indices and nominal
selections. `physical_support_proven` remains false; no LP outcome is a grasp,
release or pour success predicate.

Explicit restoration is available only after the teacher has been explicitly
discarded:

```python
from preloose_payload_allocator import restore_preloose_payload_allocator

restore_preloose_payload_allocator(task, native)
```

Restoration refuses mismatched native/task identities, foreign replacements,
changed controller/allocator bindings or an existing teacher. A failed install
transaction restores the captured private attributes. It does not overwrite
shared native source, hardware state, scene geometry or object state.

CPU checks:

```bash
PYTHONDONTWRITEBYTECODE=1 \
  /home/admin123/liminghe/miniconda3/envs/robotwin/bin/python -B -u \
  /home/admin123/liminghe/VLA-WAM-RL/30-实验记录/evidence/2026-10-22-labware-agent-pipeline/robot_cap_task/full_task_preflight/preloose_payload_allocator_v1/check_interface.py
```

`readback.json` records **38 passed synthetic checks and zero physical steps**.
The checks execute the actual original installer on **synthetic private module
objects**, with a fake controller/allocator and fake public scene interfaces.
They do not load SAPIEN, the actual native module, or a numeric native LP. They
cover attached field equality, initially loose attached caps, released mass/COM
inputs, current capsule/cap contacts, unique inventory mass accounting, frozen
configuration, installation rollback and guarded restoration.

Root still needs to validate actual attached SAPIEN readback, actual native LP
inputs and execution, then real release/hold/removal/pour. The module only
prepares that interface; no such physical outcome is claimed here.
