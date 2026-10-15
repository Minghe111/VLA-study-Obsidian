"""v13 actual trace reference loader: no extra hold/advance or fabricated state.

A successful official closing action already includes native gripper padding.
The true LAST closing sample is the initial carry reference. Its explicit step
must match pre_retreat_grip.reference_step and be followed immediately by a real
recorded after_grasp sample. Original pose/FK helpers and v11 sources are retained.
"""
from pathlib import Path
import json
import numpy as np
from observed_trace_audit import _matrix,_relative_metrics,_angle_deg,_phase


def _load_trace(trace_path,report_path):
    trace=json.loads(Path(trace_path).read_text(encoding='utf-8'))
    report=json.loads(Path(report_path).read_text(encoding='utf-8'))
    if not isinstance(trace,list) or not trace:raise ValueError('Expected nonempty actual trace')
    for row in trace:
        step=row.get('step')
        if isinstance(step,bool) or not isinstance(step,int) or step<1:raise ValueError('Actual physical step must be a positive integer')
    steps=np.array([row['step'] for row in trace])
    if np.any(np.diff(steps)<=0):raise ValueError('Recorded actual steps must strictly increase')
    grip=report.get('pre_retreat_grip',{})
    if grip.get('passed') is not True:raise ValueError('Historical pre-retreat grip failed')
    reference_step=grip.get('reference_step')
    if isinstance(reference_step,bool) or not isinstance(reference_step,int) or reference_step<1:raise ValueError('Missing exact positive pre-retreat reference_step')
    closes=[i for i,row in enumerate(trace) if _phase(row)=='close']
    if not closes:raise ValueError('Missing real historical closing samples')
    anchor=closes[-1]
    if steps[anchor]!=reference_step:raise ValueError('Pre-retreat reference_step differs from true LAST close sample')
    if any(_phase(row)=='after_grasp' for row in trace[:anchor+1]):raise ValueError('Initial reference cannot be reset after carrying already began')
    if anchor+1>=len(trace) or _phase(trace[anchor+1])!='after_grasp' or steps[anchor+1]!=reference_step+1:raise ValueError('Last actual close must be immediately followed by next physical after_grasp step')
    after=[i for i,row in enumerate(trace) if i>anchor and _phase(row)=='after_grasp']
    if any(_phase(row)!='after_grasp' for row in trace[anchor+1:]):raise ValueError('Unexpected recorded phase after initial carry reference')
    reference=_matrix(grip['actual']['relative_bottle_palm'])
    recorded=trace[anchor].get('grip',{}).get('relative_bottle_palm')
    if recorded is not None:
        actual=_matrix(recorded)
        if np.linalg.norm(actual[:3,3]-reference[:3,3])>2e-5 or _angle_deg(actual,reference)>.01:raise ValueError('Direct last-close grip differs from report initial reference')
    for row in trace:
        for side in ('left','right'):
            q=np.asarray(row['arm_qpos'][side],dtype=float)
            if q.shape!=(6,) or not np.isfinite(q).all():raise ValueError('Invalid recorded actual six-arm state')
        _matrix(row['bottle']['pose'])
    return trace,report,anchor,after,reference
