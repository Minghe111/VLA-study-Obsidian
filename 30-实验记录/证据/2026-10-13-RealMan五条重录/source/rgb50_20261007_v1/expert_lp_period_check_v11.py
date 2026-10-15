"""Zero-step scope and elapsed-failure-clock verification; no robot motion."""
from pathlib import Path
import ast,copy,json,numpy as np
import expert_generation_v11 as v11
from expert_generation_v9 import ExpertSession as Baseline
path=Path(v11.__file__).parent.parent/'adjust_bottle_20261002/contact_capture_control_v26.py'
tree=ast.parse(path.read_text(encoding='utf-8'))
original=copy.deepcopy(tree)
transformer=v11._CompositeTransformer('left')
selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('_feedback_rate_limits','install_contact_capture')]
changed=[transformer.visit(copy.deepcopy(n)) for n in selected]
count=(transformer.period.period_sites,transformer.period.failure_clock_sites,transformer.period.recovery_sites)
assert count==(1,1,1)
modified=ast.Module(body=changed,type_ignores=[]);ast.fix_missing_locations(modified)
def calls(nodes,name):
    return [ast.dump(n,include_attributes=False) for root in nodes for n in ast.walk(root)
            if isinstance(n,ast.Call) and ast.unparse(n.func)==name]
original_selected=[n for n in original.body if isinstance(n,ast.FunctionDef) and n.name in ('_feedback_rate_limits','install_contact_capture')]
for name in ('allocate_contact_forces','SDKWaypointProfile','_feedback_rate_limits'):
    assert calls(original_selected,name)==calls(changed,name)
assert any(isinstance(n,ast.Compare) and ast.unparse(n)=="state['actual_pair_streak'] >= 50" for n in ast.walk(modified))
assert v11.ExpertSession.step is Baseline.step and v11.ExpertSession.plan_path is Baseline.plan_path
allocation_if=next(n for n in ast.walk(modified) if isinstance(n,ast.If) and ast.unparse(n.test)=="lp['usable']")
function=ast.FunctionDef(name='review_clock',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='state'),ast.arg(arg='lp'),ast.arg(arg='dt')],kwonlyargs=[],kw_defaults=[],defaults=[]),body=[copy.deepcopy(allocation_if)],decorator_list=[])
module=ast.Module(body=[function],type_ignores=[]);ast.fix_missing_locations(module)
environment={'np':np};exec(compile(module,'v11_clock_check_only','exec'),environment)
clock=environment['review_clock'];elapsed=[]
for start in range(1,101):
    state={'control_steps':start,'lp_failure_steps':0,'lp_failure_first_control_step':None}
    clock(state,{'usable':False,'reason':'clock fixture'},.004)
    assert state['lp_failure_steps']==0
    for step in range(start+1,start+116):
        if step%15:continue
        state['control_steps']=step
        try:clock(state,{'usable':False,'reason':'clock fixture'},.004)
        except RuntimeError:
            elapsed.append((step-start)*.004);break
    else:raise AssertionError('Missing timeout detection')
    assert .4<=elapsed[-1]<.46
    clock(state,{'usable':True,'target_normal_forces_N':[0.,0.,1.15,1.15,2.3]},.004)
    assert state['lp_failure_steps']==0 and state['lp_failure_first_control_step'] is None
result={'passed':True,'scope':'AST feasible-region call identity and timeout arithmetic only; not physical tracking or rollout success',
    'changed_site_counts':{'period':count[0],'failure_clock':count[1],'recovery':count[2]},
    'LP_allocator_SDK_profile_and_vendor_rate_limit_calls_identical':True,
    '50_step_real_contact_pair_streak_unchanged':True,
    'physical_guards_and_planning_methods_identical':True,
    'failure_clock_start_cases':len(elapsed),'minimum_actual_timeout_s':min(elapsed),
    'maximum_actual_timeout_s':max(elapsed),'nominal_failure_timeout_s':.4,'detection_resolution_s':.06}
print(json.dumps(result,indent=2))
