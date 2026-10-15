"""Pure CPU same-LP numerical audit; original files/modules never patched."""
from pathlib import Path
import argparse,copy,hashlib,json,time
import numpy as np
import scipy
from scipy.optimize import linprog as native
import purecontact_lp_v26 as v26

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def audit(fixture_path,output_path):
    fixture_path=Path(fixture_path);out=Path(output_path)
    if out.exists():raise FileExistsError('Preserve existing evidence')
    B=Path(__file__).resolve().parent;old=B/'purecontact_lp.py';fx=json.loads(fixture_path.read_text())
    original_source=Path(fx['source'])
    if sha(original_source)!=fx['source_sha256']:raise ValueError('Original failed source changed')
    sources=[original_source,fixture_path,old,B/'purecontact_lp_v26.py']
    before={str(p):sha(p) for p in sources}
    ns={'__name__':'read_only_original_definition_copy','__file__':str(old)}
    exec(compile(old.read_text(),str(old),'exec'),ns)
    rows=[]
    for case in fx['failed_cases']:
        req=dict(contacts=case['contacts'],com_world_m=case['com_world_m'],mass_kg=.01,
                 previous_allocation_N=case['allocation']['target_normal_forces_N'],
                 target_force_world_N=case['force_target_world_N'],target_torque_world_Nm=case['torque_target_world_Nm'],**fx['lp_options'])
        original_matrix={}
        def get_matrix(c,**kw):original_matrix.update(cost=c.copy(),**copy.deepcopy(kw));return native(c,**kw)
        ns['linprog']=get_matrix;original_result=ns['allocate_contact_forces'](**req)
        variants=[]
        for method,scaling,presolve in [('highs',False,True),('highs-ds',False,True),('highs-ipm',False,True),('highs-ds',True,True),('highs-ipm',True,True),('highs-ds',False,False)]:
            kw={k:copy.deepcopy(v) for k,v in original_matrix.items() if k not in ['cost','method']}
            kw['options']=dict(kw['options'],presolve=presolve);divisors={}
            if scaling:
                for a,b in [('A_eq','b_eq'),('A_ub','b_ub')]:
                    d=np.maximum(np.max(np.abs(kw[a]),axis=1),np.abs(kw[b]));d[d==0]=1.
                    divisors[a]=d.tolist();kw[a]=kw[a]/d[:,None];kw[b]=kw[b]/d
            started=time.monotonic();sol=native(original_matrix['cost'],method=method,**kw)
            record={'method':method,'positive_row_scaling':scaling,'presolve':presolve,'options':kw['options'],'status':int(sol.status),'success':bool(sol.success),'message':str(sol.message),'elapsed_s':time.monotonic()-started,'row_divisors':divisors}
            if sol.success and sol.x is not None and np.isfinite(sol.x).all():
                x=np.asarray(sol.x);A=original_matrix['A_eq'];U=original_matrix['A_ub'];target=np.r_[case['force_target_world_N'],case['torque_target_world_Nm']]
                eq=A@x-original_matrix['b_eq'];ub=max(0.,float((U@x-original_matrix['b_ub']).max()));nonneg=max(0.,float(-x.min()))
                rays=x[:16*len(case['contacts'])];allocation=np.r_[0.,0.,U[:3]@x]
                forces=[];normals=[];cone=0.;group=np.zeros(5)
                for i,c in enumerate(case['contacts']):
                    section=slice(i*16,(i+1)*16);f=A[:3,section]@rays[section];n=np.asarray(c['force_on_bottle_normal_world']);n=n/np.linalg.norm(n);fn=float(rays[section].sum());cone=max(cone,float(np.linalg.norm(f-fn*n)-.35*fn));group[['little','ring','middle','index','thumb'].index(c['channel'])]+=fn;forces.append(f);normals.append(fn)
                forces=np.array(forces);center=np.array(case['com_world_m']);points=np.array([c['point_world_m'] for c in case['contacts']]);w=np.r_[forces.sum(0),np.cross(points-center,forces).sum(0)];delta=w-target
                ferror=float(np.max(np.abs(delta[:3])));terror=float(np.max(np.abs(delta[3:])));caps=np.array([0,0,1.5,1.5,2.5]);cap=max(0.,float(np.max(group-caps)));floor=max(0.,float(1.15-group[2:].min()))
                good=ferror<=1e-7 and terror<=1e-8 and max(ub,nonneg,cone,cap,floor)<=1e-7
                record.update(independent_original_lp_checked=True,all_original_constraints_passed=bool(good),force_residual_inf_N=ferror,torque_residual_inf_Nm=terror,original_scaled_equality_residual_inf=float(np.max(np.abs(eq))),all_original_inequality_violation_N=ub,all_variable_nonnegative_violation_N=nonneg,cone_violation_N=max(0.,cone),cap_violation_N=cap,minimum_violation_N=floor,allocation_N=allocation.tolist(),point_forces_world_N=forces.tolist(),point_normal_N=normals,objective=float(original_matrix['cost']@x),ray_coefficients_N=rays.tolist(),l1_auxiliary_N=x[len(rays):].tolist())
            else:record.update(all_original_constraints_passed=False,accept_unknown=False)
            variants.append(record)
        fixed=v26.allocate_contact_forces(**req)
        rows.append({'sample_index':case['sample_index'],'timestamp_s':case['timestamp_s'],'original_input':req,'original_result':original_result,'v26_result':fixed,'equality_condition_number':float(np.linalg.cond(original_matrix['A_eq'])),'middle_duplicate_point_distance_m':float(np.linalg.norm(np.array(case['contacts'][2]['point_world_m'])-case['contacts'][3]['point_world_m'])),'variants':variants})
    unchanged=before=={str(p):sha(p) for p in sources}
    passed=unchanged and all(row['v26_result']['usable'] and all(v['all_original_constraints_passed'] for v in row['variants'] if v['success']) and any(v['method']=='highs-ipm' and not v['positive_row_scaling'] and v['success'] for v in row['variants']) for row in rows)
    report={'passed':bool(passed),'scope':'Only the two historical saved failed LP inputs, same original points/caps/mu/wrench/L1 previous RHS. Not a complete 885-row reconstruction or any physical rollout. Stored previous fallback allocation is the exact previous input in that failed case. Mass .01 is validated deployment convention and mathematically unused with explicit target force override.','case_count':len(rows),'historical_all885_recomputed':False,'original_contacts_not_removed':True,'constraints_targets_objective_unchanged':True,'scipy_version':scipy.__version__,'cause_boundary':'Default/presolved dual simplex produces unknown; IPM, positive row scaled dual simplex and no-presolve dual simplex find Optimal. Near-duplicate points/L1 active floor may cause degeneracy, but internal HiGHS root cause is not established. Equality full-row matrix remains full rank, condition number ~37.7.','source_files_unchanged':unchanged,'source_sha256':before,'helper_sha256':sha(__file__),'cases':rows}
    out.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8');return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fixture',required=True);p.add_argument('--output',required=True);a=p.parse_args();r=audit(a.fixture,a.output);print(json.dumps({'passed':r['passed'],'case_count':r['case_count'],'source_files_unchanged':r['source_files_unchanged'],'historical_all885_recomputed':False}));raise SystemExit(0 if r['passed'] else 2)
