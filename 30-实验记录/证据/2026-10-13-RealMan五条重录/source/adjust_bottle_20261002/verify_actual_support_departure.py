"""Pure CPU readback of recorded actual support depth and table normal impulses.

No SAPIEN/FCL imports, worker queue, new planning, physics, rendering or robot IO.
Signed-gap checks and physical normal-contact evidence are reported separately:
positive gap is not a claim of zero PhysX contact-offset impulse. Missing contact
telemetry is marked unavailable, never interpreted as zero.
"""
from __future__ import annotations
from pathlib import Path
import argparse,hashlib,json,math
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def iter_rows(path,chunk_size=1024*1024):
    decoder=json.JSONDecoder()
    with Path(path).open('r',encoding='utf-8') as f:
        buf='';eof=False
        def fill():
            nonlocal buf,eof
            x=f.read(chunk_size)
            if x:buf+=x
            else:eof=True
        fill();buf=buf.lstrip()
        if not buf.startswith('['):raise ValueError('Expected JSON trace array')
        buf=buf[1:];first=True
        while True:
            buf=buf.lstrip()
            while not buf and not eof:fill();buf=buf.lstrip()
            if buf.startswith(']'):
                buf=buf[1:]
                while not eof:fill()
                if buf.strip():raise ValueError('Trailing trace JSON data')
                return
            if not first:
                if not buf.startswith(','):raise ValueError('Missing trace comma')
                buf=buf[1:].lstrip()
            while True:
                try:r,end=decoder.raw_decode(buf.lstrip());buf=buf.lstrip();break
                except json.JSONDecodeError:
                    if eof:raise ValueError('Truncated physics trace')
                    fill()
            if not isinstance(r,dict):raise ValueError('Nonobject trace row')
            buf=buf[end:];first=False;yield r


def table_impulses(row):
    if 'bottle_actual_contacts' not in row:return None
    total=0.;points=[]
    for contact in row['bottle_actual_contacts']:
        if contact['other_link'].lower()!='table':continue
        ns=float(contact['normal_impulse_Ns'])
        if not math.isfinite(ns) or ns<0:raise ValueError('Invalid actual table normal impulse')
        total+=ns
        if 'points' in contact:
            summed=0.
            for p in contact['points']:
                v=float(p['normal_impulse_Ns']);summed+=v
                if not math.isfinite(v) or v<0:raise ValueError('Invalid table point normal impulse')
                if 'impulse_Ns' in p and 'normal_world' in p:
                    actual=abs(float(np.dot(p['impulse_Ns'],p['normal_world'])))
                    if abs(actual-v)>1e-10:raise ValueError('Point impulse projection mismatch')
                if v>1e-7:points.append({'normal_impulse_Ns':v,'separation_m':p.get('separation_m')})
            if abs(ns-summed)>1e-10:raise ValueError('Table contact aggregate mismatch')
    return {'normal_impulse_Ns':total,'active_points':points}


def analyze(episode_dir,geometry_path,report_path=None,dt_s=None):
    episode=Path(episode_dir);geometry_path=Path(geometry_path)
    report_path=Path(report_path) if report_path else next((episode/n for n in ['replay_report.json','attempt_report.json'] if (episode/n).exists()),None)
    if report_path is None:raise FileNotFoundError('No completed report')
    trace_path=episode/'physics_trace.json';sources=[report_path,trace_path,geometry_path]
    stamps={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in sources}
    report=json.loads(report_path.read_text());mesh=json.loads(geometry_path.read_text())
    guard=mesh['physical_signed_height_guard'];verts=np.asarray(guard['bottle_collision_vertices_local_m'],dtype=float)
    bounds=np.asarray(guard['table_world_bounds_m'],dtype=float);top=float(guard['table_top_world_z_m'])
    if verts.ndim!=2 or verts.shape[1]!=3 or not len(verts) or not np.isfinite(verts).all() or bounds.shape!=(2,3) or not np.isfinite(bounds).all():raise ValueError('Invalid accepted actual native geometry')
    if abs(top-.74)>1e-6 or abs(guard['release_gap_m']-.001)>1e-12:raise ValueError('Unexpected table/release budget')
    initial=report['support_departure_initial'];anchor=int(initial['configured_at_step']);stored_baseline=float(initial['baseline_signed_gap_m'])
    if not math.isfinite(stored_baseline) or stored_baseline<-.0005 or stored_baseline>=.001:raise ValueError('Invalid actual support baseline')
    floor=stored_baseline-.000020
    if abs(float(initial['floor_signed_gap_m'])-floor)>1e-12:raise ValueError('Actual support floor differs from 20micrometre budget')
    if dt_s is None:
        dt_s=report.get('dataset',{}).get('physical_step_s')
        if dt_s is None:
            action=next(a for a in report['action_timing'] if a['end_step']>a['start_step'])
            dt_s=action['duration_s']/(action['end_step']-action['start_step'])
    dt=float(dt_s)
    if not math.isfinite(dt) or dt<=0:raise ValueError('Invalid exact physics dt')
    rows=[];events=[];release=None;baseline=None;previous=0;contact_available=0;active_after=[];positive_gap_active=[]
    min_gap=math.inf;support_gap_error=0.;step_count=0;anchor_found=False;scope_count=0;all_scope_telemetry=True
    self_available=0;self_active=[];fullq_available=0;names=report.get('full_qpos_joint_names')
    for row in iter_rows(trace_path):
        step=int(row['step']);step_count+=1
        if step!=previous+1:raise ValueError('Physics trace does not cover consecutive steps from1')
        previous=step
        if 'sim_time_s' in row and abs(float(row['sim_time_s'])-step*dt)>1e-6:raise ValueError('Physics clock mismatch')
        pose=np.asarray(row['bottle']['pose'],dtype=float)
        if pose.shape!=(4,4) or not np.isfinite(pose).all():raise ValueError('Invalid actual bottle pose')
        if not np.allclose(pose[3],[0,0,0,1],atol=1e-8,rtol=0) or not np.allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3),atol=1e-5,rtol=0) or abs(np.linalg.det(pose[:3,:3])-1)>1e-5:raise ValueError('Nonrigid actual bottle pose')
        if 'full_qpos_actual_rad' in row:
            fullq=np.asarray(row['full_qpos_actual_rad'],dtype=float)
            if fullq.shape!=(36,) or not np.isfinite(fullq).all() or not isinstance(names,list) or len(names)!=36 or len(set(names))!=36:raise ValueError('Invalid recorded actual36/joint order')
            fullq_available+=1
        if 'actual_self_contacts' in row:
            self_available+=1
            for c in row['actual_self_contacts']:
                v=float(c['normal_impulse_Ns'])
                if not math.isfinite(v) or v<0:raise ValueError('Invalid recorded self normal impulse')
                if v>1e-7:self_active.append({'step':step,'names':c.get('names',c.get('links')),'normal_impulse_Ns':v})
        world=verts@pose[:3,:3].T+pose[:3,3];gap=float(world[:,2].min()-top)
        if step==anchor:
            anchor_found=True;baseline=gap
            if abs(gap-stored_baseline)>1e-9:raise ValueError('Recomputed fresh actual baseline differs from report')
        impulses=table_impulses(row)
        if impulses is not None:contact_available+=1
        after_release=release is not None
        if step>anchor:
            scope_count+=1;min_gap=min(min_gap,gap)
            threshold=0. if after_release else floor
            if gap<threshold:events.append({'step':step,'kind':'signed_depth_or_released_clearance','gap_m':gap,'threshold_m':threshold})
            if np.any(world[:,:2]<bounds[0,:2]) or np.any(world[:,:2]>bounds[1,:2]):events.append({'step':step,'kind':'tabletop_footprint'})
            telemetry=row.get('support_departure');all_scope_telemetry=all_scope_telemetry and telemetry is not None
            if telemetry is not None:
                support_gap_error=max(support_gap_error,abs(float(telemetry['actual_signed_gap_m'])-gap))
                if abs(float(telemetry['floor_signed_gap_m'])-floor)>1e-12:raise ValueError('Perstep support floor drift')
                if abs(float(telemetry['baseline_signed_gap_m'])-baseline)>1e-9:raise ValueError('Perstep baseline drift')
            if release is None and gap>=.001:release={'step':step,'time_s':step*dt,'signed_gap_m':gap}
        normal=None if impulses is None else float(impulses['normal_impulse_Ns'])
        rec={'step':step,'time_s':step*dt,'phase':row['phase'],'signed_gap_m':gap,'after_release_before_step':after_release,
             'table_normal_impulse_Ns':normal,'table_normal_force_N':None if normal is None else normal/dt}
        rows.append(rec)
        if normal is not None and normal>1e-7:
            info={**rec,'active_points':impulses['active_points']}
            if after_release:active_after.append(info)
            if gap>0:positive_gap_active.append(info)
    if not anchor_found:raise ValueError('Missing actual configured support anchor row')
    accepted=report.get('accepted') is True and report.get('result') in ['physics_success','one_strictly_validated_episode_collected']
    expected=report.get('physical_steps')
    if expected is None and accepted:expected=max(a['end_step'] for a in report['action_timing'])
    complete=bool(expected is not None and int(expected)==previous)
    stable=all(stamps[str(p)]==(p.stat().st_size,p.stat().st_mtime_ns) for p in sources)
    final=report['support_departure_actual'];reported_release=final.get('release_step')
    release_matches=(release is not None and reported_release is not None and int(reported_release)==release['step'])
    prefix_pass=bool(stable and release_matches and not events and all_scope_telemetry and scope_count==final['checked_steps'] and support_gap_error<=1e-9)
    geometry_pass=bool(prefix_pass and complete)
    return {'passed':bool(accepted and geometry_pass),'accepted_report_required':True,'report_accepted':report.get('accepted'),
        'scope':'Pure CPU actual pose/native vertex signed support readback. Physical normal impulses are separate telemetry, not inferred from gap and not a replacement for full FCL actual36 audit.',
        'geometry_passed':geometry_pass,'recorded_support_prefix_passed':prefix_pass,
        'completed_episode_endpoint_covered':complete,'report_result':report.get('result'),
        'endpoint_limitation':None if complete else 'Incomplete/exception source: recorded prefix only. Any physical step throwing before trace append has no reconstructable pose/contact telemetry here.',
        'reported_carry_audit':report.get('carry_audit'),'failure_traceback_last_line':report.get('traceback','').strip().split('\n')[-1] if report.get('traceback') else None,
        'recorded_actual_self_contacts':{'full_coverage':self_available==step_count,'recorded_steps':self_available,'active_normal_contacts':self_active,'threshold_Ns':1e-7},
        'full_actual36_finite_coverage':fullq_available==step_count,
        'robot_table_collision_limitation':'No robot/table physical contact telemetry or independent robot/table FCL recomputation in this helper; do not infer absence from bottle signed height.',
        'table_top_z':top,'actual_native_vertices':len(verts),'baseline_signed_gap_m':baseline,'floor_signed_gap_m':floor,
        'minimum_post_anchor_signed_gap_m':None if min_gap==math.inf else min_gap,'maximum_additional_depth_m':None if min_gap==math.inf else max(0.,baseline-min_gap),
        'release':release,'reported_release_step':reported_release,'all_step_support_telemetry_verified':all_scope_telemetry,
        'max_recorded_vs_recomputed_gap_error_m':support_gap_error,'checked_post_anchor_steps':scope_count,'physics_steps':step_count,'physics_dt_s':dt,
        'signed_gap_or_footprint_violations':events,'contact_telemetry':{'recorded_steps':contact_available,'full_coverage':contact_available==step_count,
            'missing_steps':step_count-contact_available,'normal_impulse_threshold_Ns':1e-7,
            'positive_geometry_gap_with_active_table_impulse':positive_gap_active,'active_table_impulse_after_release':active_after,
            'limitation':None if contact_available==step_count else 'Missing perstep table impulse evidence cannot be interpreted as zero'},
        'per_physics_step':rows,'source_sha256':{str(p):sha(p) for p in sources},'source_files_unchanged':stable,'helper_sha256':sha(__file__)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--episode-dir',required=True);p.add_argument('--geometry',required=True)
    p.add_argument('--report');p.add_argument('--dt-s',type=float);p.add_argument('--output',required=True);args=p.parse_args()
    out=Path(args.output)
    if out.exists():raise FileExistsError('Refuse replacing existing evidence')
    r=analyze(args.episode_dir,args.geometry,args.report,args.dt_s)
    out.write_text(json.dumps(r,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:r[k] for k in ['passed','geometry_passed','physics_steps','baseline_signed_gap_m','minimum_post_anchor_signed_gap_m','release']}))
    print('CONTACT_COVERAGE',r['contact_telemetry']['full_coverage'],'POSITIVE_GAP_ACTIVE',len(r['contact_telemetry']['positive_geometry_gap_with_active_table_impulse']),'AFTER_RELEASE_ACTIVE',len(r['contact_telemetry']['active_table_impulse_after_release']))
    return 0 if r['passed'] else 2


if __name__=='__main__':raise SystemExit(main())
