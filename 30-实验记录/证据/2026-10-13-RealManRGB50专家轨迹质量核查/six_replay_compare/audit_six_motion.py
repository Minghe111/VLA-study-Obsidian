"""Read-only physical arm motion audit; no renderer, policy, or robot imports."""
from pathlib import Path
import gzip,json,hashlib,xml.etree.ElementTree as ET
import numpy as np

OUT=Path(__file__).resolve().parent
BASE=Path('/home/admin123/liminghe/VLA-WAM-RL/30-实验记录/证据')
EXPERT=BASE/'2026-10-11-RealMan专家回放'
MODEL=BASE/'2026-10-12-RealMan80000原场景本机推理'
NUMERIC=OUT.parent/'numeric'
IDS=(0,38,30,31,17,7)
DEG=180/np.pi
initials={e['index']:e for e in json.loads((MODEL/'source_initial_records.json').read_text())['episodes']}
urdf=EXPERT/'shared_frozen_runtime/robot_head_down_fixed.urdf'
limits={j.attrib['name']:{'lower_rad':float(j.find('limit').attrib['lower']),'upper_rad':float(j.find('limit').attrib['upper'])}
        for j in ET.parse(urdf).getroot().findall('joint') if j.attrib['name'].startswith(('l_joint','r_joint'))}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def six_arm_metrics(q,t,names):
    """Raw physical coordinates retained: unwrap is only a diagnostic comparison."""
    q=np.asarray(q,dtype=float);t=np.asarray(t,dtype=float);dq=np.diff(q,axis=0)
    qs=q[::15];ts=t[::15]
    if ts[-1]!=t[-1]:qs=np.concatenate([qs,q[-1:]],axis=0);ts=np.append(ts,t[-1])
    metrics=[]
    for j,name in enumerate(names):
        x=q[:,j];d=dq[:,j];lo=int(np.argmin(x));hi=int(np.argmax(x));net=float(x[-1]-x[0]);span=float(x.max()-x.min())
        q60=qs[:,j];jump60=np.diff(q60);k=int(np.argmax(abs(jump60))) if len(jump60) else 0
        rolling60=x[15:]-x[:-15]
        unwrap=np.unwrap(x);wrap_candidates=np.where(abs(d)>np.pi)[0]
        # No angular modulo is used for control or path metrics. A smooth
        # bounded revolute coordinate can validly travel more than pi.
        metrics.append({'joint':name,'initial_deg':float(x[0]*DEG),'terminal_deg':float(x[-1]*DEG),
            'minimum_deg':float(x.min()*DEG),'maximum_deg':float(x.max()*DEG),'range_deg':span*DEG,
            'minimum_time_s':float(t[lo]),'maximum_time_s':float(t[hi]),'net_turn_deg':net*DEG,
            'accumulated_travel_4ms_deg':float(abs(d).sum()*DEG),
            'accumulated_travel_60ms_deg':float(abs(jump60).sum()*DEG),
            'max_single_4ms_increment_deg':float(abs(d).max()*DEG) if len(d) else 0.,
            'max_sampled_60ms_increment_deg':float(abs(jump60).max()*DEG) if len(jump60) else 0.,
            'max_sampled_60ms_increment_interval_s':[float(ts[k]),float(ts[k+1])] if len(jump60) else None,
            'max_any_dense_60ms_interval_increment_deg':float(abs(rolling60).max()*DEG) if len(rolling60) else 0.,
            'raw_increment_over180deg_count':len(wrap_candidates),
            'unwrap_changes_raw_coordinate':bool(np.max(abs(unwrap-x))>1e-8),
            'raw_range_at_least_one_turn':bool(span>=2*np.pi),'raw_net_at_least_one_turn':bool(abs(net)>=2*np.pi),
            'accumulated_60ms_at_least_one_turn':bool(abs(jump60).sum()>=2*np.pi),
            'counter_direction_travel_60ms_deg':float((abs(jump60).sum()-abs(net))*DEG/2),
            'mechanical_revolute_limits_deg':{k:v*DEG for k,v in limits[name].items()}})
    return metrics

def onset(sdk,t):
    # The last SDK channel is thumb opposition/rotation=0 when open; exclude it.
    # Event threshold is an analysis definition, never an execution rule.
    idx=np.flatnonzero(np.min(sdk[:,:5],axis=1)<=950.)
    return None if not len(idx) else int(idx[0])

def load_replay(kind,index):
    folder=EXPERT/f'episode_{index:07d}' if kind=='expert_replay' else MODEL/'episodes'/f'ep{index:02d}_model80000'
    meta=json.loads((folder/'initial_state.json').read_text());report=json.loads((folder/'report.json').read_text())
    names=meta['actual_physical_contract']['full_qpos_joint_names']
    init=initials[index];assert init['source_hdf_sha256']==meta['source_hdf_sha256']
    q=[init['source_initial_qpos36']];state=[init['source_initial_state24']];times=[0.];drive=[];steps=[]
    with gzip.open(folder/'actual_physical_trace.jsonl.gz','rt') as f:
        for line in f:
            row=json.loads(line);q.append(row['actual_qpos36_rad']);state.append(row['actual_state24']);times.append(row['sim_time_s']);drive.append(row['actual_drive36_rad']);steps.append(row['step'])
    q=np.asarray(q);s=np.asarray(state);t=np.asarray(times);d=np.asarray(drive)
    assert len(steps)==report['actual_physics_steps'] and np.array_equal(steps,np.arange(1,len(steps)+1))
    assert np.max(abs(np.diff(t)-.004))<1e-8
    data={'kind':kind,'index':index,'selected_arm':report['source_episode']['arm'],'duration_s':float(t[-1]),'physics_trace_rows':len(steps),
          'actual_qpos_joint_order':names,'trace_sha256':sha(folder/'actual_physical_trace.jsonl.gz'),'report_sha256':sha(folder/'report.json'),
          'source_hdf_sha256':meta['source_hdf_sha256'],'source_initial_qpos_verified_by_prior_restore_zero_error':True,
          'native_success':report['native_task_success'],'first_positive_hand_contact_s':None if report['observation']['first_positive_hand_contact_step'] is None else report['observation']['first_positive_hand_contact_step']*.004,
          'first_opposed_contact_s':None if report['observation']['first_opposed_contact_step'] is None else report['observation']['first_opposed_contact_step']*.004,'sides':{}}
    for side,letter,offset in [('left','l',0),('right','r',12)]:
        armnames=[f'{letter}_joint{j}' for j in range(1,7)];idx=[names.index(n) for n in armnames];a=q[:,idx];armdrive=d[:,idx]
        error=float(np.max(abs(a-s[:,offset:offset+6])));assert error==0
        close=onset(s[:,offset+6:offset+12],t)
        data['sides'][side]={'measured_state24_vs_named_full36_max_abs_error_rad':error,'joints':six_arm_metrics(a,t,armnames),
          'first_primary_sdk_channel_below950_s':None if close is None else float(t[close]),
          'arm_net_at_first_close_deg':None if close is None else ((a[close]-a[0])*DEG).tolist(),
          'pre_first_close_joints':None if close is None else six_arm_metrics(a[:close+1],t[:close+1],armnames),
          'actual_drive_joints':six_arm_metrics(np.vstack([a[0],armdrive]),t,armnames),
          'max_measured_vs_drive_difference_deg':float(np.max(abs(a[1:]-armdrive))*DEG)}
        contact=data['first_positive_hand_contact_s'] if side==data['selected_arm'] else None
        k=None if contact is None else int(np.searchsorted(t,contact+1e-8,side='right'))
        data['sides'][side]['pre_first_selected_hand_contact_joints']=None if k is None else six_arm_metrics(a[:k],t[:k],armnames)
    return data

def load_original31():
    p=NUMERIC/'episode_0000031.npz'
    with np.load(p) as z:
        q0=z['realman_dexterous_physics/full36_qpos_before_rad'][0]
        q=np.concatenate([q0[None],z['realman_dexterous_physics/full36_qpos_after_rad']],axis=0)
        s=np.concatenate([z['realman_dexterous_physics/measured_state24_before'][0:1],z['realman_dexterous_physics/measured_state24_after']],axis=0)
        t=np.concatenate([[0.],z['realman_dexterous_physics/sim_time_after_s']])
        c=z['realman_dexterous_physics/applied_command24']
        names=json.loads((EXPERT/'episode_0000031/initial_state.json').read_text())['actual_physical_contract']['full_qpos_joint_names']
        result={'kind':'original_source31_numeric_HDF_readback','index':31,'numeric_npz':str(p),'numeric_npz_sha256':sha(p),'duration_s':float(t[-1]),'sides':{}}
        for side,letter,offset in [('left','l',0),('right','r',12)]:
            armnames=[f'{letter}_joint{j}' for j in range(1,7)];idx=[names.index(n) for n in armnames];a=q[:,idx]
            assert np.max(abs(a-s[:,offset:offset+6]))==0
            close=onset(s[:,offset+6:offset+12],t)
            result['sides'][side]={'joints':six_arm_metrics(a,t,armnames),'first_primary_sdk_channel_below950_s':None if close is None else float(t[close]),
                'arm_net_at_first_close_deg':None if close is None else ((a[close]-a[0])*DEG).tolist(),
                'pre_first_close_joints':None if close is None else six_arm_metrics(a[:close+1],t[:close+1],armnames),
                'applied_command_joints':six_arm_metrics(np.vstack([a[0],c[:,offset:offset+6]]),t,armnames)}
            ee=z['state/'+side+'_ee_poses'];xyz=ee[:,:3];ds=np.linalg.norm(np.diff(xyz,axis=0),axis=1);straight=float(np.linalg.norm(xyz[-1]-xyz[0]))
            result['sides'][side]['source_measured_ee_path']={'source_dataset':'state/'+side+'_ee_poses','sample_count':len(ee),'polyline_length_m':float(ds.sum()),'initial_to_terminal_straight_distance_m':straight,'path_to_net_displacement_ratio':None if straight==0 else float(ds.sum()/straight),'bounds_xyz_m':{'min':xyz.min(0).tolist(),'max':xyz.max(0).tolist()},'interpretation_limit':'Contains reach, grasp, lift and hold; not a pure pregrasp shortest-path efficiency score'}
            if close is not None:
                observed_times=z['realman_dexterous/sample_time_s'];before=xyz[observed_times<=t[close]];path=float(np.linalg.norm(np.diff(before,axis=0),axis=1).sum());direct=float(np.linalg.norm(before[-1]-before[0]))
                result['sides'][side]['source_measured_ee_preclose_path']={'end_time_s':float(observed_times[len(before)-1]),'sample_count':len(before),'polyline_length_m':path,'initial_to_preclose_straight_distance_m':direct,'path_to_displacement_ratio':None if direct==0 else path/direct,'interpretation_limit':'Observed motion before primary SDK threshold; curvature can reflect constraints, not independently proof of avoidable/collision-free shortcut'}
    return result

def source17_literal_turn_confirmation():
    p=NUMERIC/'episode_0000017.npz';names=json.loads((EXPERT/'episode_0000017/initial_state.json').read_text())['actual_physical_contract']['full_qpos_joint_names'];j=names.index('l_joint6')
    with np.load(p) as z:
        q=np.r_[z['realman_dexterous_physics/full36_qpos_before_rad'][0,j],z['realman_dexterous_physics/full36_qpos_after_rad'][:,j]]
        t=np.r_[0.,z['realman_dexterous_physics/sim_time_after_s']]
        sdk=np.concatenate([z['realman_dexterous_physics/measured_state24_before'][0:1,6:12],z['realman_dexterous_physics/measured_state24_after'][:,6:12]],axis=0)
        close=onset(sdk,t);first=np.flatnonzero(q-q[0]<=-2*np.pi)
        metrics=six_arm_metrics(q[:,None],t,['l_joint6'])[0];pre=six_arm_metrics(q[:close+1,None],t[:close+1],['l_joint6'])[0]
        return {'index':17,'kind':'original_source17_numeric_confirmation','numeric_npz_sha256':sha(p),'joint':'l_joint6','first_minus360_from_initial_time_s':None if not len(first) else float(t[first[0]]),
                'actual_primary_close_onset_s':float(t[close]),'whole_episode_metrics':metrics,'before_primary_close_metrics':pre,
                'actual_minimum_before_grasp_deg':pre['minimum_deg'],'true_full_turn_before_close':bool(pre['range_deg']>=360.)}

def main():
    rows=[load_replay(kind,i) for kind in ('expert_replay','model80000') for i in IDS]
    original=load_original31();result={'date':'2026-10-13','scope':'Read-only12physical replay traces plus31original source HDF numeric readback; no simulation/model/hardware execution',
      'selected_indices':list(IDS),'definitions':{'range':'raw maximum minus minimum actual revolute coordinate','net':'last minus restored original initial coordinate','cumulative':'sum of absolute changes, distinguished from one-direction rotation','primary_close_onset':'first any of first5actual SDK channels≤950; last channel0open excluded','wrap_diagnostic':'raw single4ms increment>pi and np.unwrap comparison; metrics never use modulo shortest angles','full_turn':'literal single-joint raw range or net≥360°, separate from cumulative back-and-forth'},
      'trace_schema':{'actual_state24':'Larm6rad,Lhand6SDK,Rarm6rad,Rhand6SDK','actual_qpos36_rad':'named mapping from persisted initial physical contract; r/l arm joints interleaved among first12','actual_drive36_rad':'same36named order','trace_period_s':.004,'video_observation_period_s':.06},
      'arm_urdf_limits_source':str(urdf),'arm_urdf_limits_source_sha256':sha(urdf),'episodes':rows,'original_source31':original,
      'original_source17_literal_turn_confirmation':source17_literal_turn_confirmation()}
    (OUT/'six_replay_motion_comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    for row in rows:
        arm=row['selected_arm'];j=row['sides'][arm]['joints'];print(row['kind'],row['index'],arm,'net_deg',[round(k['net_turn_deg'],1) for k in j],'range_deg',[round(k['range_deg'],1) for k in j],'cum60_deg',[round(k['accumulated_travel_60ms_deg'],1) for k in j],'close_s',row['sides'][arm]['first_primary_sdk_channel_below950_s'])
    print('AUDIT_JSON',str(OUT/'six_replay_motion_comparison.json'),flush=True)
if __name__=='__main__':main()
