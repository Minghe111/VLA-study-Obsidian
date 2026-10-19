"""Read-only CPU legacy/new physical-observation overlap audit.

No arguments builds ONLY the legacy index. A later --new-manifest must point to
a completed stable capture; this script never reads a growing manifest by
default. Official OpenPI resize_with_pad runs on JAX CPU, with no model loaded.
"""
from __future__ import annotations

import os
os.environ['JAX_PLATFORMS']='cpu'
os.environ['CUDA_VISIBLE_DEVICES']=''

import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

BASE=Path('/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/recap_memory_20261017/experiments')
LEGACY={
    'fit96':BASE/'protocol_repair_20261019_v2/data/fit_subsets_v1/fit96.train.jsonl',
    'original_train':BASE/'protocol_sft_20261018_v1/data/protocol_labels_reviewed_v1/train.reviewed.jsonl',
    'original_val':BASE/'protocol_sft_20261018_v1/data/protocol_labels_reviewed_v1/val.reviewed.jsonl',
    'original_test':BASE/'protocol_sft_20261018_v1/data/protocol_labels_reviewed_v1/test.reviewed.jsonl',
}
CAMERAS=('full_image','left_wrist_image','right_wrist_image')
MATCH_KEYS=('path','family_id','episode_id','seed','npz_sha256','raw_triplet_sha256','resize224_triplet_sha256')


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):digest.update(block)
    return digest.hexdigest()


def pixels_sha(value):return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def now():return datetime.now(timezone.utc).isoformat()
def canonical_path(path,base):
    path=Path(path);return str((base/path).resolve() if not path.is_absolute() else path.resolve())


class OfficialCPUResize:
    def __init__(self):
        import jax
        from openpi.shared import image_tools
        resize_with_pad=image_tools.resize_with_pad
        self.jax=jax;self.function=resize_with_pad
        devices=jax.devices()
        if not devices or any(device.platform!='cpu' for device in devices):
            raise RuntimeError('Overlap audit requires CPU-only JAX devices')
        path=Path(inspect.getfile(image_tools))
        self.receipt={'function':'openpi.shared.image_tools.resize_with_pad',
            'source_path':str(path),'source_sha256':sha(path),'height':224,'width':224,
            'method':'default JAX LINEAR; uint8 round/clip; zero padding',
            'device_platforms':sorted({device.platform for device in devices}),
            'jax_version':jax.__version__,
            'scope':'Official upstream ResizeImages uses this exact function; no torch/PIL substitute.'}
    def __call__(self,image):
        with self.jax.default_device(self.jax.devices('cpu')[0]):
            result=np.asarray(self.function(image,224,224))
        if result.shape!=(224,224,3) or result.dtype!=np.uint8:
            raise ValueError('Official 224 output must be RGB8[224,224,3]')
        return result


def triplet_sha(cameras):
    # The order/names, dtype and shape are part of the comparison, not just a
    # concatenation that might confuse array boundaries.
    text=json.dumps([[key,cameras[key]['shape'],cameras[key]['dtype'],cameras[key]['pixel_sha256']]
        for key in CAMERAS],separators=(',',':'))
    return hashlib.sha256(text.encode()).hexdigest()


def observation_fingerprint(path,resize):
    path=Path(path);before=path.stat();file_sha=sha(path)
    with np.load(path,allow_pickle=False) as archive:
        raw={key:archive[key].copy() for key in CAMERAS}
    raw_receipt={};small_receipt={}
    for key,image in raw.items():
        if image.shape!=(480,640,3) or image.dtype!=np.uint8:
            raise ValueError(f'{key} must be actual uint8 RGB[480,640,3]')
        small=resize(image)
        raw_receipt[key]={'shape':list(image.shape),'dtype':str(image.dtype),'pixel_sha256':pixels_sha(image)}
        small_receipt[key]={'shape':list(small.shape),'dtype':str(small.dtype),'pixel_sha256':pixels_sha(small)}
    after=path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
        raise ValueError('Source NPZ changed during fingerprinting')
    return {'path':str(path.resolve()),'npz_sha256':file_sha,'bytes':after.st_size,'mtime_ns':after.st_mtime_ns,
        'raw_cameras':raw_receipt,'raw_triplet_sha256':triplet_sha(raw_receipt),
        'resize224_cameras':small_receipt,'resize224_triplet_sha256':triplet_sha(small_receipt)}


def build_legacy(index_path,resize):
    labels={};observations={};errors=[]
    for name,path in LEGACY.items():
        if not path.is_file():
            labels[name]={'path':str(path),'exists':False,'required':name!='original_train',
                'coverage':'not_indexed_missing_file'}
            continue
        digest=sha(path);rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        paths=set()
        for line,row in enumerate(rows,1):
            reference=row.get('observation',{}).get('npz_path')
            if reference is None:
                errors.append({'label_source':name,'line_1based':line,'error':'Missing actual observation.npz_path'})
                continue
            resolved=canonical_path(reference,path.parent);paths.add(resolved)
            entry=observations.setdefault(resolved,{'path':resolved,'legacy_rows':[]})
            metadata=row.get('metadata',{})
            entry['legacy_rows'].append({'label_source':name,'label_path':str(path),'line_1based':line,
                'sample_id':row.get('sample_id'),'split':row.get('split',metadata.get('split')),
                'family_id':row.get('family_id',metadata.get('family_id')),
                'episode_id':row.get('episode_id',metadata.get('episode_id')),
                'seed':row.get('seed',metadata.get('seed')),'mode':row.get('mode')})
        labels[name]={'path':str(path),'exists':True,'required':name!='original_train','sha256':digest,
            'rows':len(rows),'unique_npz_paths':len(paths),'mode_counts':dict(Counter(r.get('mode') for r in rows)),
            'coverage':'JSONL_indexed; NPZ coverage separately enumerated'}
        if sha(path)!=digest:raise ValueError('Legacy labels changed during read')
    for path,entry in observations.items():
        try:entry.update(observation_fingerprint(path,resize));entry['fingerprinted']=True
        except Exception as exc:
            entry['fingerprinted']=False;entry['error']=f'{type(exc).__name__}: {exc}'
            errors.append({'npz_path':path,'error':entry['error']})
    required_present=all(labels[name]['exists'] for name in ('fit96','original_val','original_test'))
    complete=required_present and not errors
    result={'schema':'history3000_legacy_pixel_index_v1','created_at_utc':now(),
        'status':'complete_for_available_declared_sources' if complete else 'partial_coverage',
        'scope':'Old fit96 plus original val/test and original train if present; no new manifest read.',
        'all_required_label_files_present':required_present,'optional_original_train_present':labels['original_train']['exists'],
        'all_referenced_NPZ_fingerprinted':not errors,'official_resize':resize.receipt,'camera_order':list(CAMERAS),
        'triplet_definition':'sha256(canonical JSON [[camera,shape,dtype,pixel_sha256],...] in camera_order)',
        'labels':labels,'total_label_rows':sum(v.get('rows',0) for v in labels.values()),
        'unique_NPZ_paths':len(observations),'unique_raw_triplets':len({e['raw_triplet_sha256'] for e in observations.values() if e.get('fingerprinted')}),
        'unique_224_triplets':len({e['resize224_triplet_sha256'] for e in observations.values() if e.get('fingerprinted')}),
        'observations':list(observations.values()),'errors':errors,
        'limitations':['Labels overlap (fit96 is a subset of the original train pool); counts are not independent episodes.',
            'Identifier matches are provenance warnings, not sufficient proof of image leakage.',
            'Raw/224 fingerprints compare the ordered three RGB views exactly; no similarity threshold or semantic claim.',
            'Missing original data are explicitly unindexed; no global no-overlap claim is permitted with partial coverage.',
            'No learned model, policy generation, optimizer, training or GPU resize was used.']}
    dump(index_path,result);return result


def validate_index(index,resize):
    if index.get('schema')!='history3000_legacy_pixel_index_v1':raise ValueError('Unknown legacy index schema')
    if index['official_resize']!=resize.receipt:raise ValueError('Official resize implementation/version changed; rebuild index')
    for source in index['labels'].values():
        if source['exists'] and sha(source['path'])!=source['sha256']:
            raise ValueError('Legacy JSONL changed since index')
    for entry in index['observations']:
        if entry.get('fingerprinted') and sha(entry['path'])!=entry['npz_sha256']:
            raise ValueError('Legacy NPZ changed since index')


def values_for(entry,key):
    if key in ('family_id','episode_id','seed'):
        if 'legacy_rows' in entry:return {r[key] for r in entry['legacy_rows'] if r.get(key) is not None}
        return {entry[key]} if entry.get(key) is not None else set()
    return {entry[key]} if entry.get(key) is not None else set()


def compare_completed(index,new_manifest,resize,*,expected_new_points=696):
    path=Path(new_manifest).resolve();manifest_sha=sha(path)
    capture=json.loads(path.read_text())
    if capture.get('status')!='completed' or capture.get('completed') is not True:
        raise ValueError('Only a completed stable new capture manifest may be compared')
    validate_index(index,resize)
    old_index={key:defaultdict(list) for key in MATCH_KEYS}
    for old_id,entry in enumerate(index['observations']):
        for key in MATCH_KEYS:
            for value in values_for(entry,key):old_index[key][value].append(old_id)
    new_observations=[];errors=[];cache={}
    for episode in capture.get('episodes',[]):
        if not episode.get('completed') or not episode.get('worker_shutdown_confirmed'):
            raise ValueError('New manifest contains an unfinished source worker/episode')
        for observation in episode.get('observations',[]):
            resolved=canonical_path(observation['npz_path'],path.parent)
            entry={'path':resolved,'split':episode.get('split'),'family_id':episode.get('family_id'),
                'episode_id':episode.get('episode_id'),'seed':episode.get('seed'),
                'step':observation.get('step'),'physics_step':observation.get('physics_step')}
            try:
                if resolved not in cache:cache[resolved]=observation_fingerprint(resolved,resize)
                entry.update(cache[resolved])
                if observation.get('sha256') and entry['npz_sha256']!=observation['sha256']:
                    raise ValueError('New observation differs from final manifest SHA')
                entry['fingerprinted']=True
            except Exception as exc:
                entry['fingerprinted']=False;entry['error']=f'{type(exc).__name__}: {exc}'
                errors.append({'new_observation':len(new_observations),'path':resolved,'error':entry['error']})
            new_observations.append(entry)
    intersections={key:[] for key in MATCH_KEYS}
    for key in MATCH_KEYS:
        grouped=defaultdict(list)
        for new_id,entry in enumerate(new_observations):
            for value in values_for(entry,key):
                if value in old_index[key]:grouped[value].append(new_id)
        for value,new_ids in grouped.items():
            intersections[key].append({'value':value,'new_observation_indices':new_ids,
                'legacy_observation_indices':old_index[key][value]})
    if sha(path)!=manifest_sha:raise ValueError('Final new manifest changed during comparison')
    complete=(index['status']=='complete_for_available_declared_sources' and not errors
              and len(new_observations)==expected_new_points)
    exact_pixel_overlap=bool(intersections['raw_triplet_sha256'] or intersections['resize224_triplet_sha256'])
    return {'schema':'history3000_legacy_overlap_v1','created_at_utc':now(),
        'status':'complete_with_exact_pixel_overlap' if complete and exact_pixel_overlap else ('complete_no_exact_pixel_overlap' if complete else 'partial_coverage_no_global_conclusion'),
        'expected_new_source_points':expected_new_points,'new_source_points':len(new_observations),
        'new_manifest':str(path),'new_manifest_sha256':manifest_sha,'new_manifest_unchanged':True,
        'legacy_sources':index['labels'],'official_resize':resize.receipt,
        'coverage_complete_for_declared_sources':complete,'exact_three_camera_pixel_overlap_found':exact_pixel_overlap,
        'intersections':intersections,
        'intersection_counts':{key:{'distinct_shared_values':len(groups),
            'new_source_points_matched':len({i for g in groups for i in g['new_observation_indices']}),
            'legacy_source_points_matched':len({i for g in groups for i in g['legacy_observation_indices']})} for key,groups in intersections.items()},
        'new_observations':new_observations,'legacy_observations':index['observations'],'errors':errors,
        'scope':'Exact source-observation/path/provenance and ordered raw/official224 RGB equality only. No policy inference, fitting or semantic similarity claim.',
        'limitations':index['limitations']+['No-overlap is limited to these indexed legacy pools and the completed new capture; unknown historical data are outside coverage.']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-index',type=Path,default=Path('/tmp/history3000_legacy_index.json'))
    parser.add_argument('--rebuild-legacy-index',action='store_true')
    parser.add_argument('--new-manifest',type=Path)
    parser.add_argument('--expected-new-points',type=int,default=696)
    parser.add_argument('--output',type=Path,default=Path('/tmp/history3000_legacy_overlap_report.json'))
    args=parser.parse_args()
    resize=OfficialCPUResize()
    if args.rebuild_legacy_index or not args.legacy_index.is_file():
        index=build_legacy(args.legacy_index,resize)
    else:
        index=json.loads(args.legacy_index.read_text());validate_index(index,resize)
    if not args.new_manifest:
        print(json.dumps({'mode':'legacy_index_only','index':str(args.legacy_index),'status':index['status'],
            'rows':index['total_label_rows'],'unique_paths':index['unique_NPZ_paths'],
            'unique_raw_triplets':index['unique_raw_triplets'],'unique_224_triplets':index['unique_224_triplets'],
            'new_manifest_read':False,'errors':len(index['errors'])},ensure_ascii=False))
        return
    report=compare_completed(index,args.new_manifest,resize,expected_new_points=args.expected_new_points)
    dump(args.output,report)
    print(json.dumps({'report':str(args.output),'status':report['status'],'intersection_counts':report['intersection_counts']},ensure_ascii=False))


if __name__=='__main__':main()
