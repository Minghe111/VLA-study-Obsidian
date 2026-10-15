"""Static source inventory/import/file-reference audit, never imports workers."""
from pathlib import Path
import ast,json,hashlib,sys
ROOT=Path(__file__).resolve().parent.parent
LOCAL_LEGACY=Path('/home/admin123/liminghe/vla-benchmark/deploy/realman_gen4/adjust_bottle_20261002')
ARCHIVE=Path('/home/admin123/liminghe/VLA-WAM-RL/30-实验记录/证据/2026-10-11-RealMan专家回放/read_only_collector_sources')
SEARCH=[ROOT/'source/rgb50_20261007_v1',ROOT/'source/adjust_bottle_20261002',ROOT/'source',ARCHIVE,LOCAL_LEGACY]
ROOT_MODULES=['worker_settle_wake_v5','collector_settle_wake_v5','generation_worker_v2','generation_runner_v2','expert_v9_settle_wake_v5','expert_generation_v9']
MANDATORY_GENERATION_MODULES=['expert_generation']+[f'expert_generation_v{i}' for i in range(2,10)]+['geometry_vector_overlay','geometry_overlay','closure_boundary_audit_v14','benchmark_geometry_cache','collector_settle_wake_v5','expert_v9_settle_wake_v5','worker_settle_wake_v5','generation_runner_v2','generation_worker_v2']
def find(module):
    for folder in SEARCH:
        p=folder/(module.replace('.','/')+'.py')
        if p.is_file():return p
    return None
external=set();missing=set();seen={};todo=list(dict.fromkeys(ROOT_MODULES+MANDATORY_GENERATION_MODULES))
while todo:
    name=todo.pop(0)
    if name in seen:continue
    p=find(name)
    if p is None:missing.add(name);continue
    text=p.read_text(encoding='utf-8-sig');tree=ast.parse(text);imports=set();filerefs=set();absolute=set()
    for n in ast.walk(tree):
        if isinstance(n,ast.Import):imports.update(a.name for a in n.names)
        elif isinstance(n,ast.ImportFrom) and n.module and n.level==0:imports.add(n.module)
        elif isinstance(n,ast.Constant) and isinstance(n.value,str):
            v=n.value
            if v.endswith('.py') and '/' not in v:filerefs.add(v)
            if v.startswith(('/bigdata/','/bigdata2/')):absolute.add(v)
    for imp in sorted(imports):
        if find(imp):todo.append(imp)
        else:external.add(imp)
    for filename in sorted(filerefs):
        stem=filename[:-3]
        if find(stem):todo.append(stem)
        else:missing.add(stem)
    seen[name]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'imports':sorted(imports),'literal_py_dependencies':sorted(filerefs),'hardcoded_remote_absolute_paths':sorted(absolute)}
missingmandatory=[n for n in MANDATORY_GENERATION_MODULES if find(n) is None]
result={'scope':'Static import/literal-source dependency closure; conservative overapproximation, no collector/GPU/physics executed',
        'search_roots':list(map(str,SEARCH)),'entry_modules':ROOT_MODULES,'modules':seen,'missing_source_modules':sorted(missing),
        'missing_mandatory_generation_modules':missingmandatory,'external_import_names_to_resolve_in_environment_or_RoboTwin':sorted(external),
        'package_layout_required':'source/rgb50_20261007_v1 sibling of source/adjust_bottle_20261002 for inherited relative _LEGACY; do not flatten without explicit mapping',
        'original_scene_metadata_needed_for_each_index':['scene_sample.json exact pose/mass geometry','sample_spec.json grasp and placement parameters','run_config.yml actual runtime/setup','original initial full36 qpos / state24 and camera matrices for restoration audit'],
        'same_hardware_rule':'Keep frozen actual SAPIEN URDF/head pitch/SDK mapping/PD/tendons/materials4ms dynamics; path-only relocation must have proof',
        'readiness':'source imports incomplete' if missingmandatory else 'mandatory modules present; must still inspect new full closure/pins and run isolated acceptance'}
(ROOT/'preflight/collector_dependency_inventory.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'resolved_source_modules':len(seen),'missing_mandatory':missingmandatory,'missing_other_sources':sorted(missing),'hardcoded_prefix_module_count':sum(bool(r['hardcoded_remote_absolute_paths']) for r in seen.values())},indent=2))
