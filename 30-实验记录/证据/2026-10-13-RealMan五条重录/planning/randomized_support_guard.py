"""Task-only strict support certificates for real randomized table geometry.

Only bottle_table may use the original signed support-exit certificate. All
other original and new clutter categories must retain their full mesh bounds.
"""
from __future__ import annotations
import ast, copy, hashlib, importlib, math, sys
from pathlib import Path
import numpy as np

ORIGINAL_SUPPORT_SHA256 = '5ca698f162bc6440dac424ddf0454202948fedfd0c585fd95808bf5f61d33a00'
REFERENCE_TABLE_TOP_Z_M = .74
OFFICIAL_TABLE_HEIGHT_RANDOMIZATION_M = .03
_INSTALLED = None

def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _finite_scalar(value):
    return isinstance(value,(int,float,np.integer,np.floating)) and not isinstance(value,(bool,np.bool_)) and math.isfinite(float(value))

def strict_other_categories_pass(raw):
    """Retain every actual continuous category except the sole replaced table pair."""
    try:
        bounds = raw['continuous_gap_lower_bound']; thresholds = raw['thresholds_m']
        categories = set(bounds)
        if categories != set(thresholds) or not {'bottle_robot','self','robot_table','bottle_table'}.issubset(categories):
            return False
        if raw['joint_limit_audit']['passed'] is not True:
            return False
        count = raw['unique_collision_pairs']
        if type(count) is not int or count < 0:
            return False
        events = raw['collisions_first_64']
        if not isinstance(events,list) or any(e.get('category') != 'bottle_table' for e in events) or count != len(events):
            return False
        for name in categories:
            bound = bounds[name]['lower_bound_m']; threshold = thresholds[name]
            if not _finite_scalar(bound) or not _finite_scalar(threshold) or float(threshold)<0:
                return False
            if name != 'bottle_table' and float(bound) < float(threshold):
                return False
        return True
    except (KeyError,TypeError,AttributeError,ValueError):
        return False

def recorded_table_expectation(task):
    """Explicit actual-versus-recorded official DR validation, no pose guessing."""
    receipt = getattr(task,'_rerecord_randomization_receipt',None)
    bias = getattr(task,'table_z_bias',None)
    if not isinstance(receipt,dict) or receipt.get('passed') is not True or not _finite_scalar(bias):
        raise ValueError('Missing finite actual table bias or verified randomized setup receipt')
    recorded = receipt.get('table_z_bias_m')
    actual = receipt.get('actual_domain_randomization'); official = receipt.get('official_domain_randomization')
    if not _finite_scalar(recorded) or abs(float(recorded)-float(bias))>1e-12:
        raise ValueError('Recorded table bias differs from actual native task.table_z_bias')
    if not isinstance(actual,dict) or not isinstance(official,dict) or actual != official:
        raise ValueError('Recorded actual/official domain randomization settings differ')
    height = actual.get('random_table_height')
    if not _finite_scalar(height) or abs(float(height)-OFFICIAL_TABLE_HEIGHT_RANDOMIZATION_M)>1e-12:
        raise ValueError('Recorded table randomization is not the requested official 0.03m range')
    current_range = getattr(task,'random_table_height',None)
    if not _finite_scalar(current_range) or abs(float(current_range)-float(height))>1e-12:
        raise ValueError('Actual native table randomization range differs from recorded settings')
    if not -float(height)-1e-12 <= float(bias) <= 1e-12:
        raise ValueError('Actual table bias exceeds official [-0.03,0]m sampling range')
    return {'passed':True,'reference_table_top_z_m':REFERENCE_TABLE_TOP_Z_M,
            'actual_table_z_bias_m':float(bias),'recorded_table_z_bias_m':float(recorded),
            'expected_table_top_z_m':REFERENCE_TABLE_TOP_Z_M+float(bias),
            'official_table_randomization_range_m':[-float(height),0.],
            'same_actual_recorded_official_randomization':True}

def _expected_table_top_for_task(task):
    return recorded_table_expectation(task)['expected_table_top_z_m']

def derive_support_functions(payload):
    tree = ast.parse(payload)
    functions = {n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    geometry = copy.deepcopy(functions['native_tabletop_geometry'])
    geometry.args.kwonlyargs.append(ast.arg(arg='expected_table_top_z_m'))
    geometry.args.kw_defaults.append(None)
    class Height(ast.NodeTransformer):
        def __init__(self):self.sites=0;self.errors=0
        def visit_Compare(self,node):
            self.generic_visit(node)
            if ast.unparse(node) == 'abs(highest - 0.74) > 1e-06':
                self.sites += 1
                return ast.copy_location(ast.parse('abs(highest - expected_table_top_z_m) > 1e-6',mode='eval').body,node)
            return node
        def visit_Constant(self,node):
            if node.value=='Actual whole table top must be official WORLD z=.74m':
                self.errors+=1
                return ast.copy_location(ast.Constant(value='Actual native whole table top differs from recorded randomized WORLD top'),node)
            return node
    height = Height();height.visit(geometry)
    if height.sites != 1 or height.errors != 1:
        raise RuntimeError('Exactly one original fixed table-top guard is required')
    certificate = copy.deepcopy(functions['certify_first_world_support_departure'])
    class ExplicitCall(ast.NodeTransformer):
        def __init__(self):self.sites=0
        def visit_Call(self,node):
            self.generic_visit(node)
            if isinstance(node.func,ast.Name) and node.func.id=='native_tabletop_geometry':
                self.sites+=1
                if node.keywords:raise RuntimeError('Original native tabletop call keywords changed')
                node.keywords.append(ast.keyword(arg='expected_table_top_z_m',value=ast.parse('_expected_table_top_for_task(task)',mode='eval').body))
            return node
    explicit = ExplicitCall();explicit.visit(certificate)
    if explicit.sites != 1:
        raise RuntimeError('Exactly one original task-specific tabletop call required')
    ast.fix_missing_locations(geometry);ast.fix_missing_locations(certificate)
    return geometry,certificate,{'fixed_native_height_guard_sites_changed':height.sites,
        'explicit_actual_recorded_task_table_height_call_sites_changed':explicit.sites,
        'original_native_box_geometry_horizontal_unique_top_and_legs_footprint_checks_retained':True,
        'original_signed_support_path_floor_release_and_footprint_checks_retained':True,
        'support_replaces_only_bottle_table_category':True,
        'all_other_actual_continuous_categories_required':True,
        'core_bottle_robot_self_robot_table_categories_mandatory':True,
        'geometry_derived_ast_sha256':hashlib.sha256(ast.dump(geometry,include_attributes=False).encode()).hexdigest(),
        'certificate_derived_ast_sha256':hashlib.sha256(ast.dump(certificate,include_attributes=False).encode()).hexdigest()}

def build_geometry_callable(original_module,payload):
    geometry,certificate,receipt=derive_support_functions(payload)
    tree=ast.Module(body=[geometry],type_ignores=[]);ast.fix_missing_locations(tree)
    namespace=dict(vars(original_module))
    exec(compile(tree,str(Path(__file__))+'::recorded_native_tabletop','exec'),namespace)
    private=namespace['native_tabletop_geometry']
    def native_tabletop_geometry(component,*,expected_table_top_z_m):
        if not _finite_scalar(expected_table_top_z_m) or not .71-1e-12 <= float(expected_table_top_z_m) <= .74+1e-12:
            raise ValueError('Explicit finite official randomized table top [.71,.74]m required')
        result=private(component,expected_table_top_z_m=float(expected_table_top_z_m))
        result.update(expected_table_top_world_z_m=float(expected_table_top_z_m),
                      actual_native_top_matches_recorded_randomized_height=True,
                      physical_table_height_or_collision_shapes_modified=False)
        return result
    return native_tabletop_geometry,certificate,receipt

def install_randomized_support_guard(legacy_directory):
    """Install once inside the task worker, after original producer imports."""
    global _INSTALLED
    if _INSTALLED is not None:return copy.deepcopy(_INSTALLED)
    legacy_directory=Path(legacy_directory).resolve()
    if str(legacy_directory) not in sys.path:sys.path.insert(0,str(legacy_directory))
    source=legacy_directory/'support_departure_certificate_v20.py'
    if _sha(source)!=ORIGINAL_SUPPORT_SHA256:
        raise RuntimeError('Original support certificate SHA pin differs')
    module=importlib.import_module('support_departure_certificate_v20')
    if Path(module.__file__).resolve()!=source:
        raise RuntimeError('Wrong support certificate module selected')
    geometry,certificate,details=build_geometry_callable(module,source.read_text())
    module.native_tabletop_geometry=geometry
    module._other_categories_pass=strict_other_categories_pass
    tree=ast.Module(body=[certificate],type_ignores=[]);ast.fix_missing_locations(tree)
    namespace=dict(vars(module));namespace['_expected_table_top_for_task']=_expected_table_top_for_task
    exec(compile(tree,str(Path(__file__))+'::strict_randomized_support_certificate','exec'),namespace)
    private=namespace['certify_first_world_support_departure']
    receipt={'installed':True,'source':str(source),'original_source_sha256':ORIGINAL_SUPPORT_SHA256,
        'overlay_source':str(Path(__file__).resolve()),'overlay_source_sha256':_sha(__file__),
        'source_files_modified':False,'physical_state_writes':0,'physics_steps_during_install':0,**details}
    def certify_first_world_support_departure(task,*args,**kwargs):
        expectation=recorded_table_expectation(task)
        result=private(task,*args,**kwargs)
        result['randomized_support_table_expectation']=expectation
        result['randomized_support_overlay']=copy.deepcopy(receipt)
        return result
    module.certify_first_world_support_departure=certify_first_world_support_departure
    # Register this actual executing overlay in the original final source pins.
    expert=importlib.import_module('expert_generation').ExpertSession
    old_init=expert.__init__
    def init(session,*args,**kwargs):
        old_init(session,*args,**kwargs)
        session.sources[str(Path(__file__).resolve())]=receipt['overlay_source_sha256']
        session.report['randomized_support_overlay']=copy.deepcopy(receipt)
        session.report['source_hashes']=session.sources
    expert.__init__=init
    _INSTALLED=receipt
    return copy.deepcopy(receipt)
