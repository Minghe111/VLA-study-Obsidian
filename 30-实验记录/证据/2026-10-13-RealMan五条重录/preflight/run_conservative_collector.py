"""Run one new physical expert candidate with protected source bytes intact.

Default is a CPU/static preparation. --run is coordinated by the parent agent.
The old dataset is never loaded; source metadata supplies only slot and seed.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import sys
import types

import numpy as np
import yaml

TASK_ROOT = Path(__file__).resolve().parent.parent
REPOSITORY = Path('/home/admin123/liminghe/vla-benchmark/RoboTwin')
SOURCE = TASK_ROOT / 'source'
PACKAGE = SOURCE / 'rgb50_20261007_v1'
LEGACY = SOURCE / 'adjust_bottle_20261002'
OLD20 = SOURCE / 'baseline_rgb20/rgb20_final_20261004/dataset'
ARCHIVE = Path('/home/admin123/liminghe/VLA-WAM-RL/30-实验记录/证据/2026-10-11-RealMan专家回放')
FROZEN_BASE = ARCHIVE / 'shared_frozen_runtime/remote_base_task_snapshot.py'
FROZEN_URDF = ARCHIVE / 'shared_frozen_runtime/robot_head_down_fixed.urdf'
OFFICIAL_CONFIG = REPOSITORY / 'env_cfg/task_config/demo_randomized.yml'
INDICES = (17, 18, 20, 29, 34)
PATH_BINDINGS = {
    '/bigdata2/liminghe/VLA-benchmark/RoboTwin': str(REPOSITORY),
    '/bigdata/liminghe/VLA-benchmark/RoboTwin': str(REPOSITORY),
    '/bigdata2/liminghe/VLA-benchmark/deploy/realman_gen4': str(SOURCE),
    '/bigdata/liminghe/VLA-benchmark/deploy/realman_gen4': str(SOURCE),
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2,
        default=lambda v: v.tolist() if isinstance(v, np.ndarray) else str(v)) + '\n')


def map_string(value):
    if value.endswith('/robot_head_down_fixed.urdf'):
        return str(FROZEN_URDF)
    for prefix, local in PATH_BINDINGS.items():
        if value == prefix or value.startswith(prefix + '/'):
            return local + value[len(prefix):]
    return value


def map_config(value):
    if isinstance(value, dict):
        return {key: map_config(item) for key, item in value.items()}
    if isinstance(value, list):
        return [map_config(item) for item in value]
    return map_string(value) if isinstance(value, str) else value


class PathConstantBinding(ast.NodeTransformer):
    def __init__(self):
        self.sites = []

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            local = map_string(node.value)
            if local != node.value:
                self.sites.append({'line': getattr(node, 'lineno', None),
                                   'original': node.value, 'local': local})
                return ast.copy_location(ast.Constant(value=local), node)
        return node


class ProtectedSourceLoader(importlib.machinery.SourceFileLoader):
    """Compile task-local path bindings while inspect/hash reads original bytes."""
    def __init__(self, name, path, receipts):
        super().__init__(name, path)
        self.receipts = receipts

    def get_code(self, fullname):
        path = Path(self.path)
        tree = ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
        binding = PathConstantBinding()
        tree = binding.visit(tree)
        ast.fix_missing_locations(tree)
        self.receipts[fullname] = {'source': str(path), 'original_source_sha256': sha(path),
            'path_constant_sites': binding.sites,
            'compiled_ast_sha256': hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest(),
            'source_bytes_modified': False}
        return compile(tree, str(path), 'exec')


class ProtectedSourceFinder(importlib.abc.MetaPathFinder):
    def __init__(self, receipts):
        self.receipts = receipts

    def find_spec(self, fullname, path=None, target=None):
        if '.' in fullname:
            return None
        for directory in (PACKAGE, LEGACY):
            candidate = directory / (fullname + '.py')
            if candidate.is_file():
                loader = ProtectedSourceLoader(fullname, str(candidate), self.receipts)
                return importlib.util.spec_from_file_location(fullname, candidate, loader=loader)
        return None


def resolve_domain_seed(source_seed, attempt, domain_seed=None):
    if type(source_seed) is not int or type(attempt) is not int or attempt < 0:
        raise ValueError('Source seed and nonnegative attempt must be exact integers')
    actual = source_seed + 10_000_000 * attempt if domain_seed is None else domain_seed
    if type(actual) is not int or not 0 <= actual <= 2**32 - 1:
        raise ValueError('Actual domain seed must be an integer supported by native NumPy initialization')
    return actual


def verify_inputs(index, attempt, output, domain_seed=None):
    selection = SOURCE / 'selected_scenes' / f'episode_{index:07d}'
    spec = json.loads((selection / 'sample_spec.json').read_text())
    if spec['index'] != index or spec['pose_class'] != 'upright':
        raise RuntimeError('Authorized upright slot metadata changed')
    source_seed = int(spec['seed'])
    actual_seed = resolve_domain_seed(source_seed, attempt, domain_seed)
    config = map_config(yaml.safe_load((selection / 'run_config.yml').read_text()))
    official = yaml.safe_load(OFFICIAL_CONFIG.read_text())['domain_randomization']
    expected = {'random_background': True, 'cluttered_table': True,
        'clean_background_rate': .02, 'random_head_camera_dis': 0,
        'random_table_height': .03, 'random_light': True,
        'crazy_random_light_rate': .02}
    if official != expected:
        raise RuntimeError('Official randomized settings differ from reviewed exact configuration')
    config['domain_randomization'] = copy.deepcopy(official)
    config['save_path'] = str(output)
    config['seed'] = actual_seed
    for side in ('left', 'right'):
        if config[side + '_embodiment_config']['urdf_path'] != str(FROZEN_URDF):
            raise RuntimeError('Frozen SAPIEN hardware URDF binding missing')
    files = [FROZEN_BASE, FROZEN_URDF, OFFICIAL_CONFIG,
             OLD20 / 'episodes/episode_0000000/profile_used.yml', OLD20 / 'manifest.json',
             OLD20.parent / 'validate_rgb_three_camera.py']
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise RuntimeError('Original required inputs missing: ' + repr(missing))
    original_scene = json.loads((selection / 'scene_sample.json').read_text())
    if sha(OLD20 / 'episodes/episode_0000000/profile_used.yml') != original_scene['camera_profile_sha256']:
        raise RuntimeError('Original camera profile provenance SHA differs')
    if sha(OLD20 / 'manifest.json') != original_scene['old20_manifest_sha256']:
        raise RuntimeError('Original baseline manifest provenance SHA differs')
    guard = json.loads((ARCHIVE / 'hardware_guard.json').read_text())
    aliases = {
        '/tmp/realman-eval60k-20261011/local_eval/hardware/robot_head_down_fixed.urdf': FROZEN_URDF,
        '/tmp/realman-eval60k-20261011/local_eval/scripts/remote_base_task_snapshot.py': FROZEN_BASE,
    }
    protected = []
    for old, digest in guard['expected_sha256'].items():
        path = aliases.get(old, Path(old))
        if not path.is_file() or sha(path) != digest:
            raise RuntimeError('Protected hardware/runtime source changed: ' + str(path))
        protected.append({'path': str(path), 'sha256': digest})
    download = json.loads((SOURCE / 'download_receipt.json').read_text())
    changed = [name for name, item in download['files'].items()
               if not (SOURCE / name).is_file() or sha(SOURCE / name) != item['sha256']]
    if changed:
        raise RuntimeError('Downloaded source bytes changed: ' + repr(changed[:5]))
    target_config = TASK_ROOT / 'prepared' / f'slot_{index:07d}_attempt_{attempt:03d}_run_config.yml'
    target_config.parent.mkdir(exist_ok=True)
    target_config.write_text(yaml.safe_dump(config, sort_keys=False))
    spec = copy.deepcopy(spec)
    spec['sampling_revision'] = ('Fresh physical randomized candidate, retaining source slot brand/arm/upright '
        'and original sampling metadata; original seed saved separately and actual domain seed changed per attempt '
        'or explicitly selected; official demo_randomized DR enabled; nearest equivalent-goal and quality guards applied. '
        'Original bottle/robot pose and future command arrays are not restored or loaded.')
    spec['source_slot_index'] = index
    spec['source_seed'] = source_seed
    spec['seed'] = actual_seed
    spec['actual_domain_seed'] = actual_seed
    spec['domain_seed_policy'] = 'source_seed + 10000000 * attempt' if domain_seed is None else 'explicit --domain-seed'
    spec['rerecord_attempt'] = attempt
    job = {'name': f'slot_{index:07d}_randomized_attempt_{attempt:03d}', 'spec': spec}
    jobs_path = target_config.with_suffix('.jobs.json')
    write_json(jobs_path, [job])
    receipt = {'index': index, 'attempt': attempt, 'slot': {k: spec[k] for k in ('brand', 'arm', 'pose_class', 'seed')},
        'source_seed': source_seed, 'actual_domain_seed': actual_seed, 'domain_seed_policy': spec['domain_seed_policy'],
        'new_candidate_name': job['name'], 'source_sample_spec_sha256': sha(selection / 'sample_spec.json'),
        'source_run_config_sha256': sha(selection / 'run_config.yml'), 'derived_local_config': str(target_config),
        'derived_local_config_sha256': sha(target_config), 'official_randomization_config': str(OFFICIAL_CONFIG),
        'official_randomization_config_sha256': sha(OFFICIAL_CONFIG), 'domain_randomization': official,
        'source_guard_count': len(download['files']), 'hardware_guard_count': len(protected),
        'protected_input_sha256': protected, 'runtime_path_bindings': PATH_BINDINGS,
        'original_scene_restored': False, 'original_future_commands_loaded': False,
        'original_dataset_unchanged': True, 'new_physical_episode_started': False,
        'source_bytes_modified': False, 'jobs': str(jobs_path)}
    return spec, target_config, jobs_path, receipt


def install_cpu_native_planner_import(path_receipts):
    path = REPOSITORY / 'envs/robot/planner.py'
    tree = ast.parse(path.read_text())
    sites = 0
    body = []
    for statement in tree.body:
        if isinstance(statement, ast.Try) and any(isinstance(node, ast.ClassDef) and node.name == 'CuroboPlanner' for node in ast.walk(statement)):
            body.extend(ast.parse("class CuroboPlanner:\n    def __init__(self,*args,**kwargs):\n        raise RuntimeError('Task requires original native MPLib; optional CuRobo backend must not be instantiated')").body)
            sites += 1
        else:
            body.append(statement)
    if sites != 1:
        raise RuntimeError('Exactly one optional CuRobo import/class block required')
    tree.body = body
    ast.fix_missing_locations(tree)
    module = types.ModuleType('envs.robot.planner')
    module.__package__, module.__file__ = 'envs.robot', str(path)
    sys.modules['envs.robot.planner'] = module
    exec(compile(tree, str(path) + ':native_only_import', 'exec'), module.__dict__)
    path_receipts['envs.robot.planner'] = {'source': str(path), 'original_source_sha256': sha(path),
        'compiled_ast_sha256': hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest(),
        'optional_CuRobo_import_block_skipped': sites, 'CuRobo_instantiation_fails_closed': True,
        'original_MplibPlanner_preserved': True, 'source_bytes_modified': False}


def install_frozen_base_task(path_receipts):
    # Retain references after the official method clears its scratch list.
    # These inserts add no RNG calls, scene steps, object/robot writes or images.
    tree = ast.parse(FROZEN_BASE.read_text())
    clutter_sites = light_sites = 0
    for cls in tree.body:
        if not isinstance(cls, ast.ClassDef) or cls.name != 'Base_Task':
            continue
        for method in cls.body:
            if not isinstance(method, ast.FunctionDef):
                continue
            if method.name == 'get_cluttered_table':
                body = []
                for statement in method.body:
                    if isinstance(statement, ast.If) and len(statement.body) == 1 and isinstance(statement.body[0], ast.Return):
                        statement.body = ast.parse("self._rerecord_clutter_sampling_outcome = 'native_clean_background_draw'").body + statement.body
                    target = statement.targets[0] if isinstance(statement, ast.Assign) and len(statement.targets) == 1 else None
                    if isinstance(target, ast.Attribute) and target.attr == 'cluttered_objs' and isinstance(statement.value, ast.List) and not statement.value.elts:
                        body.extend(ast.parse('self._rerecord_clutter_entities = list(self.cluttered_objs)').body)
                        clutter_sites += 1
                    body.append(statement)
                # Also covers the native 2% clean early-return, without extra draws.
                method.body = ast.parse("self._rerecord_clutter_entities = []\nself._rerecord_clutter_sampling_outcome = 'placement_attempted'").body + body
            elif method.name == 'setup_scene':
                body = []
                for statement in method.body:
                    if isinstance(statement, ast.If) and isinstance(statement.test, ast.Attribute) and statement.test.attr == 'render_freq':
                        body.extend(ast.parse('self._rerecord_initial_lighting = _rerecord_copy_lighting(kwargs, direction_lights, point_lights, shadow)').body)
                        light_sites += 1
                    body.append(statement)
                method.body = body
    if (clutter_sites, light_sites) != (1, 1):
        raise RuntimeError('Official frozen clutter/light instrumentation sites changed')
    ast.fix_missing_locations(tree)
    import envs
    module = types.ModuleType('envs._base_task')
    module.__package__ = 'envs'
    module.__file__ = str(FROZEN_BASE)
    module.__dict__['_rerecord_copy_lighting'] = lambda kwargs, direction, point, shadow: copy.deepcopy({
        'ambient_light': kwargs.get('ambient_light', [.5, .5, .5]), 'direction_lights': direction,
        'point_lights': point, 'shadow': shadow})
    sys.modules['envs._base_task'] = module
    exec(compile(tree, str(FROZEN_BASE) + ':readonly_instrumentation', 'exec'), module.__dict__)
    path_receipts['envs._base_task'] = {'source': str(FROZEN_BASE), 'original_source_sha256': sha(FROZEN_BASE),
        'compiled_ast_sha256': hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest(),
        'clutter_reference_copy_sites': clutter_sites, 'lighting_parameter_copy_sites': light_sites,
        'physical_behavior_changed': False, 'extra_random_draws': 0, 'extra_physics_steps': 0,
        'source_bytes_modified': False}


def randomization_receipt(task, spec, scene):
    actual = {key: getattr(task, key) for key in (
        'random_background', 'cluttered_table', 'clean_background_rate', 'random_head_camera_dis',
        'random_table_height', 'random_light', 'crazy_random_light_rate')}
    expected = yaml.safe_load(OFFICIAL_CONFIG.read_text())['domain_randomization']
    if actual != expected:
        raise RuntimeError('Actual native DR fields differ from official requested settings')
    clutter = list(task._rerecord_clutter_entities)
    records = copy.deepcopy(task.record_cluttered_objects)
    if len(clutter) != len(records):
        raise RuntimeError('Native retained clutter count differs from official object records')
    actors = []
    for actor in task.scene.get_all_actors():
        actors.append({'name': actor.get_name(), 'pose_matrix': np.asarray(actor.get_pose().to_transformation_matrix()).tolist()})
    cameras = {}
    task.cameras.update_mount(task.robot)
    task.scene.update_render()
    for name, camera in task.cameras.camera_by_name.items():
        cameras[name] = {'intrinsic_matrix': np.asarray(camera.get_intrinsic_matrix()).tolist(),
                         'native_model_matrix': np.asarray(camera.get_model_matrix()).tolist()}
    return {'passed': True, 'official_domain_randomization': expected, 'actual_domain_randomization': actual,
        'source_slot_index': int(spec['index']), 'source_seed': int(spec['source_seed']),
        'actual_seed': int(spec['seed']), 'table_z_bias_m': float(task.table_z_bias),
        'wall_texture': task.wall_texture, 'table_texture': task.table_texture,
        'crazy_random_light_sampled': bool(task.crazy_random_light),
        'initial_light_parameters': task._rerecord_initial_lighting,
        'native_ambient_light_at_setup_end': np.asarray(task.scene.ambient_light).tolist(),
        'clutter_count': len(clutter), 'official_clutter_records': records,
        'clutter_sampling_outcome': task._rerecord_clutter_sampling_outcome,
        'all_actor_initial_poses': actors, 'actual_initial_bottle_pose': scene['initial_pose'],
        'initial_robot_full_qpos_rad': np.asarray(task.robot.left_entity.get_qpos()).tolist(),
        'camera_mount_info': task.cameras.mount_info(), 'camera_native_initial_matrices': cameras,
        'original_source_pose_restored': False, 'objects_settled_by_original_gravity_and_stability_chain': True,
        'hardware_io': False}


def read_hardware_contract(task):
    profile = yaml.safe_load((OLD20 / 'episodes/episode_0000000/profile_used.yml').read_text())
    robot = task.robot
    if robot.communication_flag:
        raise RuntimeError('Robot IO unexpectedly enabled')
    names = [joint.get_name() for joint in robot.left_entity.get_active_joints()]
    if len(names) != 36 or len(robot.left_entity.get_links()) != 93:
        raise RuntimeError('Current frozen native robot joint/link inventory differs')
    joints = {joint.get_name(): joint for joint in robot.left_entity.get_active_joints()}
    arm, hand = {}, {}
    for side in ('left', 'right'):
        cfg = robot._hand_config[side]
        if cfg['urdf_path'] != str(FROZEN_URDF):
            raise RuntimeError('Actual hardware config uses another URDF')
        for name in getattr(robot, side + '_arm_joints_name'):
            joint = joints[name]
            actual = {'stiffness': float(joint.stiffness), 'damping': float(joint.damping),
                'force_limit': float(joint.force_limit), 'drive_mode': str(joint.drive_mode)}
            expected = {'stiffness': float(cfg.get('joint_stiffness', 1000)),
                'damping': float(cfg.get('joint_damping', 200)),
                'force_limit': float(robot._manifest['arm_limits'][name]['effort']), 'drive_mode': 'force'}
            if not all(np.isfinite(actual[key]) for key in ('stiffness', 'damping', 'force_limit')) or actual != expected:
                raise RuntimeError('Actual arm drive contract differs: ' + name)
            arm[name] = actual
        primary = set(robot._hand_maps[side].channels)
        for name in robot._hand_maps[side].limits:
            joint = joints[name]
            actual = {'stiffness': float(joint.stiffness), 'damping': float(joint.damping),
                'force_limit': float(joint.force_limit), 'drive_mode': str(joint.drive_mode),
                'friction': float(joint.friction)}
            expected = {'stiffness': 20. if name in primary else 0., 'damping': .2 if name in primary else 0.,
                'force_limit': 10. if name in primary else 0., 'drive_mode': 'force', 'friction': .001}
            if (not all(np.isfinite(actual[key]) and abs(actual[key] - expected[key]) <= 1e-7
                        for key in ('stiffness', 'damping', 'force_limit', 'friction'))
                    or actual['drive_mode'] != expected['drive_mode']):
                raise RuntimeError('Actual hand drive contract differs: ' + name)
            hand[name] = actual
    if len(arm) != 12 or len(hand) != 24:
        raise RuntimeError('Native independent arm/hand drive inventory differs')
    coupling = robot.hand_coupling_audit
    if len(coupling) != 12 or any(row['stiffness'] != 100 or row['follower_drive'] != 0 for row in coupling):
        raise RuntimeError('Native fixed-tendon installation receipt differs')
    if task.cameras.mounts != profile['mounts'] or task.cameras.rgb_profile != profile['rgb_profile']:
        raise RuntimeError('Actual three-camera mounted profile differs from original')
    dt = float(task.scene.get_timestep())
    if not np.isfinite(dt) or abs(dt - .004) > 1e-9:
        raise RuntimeError('Actual 4ms physics timestep differs')
    from integrations.realman_rm65b_gen4.hand_coupling import material_snapshot
    materials = material_snapshot(robot.left_entity, robot._manifest)
    if not any(row['is_hand'] for row in materials):
        raise RuntimeError('No native hand collision shapes read')
    for row in materials:
        if not np.isfinite(row['material']).all():
            raise RuntimeError('Nonfinite native collision material: ' + row['link'])
        if row['is_hand'] and not np.allclose(row['material'], [.5, .5, 0.], rtol=0., atol=1e-7):
            raise RuntimeError('Current hand collision material differs: ' + row['link'])
    masses, offsets = {}, []
    for link in robot.left_entity.get_links():
        mass = float(link.mass)
        inertia = np.asarray(link.inertia, dtype=float)
        cmass = np.asarray(link.cmass_local_pose.to_transformation_matrix(), dtype=float)
        if not np.isfinite(mass) or mass < 0 or not np.isfinite(inertia).all() or not np.isfinite(cmass).all():
            raise RuntimeError('Nonfinite native mass/inertia state: ' + link.get_name())
        masses[link.get_name()] = {'mass_kg': mass, 'principal_inertia_kg_m2': inertia.tolist(),
                                  'cmass_local_matrix': cmass.tolist()}
        for i, shape in enumerate(link.get_collision_shapes()):
            contact = float(shape.contact_offset)
            rest = float(shape.rest_offset)
            if not np.isfinite([contact, rest]).all() or abs(contact - .01) > 1e-7 or abs(rest) > 1e-7:
                raise RuntimeError('Current native robot collision offsets differ: ' + link.get_name())
            offsets.append({'link': link.get_name(), 'shape': i, 'contact_offset_m': contact,
                            'rest_offset_m': rest, 'collision_groups': list(shape.get_collision_groups())})
    if not getattr(robot, 'preserve_vendor_inertia', False):
        raise RuntimeError('Native vendor-inertia preservation path disabled')
    return {'passed': True, 'frozen_sapien_urdf': str(FROZEN_URDF), 'frozen_sapien_urdf_sha256': sha(FROZEN_URDF),
        'actual_joint_names': names, 'actual_joint_count': len(names), 'actual_link_count': len(robot.left_entity.get_links()),
        'actual_arm_drives': arm, 'actual_hand_drives': hand, 'actual_tendon_installation_receipt': coupling,
        'tendon_readback_scope': 'Original SHA-pinned callback installation receipt; native tendon properties are not claimed independently read back',
        'actual_hand_material': robot.hand_material_audit, 'actual_hand_friction': robot.hand_friction_audit,
        'current_native_collision_materials': materials, 'current_native_collision_offsets_and_groups': offsets,
        'current_native_link_mass_and_inertia': masses, 'vendor_inertia_preservation_enabled': True,
        'mass_inertia_scope': 'Actual native readback and frozen URDF/source identity; no additional runtime mass write',
        'actual_three_camera_mounts': task.cameras.mount_info(), 'actual_physics_timestep_s': dt,
        'observation_physics_step_stride': 15, 'hardware_io': False}


def export_branch_receipt(out, spec):
    events, limits, types_by_arm = [], {}, {}
    for path in sorted(Path(out).glob('free_approach_*.json')):
        data = json.loads(path.read_text())
        meta = data['planning_metadata']
        if data['status'] != 'Success':
            continue
        side = data['side']
        indices = meta['selected_arm_indices']
        reference = np.asarray(meta['initial_actual_full_qpos_rad'])[indices].tolist()
        limits[side] = meta['native_commanded_limits_rad']
        types_by_arm[side] = meta['native_commanded_URDF_types']
        accepted = [row for row in meta['RRT_attempts'] if 'rejection_reason' not in row and
                    row.get('RRT_status', '').rstrip('.') == 'Exact solution']
        if len(accepted) != 1:
            raise RuntimeError('Cannot identify unique actually accepted approach branch')
        row = accepted[0]
        selection = row['near_numeric_goal_selection']
        endpoint = next(item for item in meta['endpoint_attempts'] if item['attempt'] == row['IK_attempt'])
        declaration = endpoint['equivalent_goal_declaration']
        candidates = [{'q_rad': item['arm_qpos_rad'], 'native_FK_equivalent': item['native_FK_equivalent']}
                      for item in declaration['goal_attempts'] if item['inside_exact_native_commanded_limits'] and item.get('native_FK_equivalent')]
        events.append({'arm': side, 'stage': 'free_approach_' + str(data['index']),
            'sim_time_s': meta['rerecord_planning_sim_time_s'], 'reference_kind': 'measured',
            'reference_q_rad': reference, 'raw_ik_q_rad': np.asarray(row['original_IK_full_qpos_rad'])[indices].tolist(),
            'selected_q_rad': selection['chosen_arm_qpos_rad'], 'candidates': candidates,
            'selection_policy': 'nearest_periodic_equivalent', 'actual_source_receipt_path': str(path),
            'actual_source_receipt_sha256': sha(path), 'selected_RRT_branch': row['branch'],
            'candidate_source': 'Actual original finite native-FK declared goal_attempts, unmodified',
            'original_full_native_collision_checks': endpoint['declared_goal_collision_checks'],
            'chosen_target_FK_proof': row['chosen_goal_target_FK_proof']})
    if not events:
        raise RuntimeError('Actual branch receipt has no executed expert approaches')
    path = Path(out) / 'nearest_periodic_branch_receipt.json'
    write_json(path, {'schema': 'nearest-periodic-branch-v1', 'source_slot_index': int(spec['index']),
        'arm_joint_limits_rad': limits, 'arm_joint_types': types_by_arm, 'events': events,
        'events_from_actual_planner_receipts': True, 'future_original_commands_used': False})
    return path


def install_runtime(index, config, output, receipts):
    os.environ.setdefault('ROBOTWIN_RT_DENOISER', 'none')
    os.environ['RGB50_EXPERT_MODULE'] = 'expert_v9_settle_wake_v5'
    sys.path[:0] = [str(TASK_ROOT / 'planning'), str(TASK_ROOT / 'quality'), str(PACKAGE), str(LEGACY), str(REPOSITORY)]
    sys.meta_path.insert(0, ProtectedSourceFinder(receipts))
    os.chdir(REPOSITORY)
    install_cpu_native_planner_import(receipts)
    install_frozen_base_task(receipts)
    original = importlib.import_module('generation_runner_v2')
    original.ROOT, original.LEGACY, original.OLD20, original.CONFIG = REPOSITORY, LEGACY, OLD20, config
    collector = importlib.import_module('collector_settle_wake_v5')
    tree = copy.deepcopy(collector._TREE)
    function = tree.body[0]
    body = []
    sites = 0
    for statement in function.body:
        call = statement.value if isinstance(statement, ast.Expr) else None
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name) and call.func.value.id == 'args' and call.func.attr == 'update':
            body.extend(ast.parse("args['domain_randomization'] = copy.deepcopy(_RERECORD_OFFICIAL_DR)").body)
            sites += 1
        body.append(statement)
    if sites != 1:
        raise RuntimeError('Exactly one original args update site required')
    function.body = body
    ast.fix_missing_locations(tree)
    collector._ENV['_RERECORD_OFFICIAL_DR'] = yaml.safe_load(OFFICIAL_CONFIG.read_text())['domain_randomization']
    exec(compile(tree, str(PACKAGE / 'collector_settle_wake_v5.py') + ':randomized_prepare', 'exec'), collector._ENV)
    collector._PRIVATE_PREPARE = collector._ENV['prepare_task']
    receipts['randomized_prepare'] = {'source': str(PACKAGE / 'generation_runner_v2.py'),
        'original_source_sha256': sha(PACKAGE / 'generation_runner_v2.py'),
        'compiled_ast_sha256': hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest(),
        'official_DR_assignment_sites': sites, 'load_actors_unchanged': True,
        'initial_bottle_pose_restored': False, 'source_bytes_modified': False}
    import randomized_environment_guard as environment
    environment_receipt = environment.install_environment_overlay(PACKAGE, LEGACY)
    import collector_quality_overlay as planning
    planning_receipt = planning.install_planning_quality(source_package=PACKAGE, legacy_directory=LEGACY)
    import auxiliary_window_overlay as numeric_window
    window_receipt = numeric_window.install_auxiliary_window_overlay()
    import conservative_ompl_overlay as conservative_ompl
    conservative_receipt = conservative_ompl.install_conservative_ompl_overlay()
    for module_name in ('full_native_approach_equivalent_goals_v6', 'full_native_approach_equivalent_goals_dense_v7'):
        module = importlib.import_module(module_name)
        original_plan = module.plan_free_approach
        def plan(*args, _original=original_plan, **kwargs):
            task = args[0]
            guard = task._rerecord_environment_guard
            planning_time = float(guard['steps_checked'] * task.scene.get_timestep())
            result = _original(*args, **kwargs)
            result.setdefault('planning_metadata', {})['rerecord_planning_sim_time_s'] = planning_time
            return result
        module.plan_free_approach = plan
    old_prepare = collector.prepare_task

    def prepare(spec, out, **kwargs):
        task, scene, initializer = old_prepare(spec, out, **kwargs)
        try:
            random = randomization_receipt(task, spec, scene)
            task._rerecord_randomization_receipt = random
            write_json(Path(out) / 'randomized_scene_receipt.json', random)
            hardware = read_hardware_contract(task)
            write_json(Path(out) / 'actual_hardware_camera_clock_receipt.json', hardware)
            environment.prepare_environment(task, out, scene, spec)
            scene['rerecord_randomization'] = random
            collector.json_write(Path(out) / 'scene_sample.json', scene)
            write_json(Path(out) / 'runtime_binding_receipt.json', receipts)
            return task, scene, initializer
        except BaseException:
            task.close_env()
            raise

    collector.prepare_task = prepare
    original_export = collector.export_recording
    def export(task, out, report, frames):
        out = Path(out)
        env_receipt = environment.finalize_environment(task, out, report)
        report_path = out / 'expert_report.json'
        write_json(report_path, report)
        # Preserve the real original exporter and every RGB/SDK timeline field.
        hdf = original_export(task, out, report, frames)
        hardware = read_hardware_contract(task)
        write_json(out / 'actual_hardware_camera_clock_after_receipt.json', hardware)
        spec = json.loads((out / 'sample_spec.json').read_text())
        branch = export_branch_receipt(out, spec)
        def evidence(path, passed):
            return {'passed': bool(passed), 'evidence_path': str(path), 'evidence_sha256': sha(path)}
        geometry_path = out / 'actual36_geometry_audit.json'
        geometry = json.loads(geometry_path.read_text())
        upstream = {
            'original_physical_success': evidence(report_path, report['accepted'] is True and report['result'] == 'physics_success'),
            'original_airhold_acceptance': evidence(report_path, report['hold_passed'] is True and report['hold_validation_s'] >= 1 - 1e-7 and report['minimum_hold_bottle_gap_m'] >= .01),
            'original_safety_acceptance': evidence(geometry_path, geometry.get('passed') is True and report['canonical_sources_unchanged'] is True),
            'official_demo_randomized_actual_parameters': evidence(out / 'randomized_scene_receipt.json', task._rerecord_randomization_receipt['passed']),
            'actual_clutter_collision_guard': evidence(out / 'randomized_environment_execution.json', env_receipt['passed']),
            'hardware_camera_physics_clock_unchanged': evidence(out / 'actual_hardware_camera_clock_after_receipt.json', hardware['passed']),
        }
        upstream_path = out / 'upstream_acceptance.json'
        write_json(upstream_path, upstream)
        from motion_gate import evaluate_candidate
        quality = evaluate_candidate(hdf, {key: spec[key] for key in ('index', 'arm', 'pose_class', 'brand', 'seed')},
            config_path=TASK_ROOT / 'quality/thresholds_confirmed.json', branch_receipt_path=branch,
            upstream_acceptance_path=upstream_path)
        write_json(out / 'independent_motion_quality.json', quality)
        recording_path = out / 'recording_report.json'
        recording = json.loads(recording_path.read_text())
        recording['original_recording_export_accepted'] = recording['accepted']
        recording['accepted'] = bool(quality['overall_publish_gate_passed'])
        recording['additional_quality_report'] = str(out / 'independent_motion_quality.json')
        recording['additional_quality_report_sha256'] = sha(out / 'independent_motion_quality.json')
        write_json(recording_path, recording)
        if not recording['accepted']:
            raise RuntimeError('Independent fresh demonstration quality gate rejected candidate; retained diagnostic files')
        return hdf
    collector.export_recording = export
    worker = importlib.import_module('worker_settle_wake_v5')
    old_worker_tree = worker.private_worker_tree

    def worker_tree(payload):
        tree = old_worker_tree(payload)
        binding = PathConstantBinding()
        tree = binding.visit(tree)
        ast.fix_missing_locations(tree)
        if not any(site['original'].endswith('/RoboTwin') for site in binding.sites):
            raise RuntimeError('Original worker ROOT path binding absent')
        receipts['original_worker_AST'] = {'original_source_sha256': sha(PACKAGE / 'generation_worker_v2.py'),
            'path_constant_sites': binding.sites, 'source_bytes_modified': False,
            'compiled_ast_sha256': hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()}
        return tree

    worker.private_worker_tree = worker_tree
    return worker, {'environment': environment_receipt, 'planning': planning_receipt,
                    'auxiliary_numeric_window': window_receipt,
                    'conservative_native_ompl': conservative_receipt}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, choices=INDICES, default=17)
    parser.add_argument('--attempt', type=int, default=0)
    parser.add_argument('--domain-seed', type=int, default=None,
        help='Explicit actual scene/IK RNG seed; default source_seed + 10000000 * attempt')
    parser.add_argument('--output', type=Path, default=TASK_ROOT / 'new_physical_episodes')
    parser.add_argument('--timeout', type=int, default=1800)
    parser.add_argument('--prepare-runtime', action='store_true', help='CPU import/AST binding only, no scene constructed')
    parser.add_argument('--run', action='store_true', help='Run the original genuine physical collector after parent coordination')
    args = parser.parse_args()
    if args.attempt < 0:
        raise ValueError('Attempt must be nonnegative')
    output = args.output.resolve()
    output.relative_to(TASK_ROOT)
    spec, config, jobs, receipt = verify_inputs(args.index, args.attempt, output, args.domain_seed)
    destination = TASK_ROOT / 'preflight' / f'randomized_slot_{args.index:07d}_attempt_{args.attempt:03d}_prepare_receipt.json'
    write_json(destination, receipt)
    print(json.dumps({key: receipt[key] for key in ('slot', 'jobs', 'domain_randomization', 'source_guard_count', 'hardware_guard_count')}, ensure_ascii=False), flush=True)
    if not args.run and not args.prepare_runtime:
        return 0
    receipts = {}
    worker, addons = install_runtime(args.index, config, output, receipts)
    receipt.update(runtime_binding_installed=True, runtime_binding_receipt=receipts, addons=addons)
    write_json(destination, receipt)
    if not args.run:
        print('CPU_RUNTIME_BINDING_READY_NO_SCENE', flush=True)
        return 0
    candidate = output / ('slot_%07d_randomized_attempt_%03d' % (args.index, args.attempt))
    if candidate.exists():
        raise RuntimeError('Refusing to overwrite/skip a prior candidate: ' + str(candidate))
    receipt['new_physical_episode_started'] = True
    write_json(destination, receipt)
    previous = sys.argv
    sys.argv = [str(PACKAGE / 'worker_settle_wake_v5.py'), '--jobs', str(jobs), '--output', str(output),
        '--backend', 'full_native', '--timeout', str(args.timeout), '--expert-module', 'expert_v9_settle_wake_v5']
    try:
        worker.main()
    finally:
        sys.argv = previous
        write_json(TASK_ROOT / 'preflight' / f'randomized_slot_{args.index:07d}_runtime_source_receipt.json', receipts)
    status = json.loads((output / 'worker_status.json').read_text())
    if not status.get('worker_ended') or len(status['receipts']) != 1 or not status['receipts'][0].get('accepted'):
        raise RuntimeError('Physical candidate failed; see retained native worker report')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
