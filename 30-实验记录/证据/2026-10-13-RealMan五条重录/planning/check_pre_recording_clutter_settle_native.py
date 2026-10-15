"""Real SAPIEN CPU physics/component checks; no real robot or renderer."""
from pathlib import Path
import argparse
import json
import sys
import types

import numpy as np
import sapien
from sapien.physx import PhysxCpuSystem

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from test_pre_recording_clutter_settle import Art, Entity, extracted_native_geometry
import pre_recording_clutter_settle as settle

# The exact unmodified native extraction functions have no renderer/robot need.
sys.modules['full_mesh_path_audit'] = extracted_native_geometry()


def native_task(moving_child=False, gravity_drop=False):
    scene = sapien.Scene([PhysxCpuSystem()])
    scene.set_timestep(.004)
    refs = []
    for i in range(9):
        builder = scene.create_actor_builder()
        builder.add_box_collision(half_size=[.02, .02, .02])
        actor = builder.build(name='native_clutter_%d' % i) if gravity_drop else builder.build_static(name='native_clutter_%d' % i)
        actor.set_pose(sapien.Pose([2 + .2 * i, 2, .5]))
        refs.append(actor)
    builder = scene.create_articulation_builder()
    root = builder.create_link_builder()
    root.set_name('native_fixed_root')
    root.add_box_collision(half_size=[.02, .02, .02])
    child = builder.create_link_builder(root)
    child.set_name('native_gravity_child')
    child.add_box_collision(half_size=[.02, .02, .02])
    child.set_joint_properties('prismatic' if moving_child or gravity_drop else 'fixed',
        [[-1, 1]] if moving_child or gravity_drop else [],
        sapien.Pose([0, 0, .5], [2 ** -.5, 0, 2 ** -.5, 0]), sapien.Pose())
    articulation = builder.build(fix_root_link=True)
    articulation.set_root_pose(sapien.Pose([4, 2, 1]))
    refs.append(articulation)
    if gravity_drop:
        builder = scene.create_actor_builder()
        builder.add_box_collision(half_size=[5, 5, .05])
        floor = builder.build_static(name='ground')
        floor.set_pose(sapien.Pose([0, 0, -.05]))
    builder = scene.create_actor_builder()
    builder.add_box_collision(half_size=[.02, .03, .02])
    bottle = builder.build_static(name='native_bottle')
    bottle.set_pose(sapien.Pose([-.2, 0, .8], [2 ** -.5, 2 ** -.5, 0, 0]))
    task = types.SimpleNamespace(scene=scene, _rerecord_clutter_entities=refs,
        record_cluttered_objects=[{'object_type': 'native_component_fixture', 'object_index': i} for i in range(10)],
        robot=types.SimpleNamespace(communication_flag=False, left_entity=Art([Entity('mock_robot_readback')]), right_entity=None),
        bottle=types.SimpleNamespace(actor=bottle, get_pose=bottle.get_pose))
    return task, articulation


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    spec = {'arm': 'left', 'pose_class': 'upright'}
    results = []
    task, articulation = native_task()
    receipt = settle.settle_clutter(task, output / 'stable_native_multilink', {}, spec,
                                    max_extra_s=.08, window_s=.02)
    settle.validate_before_expert(task, spec)
    assert receipt['passed'] and receipt['completed_extra_physics_steps'] == 5
    assert receipt['coverage']['covered_collision_component_count'] == 11
    results.append({'case': 'stable_native_multilink', 'passed': True,
        'observed_preparation_steps': receipt['completed_extra_physics_steps'],
        'observed_native_collision_components': receipt['coverage']['covered_collision_component_count']})
    task, articulation = native_task(moving_child=True)
    before = [np.asarray(link.entity.pose.to_transformation_matrix()).copy() for link in articulation.get_links()]
    try:
        settle.settle_clutter(task, output / 'moving_native_child_rejected', {}, spec,
                             max_extra_s=.08, window_s=.02)
        raise AssertionError('A real gravity-driven child was incorrectly accepted')
    except RuntimeError as error:
        assert 'did not stabilize' in str(error)
    after = [np.asarray(link.entity.pose.to_transformation_matrix()).copy() for link in articulation.get_links()]
    assert np.array_equal(before[0], after[0])
    child_displacement = float(np.linalg.norm(after[1][:3, 3] - before[1][:3, 3]))
    assert child_displacement > .001
    receipt = json.loads((output / 'moving_native_child_rejected/pre_recording_clutter_settle_receipt.json').read_text())
    assert receipt['completed_extra_physics_steps'] == 20
    assert receipt['last_step']['worst_component_from_preparation_start'] == 'clutter_09_01'
    results.append({'case': 'moving_native_child_rejected', 'passed': True,
        'actual_fixed_root_pose_unchanged': True, 'actual_child_translation_m': child_displacement,
        'observed_preparation_steps': 20, 'correctly_identified_component': 'clutter_09_01'})
    task, articulation = native_task(gravity_drop=True)
    receipt = settle.settle_clutter(task, output / 'real_gravity_settle_default_window', {}, spec)
    settle.validate_before_expert(task, spec)
    assert receipt['passed'] and 250 <= receipt['completed_extra_physics_steps'] <= 1000
    assert receipt['coverage']['covered_collision_component_count'] == 11
    results.append({'case': 'real_gravity_settle_default_window', 'passed': True,
        'observed_preparation_steps': receipt['completed_extra_physics_steps'],
        'observed_extra_simulated_seconds': receipt['actual_extra_simulated_seconds'],
        'actual_final_window_displacement_upper_bound_m': receipt['last_step']['window_pairwise_vertex_displacement_upper_bound_m'],
        'window_steps': 250, 'maximum_steps': 1000})
    final = {'passed': True, 'native_sapien_version': sapien.__version__, 'results': results,
        'component_scope': 'Real PhysX CPU scene with nine rigid colliders and a two-collidable-link articulation; mocked fixed robot shoulder/qpos readback only.',
        'official_randomized_scene_or_expert_episode_run': False,
        'real_robot_hardware_io': False, 'renderer_created': False,
        'physics_rate_s': .004, 'source_hashes': settle.producer_pins()}
    (output / 'native_component_checks.json').write_text(json.dumps(final, indent=2) + '\n')
    print(json.dumps(final, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
