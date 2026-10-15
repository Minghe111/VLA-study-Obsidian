"""Real installed native component APIs, without a Scene, renderer or GPU run."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
import numpy as np
import sapien
import sapien.physx as physx

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / 'planning'), str(ROOT / 'source/adjust_bottle_20261002')]
import randomized_environment_guard as environment


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def entity(name, component, pose=(0., 0., 0.)):
    result = sapien.Entity()
    result.name = name
    result.add_component(component)
    result.set_pose(sapien.Pose(pose))
    return result


def main():
    parent = physx.PhysxArticulationLinkComponent()
    parent_entity = entity('api_cpu_root', parent)
    child = physx.PhysxArticulationLinkComponent(parent)
    child_entity = entity('api_cpu_joint_child', child)
    joint = child.joint
    joint.type = 'revolute'
    joint.set_limits([[-6.3, 6.3]])
    joint.set_drive_property(20., .2, 10., mode='force')
    joint.set_friction(.001)
    commanded = .912345678
    joint.set_drive_target(commanded)
    joint.set_drive_velocity_target(0.)
    target = joint.get_drive_target()
    properties = {key: getattr(joint, key) for key in ('stiffness', 'damping', 'force_limit', 'drive_mode', 'friction')}
    expected = {'stiffness': 20., 'damping': .2, 'force_limit': 10., 'drive_mode': 'force', 'friction': .001}
    assert all(abs(properties[key] - value) <= 1e-7 if isinstance(value, float) else properties[key] == value for key, value in expected.items())
    assert target.dtype == np.float32
    material = physx.PhysxMaterial(.5, .5, 0.)
    child.attach(physx.PhysxCollisionShapeBox(np.array([.01, .02, .03]), material))
    parent.attach(physx.PhysxCollisionShapeBox(np.array([.02, .02, .02]), material))
    rigid_component = physx.PhysxRigidDynamicComponent()
    rigid_component.attach(physx.PhysxCollisionShapeBox(np.array([.01, .02, .03]), material))
    rigid = entity('cpu_generated_clutter_rigid', rigid_component, (.1, .2, .8))
    refs = [SimpleNamespace(actor=rigid), SimpleNamespace(actor=parent.articulation)]
    task = SimpleNamespace(_rerecord_clutter_entities=refs,
        record_cluttered_objects=[{'type': 'rigid_cpu_component'}, {'type': 'articulated_cpu_component'}])
    components = environment._components(task)
    snapshots = environment._snapshot_clutter(task)
    coverage = environment._coverage(snapshots, task)
    assert len(components) == 2 and len(snapshots) == 3
    assert coverage['passed'] and coverage['expected_clutter_count'] == 2 and coverage['covered_shape_count'] == 3
    assert all(row['vertices'].shape == (8, 3) and np.isfinite(row['initial_matrix']).all() for row in snapshots)
    assert set(coverage['static_categories']) == {'clutter_00', 'clutter_01'}
    shape = rigid_component.collision_shapes[0]
    assert abs(shape.contact_offset - .01) <= 1e-7 and shape.rest_offset == 0.
    manifest_path = Path('/home/admin123/liminghe/vla-benchmark/RoboTwin/assets/embodiments/realman-rm65b-inspire-gen4/source_manifest.json')
    manifest = json.loads(manifest_path.read_text())
    xml_path = Path('/home/admin123/liminghe/VLA-WAM-RL/30-实验记录/证据/2026-10-11-RealMan专家回放/shared_frozen_runtime/robot_head_down_fixed.urdf')
    xml = ET.parse(xml_path).getroot()
    masses = {link.attrib['name']: float(link.find('inertial/mass').attrib['value']) for link in xml.findall('link') if link.find('inertial/mass') is not None}
    assert np.isfinite(list(masses.values())).all()
    report = {'passed': True, 'scope': 'Installed SAPIEN standalone native CPU component/API probes and static hardware sources; not a completed randomized robot scene or collection',
        'scene_constructed': False, 'renderer_constructed': False, 'gpu_run': False,
        'actual_native_joint_class': type(joint).__name__, 'actual_joint_drive_properties': properties,
        'hand_readback_representation_tolerance_rad_or_scalar': 1e-7,
        'drive_target': target.tolist(), 'drive_target_dtype': str(target.dtype),
        'drive_target_float32_storage_error': float(abs(target[0] - commanded)),
        'native_default_contact_offset_m': float(shape.contact_offset), 'native_default_rest_offset_m': float(shape.rest_offset),
        'native_material': [material.static_friction, material.dynamic_friction, material.restitution],
        'rigid_and_articulated_clutter_component_expansion': coverage,
        'real_native_component_snapshot_count': len(snapshots),
        'actual_articulation_links_and_get_links_agree': len(parent.articulation.links) == len(parent.articulation.get_links()),
        'static_vendor_arm_effort_limits': {key: value['effort'] for key, value in manifest['arm_limits'].items()},
        'vendor_primary_hand_channel_counts': {side: len(item['channels']) for side, item in manifest['hands'].items()},
        'vendor_hand_mimic_counts': {side: len(item['mimic']) for side, item in manifest['hands'].items()},
        'frozen_urdf_link_count': len(xml.findall('link')), 'frozen_urdf_inertial_mass_values': masses,
        'source_manifest_sha256': digest(manifest_path), 'frozen_urdf_sha256': digest(xml_path),
        'probe_sha256': digest(__file__), 'environment_overlay_sha256': digest(environment.__file__)}
    output = ROOT / 'preflight/native_cpu_api_probe_receipt.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'passed': True, 'target_dtype': str(target.dtype), 'native_clutter_components': len(snapshots),
                      'receipt': str(output)}, indent=2))


if __name__ == '__main__':
    main()
