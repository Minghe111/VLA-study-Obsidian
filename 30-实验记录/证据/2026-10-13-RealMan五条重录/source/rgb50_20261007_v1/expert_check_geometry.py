"""CPU-only check of reflected task/palm frame and approach direction."""
import numpy as np
from types import SimpleNamespace
from transforms3d.quaternions import quat2mat
from expert_generation import ExpertSession, DEFAULT_SETTING

for axis in ([1., 0., 0.], [.8, .6, 0.], [0., 0., 1.]):
    matrix = np.eye(4)
    axis = np.asarray(axis)
    matrix[:3, 1] = axis
    matrix[:3, 3] = [.14, -.105, .774 if axis[2] == 0 else .864]
    actor = SimpleNamespace(get_pose=lambda: SimpleNamespace(to_transformation_matrix=lambda: matrix),
                            get_contact_point=lambda *args: matrix)
    for side in ("left", "right"):
        session = SimpleNamespace(side=side, setting=DEFAULT_SETTING, report={})
        pre, goal = ExpertSession.grasp_pose(session, actor, side)
        rotation = quat2mat(goal[-4:])
        assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        assert abs(np.linalg.det(rotation) - 1.) < 1e-12
        if axis[2] == 0: assert goal[2] > .74 and pre[2] > goal[2]
        print(side, axis.tolist(), "pre", np.round(pre[:3], 6).tolist(), "goal", np.round(goal[:3], 6).tolist(), "proper_rotation", True)
