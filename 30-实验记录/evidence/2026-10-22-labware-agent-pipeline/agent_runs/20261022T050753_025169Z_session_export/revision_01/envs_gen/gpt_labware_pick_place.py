"""Generated in the authorized Codex conversation, not by an API request.

Only play_once is generated. Reset, object loading, physical success and the
explicit RM65-B SDK24 grasp primitive belong to the task implementation.
"""
from envs.labware_pick_place import labware_pick_place
from envs.utils.action import ArmTag


class gpt_labware_pick_place(labware_pick_place):
    def play_once(self):
        self.grasp_lift_place(self.bottle, ArmTag('right'))
        # setup_demo already records the exact object description and arm.
        return self.info
