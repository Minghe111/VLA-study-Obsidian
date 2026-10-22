
from envs._base_task import Base_Task
from envs.labware_pick_place import labware_pick_place
from envs.utils import *
import sapien

class gpt_labware_pick_place(labware_pick_place):
    def play_once(self):
        self.grasp_lift_place(self.bottle, ArmTag('right'))
        # setup_demo already records the exact object description and arm.
        return self.info

