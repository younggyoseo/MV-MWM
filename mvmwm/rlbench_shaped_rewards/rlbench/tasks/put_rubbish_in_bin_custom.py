from typing import List, Tuple
import numpy as np
import copy
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape
from rlbench.backend.task import Task
from rlbench.backend.conditions import DetectedCondition, GraspedCondition


class PutRubbishInBin(Task):
    def init_task(self):
        self.success_sensor = ProximitySensor("success")
        self.rubbish = Shape("rubbish")
        self.register_graspable_objects([self.rubbish])
        self.register_success_conditions(
            [DetectedCondition(self.rubbish, self.success_sensor)]
        )
        self._grasped_cond = GraspedCondition(self.robot.gripper, self.rubbish)
        self._detected_cond = DetectedCondition(self.rubbish, self.success_sensor)
        self.Z_TARGET = 1.0

    def init_episode(self, index: int) -> List[str]:
        tomato1 = Shape("tomato1")
        tomato2 = Shape("tomato2")
        x1, y1, z1 = tomato2.get_position()
        x2, y2, z2 = self.rubbish.get_position()
        x3, y3, z3 = tomato1.get_position()
        # pos = np.random.randint(3)
        pos = 0
        if pos == 0:
            self.rubbish.set_position([x1, y1, z2])
            tomato2.set_position([x2, y2, z1])
        elif pos == 2:
            self.rubbish.set_position([x3, y3, z2])
            tomato1.set_position([x2, y2, z3])

        self.target1_pos = copy.deepcopy(self.rubbish.get_position())
        self.target1_pos[-1] = self.Z_TARGET
        self.target2_pos = copy.deepcopy(self.success_sensor.get_position())
        self.target2_pos[-1] = self.Z_TARGET
        self.lifted = False

        return [
            "put rubbish in bin",
            "drop the rubbish into the bin",
            "pick up the rubbish and leave it in the trash can",
            "throw away the trash, leaving any other objects alone",
            "chuck way any rubbish on the table rubbish",
        ]

    def variation_count(self) -> int:
        return 1

    def reward(self) -> float:
        grasped = self._grasped_cond.condition_met()[0]

        if not grasped:
            if self._detected_cond.condition_met()[0]:
                reward = 4.0
            else:
                grasp_rubbish_reward = np.exp(
                    -np.linalg.norm(
                        self.rubbish.get_position()
                        - self.robot.arm.get_tip().get_position()
                    )
                )
                reward = grasp_rubbish_reward
        else:
            lift_rubbish_reward = np.exp(
                -np.linalg.norm(self.rubbish.get_position() - self.target1_pos)
            )

            # TODO: Add more formal condition for reaching
            if not self.lifted and lift_rubbish_reward > 0.9:
                self.lifted = True

            if self.lifted:
                lift_rubbish_reward = 1.0
                reach_target_reward = np.exp(
                    -np.linalg.norm(self.rubbish.get_position() - self.target2_pos)
                )
            else:
                reach_target_reward = 0.0

            grasp_rubbish_reward = 1.0

            reward = grasp_rubbish_reward + lift_rubbish_reward + reach_target_reward

        return reward

    def get_low_dim_state(self) -> np.ndarray:
        # For ad-hoc reward computation, attach reward
        reward = self.reward()
        state = super().get_low_dim_state()
        return np.hstack([reward, state])

    def base_rotation_bounds(
        self,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
