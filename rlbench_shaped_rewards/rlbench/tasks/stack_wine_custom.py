from typing import List, Tuple

import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task
from rlbench.backend.conditions import (
    DetectedCondition,
    GraspedCondition,
    NothingGrasped,
)


class StackWine(Task):
    def init_task(self):
        self.wine_bottle = Shape("wine_bottle")
        self.register_graspable_objects([self.wine_bottle])

        self._success_sensor = ProximitySensor("success")
        self._grasped_cond = GraspedCondition(self.robot.gripper, self.wine_bottle)
        self._detected_cond = DetectedCondition(self.wine_bottle, self._success_sensor)

        self.register_success_conditions([self._detected_cond])

    def init_episode(self, index: int) -> List[str]:
        return [
            "stack wine bottle",
            "slide the bottle onto the wine rack",
            "put the wine away",
            "leave the wine on the shelf",
            "grasp the bottle and put it away",
            "place the wine bottle on the wine rack",
        ]

    def variation_count(self) -> int:
        return 1

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0, 0, -np.pi / 4.0], [0, 0, np.pi / 4.0]

    def reward(self) -> float:
        grasped = self._grasped_cond.condition_met()[0]
        if not grasped:
            grasp_wine_reward = np.exp(
                -np.linalg.norm(
                    self.wine_bottle.get_position()
                    - self.robot.arm.get_tip().get_position()
                )
            )
            reach_target_reward = 0.0
        else:
            grasp_wine_reward = 1.0
            reach_target_reward = np.exp(
                -np.linalg.norm(
                    self.wine_bottle.get_position()
                    - self._success_sensor.get_position()
                )
            )
        reward = grasp_wine_reward + reach_target_reward

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