from typing import List, Tuple

import numpy as np
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.shape import Shape

from rlbench.backend.conditions import (
    DetectedCondition,
    GraspedCondition,
    NothingGrasped,
)
from rlbench.backend.task import Task


class PhoneOnBase(Task):
    def init_task(self) -> None:
        self.phone = Shape("phone")
        self.success_detector = ProximitySensor("success")
        self.register_graspable_objects([self.phone])
        self._grasped_cond = GraspedCondition(self.robot.gripper, self.phone)
        self._nothing_grapsed_cond = NothingGrasped(self.robot.gripper)
        self._phone_cond = DetectedCondition(self.phone, self.success_detector)

        self.register_success_conditions(
            [self._phone_cond, NothingGrasped(self.robot.gripper)]
        )
        print("Easy version of PhoneOnBase")

    def init_episode(self, index: int) -> List[str]:
        return [
            "put the phone on the base",
            "put the phone on the stand",
            "put the hone on the hub",
            "grasp the phone and put it on the base",
            "place the phone on the base",
            "put the phone back on the base",
        ]

    def variation_count(self) -> int:
        return 1

    def reward(self) -> float:
        grasped = self._grasped_cond.condition_met()[0]
        nothing_grasped = self._nothing_grapsed_cond.condition_met()[0]
        phone_on_base = self._phone_cond.condition_met()[0]

        if phone_on_base:
            if nothing_grasped:
                # phone is not grasped anymore
                reward = 4.0
            else:
                # phone is in base, but gripper still holds the phone
                reward = 3.0
        else:
            if not grasped:
                # reaching the phone
                grasp_phone_reward = np.exp(
                    -np.linalg.norm(
                        self.phone.get_position()
                        - self.robot.arm.get_tip().get_position()
                    )
                )
                reward = grasp_phone_reward
            else:
                # moving the phone toward base
                move_phone_reward = np.exp(
                    -np.linalg.norm(
                        self.phone.get_position() - self.success_detector.get_position()
                    )
                )
                reward = 1.0 + move_phone_reward

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
