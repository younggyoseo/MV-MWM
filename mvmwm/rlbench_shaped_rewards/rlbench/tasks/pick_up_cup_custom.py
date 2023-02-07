from typing import List
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.const import colors
from rlbench.backend.task import Task
from rlbench.backend.conditions import (
    DetectedCondition,
    NothingGrasped,
    GraspedCondition,
)
from rlbench.backend.spawn_boundary import SpawnBoundary

# colors = [
#     ("red", (1.0, 0.0, 0.0)),
#     ("maroon", (0.5, 0.0, 0.0)),
#     ("lime", (0.0, 1.0, 0.0)),
#     ("green", (0.0, 0.5, 0.0)),
#     ("blue", (0.0, 0.0, 1.0)),
#     ("navy", (0.0, 0.0, 0.5)),
#     ("yellow", (1.0, 1.0, 0.0)),
#     ("cyan", (0.0, 1.0, 1.0)),
#     ("magenta", (1.0, 0.0, 1.0)),
#     ("silver", (0.75, 0.75, 0.75)),
#     ("gray", (0.5, 0.5, 0.5)),
#     # ('orange', (1.0, 0.5, 0.0)),
#     ("olive", (0.5, 0.5, 0.0)),
#     ("purple", (0.5, 0.0, 0.5)),
#     ("teal", (0, 0.5, 0.5)),
#     ("azure", (0.0, 0.5, 1.0)),
#     ("violet", (0.5, 0.0, 1.0)),
#     ("rose", (1.0, 0.0, 0.5)),
#     ("black", (0.0, 0.0, 0.0)),
#     ("white", (1.0, 1.0, 1.0)),
#     ("yellow_real", (226 / 255, 216 / 255, 103 / 255)),
# ]


class PickUpCup(Task):
    def init_task(self) -> None:
        self.cup1 = Shape("cup1")
        self.cup2 = Shape("cup2")
        self.cup1_visual = Shape("cup1_visual")
        self.cup2_visual = Shape("cup2_visual")
        self.boundary = SpawnBoundary([Shape("boundary")])
        self.success_sensor = ProximitySensor("success")
        self.register_graspable_objects([self.cup1, self.cup2])

        self._grasped_cond = GraspedCondition(self.robot.gripper, self.cup1)
        self._detected_cond = DetectedCondition(
            self.cup1, self.success_sensor, negated=True
        )

        self.register_success_conditions([self._detected_cond, self._grasped_cond])

    def init_episode(self, index: int) -> List[str]:
        self.variation_index = index
        target_color_name, target_rgb = (
            "orange_real",
            (230 / 255, 96 / 255, 51 / 255),
        )

        _, other1_rgb = (
            "yellow_real",
            (226 / 255, 216 / 255, 103 / 255),
        )

        self.cup1_visual.set_color(target_rgb)
        self.cup2_visual.set_color(other1_rgb)

        self.boundary.clear()
        self.boundary.sample(self.cup2, min_distance=0.1)
        self.boundary.sample(self.success_sensor, min_distance=0.1)

        return [
            "pick up the %s cup" % target_color_name,
            "grasp the %s cup and lift it" % target_color_name,
            "lift the %s cup" % target_color_name,
        ]

    def variation_count(self) -> int:
        return 1

    def reward(self) -> float:
        grasped = self._grasped_cond.condition_met()[0]

        if not grasped:
            grasp_cup1_reward = np.exp(
                -np.linalg.norm(
                    self.cup1.get_position() - self.robot.arm.get_tip().get_position()
                )
            )
            reward = grasp_cup1_reward
        else:
            lift_cup1_reward = np.exp(
                -np.linalg.norm(
                    self.cup1.get_position() - self.success_sensor.get_position()
                )
            )
            reward = 1.0 + lift_cup1_reward
        return reward

    def get_low_dim_state(self) -> np.ndarray:
        # For ad-hoc reward computation, attach reward
        reward = self.reward()
        state = super().get_low_dim_state()
        return np.hstack([reward, state])
