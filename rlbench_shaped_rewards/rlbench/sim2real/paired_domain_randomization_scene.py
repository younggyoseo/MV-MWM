from typing import List
import itertools

import numpy as np
from pyrep import PyRep
from pyrep.const import ObjectType, TextureMappingMode
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor

from rlbench.backend.scene import Scene
from rlbench.observation_config import ObservationConfig
from rlbench.backend.robot import Robot
from rlbench.sim2real.domain_randomization import RandomizeEvery
from rlbench.backend.observation import Observation
from rlbench.backend.utils import rgb_handles_to_mask
from rlbench.noise_model import NoiseModel


SCENE_OBJECTS = [
    "Floor",
    "Roof",
    "Wall1",
    "Wall2",
    "Wall3",
    "Wall4",
    "diningTable_visible",
]

TEX_KWARGS = {
    "mapping_mode": TextureMappingMode.PLANE,
    "repeat_along_u": True,
    "repeat_along_v": True,
    "uv_scaling": [4.0, 4.0],
}


class PairedDomainRandomizationScene(Scene):
    def __init__(
        self,
        pyrep: PyRep,
        robot: Robot,
        obs_config: ObservationConfig = ObservationConfig(),
        robot_setup: str = "Panda",
        randomize_every: RandomizeEvery = RandomizeEvery.EPISODE,
        frequency: int = 1,
        visual_randomization_config=None,
        dynamics_randomization_config=None,
    ):
        super().__init__(pyrep, robot, obs_config, robot_setup)
        self._randomize_every = randomize_every
        self._frequency = frequency
        self._visual_rand_config = visual_randomization_config
        self._dynamics_rand_config = dynamics_randomization_config
        self._previous_index = -1
        self._count = 0

        if self._dynamics_rand_config is not None:
            raise NotImplementedError(
                "Dynamics randomization coming soon! "
                "Only visual randomization available."
            )

        self._scene_objects = [Shape(name) for name in SCENE_OBJECTS]
        self._scene_objects += self.robot.arm.get_visuals()
        self._scene_objects += self.robot.gripper.get_visuals()
        if self._visual_rand_config is not None:
            # Make the floor plane renderable (to cover old floor)
            self._scene_objects[0].set_position([0, 0, 0.01])
            self._scene_objects[0].set_renderable(True)

    def get_observation(self) -> Observation:
        tip = self.robot.arm.get_tip()

        joint_forces = None
        if self._obs_config.joint_forces:
            fs = self.robot.arm.get_joint_forces()
            vels = self.robot.arm.get_joint_target_velocities()
            joint_forces = self._obs_config.joint_forces_noise.apply(
                np.array([-f if v < 0 else f for f, v in zip(fs, vels)])
            )

        ee_forces_flat = None
        if self._obs_config.gripper_touch_forces:
            ee_forces = self.robot.gripper.get_touch_sensor_forces()
            ee_forces_flat = []
            for eef in ee_forces:
                ee_forces_flat.extend(eef)
            ee_forces_flat = np.array(ee_forces_flat)

        (
            left_shoulder_rgb_list,
            left_shoulder_depth_list,
            left_shoulder_pcd_list,
            left_shoulder_mask_list,
        ) = ([], [], [], [])
        (
            right_shoulder_rgb_list,
            right_shoulder_depth_list,
            right_shoulder_pcd_list,
            right_shoulder_mask_list,
        ) = ([], [], [], [])
        (
            overhead_rgb_list,
            overhead_depth_list,
            overhead_pcd_list,
            overhead_mask_list,
        ) = ([], [], [], [])
        wrist_rgb_list, wrist_depth_list, wrist_pcd_list, wrist_mask_list = (
            [],
            [],
            [],
            [],
        )
        front_rgb_list, front_depth_list, front_pcd_list, front_mask_list = (
            [],
            [],
            [],
            [],
        )

        tree = self.task.get_base().get_objects_in_tree(ObjectType.SHAPE)
        tree = [Shape(obj.get_handle()) for obj in tree + self._scene_objects]
        # files = self._visual_rand_config.sample(len(tree))[:2]
        # files = self._visual_rand_config.select(1)
        files = self._visual_rand_config.sample(1)
        files = files.tolist()
        files.append("canonical")
        for file in files:
            if file == "canonical":
                background_obj_list = self.task.get_base().get_objects_in_tree(
                    ObjectType.SHAPE
                ) + [Shape(name) for name in SCENE_OBJECTS]
                arm_obj_list = self.robot.arm.get_visuals()
                gripper_obj_list = self.robot.gripper.get_visuals()
                for obj in background_obj_list:
                    text_ob, texture = self.pyrep.create_texture(
                        "rlbench_shaped_rewards/tests/unit/assets/canonical/white.png"
                    )
                    try:
                        obj.set_texture(texture, **TEX_KWARGS)
                    except RuntimeError:
                        ungrouped = obj.ungroup()
                        for o in ungrouped:
                            o.set_texture(texture, **TEX_KWARGS)
                        self.pyrep.group_objects(ungrouped)
                    text_ob.remove()
                canonical_texture_path = (
                    "rlbench_shaped_rewards/tests/unit/assets/canonical/"
                )
                texture_iterator = itertools.cycle(
                    [
                        canonical_texture_path + "blue.png",
                        canonical_texture_path + "green.png",
                        canonical_texture_path + "red.png",
                        canonical_texture_path + "white.png",
                        canonical_texture_path + "yellow.png",
                    ]
                )
                for obj in arm_obj_list:
                    text_ob, texture = self.pyrep.create_texture(next(texture_iterator))
                    try:
                        obj.set_texture(texture, **TEX_KWARGS)
                    except RuntimeError:
                        ungrouped = obj.ungroup()
                        for o in ungrouped:
                            o.set_texture(texture, **TEX_KWARGS)
                        self.pyrep.group_objects(ungrouped)
                    text_ob.remove()
                for obj in gripper_obj_list:
                    text_ob, texture = self.pyrep.create_texture(next(texture_iterator))
                    try:
                        obj.set_texture(texture, **TEX_KWARGS)
                    except RuntimeError:
                        ungrouped = obj.ungroup()
                        for o in ungrouped:
                            o.set_texture(texture, **TEX_KWARGS)
                        self.pyrep.group_objects(ungrouped)
                    text_ob.remove()
            else:
                for obj in tree:
                    text_ob, texture = self.pyrep.create_texture(file)
                    try:
                        obj.set_texture(texture, **TEX_KWARGS)
                    except RuntimeError:
                        ungrouped = obj.ungroup()
                        for o in ungrouped:
                            o.set_texture(texture, **TEX_KWARGS)
                        self.pyrep.group_objects(ungrouped)
                    text_ob.remove()

            lsc_ob = self._obs_config.left_shoulder_camera
            rsc_ob = self._obs_config.right_shoulder_camera
            oc_ob = self._obs_config.overhead_camera
            wc_ob = self._obs_config.wrist_camera
            fc_ob = self._obs_config.front_camera

            lsc_mask_fn, rsc_mask_fn, oc_mask_fn, wc_mask_fn, fc_mask_fn = [
                (rgb_handles_to_mask if c.masks_as_one_channel else lambda x: x)
                for c in [lsc_ob, rsc_ob, oc_ob, wc_ob, fc_ob]
            ]

            def get_rgb_depth(
                sensor: VisionSensor,
                get_rgb: bool,
                get_depth: bool,
                get_pcd: bool,
                rgb_noise: NoiseModel,
                depth_noise: NoiseModel,
                depth_in_meters: bool,
            ):
                rgb = depth = pcd = None
                if sensor is not None and (get_rgb or get_depth):
                    sensor.handle_explicitly()
                    if get_rgb:
                        rgb = sensor.capture_rgb()
                        if rgb_noise is not None:
                            rgb = rgb_noise.apply(rgb)
                        rgb = np.clip((rgb * 255.0).astype(np.uint8), 0, 255)
                    if get_depth or get_pcd:
                        depth = sensor.capture_depth(depth_in_meters)
                        if depth_noise is not None:
                            depth = depth_noise.apply(depth)
                    if get_pcd:
                        depth_m = depth
                        if not depth_in_meters:
                            near = sensor.get_near_clipping_plane()
                            far = sensor.get_far_clipping_plane()
                            depth_m = near + depth * (far - near)
                        pcd = sensor.pointcloud_from_depth(depth_m)
                        if not get_depth:
                            depth = None
                return rgb, depth, pcd

            def get_mask(sensor: VisionSensor, mask_fn):
                mask = None
                if sensor is not None:
                    sensor.handle_explicitly()
                    mask = mask_fn(sensor.capture_rgb())
                return mask

            left_shoulder_rgb, left_shoulder_depth, left_shoulder_pcd = get_rgb_depth(
                self._cam_over_shoulder_left,
                lsc_ob.rgb,
                lsc_ob.depth,
                lsc_ob.point_cloud,
                lsc_ob.rgb_noise,
                lsc_ob.depth_noise,
                lsc_ob.depth_in_meters,
            )
            (
                right_shoulder_rgb,
                right_shoulder_depth,
                right_shoulder_pcd,
            ) = get_rgb_depth(
                self._cam_over_shoulder_right,
                rsc_ob.rgb,
                rsc_ob.depth,
                rsc_ob.point_cloud,
                rsc_ob.rgb_noise,
                rsc_ob.depth_noise,
                rsc_ob.depth_in_meters,
            )
            overhead_rgb, overhead_depth, overhead_pcd = get_rgb_depth(
                self._cam_overhead,
                oc_ob.rgb,
                oc_ob.depth,
                oc_ob.point_cloud,
                oc_ob.rgb_noise,
                oc_ob.depth_noise,
                oc_ob.depth_in_meters,
            )
            wrist_rgb, wrist_depth, wrist_pcd = get_rgb_depth(
                self._cam_wrist,
                wc_ob.rgb,
                wc_ob.depth,
                wc_ob.point_cloud,
                wc_ob.rgb_noise,
                wc_ob.depth_noise,
                wc_ob.depth_in_meters,
            )
            front_rgb, front_depth, front_pcd = get_rgb_depth(
                self._cam_front,
                fc_ob.rgb,
                fc_ob.depth,
                fc_ob.point_cloud,
                fc_ob.rgb_noise,
                fc_ob.depth_noise,
                fc_ob.depth_in_meters,
            )

            left_shoulder_mask = (
                get_mask(self._cam_over_shoulder_left_mask, lsc_mask_fn)
                if lsc_ob.mask
                else None
            )
            right_shoulder_mask = (
                get_mask(self._cam_over_shoulder_right_mask, rsc_mask_fn)
                if rsc_ob.mask
                else None
            )
            overhead_mask = (
                get_mask(self._cam_overhead_mask, oc_mask_fn) if oc_ob.mask else None
            )
            wrist_mask = (
                get_mask(self._cam_wrist_mask, wc_mask_fn) if wc_ob.mask else None
            )
            front_mask = (
                get_mask(self._cam_front_mask, fc_mask_fn) if fc_ob.mask else None
            )

            left_shoulder_rgb_list.append(left_shoulder_rgb)
            left_shoulder_depth_list.append(left_shoulder_depth)
            left_shoulder_pcd_list.append(left_shoulder_pcd)
            left_shoulder_mask_list.append(left_shoulder_mask)
            right_shoulder_rgb_list.append(right_shoulder_rgb)
            right_shoulder_depth_list.append(right_shoulder_depth)
            right_shoulder_pcd_list.append(right_shoulder_pcd)
            right_shoulder_mask_list.append(right_shoulder_mask)
            overhead_rgb_list.append(overhead_rgb)
            overhead_depth_list.append(overhead_depth)
            overhead_pcd_list.append(overhead_pcd)
            overhead_mask_list.append(overhead_mask)
            wrist_rgb_list.append(wrist_rgb)
            wrist_depth_list.append(wrist_depth)
            wrist_pcd_list.append(wrist_pcd)
            wrist_mask_list.append(wrist_mask)
            front_rgb_list.append(front_rgb)
            front_depth_list.append(front_depth)
            front_pcd_list.append(front_pcd)
            front_mask_list.append(front_mask)

        left_shoulder_rgb = (
            None
            if type(left_shoulder_rgb_list[0]) == type(None)
            else np.concatenate(left_shoulder_rgb_list, axis=1)
        )
        left_shoulder_depth = (
            None
            if type(left_shoulder_depth_list[0]) == type(None)
            else np.concatenate(left_shoulder_depth_list, axis=1)
        )
        left_shoulder_pcd = (
            None
            if type(left_shoulder_pcd_list[0]) == type(None)
            else np.concatenate(left_shoulder_pcd_list, axis=1)
        )
        left_shoulder_mask = (
            None
            if type(left_shoulder_mask_list[0]) == type(None)
            else np.concatenate(left_shoulder_mask_list, axis=1)
        )
        right_shoulder_rgb = (
            None
            if type(right_shoulder_rgb_list[0]) == type(None)
            else np.concatenate(right_shoulder_rgb_list, axis=1)
        )
        right_shoulder_depth = (
            None
            if type(right_shoulder_depth_list[0]) == type(None)
            else np.concatenate(right_shoulder_depth_list, axis=1)
        )
        right_shoulder_pcd = (
            None
            if type(right_shoulder_pcd_list[0]) == type(None)
            else np.concatenate(right_shoulder_pcd_list, axis=1)
        )
        right_shoulder_mask = (
            None
            if type(right_shoulder_mask_list[0]) == type(None)
            else np.concatenate(right_shoulder_mask_list, axis=1)
        )
        overhead_rgb = (
            None
            if type(overhead_rgb_list[0]) == type(None)
            else np.concatenate(overhead_rgb_list, axis=1)
        )
        overhead_depth = (
            None
            if type(overhead_depth_list[0]) == type(None)
            else np.concatenate(overhead_depth_list, axis=1)
        )
        overhead_pcd = (
            None
            if type(overhead_pcd_list[0]) == type(None)
            else np.concatenate(overhead_pcd_list, axis=1)
        )
        overhead_mask = (
            None
            if type(overhead_mask_list[0]) == type(None)
            else np.concatenate(overhead_mask_list, axis=1)
        )
        wrist_rgb = (
            None
            if type(wrist_rgb_list[0]) == type(None)
            else np.concatenate(wrist_rgb_list, axis=1)
        )
        wrist_depth = (
            None
            if type(wrist_depth_list[0]) == type(None)
            else np.concatenate(wrist_depth_list, axis=1)
        )
        wrist_pcd = (
            None
            if type(wrist_pcd_list[0]) == type(None)
            else np.concatenate(wrist_pcd_list, axis=1)
        )
        wrist_mask = (
            None
            if type(wrist_mask_list[0]) == type(None)
            else np.concatenate(wrist_mask_list, axis=1)
        )
        front_rgb = (
            None
            if type(front_rgb_list[0]) == type(None)
            else np.concatenate(front_rgb_list, axis=1)
        )
        front_depth = (
            None
            if type(front_depth_list[0]) == type(None)
            else np.concatenate(front_depth_list, axis=1)
        )
        front_pcd = (
            None
            if type(front_pcd_list[0]) == type(None)
            else np.concatenate(front_pcd_list, axis=1)
        )
        front_mask = (
            None
            if type(front_mask_list[0]) == type(None)
            else np.concatenate(front_mask_list, axis=1)
        )

        obs = Observation(
            left_shoulder_rgb=left_shoulder_rgb,
            left_shoulder_depth=left_shoulder_depth,
            left_shoulder_point_cloud=left_shoulder_pcd,
            right_shoulder_rgb=right_shoulder_rgb,
            right_shoulder_depth=right_shoulder_depth,
            right_shoulder_point_cloud=right_shoulder_pcd,
            overhead_rgb=overhead_rgb,
            overhead_depth=overhead_depth,
            overhead_point_cloud=overhead_pcd,
            wrist_rgb=wrist_rgb,
            wrist_depth=wrist_depth,
            wrist_point_cloud=wrist_pcd,
            front_rgb=front_rgb,
            front_depth=front_depth,
            front_point_cloud=front_pcd,
            left_shoulder_mask=left_shoulder_mask,
            right_shoulder_mask=right_shoulder_mask,
            overhead_mask=overhead_mask,
            wrist_mask=wrist_mask,
            front_mask=front_mask,
            joint_velocities=(
                self._obs_config.joint_velocities_noise.apply(
                    np.array(self.robot.arm.get_joint_velocities())
                )
                if self._obs_config.joint_velocities
                else None
            ),
            joint_positions=(
                self._obs_config.joint_positions_noise.apply(
                    np.array(self.robot.arm.get_joint_positions())
                )
                if self._obs_config.joint_positions
                else None
            ),
            joint_forces=(joint_forces if self._obs_config.joint_forces else None),
            gripper_open=(
                (1.0 if self.robot.gripper.get_open_amount()[0] > 0.9 else 0.0)
                if self._obs_config.gripper_open
                else None
            ),
            gripper_pose=(
                np.array(tip.get_pose()) if self._obs_config.gripper_pose else None
            ),
            gripper_matrix=(
                tip.get_matrix() if self._obs_config.gripper_matrix else None
            ),
            gripper_touch_forces=(
                ee_forces_flat if self._obs_config.gripper_touch_forces else None
            ),
            gripper_joint_positions=(
                np.array(self.robot.gripper.get_joint_positions())
                if self._obs_config.gripper_joint_positions
                else None
            ),
            task_low_dim_state=(
                self.task.get_low_dim_state()
                if self._obs_config.task_low_dim_state
                else None
            ),
            misc=self._get_misc(),
        )
        obs = self.task.decorate_observation(obs)
        return obs
