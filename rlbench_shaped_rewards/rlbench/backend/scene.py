from typing import List, Callable
from math import sin, cos, sqrt
import random
import glob

import numpy as np
from pyrep import PyRep
from pyrep.const import ObjectType, TextureMappingMode
from pyrep.errors import ConfigurationPathError
from pyrep.objects import Dummy
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor

from rlbench.backend.exceptions import (
    WaypointError,
    BoundaryError,
    NoWaypointsError,
    DemoError,
)
from rlbench.backend.observation import Observation
from rlbench.backend.robot import Robot
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.backend.task import Task
from rlbench.backend.utils import rgb_handles_to_mask
from rlbench.demo import Demo
from rlbench.noise_model import NoiseModel
from rlbench.observation_config import ObservationConfig, CameraConfig

from rlbench.sim2real.domain_randomization import (
    RandomizeEvery,
)

STEPS_BEFORE_EPISODE_START = 10

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


class Scene(object):
    """Controls what is currently in the vrep scene. This is used for making
    sure that the tasks are easily reachable. This may be just replaced by
    environment. Responsible for moving all the objects."""

    def __init__(
        self,
        pyrep: PyRep,
        robot: Robot,
        obs_config: ObservationConfig = ObservationConfig(),
        robot_setup: str = "panda",
        add_cam_names: dict = {},
        verbose: bool = False,
        default_texture: str = "default",
        input_texture: str = "random",
        randomize_every: RandomizeEvery = None,
    ):
        self.pyrep = pyrep
        self.robot = robot
        self.robot_setup = robot_setup
        self.task = None
        self._obs_config = obs_config
        self._initial_task_state = None
        self._start_arm_joint_pos = robot.arm.get_joint_positions()
        self._starting_gripper_joint_pos = robot.gripper.get_joint_positions()
        self._workspace = Shape("workspace")
        self._workspace_boundary = SpawnBoundary([self._workspace])
        self._cam_over_shoulder_left = VisionSensor("cam_over_shoulder_left")
        self._cam_over_shoulder_right = VisionSensor("cam_over_shoulder_right")
        self._cam_overhead = VisionSensor("cam_overhead")
        self._cam_wrist = VisionSensor("cam_wrist")
        self._cam_front = VisionSensor("cam_front")
        self._cam_over_shoulder_left_mask = VisionSensor("cam_over_shoulder_left_mask")
        self._cam_over_shoulder_right_mask = VisionSensor(
            "cam_over_shoulder_right_mask"
        )
        self._cam_overhead_mask = VisionSensor("cam_overhead_mask")
        self._cam_wrist_mask = VisionSensor("cam_wrist_mask")
        self._cam_front_mask = VisionSensor("cam_front_mask")

        # Additional Cameras
        self._add_cam_dict = dict()
        self._add_cam_theta_dict = dict()
        self._add_cam_phi_dict = dict()
        self._add_cam_radius_dict = dict()
        self._add_cam_height_dict = dict()
        self._sign = 1
        self._add_cam_names = add_cam_names
        self._moving_camera = {key: False for key in add_cam_names.keys()}
        self._verbose = verbose
        if len(add_cam_names) > 0:
            try:
                self.set_additional_cams()
            except Exception as e:
                print("ERROR", e)
                raise e
        self._has_init_task = self._has_init_episode = False
        self._variation_index = 0

        self._initial_robot_state = (
            robot.arm.get_configuration_tree(),
            robot.gripper.get_configuration_tree(),
        )

        # Set camera properties from observation config
        self._set_camera_properties()

        x, y, z = self._workspace.get_position()
        minx, maxx, miny, maxy, _, _ = self._workspace.get_bounding_box()
        self._workspace_minx = x - np.fabs(minx) - 0.2
        self._workspace_maxx = x + maxx + 0.2
        self._workspace_miny = y - np.fabs(miny) - 0.2
        self._workspace_maxy = y + maxy + 0.2
        self._workspace_minz = z
        self._workspace_maxz = z + 1.0  # 1M above workspace

        self.target_workspace_check = Dummy.create()
        self._step_callback = None

        self._robot_shapes = self.robot.arm.get_objects_in_tree(
            object_type=ObjectType.SHAPE
        )

        self.default_texture = default_texture
        self.input_texture = input_texture
        self._scene_objects = [Shape(name) for name in SCENE_OBJECTS]
        self._default_colorset = [obj.get_color() for obj in self._scene_objects]

        self._scene_objects += self.robot.arm.get_visuals()
        self._scene_objects += self.robot.gripper.get_visuals()
        # Make the floor plane renderable (to cover old floor)
        self._scene_objects[0].set_position([0, 0, 0.01])
        self._scene_objects[0].set_renderable(True)

        self.cur_texture = None
        self.randomize_every = randomize_every

    def _process_randomization_range(self, intervals):
        intervals_list = []
        for itv in intervals:
            low = float(itv.split("|")[0])
            high = float(itv.split("|")[1])
            assert low <= high
            intervals_list.append([low, high])
        output = random.uniform(
            *random.choices(
                intervals_list,
                weights=[r[1] - r[0] for r in intervals_list],
            )[0]
        )
        return output

    def set_additional_cams(self):
        for cam_name in self._add_cam_names.keys():
            if cam_name not in self._add_cam_dict:
                cam = self._cam_front.copy()
                cam.set_explicit_handling(1)
                cam.set_resolution(self._obs_config.front_camera.image_size)
                cam.set_render_mode(self._obs_config.front_camera.render_mode)
                self._add_cam_dict[cam_name] = cam
            else:
                cam = self._add_cam_dict[cam_name]

            if "|" in self._add_cam_names[cam_name]:
                theta_phi_radius_height_axis_range = self._add_cam_names[
                    cam_name
                ].split("|")
                assert len(theta_phi_radius_height_axis_range) == 15
                if (
                    float(theta_phi_radius_height_axis_range[4]) != 0
                    or float(theta_phi_radius_height_axis_range[5]) != 0
                    or float(theta_phi_radius_height_axis_range[10]) != 0
                    or float(theta_phi_radius_height_axis_range[11]) != 0
                    or float(theta_phi_radius_height_axis_range[14]) != 0
                ):
                    self._moving_camera[cam_name] = True
            vp_rand_range = self._add_cam_names[cam_name]
            if (
                float(vp_rand_range["delta_theta"]) != 0
                or float(vp_rand_range["delta_phi"]) != 0
                or float(vp_rand_range["delta_radius"]) != 0
                or float(vp_rand_range["delta_height"]) != 0
                or float(vp_rand_range["delta_axis"]) != 0
            ):
                self._moving_camera[cam_name] = True

            # theta range
            theta = self._process_randomization_range(vp_rand_range["theta"])
            phi = self._process_randomization_range(vp_rand_range["phi"])
            radius = self._process_randomization_range(vp_rand_range["radius"])
            height = self._process_randomization_range(vp_rand_range["height"])
            axis = self._process_randomization_range(vp_rand_range["axis"])

            # camera set-up
            cam.set_orientation(
                np.array(
                    [axis * np.pi / 180.0, -90 * np.pi / 180.0, -90 * np.pi / 180.0]
                )
            )
            cam.rotate([phi * np.pi / 180.0, -theta * np.pi / 180.0, 0.0])
            cam.set_position(
                [
                    radius * cos(theta * np.pi / 180.0),
                    -radius * sin(theta * np.pi / 180.0),
                    height,
                ]
            )
            self._add_cam_theta_dict[cam_name] = theta
            self._add_cam_phi_dict[cam_name] = phi
            self._add_cam_radius_dict[cam_name] = radius
            self._add_cam_height_dict[cam_name] = height

    def move_additional_cams(self):
        raise NotImplementedError
        # for cam_name, cam in self._add_cam_dict.items():
        #     if not self._moving_camera[cam_name]:
        #         continue
        #
        #     theta_phi_radius_height_range = self._add_cam_names[cam_name].split("|")
        #     theta_low = float(theta_phi_radius_height_range[0])
        #     theta_high = float(theta_phi_radius_height_range[1])
        #     phi_low = float(theta_phi_radius_height_range[2])
        #     phi_high = float(theta_phi_radius_height_range[3])
        #     delta_theta = float(theta_phi_radius_height_range[4])
        #     delta_phi = float(theta_phi_radius_height_range[5])
        #     radius_low = float(theta_phi_radius_height_range[6])
        #     radius_high = float(theta_phi_radius_height_range[7])
        #     height_low = float(theta_phi_radius_height_range[8])
        #     height_high = float(theta_phi_radius_height_range[9])
        #     delta_radius = float(theta_phi_radius_height_range[10])
        #     delta_height = float(theta_phi_radius_height_range[11])
        #
        #     prev_theta = self._add_cam_theta_dict[cam_name]
        #     prev_phi = self._add_cam_phi_dict[cam_name]
        #     prev_radius = self._add_cam_radius_dict[cam_name]
        #     prev_height = self._add_cam_height_dict[cam_name]
        #
        #     delta_theta = delta_theta * np.pi / 180.0
        #     theta = prev_theta + delta_theta
        #
        #     delta_phi = delta_phi * np.pi / 180.0
        #     if prev_phi < phi_low:
        #         self._sign = 1
        #     elif prev_phi > phi_high:
        #         self._sign = -1
        #     phi = prev_phi + self._sign * delta_phi
        #
        #     radius = prev_radius + delta_radius
        #     height = prev_height + delta_height
        #     # camera set-up
        #     # radius, height, phi = self._radius, self._height, self._phi
        #     # radius, height = self._radius, self._height
        #     cam.set_position([0.0, 0.0, 0.0])
        #     cam.set_orientation(np.array([0.0, 0.0, 0.0]))
        #     cam.set_orientation(
        #         np.array([0.0, -90 * np.pi / 180.0, -90 * np.pi / 180.0])
        #     )
        #     cam.rotate([0.0, -theta * np.pi / 180.0, 0.0])
        #     cam.set_position(
        #         [
        #             radius * cos(theta * np.pi / 180.0),
        #             -radius * sin(theta * np.pi / 180.0),
        #             height,
        #         ]
        #     )
        #     cam.rotate([phi * np.pi / 180.0, 0.0, 0.0])
        #     cam.set_explicit_handling(1)
        #     cam.set_resolution(self._obs_config.front_camera.image_size)
        #     cam.set_render_mode(self._obs_config.front_camera.render_mode)
        #
        #     self._add_cam_dict[cam_name] = cam
        #     self._add_cam_theta_dict[cam_name] = theta
        #     self._add_cam_phi_dict[cam_name] = phi

    def load(self, task: Task) -> None:
        """Loads the task and positions at the centre of the workspace.

        :param task: The task to load in the scene.
        """
        task.load()  # Load the task in to the scene

        # Set at the centre of the workspace
        task.get_base().set_position(self._workspace.get_position())

        self._initial_task_state = task.get_state()
        self.task = task
        self._initial_task_pose = task.boundary_root().get_orientation()
        self._has_init_task = self._has_init_episode = False
        self._variation_index = 0

    def unload(self) -> None:
        """Clears the scene. i.e. removes all tasks."""
        if self.task is not None:
            self.robot.gripper.release()
            if self._has_init_task:
                self.task.cleanup_()
            self.task.unload()
        self.task = None
        self._variation_index = 0

    def init_task(self) -> None:
        self.task.init_task()
        self._initial_task_state = self.task.get_state()
        self._has_init_task = True
        self._variation_index = 0

    def init_episode(
        self, index: int, randomly_place: bool = True, max_attempts: int = 5
    ) -> List[str]:
        """Calls the task init_episode and puts randomly in the workspace."""

        self._variation_index = index

        if not self._has_init_task:
            self.init_task()

        try:
            self.set_additional_cams()
        except Exception as e:
            print("ERROR", e)
            raise e
        # Try a few times to init and place in the workspace
        attempts = 0
        descriptions = None
        while attempts < max_attempts:
            descriptions = self.task.init_episode(index)
            try:
                if randomly_place and not self.task.is_static_workspace():
                    self._place_task()
                    if self.robot.arm.check_arm_collision():
                        raise BoundaryError()
                self.task.validate()
                break
            except (BoundaryError, WaypointError) as e:
                self.task.cleanup_()
                self.task.restore_state(self._initial_task_state)
                attempts += 1
                if attempts >= max_attempts:
                    raise e

        # Let objects come to rest
        [self.pyrep.step() for _ in range(STEPS_BEFORE_EPISODE_START)]
        self._has_init_episode = True
        return descriptions

    def reset(self) -> None:
        """Resets the joint angles."""
        self.robot.gripper.release()

        arm, gripper = self._initial_robot_state
        self.pyrep.set_configuration_tree(arm)
        self.pyrep.set_configuration_tree(gripper)
        self.robot.arm.set_joint_positions(
            self._start_arm_joint_pos, disable_dynamics=True
        )
        self.robot.arm.set_joint_target_velocities([0] * len(self.robot.arm.joints))
        self.robot.gripper.set_joint_positions(
            self._starting_gripper_joint_pos, disable_dynamics=True
        )
        self.robot.gripper.set_joint_target_velocities(
            [0] * len(self.robot.gripper.joints)
        )

        if self.task is not None and self._has_init_task:
            self.task.cleanup_()
            self.task.restore_state(self._initial_task_state)
        self.task.set_initial_objects_in_scene()

    def get_observation(self, texture="default") -> Observation:
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

        lsc_ob = self._obs_config.left_shoulder_camera
        rsc_ob = self._obs_config.right_shoulder_camera
        oc_ob = self._obs_config.overhead_camera
        wc_ob = self._obs_config.wrist_camera
        fc_ob = self._obs_config.front_camera

        lsc_mask_fn, rsc_mask_fn, oc_mask_fn, wc_mask_fn, fc_mask_fn = [
            (rgb_handles_to_mask if c.masks_as_one_channel else lambda x: x)
            for c in [lsc_ob, rsc_ob, oc_ob, wc_ob, fc_ob]
        ]

        if self.cur_texture != texture or texture == "random":
            self._change_texture(texture)
            self.cur_texture = texture

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
        right_shoulder_rgb, right_shoulder_depth, right_shoulder_pcd = get_rgb_depth(
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

        add_cam_rgb_depth_pcd = dict()
        for key, val in self._add_cam_names.items():
            add_cam_rgb, add_cam_depth, add_cam_pcd = get_rgb_depth(
                self._add_cam_dict[key],
                fc_ob.rgb,
                fc_ob.depth,
                fc_ob.point_cloud,
                fc_ob.rgb_noise,
                fc_ob.depth_noise,
                fc_ob.depth_in_meters,
            )
            add_cam_theta = self._add_cam_theta_dict[key]
            add_cam_rgb_depth_pcd[key] = add_cam_rgb
            if add_cam_depth is not None:
                add_cam_rgb_depth_pcd[key + "_depth"] = add_cam_depth
            if add_cam_pcd is not None:
                add_cam_rgb_depth_pcd[key + "_pcd"] = add_cam_pcd
            add_cam_rgb_depth_pcd[key + "_theta"] = add_cam_theta

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
        wrist_mask = get_mask(self._cam_wrist_mask, wc_mask_fn) if wc_ob.mask else None
        front_mask = get_mask(self._cam_front_mask, fc_mask_fn) if fc_ob.mask else None

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
            add_cam_rgb_depth_pcd=add_cam_rgb_depth_pcd,
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

    def _change_texture(self, texture):
        assert texture in ["default", "real_proxy", "canonical", "random"]
        if texture == "default":
            background_obj_list = [Shape(name) for name in SCENE_OBJECTS]
            obj_list = background_obj_list
            for default_color, obj in zip(self._default_colorset, obj_list):
                obj.set_color(default_color)
                # text_ob.remove()

        elif texture == "real_proxy":
            background_obj_list = [Shape(name) for name in SCENE_OBJECTS]
            obj_list = background_obj_list
            colorset = [[1.0, 1.0, 1.0]] * len(obj_list)
            for default_color, obj in zip(colorset, obj_list):
                obj.set_color(default_color)

        elif texture == "canonical":
            background_obj_list = [
                Shape(name)
                for name in ["Floor", "Roof", "Wall1", "Wall2", "Wall3", "Wall4"]
            ]
            table_obj_list = [Shape("diningTable_visible")]
            # table_obj_list[0].set_color([1.0, 1.0, 1.0])  # white
            table_obj_list[0].set_color(
                [196 / 255, 196 / 255, 196 / 255]
            )  # real desk color
            arm_obj_list = self.robot.arm.get_visuals()
            gripper_obj_list = self.robot.gripper.get_visuals()

            texture_path = "rlbench_shaped_rewards/tests/unit/assets/real_proxy/"

            for obj in background_obj_list + table_obj_list:
                file = (
                    texture_path + "black.png"
                    if obj in background_obj_list
                    else texture_path + "white.png"
                )
                text_ob, texture = self.pyrep.create_texture(file)
                try:
                    obj.set_texture(texture, **TEX_KWARGS)
                except RuntimeError:
                    ungrouped = obj.ungroup()
                    for o in ungrouped:
                        o.set_texture(texture, **TEX_KWARGS)
                    self.pyrep.group_objects(ungrouped)
                text_ob.remove()

            for obj in arm_obj_list + gripper_obj_list:
                try:
                    obj.remove_texture()
                except RuntimeError:
                    ungrouped = obj.ungroup()
                    for o in ungrouped:
                        o.remove_texture()
                    self.pyrep.group_objects(ungrouped)

        elif texture == "random":
            # background_obj_list = [Shape(name) for name in SCENE_OBJECTS]
            # obj_list = background_obj_list

            background_obj_list = [
                Shape(name)
                for name in ["Floor", "Roof", "Wall1", "Wall2", "Wall3", "Wall4"]
            ]
            table_obj_list = [Shape("diningTable_visible")]

            texture_path = "rlbench_shaped_rewards/tests/unit/assets/textures"
            texture_list = glob.glob(texture_path + "/*.png")
            files = random.choices(
                texture_list, k=len(background_obj_list + table_obj_list)
            )
            # for file, obj in zip(files, obj_list):
            for file, obj in zip(files, background_obj_list + table_obj_list):
                if obj in background_obj_list:
                    obj.set_color(
                        [
                            30.0 / 255.0,
                            30.0 / 255.0,
                            30.0 / 255.0,
                        ]
                    )
                elif obj in table_obj_list:
                    obj.set_color([196 / 255, 196 / 255, 196 / 255])

                text_ob, texture = self.pyrep.create_texture(file)
                try:
                    obj.set_texture(texture, **TEX_KWARGS)
                    # obj.set_color([random.random(), random.random(), random.random()])
                except RuntimeError:
                    ungrouped = obj.ungroup()
                    for o in ungrouped:
                        # obj.set_color(
                        #     [random.random(), random.random(), random.random()]
                        # )
                        o.set_texture(texture, **TEX_KWARGS)
                    self.pyrep.group_objects(ungrouped)
                text_ob.remove()
        else:
            tree = self.task.get_base().get_objects_in_tree(ObjectType.SHAPE)
            tree = [Shape(obj.get_handle()) for obj in tree + self._scene_objects]

            for obj in tree:
                try:
                    obj.remove_texture()
                except RuntimeError:
                    pass

    def step(self):
        if True in self._moving_camera.values():
            self.move_additional_cams()
        self.pyrep.step()
        self.task.step()
        if self._step_callback is not None:
            self._step_callback()

    def register_step_callback(self, func):
        self._step_callback = func

    def get_demo(
        self,
        record: bool = True,
        callable_each_step: Callable[[Observation], None] = None,
        randomly_place: bool = True,
        randomize: bool = False,
    ) -> Demo:
        """Returns a demo (list of observations)"""

        if not self._has_init_task:
            self.init_task()
        if not self._has_init_episode:
            self.init_episode(self._variation_index, randomly_place=randomly_place)
        self._has_init_episode = False

        waypoints = self.task.get_waypoints()
        if len(waypoints) == 0:
            raise NoWaypointsError("No waypoints were found.", self.task)

        demo = []
        demo_randomize = [] if randomize else None
        if record:
            self.pyrep.step()  # Need this here or get_force doesn't work...
            demo.append(self.get_observation(texture=self.default_texture))
            if randomize:
                demo_randomize.append(self.get_observation(texture=self.input_texture))
        while True:
            success = False
            for i, point in enumerate(waypoints):
                point.start_of_path()
                if point.skip:
                    continue
                grasped_objects = self.robot.gripper.get_grasped_objects()
                colliding_shapes = [
                    s
                    for s in self.pyrep.get_objects_in_tree(
                        object_type=ObjectType.SHAPE
                    )
                    if s not in grasped_objects
                    and s not in self._robot_shapes
                    and s.is_collidable()
                    and self.robot.arm.check_arm_collision(s)
                ]
                [s.set_collidable(False) for s in colliding_shapes]
                try:
                    path = point.get_path()
                    [s.set_collidable(True) for s in colliding_shapes]
                except ConfigurationPathError as e:
                    [s.set_collidable(True) for s in colliding_shapes]
                    raise DemoError(
                        "Could not get a path for waypoint %d." % i, self.task
                    ) from e
                ext = point.get_ext()
                path.visualize()

                done = False
                success = False
                while not done:
                    done = path.step()
                    self.step()
                    self._demo_record_step(
                        demo, record, callable_each_step, demo_randomize
                    )
                    success, term = self.task.success()

                point.end_of_path()

                path.clear_visualization()

                if len(ext) > 0:
                    contains_param = False
                    start_of_bracket = -1
                    gripper = self.robot.gripper
                    if "open_gripper(" in ext:
                        gripper.release()
                        start_of_bracket = ext.index("open_gripper(") + 13
                        contains_param = ext[start_of_bracket] != ")"
                        if not contains_param:
                            done = False
                            while not done:
                                done = gripper.actuate(1.0, 0.04)
                                self.pyrep.step()
                                self.task.step()
                                if self._obs_config.record_gripper_closing:
                                    self._demo_record_step(
                                        demo, record, callable_each_step, demo_randomize
                                    )
                    elif "close_gripper(" in ext:
                        start_of_bracket = ext.index("close_gripper(") + 14
                        contains_param = ext[start_of_bracket] != ")"
                        if not contains_param:
                            done = False
                            while not done:
                                done = gripper.actuate(0.0, 0.04)
                                self.pyrep.step()
                                self.task.step()
                                if self._obs_config.record_gripper_closing:
                                    self._demo_record_step(
                                        demo, record, callable_each_step, demo_randomize
                                    )

                    if contains_param:
                        rest = ext[start_of_bracket:]
                        num = float(rest[: rest.index(")")])
                        done = False
                        while not done:
                            done = gripper.actuate(num, 0.04)
                            self.pyrep.step()
                            self.task.step()
                            if self._obs_config.record_gripper_closing:
                                self._demo_record_step(
                                    demo, record, callable_each_step, demo_randomize
                                )

                    if "close_gripper(" in ext:
                        for g_obj in self.task.get_graspable_objects():
                            gripper.grasp(g_obj)

                    self._demo_record_step(
                        demo, record, callable_each_step, demo_randomize
                    )

            if not self.task.should_repeat_waypoints() or success:
                break

        # Some tasks may need additional physics steps
        # (e.g. ball rowling to goal)
        if not success:
            for _ in range(10):
                self.pyrep.step()
                self.task.step()
                self._demo_record_step(demo, record, callable_each_step)
                success, term = self.task.success()
                if success:
                    break

        success, term = self.task.success()
        if not success:
            raise DemoError("Demo was completed, but was not successful.", self.task)

        out_canonical = Demo(demo_randomize) if randomize else None

        return Demo(demo), out_canonical

    def get_observation_config(self) -> ObservationConfig:
        return self._obs_config

    def check_target_in_workspace(self, target_pos: np.ndarray) -> bool:
        x, y, z = target_pos
        return (
            self._workspace_maxx > x > self._workspace_minx
            and self._workspace_maxy > y > self._workspace_miny
            and self._workspace_maxz > z > self._workspace_minz
        )

    def _demo_record_step(self, demo_list, record, func, demo_randomize_list=None):
        if record:
            demo_list.append(self.get_observation(texture=self.default_texture))
            if demo_randomize_list is not None:
                demo_randomize_list.append(
                    self.get_observation(texture=self.input_texture)
                )
        if func is not None:
            func(self.get_observation(texture=self.default_texture))

    def _set_camera_properties(self) -> None:
        def _set_rgb_props(
            rgb_cam: VisionSensor, rgb: bool, depth: bool, conf: CameraConfig
        ):
            if not (rgb or depth or conf.point_cloud):
                rgb_cam.remove()
            else:
                rgb_cam.set_explicit_handling(1)
                rgb_cam.set_resolution(conf.image_size)
                rgb_cam.set_render_mode(conf.render_mode)

        def _set_mask_props(mask_cam: VisionSensor, mask: bool, conf: CameraConfig):
            if not mask:
                mask_cam.remove()
            else:
                mask_cam.set_explicit_handling(1)
                mask_cam.set_resolution(conf.image_size)

        _set_rgb_props(
            self._cam_over_shoulder_left,
            self._obs_config.left_shoulder_camera.rgb,
            self._obs_config.left_shoulder_camera.depth,
            self._obs_config.left_shoulder_camera,
        )
        _set_rgb_props(
            self._cam_over_shoulder_right,
            self._obs_config.right_shoulder_camera.rgb,
            self._obs_config.right_shoulder_camera.depth,
            self._obs_config.right_shoulder_camera,
        )
        _set_rgb_props(
            self._cam_overhead,
            self._obs_config.overhead_camera.rgb,
            self._obs_config.overhead_camera.depth,
            self._obs_config.overhead_camera,
        )
        _set_rgb_props(
            self._cam_wrist,
            self._obs_config.wrist_camera.rgb,
            self._obs_config.wrist_camera.depth,
            self._obs_config.wrist_camera,
        )
        _set_rgb_props(
            self._cam_front,
            self._obs_config.front_camera.rgb,
            self._obs_config.front_camera.depth,
            self._obs_config.front_camera,
        )
        _set_mask_props(
            self._cam_over_shoulder_left_mask,
            self._obs_config.left_shoulder_camera.mask,
            self._obs_config.left_shoulder_camera,
        )
        _set_mask_props(
            self._cam_over_shoulder_right_mask,
            self._obs_config.right_shoulder_camera.mask,
            self._obs_config.right_shoulder_camera,
        )
        _set_mask_props(
            self._cam_overhead_mask,
            self._obs_config.overhead_camera.mask,
            self._obs_config.overhead_camera,
        )
        _set_mask_props(
            self._cam_wrist_mask,
            self._obs_config.wrist_camera.mask,
            self._obs_config.wrist_camera,
        )
        _set_mask_props(
            self._cam_front_mask,
            self._obs_config.front_camera.mask,
            self._obs_config.front_camera,
        )

    def _place_task(self) -> None:
        self._workspace_boundary.clear()
        # Find a place in the robot workspace for task
        self.task.boundary_root().set_orientation(self._initial_task_pose)
        min_rot, max_rot = self.task.base_rotation_bounds()
        self._workspace_boundary.sample(
            self.task.boundary_root(), min_rotation=min_rot, max_rotation=max_rot
        )

    def _get_misc(self):
        def _get_cam_data(cam: VisionSensor, name: str):
            d = {}
            if cam.still_exists():
                d = {
                    "%s_extrinsics" % name: cam.get_matrix(),
                    "%s_intrinsics" % name: cam.get_intrinsic_matrix(),
                    "%s_near" % name: cam.get_near_clipping_plane(),
                    "%s_far" % name: cam.get_far_clipping_plane(),
                }
            return d

        misc = _get_cam_data(self._cam_over_shoulder_left, "left_shoulder_camera")
        misc.update(
            _get_cam_data(self._cam_over_shoulder_right, "right_shoulder_camera")
        )
        misc.update(_get_cam_data(self._cam_overhead, "overhead_camera"))
        misc.update(_get_cam_data(self._cam_front, "front_camera"))
        misc.update(_get_cam_data(self._cam_wrist, "wrist_camera"))
        return misc
