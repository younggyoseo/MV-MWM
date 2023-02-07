import atexit
import os
import copy
import random
import sys
import threading
import traceback
import signal

import cloudpickle
from functools import partial
import gym
import numpy as np

from rlbench import RandomizeEvery

try:
    from pyrep.errors import ConfigurationPathError, IKError
    from rlbench.backend.exceptions import InvalidActionError
except:
    pass


class RLBench:
    def __init__(
        self,
        name,
        camera_keys,
        size=(64, 64),
        actions_min_max=None,
        shaped_rewards=False,
        use_rotation=False,
        additional_camera=False,
        add_cam_names={},
        verbose=False,
        randomize_texture=False,
        default_texture="default",
    ):
        from rlbench.action_modes.action_mode import MoveArmThenGripper
        from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
        from rlbench.action_modes.gripper_action_modes import Discrete
        from rlbench.environment import Environment
        from rlbench.observation_config import ObservationConfig

        # we only support reach_target in this codebase
        obs_config = ObservationConfig()

        ## Camera setups
        obs_config.front_camera.set_all(False)
        obs_config.wrist_camera.set_all(False)
        obs_config.left_shoulder_camera.set_all(False)
        obs_config.right_shoulder_camera.set_all(False)
        obs_config.overhead_camera.set_all(False)

        obs_config.front_camera.rgb = True
        obs_config.front_camera.image_size = size

        if "wrist" in camera_keys:
            obs_config.wrist_camera.rgb = True
            obs_config.wrist_camera.image_size = size

        obs_config.joint_forces = False
        obs_config.joint_positions = True
        obs_config.joint_velocities = True
        obs_config.task_low_dim_state = True
        obs_config.gripper_touch_forces = False
        obs_config.gripper_pose = True
        obs_config.gripper_open = True
        obs_config.gripper_matrix = False
        obs_config.gripper_joint_positions = True

        self._use_rotation = use_rotation
        if use_rotation:
            abs_mode = True
        else:
            abs_mode = False

        self._randomize_texture = randomize_texture
        self.default_texture = default_texture
        if randomize_texture:
            env = Environment(
                action_mode=MoveArmThenGripper(
                    arm_action_mode=EndEffectorPoseViaPlanning(abs_mode),
                    gripper_action_mode=Discrete(),
                ),
                obs_config=obs_config,
                headless=True,
                shaped_rewards=shaped_rewards,
                add_cam_names=add_cam_names,
                verbose=verbose,
                randomize_every=RandomizeEvery.EPISODE,
                frequency=1,
                default_texture=self.default_texture,
            )
        else:
            env = Environment(
                action_mode=MoveArmThenGripper(
                    arm_action_mode=EndEffectorPoseViaPlanning(abs_mode),
                    gripper_action_mode=Discrete(),
                ),
                obs_config=obs_config,
                headless=True,
                shaped_rewards=shaped_rewards,
                add_cam_names=add_cam_names,
                verbose=verbose,
            )
        env.launch()

        # Here, `custom` envs are the ones used for viewpoint-robust control experiments.
        if "phone_on_base" in name:
            if "custom" in name:
                from rlbench.tasks.phone_on_base_custom import PhoneOnBase
            else:
                from rlbench.tasks.phone_on_base import PhoneOnBase
            task = PhoneOnBase
        elif "pick_up_cup" in name:
            if "custom" in name:
                from rlbench.tasks.pick_up_cup_custom2 import PickUpCup
            else:
                from rlbench.tasks.pick_up_cup import PickUpCup
            task = PickUpCup
        elif "put_rubbish_in_bin" in name:
            if "custom" in name:
                from rlbench.tasks.put_rubbish_in_bin_custom import PutRubbishInBin
            else:
                from rlbench.tasks.put_rubbish_in_bin import PutRubbishInBin
            task = PutRubbishInBin
        elif "take_umbrella_out_of_umbrella_stand" in name:
            if "custom" in name:
                from rlbench.tasks.take_umbrella_out_of_umbrella_stand_custom import (
                    TakeUmbrellaOutOfUmbrellaStand,
                )
            else:
                from rlbench.tasks.take_umbrella_out_of_umbrella_stand import (
                    TakeUmbrellaOutOfUmbrellaStand,
                )
            task = TakeUmbrellaOutOfUmbrellaStand
        elif "stack_wine" in name:
            if "custom" in name:
                from rlbench.tasks.stack_wine_custom import StackWine
            else:
                from rlbench.tasks.stack_wine import StackWine
            task = StackWine
        else:
            raise ValueError(name)
        self._env = env
        self._task = env.get_task(task)

        _, obs = self._task.reset(self.default_texture)
        task_low_dim = obs.task_low_dim_state.shape[0]
        self._state_dim = obs.get_low_dim_data().shape[0] - 14 - task_low_dim
        self._prev_obs, self._prev_reward = None, None
        self._ep_success = None

        self._size = size
        self._shaped_rewards = shaped_rewards
        self._camera_keys = camera_keys
        self._additional_camera = additional_camera

        if actions_min_max:
            self.register_min_max(actions_min_max)
        else:
            self.low = np.array([-0.03, -0.03, -0.03])
            self.high = np.array([0.03, 0.03, 0.03])
            if self._use_rotation:
                self.rot_low = np.array([-0.05, -0.05, -0.05])
                self.rot_high = np.array([0.05, 0.05, 0.05])

    @property
    def obs_space(self):
        spaces = {
            "reward": gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (), dtype=bool),
            "success": gym.spaces.Box(0, 1, (), dtype=bool),
            "state": gym.spaces.Box(
                -np.inf, np.inf, (self._state_dim,), dtype=np.float32
            ),
            "image": gym.spaces.Box(
                0,
                255,
                (self._size[0], self._size[1] * len(self._camera_keys), 3),
                dtype=np.uint8,
            ),
        }
        return spaces

    def register_min_max(self, actions_min_max):
        if self._use_rotation:
            self.low, self.high, self.rot_low, self.rot_high = actions_min_max
        else:
            self.low, self.high = actions_min_max

    @property
    def act_space(self):
        assert self.low is not None
        if self.low.shape[0] == 3:
            self.low = np.hstack([self.low, [0.0]])
            self.high = np.hstack([self.high, [1.0]])
            if self._use_rotation:
                self.low = np.hstack([self.low, self.rot_low])
                self.high = np.hstack([self.high, self.rot_high])
        action = gym.spaces.Box(
            low=self.low, high=self.high, shape=(self.low.shape[0],), dtype=np.float32
        )
        return {"action": action}

    def unnormalize(self, a):
        # Un-normalize gripper pose normalized to [-1, 1]
        assert self.low is not None
        pose = a[:3]
        pose = (pose + 1) / 2 * (self.high[:3] - self.low[:3]) + self.low[:3]

        # Manual handling of overflow in z axis
        curr_pose = self._task._task.robot.arm.get_tip().get_pose()[:3]
        curr_z = curr_pose[2]
        init_z = self._init_pose[2]
        delta_z = pose[2]

        if curr_z + delta_z >= init_z:
            pose[2] = 0.0

        # Un-normalize gripper action normalized to [-1, 1]
        gripper = a[3:4]
        gripper = (gripper + 1) / 2 * (self.high[3:4] - self.low[3:4]) + self.low[3:4]

        if self._use_rotation:
            target_pose = curr_pose + pose
            curr_quat = self._task._task.robot.arm.get_tip().get_pose()[3:]
            d_theta = (a[4:7] + 1) / 2 * (self.high[4:7] - self.low[4:7]) + self.low[
                4:7
            ]
            curr_theta = self.quat_to_theta(curr_quat)
            theta = curr_theta + d_theta
            quat = self.theta_to_quat(theta)
            quat = quat / np.linalg.norm(quat)

        else:
            target_pose = pose
            quat = np.array([0.0, 0.0, 0.0, 1.0])

        action = np.hstack([target_pose, quat, gripper])
        assert action.shape[0] == 8
        return action

    def step(self, action):
        assert np.isfinite(action["action"]).all(), action["action"]
        try:
            original_action = self.unnormalize(action["action"])
            _obs, _reward, _ = self._task.step(original_action)
            terminal = False
            success, _ = self._task._task.success()
            if success:
                self._ep_success = True
            self._prev_obs, self._prev_reward = _obs, _reward
            if not self._shaped_rewards:
                reward = float(self._ep_success)
            else:
                reward = _reward
        except ConfigurationPathError:
            _obs = self._prev_obs
            terminal = False
            success = False
            if not self._shaped_rewards:
                reward = float(self._ep_success)
            else:
                reward = self._prev_reward

        except (IKError, InvalidActionError) as e:
            _obs = self._prev_obs
            terminal = True
            success = False
            reward = -0.05

        except Exception as e:
            print("ERROR", e)
            print("DEBUG: theta ", _obs.cam1_theta)
            self._env._scene.set_additional_cams()
            original_action = self.unnormalize(action["action"])
            _obs, _reward, _ = self._task.step(original_action)
            print("new _obs", _obs)
            raise e

        _obs.joint_velocities = None
        _obs.joint_positions = None
        _obs.task_low_dim_state = None

        obs = {
            "reward": reward,
            "is_first": False,
            "is_last": terminal,
            "is_terminal": terminal,
            "success": success,
            "state": _obs.get_low_dim_data(),
        }
        # images = []
        for key in self._camera_keys:
            if key == "front":
                obs[key] = _obs.front_rgb
            if key == "wrist":
                obs[key] = _obs.wrist_rgb

        if self._additional_camera:
            for key, val in _obs.add_cam_rgb_depth_pcd.items():
                obs[key] = val

        self._time_step += 1
        return obs

    def reset(self):
        _, _obs = self._task.reset(self.default_texture)
        self._prev_obs = _obs
        self._init_pose = copy.deepcopy(
            self._task._task.robot.arm.get_tip().get_pose()[:3]
        )
        if self._use_rotation:
            self._use_minus_dict = {i: False for i in range(3)}
            self._prev_theta_dict = {i: None for i in range(3)}

        self._time_step = 0
        self._ep_success = False

        _obs.joint_velocities = None
        _obs.joint_positions = None
        _obs.task_low_dim_state = None

        obs = {
            "reward": 0.0,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
            "success": False,
            "state": _obs.get_low_dim_data(),
        }
        for key in self._camera_keys:
            if key == "front":
                obs[key] = _obs.front_rgb
            if key == "wrist":
                obs[key] = _obs.wrist_rgb

        if self._additional_camera:
            for key, val in _obs.add_cam_rgb_depth_pcd.items():
                obs[key] = val

        return obs

    def theta_to_quat(self, thetas):
        theta1, theta2, theta3 = thetas
        x1 = np.cos(theta1)
        x2 = np.sin(theta1) * np.cos(theta2)
        x3 = np.sin(theta1) * np.sin(theta2) * np.cos(theta3)
        x4 = np.sin(theta1) * np.sin(theta2) * np.sin(theta3)
        quat = np.hstack([x2, x3, x4, x1])
        return quat

    def quat_to_theta(self, quat):
        x2, x3, x4, x1 = quat
        theta1 = np.arccos(x1 / np.sqrt(x4 ** 2 + x3 ** 2 + x2 ** 2 + x1 ** 2))
        theta2 = np.arccos(x2 / np.sqrt(x4 ** 2 + x3 ** 2 + x2 ** 2))
        theta3 = np.arccos(x3 / np.sqrt(x4 ** 2 + x3 ** 2))
        if x4 < 0:
            theta3 = 2 * np.pi - theta3
        thetas = np.hstack([theta1, theta2, theta3])
        return thetas


class TimeLimit:
    def __init__(self, env, duration):
        self._env = env
        self._duration = duration
        self._step = None

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    def step(self, action):
        assert self._step is not None, "Must reset environment."
        obs = self._env.step(action)
        self._step += 1
        if self._duration and self._step >= self._duration:
            obs["is_last"] = True
            self._step = None
        return obs

    def reset(self):
        self._step = 0
        return self._env.reset()


class ResizeImage:
    def __init__(self, env, size=(64, 64)):
        self._env = env
        self._size = size
        self._keys = [
            k
            for k, v in env.obs_space.items()
            if len(v.shape) > 1 and v.shape[:2] != size
        ]
        print(f'Resizing keys {",".join(self._keys)} to {self._size}.')
        if self._keys:
            from PIL import Image

            self._Image = Image

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def obs_space(self):
        spaces = self._env.obs_space
        for key in self._keys:
            shape = self._size + spaces[key].shape[2:]
            spaces[key] = gym.spaces.Box(0, 255, shape, np.uint8)
        return spaces

    def step(self, action):
        obs = self._env.step(action)
        for key in self._keys:
            obs[key] = self._resize(obs[key])
        return obs

    def reset(self):
        obs = self._env.reset()
        for key in self._keys:
            obs[key] = self._resize(obs[key])
        return obs

    def _resize(self, image):
        image = self._Image.fromarray(image)
        image = image.resize(self._size, self._Image.NEAREST)
        image = np.array(image)
        return image


class RenderImage:
    def __init__(self, env, key="image"):
        self._env = env
        self._key = key
        self._shape = self._env.render().shape

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        try:
            return getattr(self._env, name)
        except AttributeError:
            raise ValueError(name)

    @property
    def obs_space(self):
        spaces = self._env.obs_space
        spaces[self._key] = gym.spaces.Box(0, 255, self._shape, np.uint8)
        return spaces

    def step(self, action):
        obs = self._env.step(action)
        obs[self._key] = self._env.render("rgb_array")
        return obs

    def reset(self):
        obs = self._env.reset()
        obs[self._key] = self._env.render("rgb_array")
        return obs


class Async:

    # Message types for communication via the pipe.
    _ACCESS = 1
    _CALL = 2
    _RESULT = 3
    _CLOSE = 4
    _EXCEPTION = 5

    def __init__(self, constructor, strategy="thread"):
        self._pickled_ctor = cloudpickle.dumps(constructor)
        if strategy == "process":
            import multiprocessing as mp

            context = mp.get_context("spawn")
        elif strategy == "thread":
            import multiprocessing.dummy as context
        else:
            raise NotImplementedError(strategy)
        self._strategy = strategy
        self._conn, conn = context.Pipe()
        self._process = context.Process(target=self._worker, args=(conn,))
        atexit.register(self.close)
        self._process.start()
        self._receive()  # Ready.
        self._obs_space = None
        self._act_space = None

    def access(self, name):
        self._conn.send((self._ACCESS, name))
        return self._receive

    def call(self, name, *args, **kwargs):
        payload = name, args, kwargs
        self._conn.send((self._CALL, payload))
        return self._receive

    def close(self):
        try:
            self._conn.send((self._CLOSE, None))
            self._conn.close()
        except IOError:
            pass  # The connection was already closed.
        self._process.join(5)

    def kill(self):
        pid = self._process.pid
        try:
            os.kill(pid, signal.SIGKILL)
            print("kill process ", pid)
        except:
            pass

    @property
    def obs_space(self):
        if not self._obs_space:
            self._obs_space = self.access("obs_space")()
        return self._obs_space

    @property
    def act_space(self):
        if not self._act_space:
            self._act_space = self.access("act_space")()
        return self._act_space

    def step(self, action, blocking=False):
        promise = self.call("step", action)
        if blocking:
            return promise()
        else:
            return promise

    def reset(self, blocking=False):
        promise = self.call("reset")
        if blocking:
            return promise()
        else:
            return promise

    def _receive(self):
        try:
            message, payload = self._conn.recv()
        except (OSError, EOFError):
            raise RuntimeError("Lost connection to environment worker.")
        # Re-raise exceptions in the main process.
        if message == self._EXCEPTION:
            stacktrace = payload
            raise Exception(stacktrace)
        if message == self._RESULT:
            return payload
        raise KeyError("Received message of unexpected type {}".format(message))

    def _worker(self, conn):
        try:
            ctor = cloudpickle.loads(self._pickled_ctor)
            env = ctor()
            conn.send((self._RESULT, None))  # Ready.
            while True:
                try:
                    # Only block for short times to have keyboard exceptions be raised.
                    if not conn.poll(0.1):
                        continue
                    message, payload = conn.recv()
                except (EOFError, KeyboardInterrupt):
                    break
                if message == self._ACCESS:
                    name = payload
                    result = getattr(env, name)
                    conn.send((self._RESULT, result))
                    continue
                if message == self._CALL:
                    name, args, kwargs = payload
                    result = getattr(env, name)(*args, **kwargs)
                    conn.send((self._RESULT, result))
                    continue
                if message == self._CLOSE:
                    break
                raise KeyError("Received message of unknown type {}".format(message))
        except Exception:
            stacktrace = "".join(traceback.format_exception(*sys.exc_info()))
            print("Error in environment process: {}".format(stacktrace))
            conn.send((self._EXCEPTION, stacktrace))
        finally:
            try:
                conn.close()
            except IOError:
                pass  # The connection was already closed.
