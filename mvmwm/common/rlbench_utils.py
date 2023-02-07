import gc
import random
import re
import copy
import numpy as np
from tqdm import tqdm


THETA_LIMIT = 0.05


def collect_demo(
    env,
    replay,
    num_demos,
    camera_keys,
    shaped_rewards=False,
    demo_replay=None,
    use_rotation=False,
    additional_camera=False,
    randomize=False,
):
    transitions = []
    print("collecting demos.")
    for _ in tqdm(range(num_demos)):
        success = False
        while not success:
            try:
                demo, demo_randomize = env._task.get_demos(
                    1, live_demos=True, randomize=randomize
                )
                demo = demo[0]
                demo_randomize = demo_randomize[0]
                success = True
            except:
                pass
        if randomize:
            transitions.extend(
                extract_from_paired_demo(
                    demo,
                    demo_randomize,
                    shaped_rewards,
                    camera_keys,
                    use_rotation,
                    additional_camera=additional_camera,
                )
            )
        else:
            transitions.extend(
                extract_from_demo(
                    demo,
                    shaped_rewards,
                    camera_keys,
                    use_rotation,
                    additional_camera=additional_camera,
                )
            )

    # Restrict translation space by min_max
    actions = []

    for obs in transitions:
        if obs["is_first"]:
            continue
        action = obs["action"]
        actions.append(action)

    low, high = np.min(actions, 0)[:3], np.max(actions, 0)[:3]
    low -= 0.2 * np.fabs(low)
    high += 0.2 * np.fabs(high)

    if use_rotation:
        rot_low, rot_high = np.min(actions, 0)[4:7], np.max(actions, 0)[4:7]
        rot_low = np.clip(rot_low, -THETA_LIMIT, np.inf)
        rot_high = np.clip(rot_high, -np.inf, THETA_LIMIT)
        rot_low -= 0.2 * np.fabs(rot_low)
        rot_high += 0.2 * np.fabs(rot_high)

    for obs in transitions:
        if obs["is_first"]:
            # for first action, let's just label with zero action
            rot_dim = 3 if use_rotation else 0
            obs["action"] = np.zeros(3 + 1 + rot_dim)
        else:
            action = obs["action"]
            updated_action = []

            pose = action[:3]
            norm_pose = 2 * ((pose - low) / (high - low)) - 1
            updated_action.append(norm_pose)

            gripper = action[3:4]
            norm_gripper = gripper * 2 - 1.0
            updated_action.append(norm_gripper)

            if use_rotation:
                theta_delta = action[4:7]
                norm_theta_delta = (
                    2 * ((theta_delta - rot_low) / (rot_high - rot_low)) - 1
                )
                norm_theta_delta = np.clip(norm_theta_delta, -1.0, 1.0)
                updated_action.append(norm_theta_delta)

            obs["action"] = np.hstack(updated_action)

        replay.add_step(obs)
        if demo_replay is not None:
            demo_replay.add_step(obs)

    print(f"Position min/max: {low}/{high}")
    actions_min_max = low, high
    if use_rotation:
        print(f"Rotation min/max: {rot_low}/{rot_high}")
        actions_min_max = low, high, rot_low, rot_high

    del transitions
    gc.collect()
    return actions_min_max


def get_action(prev_obs, obs):
    prev_pose = prev_obs.gripper_pose[:3]
    cur_pose = obs.gripper_pose[:3]
    pose = cur_pose - prev_pose
    gripper_action = float(obs.gripper_open)
    prev_action = np.hstack([pose, gripper_action])
    return prev_action


def get_theta_delta(prev_theta_dict, obs, use_minus_dict):
    plus_theta = get_theta(obs, mode="plus")
    minus_theta = get_theta(obs, mode="minus")
    theta_deltas = []

    for i in range(3):
        use_minus = use_minus_dict[i]
        if use_minus:
            theta = minus_theta
        else:
            theta = plus_theta

        prev_theta = prev_theta_dict[i]
        theta_delta = theta[i] - prev_theta[i]
        if np.abs(theta_delta) < np.pi / 2:
            theta_delta = theta_delta
        else:
            theta_delta = -np.sign(theta_delta) * (np.pi - np.abs(theta_delta))

        if np.abs(theta_delta) > 0.25 * (np.pi / 2.0):
            # Extremely drastic change in delta
            # Detected change in theta2 system
            use_minus = ~use_minus

            if use_minus:
                theta = minus_theta
            else:
                theta = plus_theta

            theta_delta = theta[i] - prev_theta[i]
            if np.abs(theta_delta) < np.pi / 2:
                theta_delta = theta_delta
            else:
                theta_delta = -np.sign(theta_delta) * (np.pi - np.abs(theta_delta))

        prev_theta_dict[i] = theta
        use_minus_dict[i] = use_minus
        theta_deltas.append(np.clip(theta_delta, -THETA_LIMIT, THETA_LIMIT))

    return theta_deltas, prev_theta_dict, use_minus_dict


def extract_from_demo(
    demo,
    shaped_rewards,
    camera_keys,
    use_rotation,
    additional_camera=False,
):
    transitions = []

    for k, obs in enumerate(demo):
        if k == 0:
            prev_action = None
            if use_rotation:
                use_minus = {i: False for i in range(3)}
                prev_theta = {i: get_theta(obs) for i in range(3)}
        else:
            prev_obs = demo[k - 1]
            prev_action = get_action(prev_obs, obs)
            if use_rotation:
                theta_deltas, prev_theta, use_minus = get_theta_delta(
                    prev_theta, obs, use_minus
                )
                prev_action = np.hstack([prev_action, theta_deltas])

        terminal = k == len(demo) - 1
        first = k == 0
        success = terminal

        if shaped_rewards:
            reward = obs.task_low_dim_state[0]
        else:
            reward = float(success)

        # Not to override obs
        _obs = copy.deepcopy(obs)
        _obs.joint_velocities = None
        _obs.joint_positions = None
        _obs.task_low_dim_state = None

        transition = {
            "reward": reward,
            "is_first": first,
            "is_last": False,
            "is_terminal": False,
            "success": success,
            "action": prev_action,
            "state": _obs.get_low_dim_data(),
        }

        keys = get_camera_keys(camera_keys)
        # images = []
        for key in keys:
            if key == "front":
                transition[key] = _obs.front_rgb
            if key == "wrist":
                transition[key] = _obs.wrist_rgb

        if additional_camera:
            for key, val in _obs.add_cam_rgb_depth_pcd.items():
                transition[key] = val

        transitions.append(transition)

    if len(transitions) % 50 == 0:
        time_limit = len(transitions)
    else:
        time_limit = 50 * (1 + (len(transitions) // 50))
    while len(transitions) < time_limit:
        transitions.append(copy.deepcopy(transition))
    transitions[-1]["is_last"] = True
    return transitions


def extract_from_paired_demo(
    demo,
    randomize_demo,
    shaped_rewards,
    camera_keys,
    use_rotation,
    additional_camera=False,
):
    return extract_from_demo(
        demo,
        shaped_rewards,
        camera_keys,
        use_rotation,
        additional_camera=additional_camera,
    )


def get_camera_keys(keys):
    camera_keys = keys.split("|")
    return camera_keys


def quat_to_theta(quat):
    x2, x3, x4, x1 = quat
    theta1 = np.arccos(x1 / np.sqrt(x4 ** 2 + x3 ** 2 + x2 ** 2 + x1 ** 2))
    theta2 = np.arccos(x2 / np.sqrt(x4 ** 2 + x3 ** 2 + x2 ** 2))
    theta3 = np.arccos(x3 / np.sqrt(x4 ** 2 + x3 ** 2))
    if x4 < 0:
        theta3 = 2 * np.pi - theta3
    thetas = np.array([theta1, theta2, theta3])
    return thetas


def get_theta(obs, mode="plus"):
    quat = obs.gripper_pose[3:]
    if mode == "minus":
        quat = -quat
    thetas = quat_to_theta(quat)
    return thetas
