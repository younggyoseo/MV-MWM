import collections
import functools
import gc
import logging
import os
import pathlib
import re
import sys
import warnings
import pickle
from tqdm import tqdm

try:
    import rich.traceback

    rich.traceback.install()
except ImportError:
    pass

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger().setLevel("ERROR")
warnings.filterwarnings("ignore", ".*box bound precision lowered.*")

sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import numpy as np
from keras import backend as K
import ruamel.yaml as yaml

import agent as agent
import common


def main():

    # Load YAML Configs
    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )
    parsed, remaining = common.Flags(configs=["defaults"]).parse(known_only=True)
    config = common.Config(configs["defaults"])
    for name in parsed.configs:
        config = config.update(configs[name])
    config = common.Flags(config).parse(remaining)

    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    config.save(logdir / "config.yaml")

    print(config, "\n")
    print("Logdir", logdir)

    # Tensorflow Setups
    import tensorflow as tf

    tf.config.experimental_run_functions_eagerly(not config.jit)
    message = "No GPU found. To actually train on CPU remove this assert."
    assert tf.config.experimental.list_physical_devices("GPU"), message
    for gpu in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    assert config.precision in (16, 32), config.precision
    if config.precision == 16:
        from tensorflow.keras.mixed_precision import experimental as prec

        prec.set_policy(prec.Policy("mixed_float16"))

    ###### [1] Set Cameras ######
    # 1-1. Additional cameras.
    additional_cams = common.get_additional_cams(config)
    additional_camera = len(additional_cams) != 0
    # 1-2. Additional eval cameras. This could be empty
    eval_cams = common.get_eval_cams(config)
    eval_cams_list = [config.control_input] + list(eval_cams.keys())
    # 1-3. Cameras which will be used for control
    control_input = config.control_input.split("|")
    # 1-4. Cameras which will be used as inputs for training MVMAE
    mae_viewpoints = [cam for cam in config.camera_keys.split("|")]
    ########################

    ###### [2] Set Replay Buffers ######
    # 2-1. Set train/eval replay
    train_replay = common.Replay(
        logdir / "train_episodes",
        **config.replay,
        augment=config.augment,
    )
    eval_replay = common.Replay(
        logdir / "eval_episodes",
        **dict(
            capacity=config.replay.capacity // 10,
            minlen=config.replay.minlen,
            maxlen=config.replay.maxlen,
        ),
        load=False,
    )
    # 2-2. Set demo_replay if using demo behavior cloning
    if config.demo_bc:
        demo_replay = common.Replay(
            logdir / "demo_episodes",
            **config.replay,
            augment=config.augment,
        )
    else:
        demo_replay = None
    ########################

    ###### [3] Set Loggers ######
    step = common.Counter(train_replay.stats["total_steps"])
    outputs = [
        common.TerminalOutput(),
        common.JSONLOutput(logdir),
        common.TensorBoardOutput(logdir),
    ]
    logger = common.Logger(step, outputs, multiplier=config.action_repeat)
    metrics = collections.defaultdict(list)

    should_train = common.Every(config.train_every)
    should_train_mae = common.Every(config.train_mae_every)
    should_log = common.Every(config.log_every)
    should_video_train = common.Every(config.eval_every)
    should_video_eval = common.Every(config.eval_every)

    ###### [4] Create Environment ######
    # 4-1. Create Env Factory
    def make_env(mode, actions_min_max=None):
        camera_keys = common.get_camera_keys(config.camera_keys)
        suite, task = config.task.split("_", 1)

        all_add_cams = dict()
        all_add_cams.update(additional_cams)
        if mode == "eval":
            # Add additional eval cameras, if they exist
            all_add_cams.update(eval_cams)

        env_cls = common.RLBench
        env = env_cls(
            task,
            camera_keys,
            config.render_size,
            shaped_rewards=config.shaped_rewards,
            use_rotation=config.use_rotation,
            additional_camera=additional_camera,
            add_cam_names=all_add_cams,
            randomize_texture=config.use_randomize,
            default_texture=config.default_texture,
        )
        if actions_min_max:
            env.register_min_max(actions_min_max)

        env = common.TimeLimit(env, config.time_limit)
        return env

    # 4-2. RLBench demonstration collection
    dev_env = make_env("train")
    if config.num_demos != 0:
        collect_fn = common.collect_demo
        actions_min_max = collect_fn(
            dev_env,
            train_replay,
            config.num_demos,
            config.camera_keys,
            shaped_rewards=config.shaped_rewards,
            demo_replay=demo_replay,
            use_rotation=config.use_rotation,
            additional_camera=additional_camera,
            randomize=config.use_randomize,
        )
        with open(config.logdir + "/actions_min_max.pkl", "wb") as f:
            pickle.dump(actions_min_max, f)
    else:
        actions_min_max = None

    # 4-3. Load actions_min_max if resuming from pre-trained models
    if config.loaddir != "none":
        with open(config.loaddir + "/actions_min_max.pkl", "rb") as f:
            actions_min_max = pickle.load(f)
        print("load actions_min_max: ", actions_min_max)

    del dev_env
    gc.collect()

    # 4-4. Create envs
    make_async_env = lambda mode: common.Async(
        functools.partial(make_env, mode, actions_min_max), config.envs_parallel
    )
    train_env_fn = functools.partial(make_async_env, "train")
    eval_env_fn = functools.partial(make_async_env, "eval")
    train_envs = [train_env_fn() for _ in range(config.envs)]
    eval_envs = [eval_env_fn() for _ in range(config.eval_envs)]

    ###### [5] Set Drivers ######
    # 5-1. Setup Train Drivers
    act_space = train_envs[0].act_space
    obs_space = train_envs[0].obs_space
    train_driver = common.Driver(train_envs, train_env_fn)
    train_driver.on_episode(
        lambda ep: per_episode(
            ep, step, config, logger, should_video_train, train_replay, mode="train"
        )
    )
    train_driver.on_step(lambda tran, worker: step.increment())
    train_driver.on_step(train_replay.add_step)
    train_driver.on_reset(train_replay.add_step)

    # 5-2. Setup Eval Drivers
    eval_driver_dict = dict()
    for cam_name in eval_cams_list:
        eval_driver_dict[cam_name] = common.Driver(eval_envs, eval_env_fn)
        eval_driver_dict[cam_name].on_episode(
            lambda ep: per_episode(
                ep,
                step,
                config,
                logger,
                should_video_eval,
                eval_replay,
                mode="eval",
                prefix=cam_name,
            )
        )

    ###### [6] Set Dataset / Agent ######
    # 6-1. Set Datasets
    train_dataset = iter(train_replay.dataset(**config.dataset))
    if demo_replay is not None:
        demo_dataset = iter(demo_replay.dataset(**config.demo_dataset))
    mae_train_dataset = iter(train_replay.dataset(**config.mae_dataset))
    report_dataset = iter(train_replay.dataset(**config.dataset))

    # 6-2. Set Agent / Policies
    agnt = agent.Agent(config, obs_space, act_space, step)
    train_policy = lambda *args: agnt.policy(*args, mode="train")
    eval_policy = lambda *args: agnt.policy(*args, mode="eval")

    # 6-3. Set train functions
    train_mae = agnt.train_mae
    if config.demo_bc:
        train_agent = common.CarryOverState(agnt.train_with_bc)
    else:
        train_agent = common.CarryOverState(agnt.train)

    ###### [7] Initialize TF Graphs ######
    # 7-1. Initialize MAE
    train_mae(next(mae_train_dataset), mae_viewpoints)

    # 7-2. Initialize World Model / Behavior Learning
    if config.demo_bc:
        train_agent(next(train_dataset), next(demo_dataset), control_input)
    else:
        train_agent(next(train_dataset), control_input)

    ###### [8] Pre-training Processes ######
    # 8-1. If pre-trained model is specified, load parameters
    loaddir = (
        pathlib.Path(config.loaddir).expanduser() if config.loaddir != "none" else None
    )
    if loaddir is not None and (loaddir / "variables.pkl").exists():
        print("Load mae, world model, and policy.")
        # load pre-trained RL agents parameters
        agnt.load(loaddir / "variables.pkl")

    # 8-2. Pre-train MAE using demonstrations
    if config.mae_pretrain:
        print("Pretrain MAE.")
        for _ in tqdm(range(config.mae_pretrain)):
            train_mae(next(mae_train_dataset), mae_viewpoints)

    # 8-3. Pre-train World Model / Behaviors using demonstrations
    if config.pretrain:
        print("Pretrain agent.")
        for _ in tqdm(range(config.pretrain)):
            if config.demo_bc:
                # Switching train/demo batch here to encourage BC pretraining
                # Anyway, here, demo and training datasets are same
                train_agent(
                    next(demo_dataset),
                    next(train_dataset),
                    control_input,
                )
            else:
                train_agent(next(train_dataset), control_input)

    ###### [9] Finally, Training Processes ######
    # 9-1. Define helper function for training
    def train_step(tran, worker):
        if should_train_mae(step):
            for _ in range(config.train_mae_steps):
                mets = train_mae(next(mae_train_dataset), mae_viewpoints)
                [metrics[key].append(value) for key, value in mets.items()]
        if should_train(step):
            for _ in range(config.train_steps):
                if config.demo_bc:
                    mets = train_agent(
                        next(train_dataset),
                        next(demo_dataset),
                        control_input,
                    )
                else:
                    mets = train_agent(next(train_dataset), control_input)
                [metrics[key].append(value) for key, value in mets.items()]
        if should_log(step):
            for name, values in metrics.items():
                logger.scalar(name, np.array(values, np.float64).mean())
                metrics[name].clear()
            logger.add(
                agnt.report(
                    next(report_dataset),
                    control_input,
                ),
                prefix="train",
            )
            logger.write(fps=True)

    # 9-2. Register helper function to train_driver
    train_driver.on_step(train_step)

    # 9-3. Start training
    while step < config.steps:
        logger.write()

        print("Start evaluation.")
        for cam_name in eval_driver_dict.keys():
            eval_driver_dict[cam_name](
                eval_policy,
                viewpoint=control_input,
                episodes=config.eval_eps,
            )

        print("Start training.")
        train_driver(
            train_policy,
            viewpoint=control_input,
            steps=config.eval_every,
        )
        agnt.save(logdir / "variables.pkl")

    # 9-4. Close all envs after training ends
    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass


# Helper function executed after episode ends
def per_episode(ep, step, config, logger, should, replay, mode, prefix=""):
    if prefix != "":
        prefix = prefix + "_"

    length = len(ep["reward"]) - 1
    score = float(ep["reward"].astype(np.float64).sum())
    success = float(np.sum(ep["success"]) >= 1.0)
    print(
        f"{mode.title()} episode has {float(success)} success, {length} steps and return {score:.1f}."
    )
    logger.scalar(f"{prefix}{mode}_success", float(success))
    logger.scalar(f"{prefix}{mode}_return", score)
    logger.scalar(f"{prefix}{mode}_length", length)
    for key, value in ep.items():
        if re.match(config.log_keys_sum, key):
            logger.scalar(f"sum_{prefix}{mode}_{key}", ep[key].sum())
        if re.match(config.log_keys_mean, key):
            logger.scalar(f"mean_{prefix}{mode}_{key}", ep[key].mean())
        if re.match(config.log_keys_max, key):
            logger.scalar(f"max_{prefix}{mode}_{key}", ep[key].max(0).mean())

    if should(step):
        cam_name = prefix.split("_")[0].split("|")[0]
        if mode != "train":
            # Because we're randomizing the camera, it's a bit difficult
            # to log videos of training episodes.
            out_video = ep[cam_name]
            logger.video(f"{prefix}{mode}_policy_image", out_video)
    logger.add(replay.stats, prefix=mode)
    logger.write()


if __name__ == "__main__":
    main()
