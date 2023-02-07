import numpy as np
import tensorflow as tf
import random


class Driver:
    def __init__(self, envs, env_fn, **kwargs):
        self._envs = envs
        self._env_fn = env_fn
        self._kwargs = kwargs
        self._on_steps = []
        self._on_resets = []
        self._on_episodes = []
        self._act_spaces = [env.act_space for env in envs]
        self.reset()

    def on_step(self, callback):
        self._on_steps.append(callback)

    def on_reset(self, callback):
        self._on_resets.append(callback)

    def on_episode(self, callback):
        self._on_episodes.append(callback)

    def reset(self):
        self._obs = [None] * len(self._envs)
        self._eps = [None] * len(self._envs)
        self._state = None

    def handle_sim_failure(self, i):
        # If reset failed, re-start env
        print(f"Resetting Env {i} due to sim failure")
        self._envs[i].kill()
        self._envs[i] = self._env_fn()
        # NOTE: We need this to set a new action space
        # A bit hacky ..
        self._envs[i].act_space

        if self._state is not None:
            if len(self._envs) == 1:
                # If not using parallel envs, let's just set state to None
                self._state = None
            else:
                # If using parallel envs, manually reset state = (latent, action)
                
                # Choose different index j != i, which is index of other envs
                j = i - 1 if i != 0 else i + 1

                # Set new empty latent
                latent = self._state[0]
                new_latent = dict()
                for key in latent.keys():
                    new_latent[key] = tf.concat(
                        [
                            latent[key][:i],
                            tf.zeros_like(latent[key][j])[None],
                            latent[key][i + 1 :],
                        ],
                        axis=0,
                    )
                # Set new empty action
                action = self._state[1]
                new_action = tf.concat(
                    [
                        action[:i],
                        tf.zeros_like(action[j])[None],
                        action[i + 1 :],
                    ],
                    axis=0,
                )
                self._state = (new_latent, new_action)

    def __call__(self, policy, viewpoint, steps=0, episodes=0):
        step, episode = 0, 0
        control_input = viewpoint

        while step < steps or episode < episodes:
            for i in range(len(self._envs)):
                ob = self._obs[i]
                if ob is None or ob["is_last"]:
                    reset_success = False
                    while not reset_success:
                        try:
                            # Try resetting env_i
                            ob = self._envs[i].reset()
                            self._obs[i] = ob() if callable(ob) else ob
                            ob = self._obs[i]
                            act = {
                                k: np.zeros(v.shape)
                                for k, v in self._act_spaces[i].items()
                            }
                            tran = {
                                k: self._convert(v) for k, v in {**ob, **act}.items()
                            }
                            [
                                fn(tran, worker=i, **self._kwargs)
                                for fn in self._on_resets
                            ]
                            self._eps[i] = [tran]

                            reset_success = True
                        except:
                            self.handle_sim_failure(i)

            # Stack current obs after resetting
            obs = {k: np.stack([o[k] for o in self._obs]) for k in self._obs[0]}

            # Get actions from batch of observations
            actions, self._state = policy(
                obs, control_input, self._state, **self._kwargs
            )
            actions = [
                {k: np.array(actions[k][i]) for k in actions}
                for i in range(len(self._envs))
            ]
            assert len(actions) == len(self._envs)

            obs = [None for _ in range(len(self._envs))]
            for i in range(len(self._envs)):
                try:
                    ob = self._envs[i].step(actions[i])
                    ob = ob() if callable(ob) else ob
                    obs[i] = ob
                except:
                    # If failure occurs during episode step, let's skip this env
                    # And do reset during the next episode by setting "is_last" to True
                    print(f"Skipping step for Env {i} due to sim failure")
                    obs[i] = {"is_last": True}
                    self.handle_sim_failure(i)
                    # Let's skip to next env
                    continue

                act = actions[i]
                tran = {k: self._convert(v) for k, v in {**ob, **act}.items()}
                [fn(tran, worker=i, **self._kwargs) for fn in self._on_steps]
                self._eps[i].append(tran)
                step += 1
                if ob["is_last"]:
                    ep = self._eps[i]
                    ep = {k: self._convert([t[k] for t in ep]) for k in ep[0]}
                    [fn(ep, **self._kwargs) for fn in self._on_episodes]
                    episode += 1
                    control_input = viewpoint
            self._obs = obs

        return step, episode

    def _convert(self, value):
        value = np.array(value)
        if np.issubdtype(value.dtype, np.floating):
            return value.astype(np.float32)
        elif np.issubdtype(value.dtype, np.signedinteger):
            return value.astype(np.int32)
        elif np.issubdtype(value.dtype, np.uint8):
            return value.astype(np.uint8)
        return value
