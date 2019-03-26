# Author: Taoz
# Date  : 8/25/2018
# Time  : 12:23 PM
# FileName: environment.py

import gym
from gym.envs.atari import AtariEnv
from config import *
import gym.spaces


class GameEnv(object):
    def __init__(self, game, obs_type, frame_skip):
        self.gym_env = AtariEnv(game=game,
                                obs_type=obs_type,
                                frameskip=frame_skip,
                                repeat_action_probability=0.05)
        self.step_count = 0
        self.gym_env.reset()
        self.lives = self.gym_env.ale.lives()

    def step(self, action):
        self.step_count += 1
        observation, reward, done, _ = self.gym_env.step(action)
        score = reward
        new_lives = self.gym_env.ale.lives()
        reward = max(NEGATIVE_REWARD, min(POSITIVE_REWARD, reward))

        if self.lives > new_lives:
            reward = NEGATIVE_REWARD
        self.lives = new_lives
        return observation, reward, done, new_lives, score

    def render(self):
        return self.gym_env.render()

    def random_action(self):
        return self.gym_env.action_space.sample()

    def action_num(self):
        return self.gym_env.action_space.n

    def reset(self, skip_begin_frame=5):
        assert skip_begin_frame > 0
        self.gym_env.reset()
        obs = None
        for _ in range(skip_begin_frame):
            obs, _, _, _ = self.gym_env.step(self.gym_env.action_space.sample())
        self.lives = self.gym_env.ale.lives()
        self.step_count = 0
        return obs

    def close(self):
        self.gym_env.close()
