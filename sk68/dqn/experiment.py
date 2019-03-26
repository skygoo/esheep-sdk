# Author: Taoz
# Date  : 8/25/2018
# Time  : 12:13 PM
# FileName: experiment.py


import numpy as np

# from src.dqn.player_priority import Player
from player import Player
# from game_env import GameEnv
from esheep_env.game_env import GameEnvironment
# from src.dqn.replay_buffer import ReplayBuffer
# from priority_experience_replay import PriorityReplayBuffer
from replay_buffer import ReplayBuffer
from d3qn import D3QLearning

import g_utils
import mxnet as mx
import mxnet as nd
import ztutils

from config import *


class Experiment(object):
    ctx = g_utils.try_gpu(GPU_INDEX)

    INPUT_SAMPLE = nd.random.uniform(0, 255, (1, PHI_LENGTH * CHANNEL, HEIGHT, WIDTH), ctx=ctx) / 255.0
    # INPUT_SAMPLE = nd.random.uniform(0, 255, (1, CHANNEL, HEIGHT, WIDTH), ctx=ctx) / 255.0
    print('Input_sample', INPUT_SAMPLE.shape)

    mx.random.seed(RANDOM_SEED)
    rng = np.random.RandomState(RANDOM_SEED)

    def __init__(self, testing=False):
        ztutils.mkdir_if_not_exist(MODEL_PATH)
        self.step_count = 0
        self.episode_count = 0
        self.target_net_update_count = 0
        ip = '127.0.0.1'
        port = str(PORT)
        self.q_learning = D3QLearning(Experiment.ctx,
                                           Experiment.INPUT_SAMPLE,
                                           model_file=PRE_TRAIN_MODEL_FILE
                                           )

        self.game = GameEnvironment(ip=ip, port=port, api_token="test")

        self.player = Player(self.game,
                             self.q_learning,
                             Experiment.rng)

        # self.replay_buffer = PriorityReplayBuffer(HEIGHT,
        #                                   WIDTH,
        #                                   CHANNEL,
        #                                   Experiment.rng,
        #                                   BUFFER_MAX)
        self.replay_buffer = ReplayBuffer(HEIGHT,
                                          WIDTH,
                                          CHANNEL,
                                          Experiment.rng,
                                          BUFFER_MAX)
        self.update_target_episode = UPDATE_TARGET_BY_EPISODE_BEGIN
        self.update_target_interval = UPDATE_TARGET_BY_EPISODE_BEGIN + UPDATE_TARGET_RATE
        self.testing = testing
        self.frame_period = self.game.get_frame_period()
        self.roomId, self.state = self.game.create_room("123")
        self.move, swing, fire, apply = self.game.get_action_space()

    def start_train(self):
        print('frame_period', self.frame_period)
        for i in range(1, EPOCH_NUM + 1):
            self._run_epoch(i)
        print('train done.')
        self.game.close()

    def start_test(self, render):
        assert PRE_TRAIN_MODEL_FILE is not None
        for i in range(1, EPOCH_NUM + 1):
            self._run_epoch(i, render=render)
        print('test done.')
        self.game.close()

    def _run_epoch(self, epoch, render=False):
        print('run epoch')
        steps_left = EPOCH_LENGTH
        random_episode = True
        episode_in_epoch = 0
        step_in_epoch = 0
        reward_in_epoch = 0.0
        score_in_epoch = 0.0

        while steps_left > 0:
            if self.step_count > BEGIN_RANDOM_STEP:
                random_episode = False
            t0 = time.time()
            # ep_steps, ep_reward, ep_score, avg_loss, avg_max_q, avg_error_sum = self.player.run_episode(epoch,
            ep_steps, ep_reward, ep_score, avg_loss, avg_max_q= self.player.run_episode(epoch,
                                                                                         self.state,
                                                                                         self.move,
                                                                                         self.frame_period,
                                                                                         self.replay_buffer,
                                                                                         render=render,
                                                                                         random_action=random_episode,
                                                                                         testing=self.testing)

            self.step_count += ep_steps
            if not random_episode:
                self.episode_count += 1
                episode_in_epoch += 1
                score_in_epoch += ep_score
                step_in_epoch += ep_steps
                reward_in_epoch += ep_reward
                steps_left -= ep_steps
            t1 = time.time()

            print(
                'episode [%d], episode step=%d, total_step=%d, time=%.2fs, score=%.2f, ep_reward=%.2f, avg_loss=%.4f, avg_q=%f'
                % (self.episode_count, ep_steps, self.step_count, (t1 - t0), ep_score, ep_reward, avg_loss, avg_max_q))
            print('')
            self._update_target_net(random_episode)

        self._save_net()
        print('\n%s EPOCH finish [%d], episode=%d, step=%d, avg_step=%d, avg_score=%.2f avg_reward=%.2f \n\n\n' %
              (time.strftime("%Y-%m-%d %H:%M:%S"),
               epoch,
               self.episode_count,
               self.step_count,
               step_in_epoch // episode_in_epoch,
               score_in_epoch / episode_in_epoch,
               reward_in_epoch / episode_in_epoch))

    def _update_target_net(self, random_action=False):
        if not self.testing and self.episode_count == self.update_target_episode and not random_action:
            self.target_net_update_count += 1
            print('%s UPDATE TARGET NET, interval[%.3f], update count[%d]\n' % (
                time.strftime("%Y-%m-%d %H:%M:%S"), self.update_target_interval, self.target_net_update_count))

            self.update_target_episode = int(self.update_target_episode + self.update_target_interval)
            self.update_target_interval = min((self.update_target_interval + UPDATE_TARGET_RATE),
                                              UPDATE_TARGET_BY_EPISODE_END)

            self.q_learning.update_target_net()

    def _save_net(self):
        if not self.testing:
            self.q_learning.save_params_to_file(MODEL_PATH, MODEL_FILE_MARK + BEGIN_TIME)


def train():
    print(' ====================== START TRAIN ========================')
    exper = Experiment()
    exper.start_train()


def test(render):
    print(' ====================== START test ========================')
    exper = Experiment(testing=True)
    exper.start_test(render=render)


def test_speed():
    pass


if __name__ == '__main__':
    train()
