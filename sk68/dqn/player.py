# Author: Taoz
# Date  : 8/25/2018
# Time  : 12:22 PM
# FileName: player.py

import numpy as np
from sk68.dqn.config import *
import random
import logging
from mxnet import init, nd, autograd, gluon


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


class Player(object):
    def __init__(self, game, q_learning, rng):
        self.game = game
        self.action_num = 5  # [0,1,2,..,action_num-1]
        self.q_learning = q_learning
        self.rng = rng
        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_MIN
        self.epsilon_rate = (EPSILON_START - EPSILON_MIN) * 1.0 / EPSILON_DECAY
        self.waiting_restart = False


    def run_episode(self, epoch, state, move, frame_period, replay_buffer, render=False, random_action=False,
                    testing=False):
        episode_step = 0
        episode_reword = 0
        train_count = 0
        loss_sum = 0
        episode_score = 0.0
        q_sum = 0.0
        q_count = 0
        episode_score = 0
        score = 0
        # st = state
        # tag = True

        frame, \
        state, \
        location, \
        immutable_element, \
        mutable_element, \
        body, \
        bodies, \
        asset_ownership, \
        self_asset, \
        self_status, \
        pointer, \
        score, \
        kill, \
        health = self.game.get_observation_with_info()



        # # do no operation steps.
        # max_no_op_steps = 20
        # for _ in range(self.rng.randint(10, max_no_op_steps)):
        #     self.game.step(0)

        while True:
            if mutable_element is None or mutable_element.size == 0 or self.waiting_restart:
                time.sleep(frame_period / 1000)
                frame, \
                state, \
                location, \
                immutable_element, \
                mutable_element, \
                body, \
                bodies, \
                asset_ownership, \
                self_asset, \
                self_status, \
                pointer, \
                score, \
                kill, \
                health = self.game.get_observation_with_info()
                if mutable_element is not None:
                    self.game.submit_reincarnation()
                if health == 1:
                    self.waiting_restart = False

            else:
                # print('render')

                # st = np.concatenate((immutable_element, mutable_element, bodies, self_asset))
                st = np.empty([8, 100, 200])
                st[0] = location.reshape([1, 100, 200])
                st[1] = immutable_element.reshape([1, 100, 200])
                st[2] = mutable_element.reshape([1, 100, 200])
                st[3] = body.reshape([1, 100, 200])
                st[4] = bodies.reshape([1, 100, 200])
                st[5] = asset_ownership.reshape([1, 100, 200])
                st[6] = self_asset.reshape([1, 100, 200])
                st[7] = self_status.reshape([1, 100, 200])

                # print('next_state', st.shape)
                # print(st.shape)
                # st = np.concatenate((location, immutable_element, mutable_element, bodies))
                # print(location.shape)
                # print(st.shape)
                s = score
                # print(score)

                if state == 1:
                    if not testing and random_action:
                        action_num = random.randint(0, 4)
                        # action = move[random.randint(0, 3)]
                    else:
                        action_num, max_q = self._choose_action(st, replay_buffer, testing)
                        print('action_num_out:', action_num)
                        # action = move[action_num]
                        if max_q is not None:
                            q_count += 1
                            q_sum += max_q

                    if action_num != 4:
                        action = move[int(action_num)]
                        self.game.submit_action(frame, action, None, None, None)
                    # time.sleep(frame_period * 3 / 1000)
                    current_frame = frame
                    # print('current_frame', current_frame)

                    while frame <= current_frame:
                        frame, \
                        state, \
                        location, \
                        immutable_element, \
                        mutable_element, \
                        body, \
                        bodies, \
                        asset_ownership, \
                        self_asset, \
                        self_status, \
                        pointer, \
                        score, \
                        kill, \
                        health = self.game.get_observation_with_info()
                        time.sleep(frame_period / 1000)

                    if action_num != -1:

                        if health == 0:
                            reward = NEGATIVE_REWARD
                            terminal = True
                        else:
                            next_st = np.empty([8, 100, 200])
                            next_st[0] = location.reshape([1, 100, 200])
                            next_st[1] = immutable_element.reshape([1, 100, 200])
                            next_st[2] = mutable_element.reshape([1, 100, 200])
                            next_st[3] = body.reshape([1, 100, 200])
                            next_st[4] = bodies.reshape([1, 100, 200])
                            next_st[5] = asset_ownership.reshape([1, 100, 200])
                            next_st[6] = self_asset.reshape([1, 100, 200])
                            next_st[7] = self_status.reshape([1, 100, 200])

                            next_s = score
                            if next_s - s > 0:
                                print('get food!')
                                reward = POSITIVE_REWARD
                                episode_score += next_s - s
                                terminal = False
                            else:
                                reward = 0
                                terminal = False

                        replay_buffer.add_sample(st, action_num, reward, terminal)
                        episode_step += 1
                        episode_reword += reward

                        if terminal:
                            break
                    else:
                        reward = NEGATIVE_REWARD
                        episode_reword += reward

                elif state == 2:
                    print('killed')
                    rsp = self.game.submit_reincarnation()
                    self.waiting_restart = True
                    print('reincarnation rsp:', rsp)
                    time.sleep(frame_period / 1000)
                    break
                else:
                    time.sleep(frame_period / 1000)
                    print('get state:', state)
                    # action_num = -1
            if not testing and episode_step % TRAIN_PER_STEP == 0 and not random_action:
                logging.info('-- train_policy_net episode_step=%d' % episode_step)
                imgs, actions, rs, terminal = replay_buffer.random_batch(32)
                loss = self.q_learning.train_policy_net(imgs, actions, rs, terminal)
                loss_sum += loss
                train_count += 1

        return episode_step, episode_reword, episode_score, loss_sum / (train_count + 0.0000001), q_sum / (
                    q_count + 0.0000001)

    def _choose_action(self, img, replay_buffer, testing):
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_rate)
        max_q = None

        if not testing and self.rng.rand() < self.epsilon:
            action = self.rng.randint(0, self.action_num)
        else:
            phi = replay_buffer.phi(img)
            action, max_q = self.q_learning.choose_action(phi)
        return action, max_q


if __name__ == '__main__':
    pass
