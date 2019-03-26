# Author: Taoz
# Date  : 8/29/2018
# Time  : 5:28 PM
# FileName: runner.py


import sk68.dqn.experiment as runner
from esheep_env.game_env import GameEnvironment
if __name__ == '__main__':
    runner.train(GameEnvironment)
    # runner.test(render=True)
