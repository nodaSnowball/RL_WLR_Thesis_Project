import numpy as np
import random
import math
import gym
import torch
import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from itertools import count
import argparse
from sac import SAC
from replay_memory import ReplayMemory
import envs.register

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="Stop-v0",
                    help='Mujoco Gym environment')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=128, metavar='N',
                    help='hidden size (default: 128)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true", default=True,
                    help='run on CUDA (default: False)')
args = parser.parse_args()

# Environment, Agent
env = gym.make(args.env_name)
agent = SAC(env.observation_space.shape[0], env.action_space, args)

'''
models/forward/sac_actor_Forward-v0_lr_0.0005_ep1200_sr1.0
models/biped/sac_actor_Biped-v0_nopreproc_lr_0.0005_ep7300_sr0.9
'''

actor = os.path.join(os.path.dirname(__file__),'models/stop/actor')
critic = os.path.join(os.path.dirname(__file__),'models/stop/critic')
agent.load_model(actor, critic)

success = 0
avg = 0.0
res = []

# Testing loop
def testSAC():
    success = 0
    num_test = 100
    for i in range(num_test):
        state = env.reset() # state size: 27
        
        ret = 0.0
        for t in count():
            env.render()
            action = agent.select_action(state, evaluate=True)
            # print(action)
            # print(f'Local linear velocity: {state[6:9]}\nLocal orientation velocity: {state[12:]}')
            nextState, reward, done, _ = env.step(action)
            ret += reward
            state = nextState
            if done:
                print("Episode %d ended in %d steps" % (i + 1, t + 1))
                if reward == 100:
                    success += 1
                res.append(ret)
                # print('Return is',ret)
                break
    avg = np.average(res)
    sr = success/num_test
    return res, avg, sr


res, avg, sr = testSAC()

print(f'average reward = {avg}, success rate = {sr}')


