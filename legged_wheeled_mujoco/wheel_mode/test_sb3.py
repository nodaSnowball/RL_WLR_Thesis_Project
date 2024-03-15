import argparse
import time
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import gym
import numpy as np
import itertools
import torch
from torch.utils.tensorboard import SummaryWriter
import envs.register
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback,CheckpointCallback,CallbackList
# from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from collections import deque

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env_name', default="Biped-v0",
                    help='Mujoco Gym environment (default: Biped-v0)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--suffix', default='', type=str)
parser.add_argument('--eval', type=bool, default=False,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.001, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=1e-4, metavar='G',
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
parser.add_argument('--num_steps', type=int, default=10000001, metavar='N',
                    help='maximum number of steps (default: 500000)')
parser.add_argument('--hidden_size', type=int, default=128, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=1000, metavar='N',
                    help='Steps sampling random actions (default: 1000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true", default=True,
                    help='run on CUDA (default: False)')
args = parser.parse_args()

# Environment
env = gym.make(args.env_name)

num_test = 100
render = True

# Agent
model = SAC.load(os.path.join(os.path.dirname(__file__), 'models/best_model'))
ec = EvalCallback(env, eval_freq=1000, n_eval_episodes=20, render=0)
state = env.reset()
done = False
step = 0
c_game = 1
success = 0
avg_reward = 0
rewards = 0
steps = 0

while 1:
        action, _ = model.predict(state, deterministic=0)
        state, reward, done, _ = env.step(action)
        rewards += reward
        step += 1
        
        if done:
                print(f'Game {c_game}: total_reward={rewards}, length={step}')
                steps += step
                state = env.reset()
                step = 0
                c_game += 1
                avg_reward += rewards/num_test
                rewards = 0
                if reward == 10:
                        success += 1
        if render:
                env.render()
        
        if c_game>=num_test:
                print(f'Success rate: {success/num_test}, mean ep length: {steps/num_test}')
                break
        


