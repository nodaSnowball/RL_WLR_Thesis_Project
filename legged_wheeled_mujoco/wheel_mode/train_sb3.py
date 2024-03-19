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
from stable_baselines3 import SAC, TD3
from stable_baselines3.common.callbacks import EvalCallback,CheckpointCallback,CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from collections import deque

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env_name', default="Biped-v0",
                    help='Mujoco Gym environment (default: Biped-v0)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
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
parser.add_argument('--suffix', type=str, default='sac_dfctl',
                    help='run on CUDA (default: False)')
args = parser.parse_args()
start_time = time.strftime("%m%d%H")
model_dir = os.path.join(os.path.dirname(__file__), f'history/models/{start_time}_lr_{args.lr}_'+args.suffix)
log_dir = os.path.join(os.path.dirname(__file__), f'history/logs/{start_time}_lr_{args.lr}_'+args.suffix)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
def make_env(n=2000):
        def _init():
                env = gym.make(args.env_name)
                return env
        return _init
# Environment
env = DummyVecEnv([make_env(2000) for _ in range(5)])
eval_env = DummyVecEnv([make_env(2000) for _ in range(5)])

# Agent
model = SAC('MlpPolicy', env, learning_rate=args.lr, verbose=0, tensorboard_log=log_dir)
ec = EvalCallback(env, eval_freq=2000, n_eval_episodes=20, deterministic=1, render=0, 
                  best_model_save_path=model_dir, log_path=log_dir)
model.learn(total_timesteps=5e6, callback=ec)
model.save(model_dir+'/final_model')
end_time = time.strftime('%m%d%H')


