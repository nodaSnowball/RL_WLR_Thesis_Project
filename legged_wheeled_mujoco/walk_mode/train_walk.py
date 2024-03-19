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
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from collections import deque

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env_name', default="Walk-v0")
parser.add_argument('--policy', default="Gaussian")
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--seed', type=int, default=123456)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--num_steps', type=int, default=10000001)
parser.add_argument('--hidden_size', type=int, default=128, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true", default=True,
                    help='run on CUDA (default: False)')
parser.add_argument('--suffix', type=str, default='walk_sac_')
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
0
# Agent
model = SAC('MlpPolicy', env, learning_rate=args.lr, verbose=0, tensorboard_log=log_dir)
ec = EvalCallback(env, eval_freq=4000, n_eval_episodes=20, deterministic=1, render=0, 
                  best_model_save_path=model_dir, log_path=log_dir)
model.learn(total_timesteps=5e6, callback=ec)
model.save(model_dir+'/final_model')
end_time = time.strftime('%m%d%H')
print(f'training complete. '+start_time+' -> '+end_time)


