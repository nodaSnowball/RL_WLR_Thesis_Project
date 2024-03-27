import argparse
import time
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import gymnasium as gym
import numpy as np
import itertools
import torch
from torch.utils.tensorboard import SummaryWriter
import envs.register
from stable_baselines3 import SAC, TD3, PPO
from stable_baselines3.common.callbacks import EvalCallback,CheckpointCallback,CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from collections import deque

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env_name', default="Jump-v0")
parser.add_argument('--policy', default="Gaussian")
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--seed', type=int, default=123456)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--num_steps', type=int, default=10000001)
parser.add_argument('--hidden_size', type=int, default=128, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--suffix', type=str, default='sac_pretrain')
args = parser.parse_args()
start_time = time.strftime("%m%d%H")
model_dir = os.path.join(os.path.dirname(__file__), f'history_lidar/models/{start_time}_lr_{args.lr}_'+args.suffix)
log_dir = os.path.join(os.path.dirname(__file__), f'history_lidar/logs/{start_time}_lr_{args.lr}_'+args.suffix)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
def make_env(n=2000, render_mode=None, camera_id=None):
        def _init():
                env = gym.make(args.env_name, max_step=n, healthy_reward=.5, render_mode=render_mode, camera_id=camera_id)
                return env
        return _init
# Environment
env = DummyVecEnv([make_env(500) for _ in range(5)])
# env = VecNormalize(env)
eval_env = DummyVecEnv([make_env(500, render_mode='human', camera_id=0) for _ in range(1)])
# eval_env = VecNormalize(eval_env)

# Agent
policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                     net_arch=dict(pi=[512, 512, 256], qf=[512, 512, 256]))
model = SAC('MlpPolicy', env, learning_rate=args.lr, verbose=0, tensorboard_log=log_dir, policy_kwargs=policy_kwargs)
model_path = os.path.join(os.path.dirname(__file__), 'test_model/best_model')
# model = SAC.load(model_path,env)
ec = EvalCallback(eval_env, eval_freq=2000, n_eval_episodes=20, deterministic=0, render=1, 
                  best_model_save_path=model_dir, log_path=log_dir)
cc = CheckpointCallback(save_freq=int(5e4), save_path=model_dir,
                                             name_prefix='checkpoint_model')
model.learn(total_timesteps=5e6, callback=CallbackList([ec,cc]))
model.save(model_dir+'/final_model')
end_time = time.strftime('%m%d%H')


