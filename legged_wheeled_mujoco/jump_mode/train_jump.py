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
from jump_mode.agent_jump import CustomExtractor
from stable_baselines3 import SAC, TD3, PPO
from stable_baselines3.common.callbacks import EvalCallback,CheckpointCallback,CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from collections import deque

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env_name', default="Jump-v0")
parser.add_argument('--input', default="short", type=str)
parser.add_argument('--alg', default="sac", type=str)
parser.add_argument('--hr', default=0.3, type=float,
                    help='healthy reward')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--seed', type=int, default=123456)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--num_process', type=int, default=10)
parser.add_argument('--num_steps', type=int, default=10000001)
parser.add_argument('--hidden_size', type=int, default=128, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--suffix', type=str, default='reset_target')
args = parser.parse_args()
num_envs = args.num_process
start_time = time.strftime("%m%d%H")
dirname = args.alg +'_'+ args.input + '_hr'+str(args.hr) + '_lr'+str(args.lr) +'_'+args.suffix
model_dir = os.path.join(os.path.dirname(__file__), 'history_lidar/models/'+dirname)
log_dir = os.path.join(os.path.dirname(__file__), 'history_lidar/logs/'+dirname)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
def make_env(n=2000, render_mode=None, camera_id=None):
        def _init():
                env = gym.make(args.env_name, max_step=n, healthy_reward=.3, render_mode=render_mode, fixed_height=0, camera_id=camera_id)
                return env
        return _init
# Environment
env = DummyVecEnv([make_env(500) for _ in range(10)])
# env = VecNormalize(env)
eval_env = DummyVecEnv([make_env(500, render_mode='human', camera_id=3) for _ in range(1)])
# eval_env = VecNormalize(eval_env)


# Agent

# CNN model
if args.input == 'long':
        if args.alg == 'sac':
                policy_kwargs = dict(
                        features_extractor_class=CustomExtractor,
                        features_extractor_kwargs=dict(args=args, features_dim=600, ),  # 736, 1472, 1024
                        activation_fn=torch.nn.ReLU,
                        normalize_images=False,
                        net_arch=dict(pi=[512, 256, 128], qf=[256, 128]))
                model = SAC('CnnPolicy', env, learning_rate=args.lr, verbose=0, tensorboard_log=log_dir, policy_kwargs=policy_kwargs, buffer_size=int(2e5), seed=1)
        if args.alg == 'ppo':
                policy_kwargs = dict(
                        features_extractor_class=CustomExtractor,
                        features_extractor_kwargs=dict(args=args, features_dim=600, ),  # 736,1472
                        activation_fn=torch.nn.ReLU,
                        normalize_images=False,
                        net_arch=dict(pi=[512, 256, 128], vf=[256, 128]))
                model = PPO('CnnPolicy', env, learning_rate=args.lr, verbose=0, tensorboard_log=log_dir, policy_kwargs=policy_kwargs, buffer_size=int(2e5), seed=1)

# MLP model
if args.input == 'short':
        if args.alg == 'sac':
                policy_kwargs = dict(
                        activation_fn=torch.nn.ReLU,
                        net_arch=dict(pi=[512, 512, 256], qf=[256, 512, 256]))
                model = SAC('MlpPolicy', env, learning_rate=args.lr, verbose=0, tensorboard_log=log_dir, policy_kwargs=policy_kwargs, seed=1)
        if args.alg == 'ppo':
                policy_kwargs = dict(
                        activation_fn=torch.nn.ReLU,
                        net_arch=dict(pi=[512, 512, 256], vf=[256, 512, 256]))
                model = PPO('MlpPolicy', env, learning_rate=args.lr, verbose=0, tensorboard_log=log_dir, policy_kwargs=policy_kwargs, seed=1)
        
# Load model
# model_path = os.path.join(os.path.dirname(__file__), 'test_model/continue')
# model = SAC.load(model_path,env)


# train
ec = EvalCallback(eval_env, eval_freq=20000/num_envs, n_eval_episodes=20, deterministic=1, render=1, 
                  best_model_save_path=model_dir, log_path=log_dir)
cc = CheckpointCallback(save_freq=int(5e4/num_envs), save_path=model_dir,
                                             name_prefix='checkpoint_model')
print(f'Training start at {start_time}. Learning rate:{args.lr} healthy reward:{args.hr} env: {args.env_name} algorithm: {args.alg}\ncomment:{args.suffix}')
model.learn(total_timesteps=5e6, callback=CallbackList([ec,cc]))
model.save(model_dir+'/final_model')
end_time = time.strftime('%m%d%H')


