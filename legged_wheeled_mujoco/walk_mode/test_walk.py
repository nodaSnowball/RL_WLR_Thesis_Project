import argparse
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import gymnasium as gym
import numpy as np
import envs.register
from stable_baselines3 import SAC,PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env_name', default="Walk-v0",
                    help='Mujoco Gym environment (default: Biped-v0)')
args = parser.parse_args()

# Environment
def make_env(n=2000, render_mode=None):
        def _init():
                env = gym.make(args.env_name, max_step=n, healthy_reward=1, render_mode=render_mode, camera_name='top')
                return env
        return _init
eval_env = DummyVecEnv([make_env(2000, render_mode='human') for _ in range(1)])

num_test = 100
render = True

# Agent
model = SAC.load(os.path.join(os.path.dirname(__file__), 'test_model/best_model'))
# model = SAC.load(os.path.join(os.path.dirname(__file__), '../models/jump/land'))
evaluate_policy(model, eval_env, 100, deterministic=1, render=True, return_episode_rewards=True)
state = eval_env.reset()
done = False
step = 0
c_game = 1
success = 0
avg_reward = 0
rewards = 0
steps = 0

while 1:
        action, _ = model.predict(state, deterministic=1)
        state, reward, done, info = eval_env.step(action)
        rewards += reward
        step += 1
        
        if done:
                print(f'Game {c_game}: total_reward={rewards}, length={step}')
                steps += step
                state = eval_env.reset()
                step = 0
                c_game += 1
                avg_reward += rewards/num_test
                rewards = 0
                if reward == 100:
                        success += 1
        
        if c_game>=num_test:
                print(f'Success rate: {success/num_test}, mean ep length: {steps/num_test}')
                break
        


