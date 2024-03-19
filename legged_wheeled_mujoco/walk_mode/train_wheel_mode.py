import argparse
import time
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import gym
import numpy as np
import itertools
import torch
from sac import SAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory
import envs.register
from collections import deque

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="Biped-v0",
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
parser.add_argument('--lr', type=float, default=1e-6, metavar='G',
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
# env = NormalizedActions(gym.make(args.env_name))
env = gym.make(args.env_name)
env.seed(args.seed)
env.action_space.seed(args.seed)

torch.manual_seed(args.seed)
np.random.seed(args.seed)    # the st seed, the same number of seed, the random is same

# Agent
agent = SAC(env.observation_space.shape[0], env.action_space, args)
model_dir = os.path.join(os.path.dirname(__file__), 'models/'+time.strftime("%Y%m%d%H%M")+f'_lr_{args.lr}')
log_dir = os.path.join(os.path.dirname(__file__), 'logs/'+time.strftime("%Y%m%d%H%M")+f'_lr_{args.lr}/')
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
#Tesnorboard "{}" inside is the value behind
writer = SummaryWriter(log_dir)

# Memory
memory = ReplayMemory(args.replay_size, args.seed)

# Training Loop
total_numsteps = 0
updates = 0
c_log = 0
ll = 50
success_list = deque([], maxlen=ll)

for i_episode in range(10000):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()

    while not done:
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(state)  # Sample action from policy
        env.render()

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            c_log += 1 
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

                if c_log%5 == 0:
                    writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    writer.add_scalar('loss/policy', policy_loss, updates)
                    writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                    writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                    # writer.add_scalar('success_rate', success_rate, updates)
                    updates += 1

        next_state, reward, done, _ = env.step(action) # Step
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward


        mask = 1 if episode_steps == env._max_episode_steps else float(not done)

        memory.push(state, action, reward, next_state, mask) # Append transition to memory

        state = next_state
    
    is_success = 1 if reward==10 else 0
    success_list.append(episode_reward)
    avgrwd = sum(success_list)/ll
    
    if total_numsteps > args.num_steps:
        break

    writer.add_scalar('reward/train', episode_reward, i_episode)
    if i_episode % 50 == 0 and i_episode>500:
        print("Episode: {}, average reward: {}, episode steps: {}, reward: {}".format(i_episode, avgrwd, episode_steps, round(episode_reward, 2)))

    if (i_episode%50==0 and avgrwd>500) or i_episode%500==0:    # i_episode > 1000 and i_episode%200==0:
        agent.save_model(path=model_dir+'lr_'+str(args.lr)+'_ep'+str(i_episode)+'_ar'+str(avgrwd))

env.close()

