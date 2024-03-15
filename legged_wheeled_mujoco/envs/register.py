from gym.envs.registration import register
import numpy as np

register(
    id="Biped-v0", # 环境id
    entry_point="envs.env_wheel_mode:BipedEnv", # 环境类入口
    max_episode_steps=2000, # 一个episode的最大步数
    reward_threshold=6000.0, # 完成任务的奖励阈值
)

register(
    id="Jump-v0", # 环境id
    entry_point="envs.jump:JumpEnv", # 环境类入口
    max_episode_steps=1000, # 一个episode的最大步数
    reward_threshold=6000.0, # 完成任务的奖励阈值
)

register(
    id="Forward-v0", # 环境id
    entry_point="envs.forward:ForwardEnv", # 环境类入口
    max_episode_steps=1000, # 一个episode的最大步数
    reward_threshold=6000.0, # 完成任务的奖励阈值
)

register(
    id="Stop-v0", # 环境id
    entry_point="envs.stop:StopEnv", # 环境类入口
    max_episode_steps=2000, # 一个episode的最大步数
    reward_threshold=6000.0, # 完成任务的奖励阈值
)

