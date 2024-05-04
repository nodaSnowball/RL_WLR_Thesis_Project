from gymnasium.envs.registration import register
import numpy as np

register(
    id="Roll-v0", # 环境id
    entry_point="envs.env_roll_short_term:RollEnv", # 环境类入口
    max_episode_steps=2000, # 一个episode的最大步数
    reward_threshold=6000.0, # 完成任务的奖励阈值
)

register(
    id="Roll-v1", # 环境id
    entry_point="envs.env_roll_long_term:RollEnv", # 环境类入口
    max_episode_steps=2000, # 一个episode的最大步数
    reward_threshold=6000.0, # 完成任务的奖励阈值
)

register(
    id="Jump-v0", # 环境id
    entry_point="envs.env_jump_short_term_input:JumpEnv", # 环境类入口
    max_episode_steps=1000, # 一个episode的最大步数
    reward_threshold=6000.0, # 完成任务的奖励阈值
)
register(
    id="Jump-v1", # 环境id
    entry_point="envs.env_jump_long_term_input:JumpEnv", # 环境类入口
    max_episode_steps=1000, # 一个episode的最大步数
    reward_threshold=6000.0, # 完成任务的奖励阈值
)

register(
    id="Walk-v0", # 环境id
    entry_point="envs.env_walk_short_term:WalkEnv", # 环境类入口
    max_episode_steps=2000, # 一个episode的最大步数
    reward_threshold=6000.0, # 完成任务的奖励阈值
)

register(
    id="Walk-v1", # 环境id
    entry_point="envs.env_walk_long_term:WalkEnv", # 环境类入口
    max_episode_steps=2000, # 一个episode的最大步数
    reward_threshold=6000.0, # 完成任务的奖励阈值
)

register(
    id="Stop-v0", # 环境id
    entry_point="envs.stop:StopEnv", # 环境类入口
    max_episode_steps=2000, # 一个episode的最大步数
    reward_threshold=6000.0, # 完成任务的奖励阈值
)

register(
    id="Nav-v0", # 环境id
    entry_point="envs.env_nav_short_term_baseline:NavEnv", # 环境类入口
    max_episode_steps=1000, # 一个episode的最大步数
    reward_threshold=6000.0, # 完成任务的奖励阈值
)
