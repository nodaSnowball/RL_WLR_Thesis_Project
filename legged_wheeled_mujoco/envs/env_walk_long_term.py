import numpy as np
from typing import Dict, Tuple, Union
import gymnasium as gym
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import random 
import math
from time import sleep
from scipy.spatial.transform import Rotation
import envs.register

# xml_file='/scratch/zl4930/wlr/envs/asset/walk_model.xml',

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 2.0)),
    "elevation": -20.0,
}

# 定义一个仿真环境
class WalkEnv(MujocoEnv, utils.EzPickle):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    # 初始化环境参数
    def __init__(
        self,
        xml_file=os.path.join(os.path.dirname(__file__), 'asset', "walk_model.xml"),
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        max_step = 500,
        healthy_reward=2,
        healthy_z_range=0.25,
        random_reset=0,
        **kwargs
    ):
        utils.EzPickle.__init__(**locals())

        self._healthy_reward = healthy_reward
        self._healthy_z_range = healthy_z_range
        self.random_reset = random_reset
        self.c_step = 0
        self.max_step = max_step
        self.obs = None
        self.target_pos = np.array([[x,y] for y in range(6,-7,-3) for x in range(-6,7,3)])

        MujocoEnv.__init__(self, xml_file, 5, observation_space=None, default_camera_config=DEFAULT_CAMERA_CONFIG, **kwargs)
        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }
        # self.end_x = self.sim.model.get_joint_qpos_addr('end')

        obs_size = self.data.qpos.size-2 + self.data.qvel.size-2 + 6 + self.data.sensordata.size
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(100*obs_size,), dtype=np.float32
        )

    @property # 计算健康奖励
    def healthy_reward(self):
        return float(self.is_healthy) * self._healthy_reward

    # 计算控制成本
    def control_cost(self, action):
        control_cost = 0    # self._ctrl_cost_weight * np.sum(np.abs([action[i] for i in range(6) if i!=2 and i!=5]))
        return control_cost

    @property  # 是否倾倒
    def is_healthy(self):
        min_z = self._healthy_z_range
        is_healthy = (self.get_body_com("base_link")[2] > min_z)
        return is_healthy

    @property
    def done(self):
        done = not self.is_healthy
        return done
    
    def get_xydistance(self)->float:
        # target pos
        target_pos = self.target
        # robot pos
        rob_pos = self.curr_obs[2:4]
        xy_distance = target_pos - rob_pos
        distance = (xy_distance[0]**2 + xy_distance[1]**2)**0.5
        return distance
    
    def is_both_feet_on_ground(self) -> bool :
        body_pos = self.data.xpos.copy()
        if body_pos[-4,-1]<0.1 and body_pos[-1,-1]<0.1:
            return True
        return False
    
    # 执行仿真中的一步
    def step(self, action):
        info = {}
        reward = 0
        self.c_step+=1
        self.action_prev =self.action
        self.action = action.copy()
        d_action = self.action-self.action_prev
        d_before = self.get_xydistance()
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        d_after = self.get_xydistance()
        
        qacc = self.data.qacc[-6:]
        acc_cost = sum((.0005*qacc)**2)
        ctrl_cost = min(acc_cost,1)

        punishment = -0.3 if self.is_both_feet_on_ground() else 0
        
        approaching_reward = d_before - d_after

        done = self.done
        reward =  approaching_reward + self.healthy_reward - 1*punishment - 1*ctrl_cost

        # 判断是否到达终点
        if done == False:
            if d_after < 1:
                # success
                done = True
                reward = 100
                info.update({'is_success':True})
            else:
                # max step
                if self.c_step==2000:
                    info.update({'is_success':False})
        else:
            # Fall
            reward -= 10
            info.update({'is_success':False})
        
        # print(reward)
        return obs, reward, done, False, info

    # 获取当前状态的观察值
    def _get_obs(self):
        '''
        STATE INDEX (no preprocess)
        sensor data from base imu
        sensor: length = 87
            IDX    |DATA
            0-1    |target x,y
            2-4    |base pos x,y,z
            5-8    |base ori quat
            9-10   |left hip/knee qpos in radius
            11-12  |right hip/knee qpos in radius
            13-15  |base linear velocity
            16-18  |base angular velocity
            19-21  |left hip/knee/wheel qvel
            22-24  |right hip/knee/wheel qvel
            25-30  |actions
            31-86  |lidar readings
        '''
        # sensor data
        qpos = self.data.qpos.flatten()
        qpos = np.delete(qpos,[-1,-4])
        qvel = self.data.qvel.flatten()
        qvel = np.delete(qvel,[0,1])
        lidar = self.data.sensordata
        self.curr_obs = np.concatenate([qpos, qvel, self.action, lidar]).astype(np.float32)
        obs = self.curr_obs.copy()
        
        # normalization
        obs[:4] /= 10 
        obs[9:13] *= 3
        obs[16:25] /= 10
        obs[25:31] /= 10
        obs[31:] /= 10 

        if self.c_step==0:
            self.obs = np.repeat([obs], 100, axis=0)
        else:
            self.obs = np.concatenate([[obs], self.obs[:-1,:]], axis=0)
        
        return self.obs.flatten()

    # 重置模型
    def reset_model(self):
        self.c_step = 0
        self.action_prev = np.zeros(6,)
        self.action = np.zeros(6,)
        qpos = self.init_qpos
        qvel = self.init_qvel
        
        # reset obstacles
        # mocap_quat = self.sim.data.mocap_quat.flat.copy()
        # mpos = np.zeros((3,))
        # mocap_quat = Rotation.from_euler('zyx',[0, 0, 2*np.pi*random.random()]).as_quat()
        # self.sim.data.set_mocap_quat('mocap', mocap_quat)
        # self.sim.data.set_mocap_pos('mocap', mpos)
        
        if self.random_reset:
            # reset target
            target_idx = random.randint(0,24)
            self.target = self.pos_dict[target_idx]
            qpos[0:2] = self.target
            # reset inital robot position
            robot_idx = random.randint(0,24)
            while robot_idx==target_idx:
                robot_idx = random.randint(0,24)
            qpos[2:4] = self.pos_dict[robot_idx]
            qpos[5:9] = Rotation.from_euler('zyx',[0, 0, random.randint(0,3)*np.pi/2]).as_quat()
                
            self.set_state(qpos, qvel)
            obs = self._get_obs()
            return obs
        
        self.target = np.array([3,0])
        qpos[0:2] = self.target
        self.set_state(qpos, qvel)
        obs = self._get_obs()
        return obs


if __name__ == '__main__':
    env = gym.make('Walk-v0')
    state = env.reset()
    state = env.step(np.zeros((6,)))

    print(state)

    