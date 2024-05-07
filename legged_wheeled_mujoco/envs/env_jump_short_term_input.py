import numpy as np
from typing import Dict, Tuple, Union
import gymnasium as gym
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import matplotlib.pylab as plt
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
import random 
import math
from scipy.spatial.transform import Rotation
import envs.register

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 2.0)),
    "elevation": -20.0,
}

# 定义一个仿真环境
class JumpEnv(MujocoEnv, utils.EzPickle):

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
        xml_file=os.path.join(os.path.dirname(__file__), 'asset', "jump_model.xml"),
        # xml_file='/scratch/zl4930/wlr/envs/asset/jump_model.xml',
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        max_step = 500,
        healthy_reward=.1,
        healthy_z_range=0.20,
        random_reset=1,
        **kwargs
    ):
        utils.EzPickle.__init__(**locals())

        self._healthy_reward = healthy_reward
        self._healthy_z_range = healthy_z_range
        self.random_reset = random_reset
        self.c_step = 0
        self.obs = None
        self.max_step = max_step
        self.pos_dict = np.array([[x,y] for y in range(6,-7,-3) for x in range(-6,7,3)])


        MujocoEnv.__init__(self, xml_file, 5, observation_space=None, default_camera_config=DEFAULT_CAMERA_CONFIG, **kwargs)
        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }
        obs_size = self.data.qpos.size-2 + self.data.qvel.size-2 + 6 + self.data.sensordata.size
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

    @property # 计算健康奖励
    def healthy_reward(self):
        return float(self.is_healthy) * self._healthy_reward
    
    def bump_base(self):
        for i in range(self.data.ncon):
            geom1 = self.data.contact.geom1
            geom2 = self.data.contact.geom2
            for id in np.concatenate([geom1, geom2],axis=0):
                if self.model.geom_group[id] == 10:
                    return True 

    @property  # 是否倾倒
    def is_healthy(self):
        min_z = self._healthy_z_range
        is_healthy = (self.get_body_com("base_link")[2] > min_z) and (not self.bump_base())
        return is_healthy

    @property
    def done(self):
        done = not self.is_healthy
        return done
    
    def get_xydistance(self)->float:
        # target pos
        target_pos = self.target
        # robot pos
        rob_pos = self.obs[2:4]
        distance = target_pos - rob_pos
        distance = sum(distance**2)**0.5
        return distance
    
    def is_jump_with_both_feet(self) -> int :
        body_pos = self.data.xpos.copy()
        if body_pos[-4,-1]<0.08 and body_pos[-1,-1]<0.08:
            return 1
        if body_pos[-4,-1]>0.08 and body_pos[-1,-1]>0.08:
            return 1
        return 0
    
    # 执行仿真中的一步
    def step(self, action):
        info = {}
        reward = 0
        self.c_step+=1
        self.action_prev = self.action
        self.action = action.copy()
        d_action = self.action -self.action_prev
        d_before = self.get_xydistance()
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        robot_pos = self.obs[2:5]
        d_after = self.get_xydistance()
        
        # control cost
        qacc = self.data.qacc[-6:]
        acc_cost = sum((.001*qacc)**2)
        ctrl_cost = min(acc_cost,10)

        # insymmetric_punishment = sum(((action[:3]-action[3:])/np.array([30,30,160]))**2)
        # insymmetric_punishment = sum((abs(action[:2]-action[2:])/np.array([8,8]))**2)    # 10, 20
        
        # approcahing reward
        approaching_reward = min(d_before-d_after, .025)

        # walk pos reference
        p_jump = abs(action[0]+action[1]) + abs(action[3]+action[4]) # + abs(action[0]+action[3])
        p_jump = min(p_jump,10)
        p_roll = min(abs(action[2])+abs(action[5]), 5)
        punishment = p_jump + 0*p_roll

        # jumping reward
        v_z = self.obs[15]
        jump_reward = max(0, v_z)    # 10
        jump_reward = min(1, jump_reward)

        # total reward
        reward = 20*approaching_reward + self.healthy_reward - .02*punishment + 1*jump_reward - .03*ctrl_cost
        
        done = self.done
        # if robot_pos[0]>-0.5 and robot_pos[0]<0.5 and robot_pos[2]<(self._healthy_z_range):
        #     done = True
        if robot_pos[2]>0.8:
            done = True

        # 判断是否到达终点, pretrain 
        if done == False:
            if d_after < 0.5:
                done = True
                reward = 100
                info.update({'is_success':True, 'termination':'success'})
            elif self.c_step>=self.max_step:
                done = True
                info.update({'is_success':False, 'termination':'exceed max step'})
        else:
            reward -= 10
            info.update({'is_success':False, 'termination':'fall or collision'})

        return obs, reward, done, False, info

    # 获取当前状态的观察值
    def _get_obs(self):
        '''
        STATE INDEX (no preprocess)
        sensor data from base imu
        sensor: length = 87
            IDX    |DATA
            0-1    |target pos
            2-4    |base pos x,y,z
            5-8    |base ori quat
            9-10   |left hip/knee qpos
            11-12  |right hip/knee qpos
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
        self.obs = np.concatenate([qpos, qvel, self.action, lidar]).astype(np.float32)
        # np.delete(qpos,[-1,-4]) if delete wheel motor

        obs = self.obs.copy()
        obs[:4] /= 10
        obs[9:13] *= 3
        obs[16:25] /= 10
        obs[25:31] /= 10
        obs[24] /= 10
        obs[21] /= 10
        obs[-56:] /= 10
        
        return obs

    # 重置模型
    def reset_model(self):
        self.c_step = 0
        self.action = np.zeros((6,))
        self.action_prev = np.zeros((6,))
        qpos = self.init_qpos
        qvel = self.init_qvel

        # reset height
        for i in range(4):
            h = random.random()*0.06+0.01
            self.data.mocap_pos[i,-1] = h - .05

        # randomly reset target and robot
        if self.random_reset:
            # reset obstacle orientation
            for i in range(4):
                quat = Rotation.from_euler('zyx',[0, 0, random.randint(0,1)*np.pi/2]).as_quat()
                self.data.mocap_quat[i] = quat
            # reset target
            target_idx = random.randint(0,24)
            self.target = self.pos_dict[target_idx]
            qpos[0:2] = self.target
            # reset inital robot position
            action_idx = random.randint(0,3)
            while True:
                if action_idx==0:   # up
                    if target_idx+5<=24:
                        robot_idx = target_idx+5
                        quat = Rotation.from_euler('zyx',[0, 0, 1*np.pi/2]).as_quat()
                        break
                    action_idx+=1
                if action_idx==1:   # down
                    if target_idx-5>=0:
                        robot_idx = target_idx-5
                        quat = Rotation.from_euler('zyx',[0, 0, -1*np.pi/2]).as_quat()
                        break
                    action_idx+=1
                if action_idx==2:   # left
                    if not (target_idx+1)%5==0:
                        robot_idx = target_idx+1
                        quat = Rotation.from_euler('zyx',[0, 0, 0*np.pi/2]).as_quat()
                        break
                    action_idx+=1
                if action_idx==3:   # right
                    if not target_idx%5==0:
                        robot_idx = target_idx-1
                        quat = Rotation.from_euler('zyx',[0, 0, 2*np.pi/2]).as_quat()
                        break
                    action_idx=0
            qpos[2:4] = self.pos_dict[robot_idx]
            qpos[5:9] = quat
                
            self.set_state(qpos, qvel)
            obs = self._get_obs()
            return obs
            
        self.target = np.array([3,3])
        qpos[0:2] = self.target
        qpos[2:4] = np.array([0,3])
        self.set_state(qpos, qvel)
        obs = self._get_obs()
        return obs


if __name__ == '__main__':
    
    env = gym.make('Jump-v0')
    state = env.reset()
    print(state)

    