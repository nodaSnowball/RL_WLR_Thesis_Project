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

# xml_file='/scratch/zl4930/wlr/envs/asset/jump_model.xml',

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
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        ctrl_cost_weight=0.0001,
        max_step = 500,
        healthy_reward=.1,
        healthy_z_range=0.21,
        reset_noise_scale=0.1,
        fixed_height = True,
        **kwargs
    ):
        utils.EzPickle.__init__(**locals())

        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self.c_step = 0
        self.obs = None
        self.max_step = max_step
        self.fixed_height = fixed_height

        MujocoEnv.__init__(self, xml_file, 5, observation_space=None, default_camera_config=DEFAULT_CAMERA_CONFIG, **kwargs)
        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }
        obs_size = 2 + self.data.qpos.size + self.data.qvel.size + 4 + self.data.sensordata.size
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )
        self.tx = np.zeros([1000])
        self.ty = np.zeros([1000])
        self.tz = np.zeros([1000])

        self.tx[:100].fill(-1.5)
        self.tz[:100].fill(0.3)
        for i in range(100,172):
            self.tx[i] = -1.5 + (i-100)*0.021
            self.tz[i] = 0.3 + (i-100)*0.03639 - 0.5*9.806*(i/100-1)**2
        self.tx[172:].fill(self.tx[171])
        self.tz[172:].fill(self.tz[171])
        # plt.plot(self.tx,self.tz)
        # plt.show()


        # self.end_x = self.sim.model.get_joint_qpos_addr('end')

    @property # 计算健康奖励
    def healthy_reward(self):
        return float(self.is_healthy) * self._healthy_reward

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
        # target_pos = np.array([0,0,self.height+0.3])
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
        d_action = action-self.action
        self.action_prev = self.action
        self.action = action.copy()
        d_before = self.get_xydistance()
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        robot_pos = self.obs[2:5].copy()
        d_after = self.get_xydistance()
        
        # control cost
        qacc = self.data.qacc[-4:]
        acc_cost = sum((.001*qacc)**2)
        ctrl_cost = min(acc_cost,10)

        # insymmetric_punishment = sum(((action[:3]-action[3:])/np.array([30,30,160]))**2)
        insymmetric_punishment = sum((abs(action[:2]-action[2:])/np.array([8,8]))**2)    # 10, 20
        
        approaching_reward = (d_before-d_after) # if forward_check>0 else 0 # -5*abs(d_before-d_after)
        target_pos = np.array([self.tx[self.c_step-1], self.ty[self.c_step-1], self.tz[self.c_step-1]])

        # trajector_align = - sum((target_pos-robot_pos)**2)  # 10

        # jumping reward
        v_z = self.obs[15]
        jump_reward = max(0, v_z)    # 10
        jump_reward = min(1, jump_reward)
        # jump_reward = self.is_jump_with_both_feet()*jump_reward

        # total reward
        reward = 20*approaching_reward + self.healthy_reward - .5*insymmetric_punishment + 1*jump_reward - 0.1*ctrl_cost
        
        done = self.done
        if robot_pos[0]>-0.5 and robot_pos[0]<0.5 and robot_pos[2]<(self.height+self._healthy_z_range):
            done = True
        if robot_pos[2]>0.6:
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
            if self.c_step < 200:
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
        qvel = self.data.qvel.flatten()
        lidar = self.data.sensordata
        self.obs = np.concatenate([self.target, qpos, qvel, self.action, lidar])
        # np.delete(qpos,[-1,-4]) if delete wheel motor

        obs = self.obs.copy()
        obs[:4] /= 10
        obs[9:13] *= 3
        obs[16:23] /= 10
        obs[23:27] /= 10
        obs[-56:] /= 10
        # obs[20] /= 100
        # obs[23] /= 100
        
        return obs

    # 重置模型
    def reset_model(self):
        self.action = np.zeros((4,))
        self.action_prev = np.zeros((4,))
        self.target = np.array([1.5, 0])
        self.c_step = 0
        qpos = self.init_qpos
        qvel = self.init_qvel
        # reset target
        self.height = 0.05 if self.fixed_height else random.random()*0.06+0.01
        mpos = np.array([0,0,self.height-0.2])
        self.data.mocap_pos = mpos
        # reset inital robot position
        # qpos[0] = -1.5
        # qpos[1] = 0
        # qpos[3:7] = Rotation.from_euler('zyx',[0, 0, 0]).as_quat()
            
        self.set_state(qpos, qvel)
        self._get_obs()
        return self.obs


if __name__ == '__main__':
    
    env = gym.make('Jump-v0')
    state = env.reset()
    print(state)

    