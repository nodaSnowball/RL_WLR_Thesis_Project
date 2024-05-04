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
from collections import deque
import math
from scipy.spatial.transform import Rotation
import envs.register

# xml_file='/scratch/zl4930/wlr/envs/asset/jump_model.xml',

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 25.0,
    "lookat": np.array((0.0, 0.0, 2.0)),
    "elevation": -90.0,
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
        max_step = 500,
        healthy_reward=.1,
        healthy_z_range=0.21,
        fixed_height = True,
        **kwargs
    ):
        utils.EzPickle.__init__(**locals())

        self._healthy_reward = healthy_reward
        self._healthy_z_range = healthy_z_range
        self.c_step = 0
        self.obs = None
        self.action = None
        self.curr_obs = None
        self.slist = None
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
        self.obs_size = 2 + self.data.qpos.size + self.data.qvel.size + self.data.sensordata.size + 4
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(100*self.obs_size,), dtype=np.float32
        )
        # self.tx = np.zeros([1000])
        # self.ty = np.zeros([1000])
        # self.tz = np.zeros([1000])

        # self.tx[:100].fill(-1.5)
        # self.tz[:100].fill(0.3)
        # for i in range(100,172):
        #     self.tx[i] = -1.5 + (i-100)*0.021
        #     self.tz[i] = 0.3 + (i-100)*0.03639 - 0.5*9.806*(i/100-1)**2
        # self.tx[172:].fill(self.tx[171])
        # self.tz[172:].fill(self.tz[171])
        # plt.plot(self.tx,self.tz)
        # plt.show()


        # self.end_x = self.sim.model.get_joint_qpos_addr('end')

    @property # 计算健康奖励
    def healthy_reward(self):
        if self.c_step<500:
            return float(self.is_healthy) * self._healthy_reward
        else:
            return 0

    @property  # 是否倾倒
    def is_healthy(self):
        min_z = self._healthy_z_range
        is_healthy = (self.get_body_com("base_link")[2] > min_z)
        return is_healthy

    @property
    def done(self):
        done = not self.is_healthy
        return done

    # 碰到大腿以上
    def bump_base(self):
        # for i in range(self.sim.data.ncon):
        #     contact = self.sim.data.contact[i]
        #     geom1 = self.sim.model.geom_id2name(contact.geom1)
        #     geom2 = self.sim.model.geom_id2name(contact.geom2)
        #     if (geom1 in ['base1', 'base2', 'base3', 'base4', 'left_thigh1', 'left_thigh2', 'left_thigh3',
        #                   'right_thigh1', 'right_thigh2', 'right_thigh3']) or (
        #             geom2 in ['base1', 'base2', 'base3', 'base4', 'left_thigh1', 'left_thigh2', 'left_thigh3',
        #                       'right_thigh1', 'right_thigh2', 'right_thigh3']):
        #         return True
        return False
    
    def get_xydistance(self)->float:
        # target pos
        # target_pos = np.array([0,0,self.height+0.3])
        target_pos = self.target
        # robot pos
        rob_pos = self.curr_obs[2:4]
        distance = target_pos - rob_pos
        distance = sum(distance**2)**0.5
        return distance
    
    def is_both_feet_off_ground(self) -> bool :
        body_pos = self.data.xpos.copy()
        if body_pos[-4,-1]>0.1 and body_pos[-1,-1]>0.1:
            return 0.2
        if body_pos[-4,-1]>0.07 and body_pos[-1,-1]>0.07:
            return 0.1
        return 0
    
    # 执行仿真中的一步
    def step(self, action):
        s = False
        self.action_prev = self.action
        self.action = action.copy()
        d_action = action-self.action
        info = {}
        reward = 0
        self.c_step+=1
        d_before = self.get_xydistance()
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        robot_pos = self.curr_obs[2:5].copy()
        d_after = self.get_xydistance()

        # approching reward
        approaching_reward = (d_before-d_after)
        
        # control cost
        qacc = self.data.qacc[-4:]
        acc_cost = sum((.001*qacc)**2)
        ctrl_cost = min(acc_cost,10)

        # insymmetric_punishment = sum(((action[:3]-action[3:])/np.array([30,30,160]))**2)
        insymmetric_punishment = sum((abs(action[:2]-action[2:])/np.array([8,8]))**2)    # 10, 20
        
        # target_pos = np.array([self.tx[self.c_step-1], self.ty[self.c_step-1], self.tz[self.c_step-1]])
        
        # trajector_align = - sum((target_pos-robot_pos)**2)  # 10

        # jumping reward
        v_z = self.curr_obs[15]
        jump_reward = max(0,v_z)    # 10
        jump_reward = min(1, jump_reward)

        # total reward
        reward = 20*approaching_reward + self.healthy_reward - .5*insymmetric_punishment + 1*jump_reward - 0.1*ctrl_cost
        
        # check collision on platform
        done = self.done
        if robot_pos[0]>-0.5 and robot_pos[0]<0.5 and robot_pos[2]<(self.height+self._healthy_z_range):
            done = True
        if robot_pos[2]>0.6:
            done = True

        # 判断是否到达终点, reset at target 
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
        STATE INDEX 
        observation at a time step
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
        # qpos = np.delete(qpos,[-1,-4]) # if delete wheel motor
        qvel = self.data.qvel.flatten()
        lidar = self.data.sensordata.copy()
        self.curr_obs = np.concatenate([self.target, qpos, qvel, self.action, lidar]).astype(np.float32)

        obs = self.curr_obs.copy()
        obs[:4] /= 10
        obs[9:13] *= 3
        obs[16:23] /= 10
        obs[23:-56] /= 10
        obs[-56:] /= 10

        if self.c_step==0:
            del self.obs
            self.obs = np.repeat([obs], 100, axis=0)
        else:
            self.obs = np.concatenate([[obs], self.obs[:-1,:]], axis=0)

        return self.obs.flatten()

    # 重置模型
    def reset_model(self):
        self.c_step = 0
        self.slist = deque([False for _ in range(50)], maxlen=50)
        self.action = np.zeros(4,)
        self.action_prev = np.zeros(4,)
        qpos = self.init_qpos
        qvel = self.init_qvel
        
        # reset target
        self.height = 0.05 if self.fixed_height else random.random()*0.06+0.01
        self.target = np.array([1.5, 0])
        mpos = np.array([0,0,self.height-0.2])
        self.data.mocap_pos = mpos

        # reset inital robot position
        qpos[0:3] = np.array([-1.5,0,.35])
        # qpos[3:7] = Rotation.from_euler('zyx',[0, 0, 0]).as_quat()
            
        self.set_state(qpos, qvel)
        obs = self._get_obs()
        return obs


if __name__ == '__main__':
    
    env = gym.make('Nav-v0', render_mode='rgb', fixed_height=0, camera_name='top')
    state = env.reset()
    print(state)

    