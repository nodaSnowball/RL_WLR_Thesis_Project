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
    "trackbodyid": 0,
    "distance": 25.0,
    "lookat": np.array((0.0, 0.0, 2.0)),
    "elevation": -90.0,
}


# 定义一个仿真环境
class NavEnv(MujocoEnv, utils.EzPickle):

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
        xml_file=os.path.join(os.path.dirname(__file__), 'asset', "nav_model.xml"),
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        ctrl_cost_weight=0.0001,
        max_step = 500,
        healthy_reward=.1,
        healthy_z_range=0.2,
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
        self.obstacle_pos = np.array([[-6,6,0],[0,6,0],[6,6,0],
                                      [-6,0,0],[0,0,0],[6,0,0],
                                      [-6,-6,0],[0,-6,0],[6,-6,0]])
        self.target_pos = np.array([[x,y,0] for y in range(9,-10,-3) for x in range(-9,10,3)])
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
        obs_size = 1 + self.data.qpos.size + self.data.qvel.size + self.data.sensordata.size
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        # plt.plot(self.tx,self.tz)
        # plt.show()


        # self.end_x = self.sim.model.get_joint_qpos_addr('end')

    @property # 计算健康奖励
    def healthy_reward(self):
        if self.c_step<500:
            return float(self.is_healthy) * self._healthy_reward
        else:
            return 0

    # 计算控制成本
    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.abs([action[i] for i in range(6) if i!=2 and i!=5]))
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
        target_pos = np.array([2,0,0.3])
        # robot pos
        rob_pos = self.obs[1:4]
        distance = target_pos - rob_pos
        distance = sum(distance**2)**0.5
        return distance
    
    # 执行仿真中的一步
    def step(self, action):
        info = {}
        reward = 0
        self.c_step+=1
        self._get_obs()
        z_prev = self.obs[3]
        d_before = self.get_xydistance()
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
        z_curr = self.obs[3]
        v_z = self.obs[14]
        d_after = self.get_xydistance()
        
        # ctrl_cost = self.control_cost(action)  # 控制损失
        unstable_punishment = 0
        # if the robot is moving forward
        approch_distance = d_before-d_after
        # print(approch_distance)
        forward_check = 0
        
        # health reward: maintain standing pose

        # control cost: acceleration of motors
        # cost = ctrl_cost
        # current_qpos = np.concatenate((self.obs[8:10], self.obs[10:12]), axis=0)
        # act_qpos = np.concatenate((action[:2],action[2:]),axis=0)/50
        # control_cost = sum((current_qpos-act_qpos)**2)

        # insymmetric_punishment = sum(((action[:3]-action[3:])/np.array([30,30,160]))**2)
        insymmetric_punishment = sum(((action[:2]-action[2:])/np.array([30,30]))**2)    # 10, 20
        
        approaching_reward = (d_before-d_after) # if forward_check>0 else 0 # -5*abs(d_before-d_after)

        robot_pos = self.obs[1:4].copy()

        jump_reward = max(0,v_z)    # 10
        # reward
        #sac
        # reward = 0*trajector_align + 20*approaching_reward + self.healthy_reward - 10*insymmetric_punishment + 0.5*jump_reward
        #ppo
        reward = 20*approaching_reward + self.healthy_reward - 10*insymmetric_punishment + .5*jump_reward
        
        done = self.done
        if self.obs[1]>-1 and self.obs[1]<1 and self.obs[3]<(self.height+self._healthy_z_range):
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
        
        # 判断是否到达终点， enhance
        # reward = 0
        # if not done and robot_pos[0]>1:
        #     reward += .3
        # if d_after<0.5:
        #     reward += .3
        #     jump_reward = 0
        #     approaching_reward = 0
        # if self.c_step>=self.max_step:
        #     done = True
        #     if d_after<0.5:
        #         reward += 100
        #         info.update({'is_success':True, 'termination':'success'})
        #     else:
        #         reward -= 10
        #         info.update({'is_success':False, 'termination':'not within target boundary'})
        # elif done == True:
        #     reward -= 10
        #     info.update({'is_success':False, 'termination':'fall or collision'})
        # # print(reward)
        # reward += 0*trajector_align + 20*approaching_reward + self.healthy_reward - 10*insymmetric_punishment + 0.5*jump_reward  # - cost
        

        return obs, reward, done, False, info

    # 获取当前状态的观察值
    def _get_obs(self):
        '''
        STATE INDEX (no preprocess)
        sensor data from base imu
        sensor: length = 24
            IDX    |DATA
            0      |target h
            1-3    |base pos x,y,z
            4-7    |base ori quat
            8-9    |left hip/knee q pos
            10-11  |right hip/knee q pos
            12-14  |base linear velocity
            15-17  |base angular velocity
            18-20  |left hip/knee/wheel q velocity
            21-23  |right hip/knee/wheel q velocity
            24-79  |lidar readings
        '''
        # sensor data
        bev = self.render() # (240,240,3)
        bev = bev.astype(np.float32)/255
        qpos = self.data.qpos.flatten()
        qvel = self.data.qvel.flatten()
        lidar = self.data.sensordata
        self.obs = np.concatenate([np.array(self.height).reshape(-1), qpos, qvel, lidar])
        # np.delete(qpos,[-1,-4]) if delete wheel motor

        obs = self.obs.copy()
        obs[0] *= 5
        obs[1:3] /= 10
        obs[8:12] *= 3
        for i in range(-56,0):
            obs[i] /= 10 if obs[i]>0 else 1
        # obs[20] /= 100
        # obs[23] /= 100
        
        return obs

    # 重置模型
    def reset_model(self):
        
        self.c_step = 0
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.current_pos_idx = random.randint(0,24)
        self.target_pos_idx = random.randint(0,24)
        while self.current_pos_idx == self.target_pos_idx:
            self.target_pos_idx = random.randint(0,24)

        # reset target
        self.height = 0.05 if self.fixed_height else random.random()*0.05+0.05
        # mpos = np.array([0,0,self.height-0.2])
        # self.data.mocap_pos = mpos
        # reset inital robot position
        # qpos[0] = -1.5
        # qpos[1] = 0
        # qpos[3:7] = Rotation.from_euler('zyx',[0, 0, 0]).as_quat()

        # self.mujoco_renderer
            
        # self.set_state(qpos, qvel)
        self._get_obs()
        return self.obs


if __name__ == '__main__':
    
    env = gym.make('Nav-v0', render_mode='rgb_array', camera_name='top', fixed_height=0, width=240, height=240)
    state = env.reset()
    x = env.render()
    plt.imshow(x)
    for _ in range(100):
        env.step(np.zeros([4,]))
        x = env.render()
        plt.imshow(x)
    state = env.reset()
    env.step(np.zeros([4,]))
    x = env.render()
    plt.imshow(x)
    print(state)

    