import numpy as np
import gym
from gym import utils
from gym.envs.mujoco import mujoco_env
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import random 
import math
from time import sleep
from scipy.spatial.transform import Rotation
# import register

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}

# 定义一个仿真环境
class WalkEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    # 初始化环境参数
    def __init__(
        self,
        xml_file=os.path.join(os.path.dirname(__file__), 'asset', "walk_model.xml"),
        ctrl_cost_weight=0.0001,
        healthy_reward=2,
        healthy_z_range=0.05,
        reset_noise_scale=0.1,
    ):
        utils.EzPickle.__init__(**locals())

        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self.c_step = 0
        self.re = 0.1
        self.obs = None

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)
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
        is_healthy = (self.get_body_com("base_link")[2] > min_z) and (not self.bump_base()) 
        return is_healthy

    @property
    def done(self):
        done = not self.is_healthy
        return done

    # 碰到大腿以上
    def bump_base(self):
        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            geom1 = self.sim.model.geom_id2name(contact.geom1)
            geom2 = self.sim.model.geom_id2name(contact.geom2)
            if (geom1 in ['base1', 'base2', 'base3', 'base4', 'left_thigh1', 'left_thigh2', 'left_thigh3',
                          'right_thigh1', 'right_thigh2', 'right_thigh3']) or (
                    geom2 in ['base1', 'base2', 'base3', 'base4', 'left_thigh1', 'left_thigh2', 'left_thigh3',
                              'right_thigh1', 'right_thigh2', 'right_thigh3']):
                return True
        return False
    
    def get_xydistance(self)->float:
        # target pos
        target_pos = [self.obs[0], self.obs[1]]
        # robot pos
        rob_pos = self.obs[2:4]
        xy_distance = target_pos - rob_pos
        distance = (xy_distance[0]**2 + xy_distance[1]**2)**0.5
        return distance
    
    # 执行仿真中的一步
    def step(self, action):
        info = {}
        self.c_step+=1
        self._get_obs()
        d_before = self.get_xydistance()
        # t = action[2]
        # r = action[5]
        # action[2] = (t-r*.5)/1.5
        # action[5] = (t+r*.5)/1.5
        # action[5] = t
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()
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
        punishment = -10*(self.obs[4]-0.4)**2      # height control
        
        approaching_reward = -0.01 * d_after**2

        done = self.done
        reward =  self.re * (approaching_reward + self.healthy_reward - punishment) # - cost

        # 判断是否到达终点
        if done == False:
            if d_after < 1:
                # success
                done = True
                reward = self.re * 1000
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
        return obs, reward, done, info

    # 获取当前状态的观察值
    def _get_obs(self):
        '''
        STATE INDEX (no preprocess)
        sensor data from base imu
        sensor: length = 24
            IDX    |DATA
            0-1    |target x,y
            2-4    |base pos x,y,z
            5-8    |base ori quat
            9/11   |left/right hip joint pos in radius
            10/12  |left/right knee joint pos in radius
            13/14  |left/right wheel velocity
            15-17  |base local linear velocity
            18-20  |base local linear acceleration
            21-23  |base local angular velocity
        '''
        # sensor data
        self.obs = np.array(self.sim.data.sensordata) 
        qpos = np.array(self.sim.data.qpos.flat.copy())
        qvel = np.array(self.sim.data.qvel.flat.copy())
        self.obs[:8] = qpos[:8]
        obs = self.obs.copy()
        
        # normalization
        obs[:2] /= 10      # target        xy
        obs[2:4] /= 10     # robot         xy
        # obs[5:9]           robot         z, quat ori
        # obs[9:13]          hip, keen     qpos
        obs[13:15] /= 30   # wheel         qvel
        
        return obs

    # 重置模型
    def reset_model(self):
        random_pos = 1
        self.c_step = 0
        qpos = self.init_qpos
        qvel = self.init_qvel
        
        # reset obstacles
        # mocap_quat = self.sim.data.mocap_quat.flat.copy()
        mpos = np.zeros((3,))
        mocap_quat = Rotation.from_euler('zyx',[0, 0, 2*np.pi*random.random()]).as_quat()
        self.sim.data.set_mocap_quat('mocap', mocap_quat)
        self.sim.data.set_mocap_pos('mocap', mpos)
        
        # reset target
        qpos[0] = 20*random.random()-10
        qpos[1] = 20*random.random()-10
        # reset inital robot position
        qpos[2] = 20*random.random()-10 if random_pos else 0
        qpos[3] = 20*random.random()-10 if random_pos else 0
        qpos[4] = 0.45
        qpos[5:9] = Rotation.from_euler('zyx',[0, 0, 2*np.pi*random.random()]).as_quat()
        while ((qpos[0]-qpos[2])**2+(qpos[1]-qpos[3])**2)<1:
            qpos[0] = 20*random.random()-10
            qpos[1] = 20*random.random()-10
            
        self.set_state(qpos, qvel)
        self._get_obs()
        # self.render()
        return self.obs

    # 可视化查看器
    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)


if __name__ == '__main__':
    
    env = gym.make('Walk-v0')
    state = env.reset()
    i = 0
    while 1:
        i+=1
        env.render()
        env.step(np.zeros((6,)))
        if i %1000 == 0:
            env.reset()

        

    print(state)

    