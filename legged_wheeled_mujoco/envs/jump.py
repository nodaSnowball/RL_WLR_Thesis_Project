import numpy as np
import gym
from gym import utils
from gym.envs.mujoco import mujoco_env
import os
import random 
import math
from scipy.spatial.transform import Rotation
#import register

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}

# 定义一个仿真环境
class JumpEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    # 初始化环境参数
    def __init__(
        self,
        xml_file=os.path.join(os.path.dirname(__file__), 'asset', "Legged_wheel_jump.xml"),
        ctrl_cost_weight=0.0001,
        healthy_reward=1,
        healthy_z_range=0.05,
        reset_noise_scale=0.1,
    ):
        utils.EzPickle.__init__(**locals())

        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self.c_step = 0
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
        rob_pos = self.obs[3:5]
        xy_distance = target_pos - rob_pos
        distance = (xy_distance[0]**2 + xy_distance[1]**2)**0.5
        return distance
    
    # 执行仿真中的一步
    def step(self, action):
        self.c_step+=1
        self._get_obs()
        d_before = self.get_xydistance()
        self.do_simulation(action, self.frame_skip)
        self._get_obs()
        d_after = self.get_xydistance()
        
        # ctrl_cost = self.control_cost(action)  # 控制损失
        unstable_punishment = 0
        # if the robot is moving forward
        approch_distance = d_before-d_after
        forward_check = 0
        
        # health reward: maintain standing pose
        # control cost: acceleration of motors
        # cost = ctrl_cost
        punishment = 0      # 0.5*abs(self.obs[-2])
        
        approaching_reward = 100*approch_distance # if forward_check>0 else 0 # -5*abs(d_before-d_after)

        done = self.done
        reward =  approaching_reward + self.healthy_reward - punishment # - cost

        # 判断是否到达终点
        if done == False:
            if d_after < 0.3:
                done = True
                reward = 1000
        # else:
        #     # if self.c_step < 100:
        #     reward -= 500
        
        info = {}
        # print(reward)
        return self.obs, reward, done, info

    # 获取当前状态的观察值
    def _get_obs(self):
        '''
        STATE INDEX (no preprocess)
        sensor data from base imu
        sensor: length = 24
            IDX    |DATA
            0-2    |target x,y,z
            3-5    |base pos x,y,z
            6-9    |base ori quat
            10/12  |left/right hip joint pos in radius
            11/13  |left/right knee joint pos in radius
            14/15  |distance and orientatiion diff
        '''
        # sensor data
        # self.obs = np.array(self.sim.data.sensordata) 
        qpos = np.array(self.sim.data.qpos.flat.copy())
        self.obs = qpos
        
        # preprocess
        diff = self.obs[0:2]-self.obs[3:5]
        twd_target = math.atan2(diff[1],diff[0])
        if self.obs[6:10].all() == 0:
            base_ori_z = 0
        else:
            base_ori_z = Rotation.from_quat(self.obs[6:10]).as_euler('xyz')
            base_ori_z = base_ori_z[2]
        base_ori_z+=2*np.pi if base_ori_z < 0 else 0
        twd_target+=2*np.pi if twd_target < 0 else 0
        angle_diff =  twd_target - base_ori_z
        angle_diff-=2*np.pi if angle_diff>np.pi else 0
        angle_diff+=2*np.pi if angle_diff<-np.pi else 0
        # degree = 180*angle_diff/np.pi # for debugging
        self.obs = np.append(self.obs, angle_diff)
        self.obs = np.append(self.obs, (diff[0]**2+diff[1]**2)**0.5)

        # return self.obs

    # 重置模型
    def reset_model(self):
        random_pos = False
        self.c_step = 0
        qpos = self.init_qpos
        qvel = self.init_qvel
        # reset target
        # qpos[0] = 20*random.random()-10
        # qpos[1] = 20*random.random()-10
        qpos[0] = 1.2
        qpos[1] = 0
        qpos[2] = 0.08
        # reset inital robot position
        qpos[3] = 20*random.random()-10 if random_pos else 0
        qpos[4] = 20*random.random()-10 if random_pos else 0
        # qpos[5:9] = Rotation.from_euler('zyx',[0, 0, 2*np.pi*random.random()]).as_quat()
        # while ((qpos[0]-qpos[2])**2+(qpos[1]-qpos[3])**2)**0.5<1:
        #     qpos[0] = 20*random.random()-10
        #     qpos[1] = 20*random.random()-10
            
        self.set_state(qpos, qvel)
        self._get_obs()
        return self.obs

    # 可视化查看器
    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)


if __name__ == '__main__':
    
    env = gym.make('Biped-v0')
    state = env.reset()
    print(state)

    