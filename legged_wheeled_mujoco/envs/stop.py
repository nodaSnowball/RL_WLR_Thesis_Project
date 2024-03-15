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
class StopEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    # 初始化环境参数
    def __init__(
        self,
        xml_file=os.path.join(os.path.dirname(__file__), 'asset', "Legged_wheel_forward.xml"),
        ctrl_cost_weight=0.0001,
        healthy_reward=1,
        healthy_z_range=0.13,
        reset_noise_scale=0.1,
    ):
        utils.EzPickle.__init__(**locals())

        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self.c_step = 0
        self.obs = None
        self.target = [0,0,0,0,0,0]

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)   # 50Hz = 1/(0.02*10)
        # self.end_x = self.sim.model.get_joint_qpos_addr('end')

    @property # 计算健康奖励
    def healthy_reward(self):
        return float(self.is_healthy) * self._healthy_reward

    # 计算控制成本
    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.abs([action[i] for i in range(6) if i!=2 and i!=5]))
        return control_cost

    @property  # 是否倾倒
    def is_healthy(self):
        min_z = self._healthy_z_range
        is_healthy = (self.sim.data.qpos[2] > min_z) and (not self.bump_base())
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
    
    # close loop
    def step(self, action):
        # calculate approaching distance
        reward = 0
        done = False
        info = {}
        self.c_step+=1
        self._get_obs()
        qpos1 = np.array(self.sim.data.qpos.flat.copy())
        self.do_simulation(action, self.frame_skip)
        self._get_obs()
        qpos2 = np.array(self.sim.data.qpos.flat.copy())  
        
        off_track_distance = 0
        approch_distance = 0
        if qpos1[3:7].all() == True:
            xy1 = qpos1[:2]
            xy2 = qpos2[:2]
            displacement = sum((xy1-xy2)**2)
            control_cost = sum(abs(self.obs[11:13]))**2
            
            punish = 3000*displacement + 0.002 * control_cost
            done = self.done 
        
            reward =  self.healthy_reward - punish 

        return self.obs, reward, done, info

    # 获取当前状态的观察值
    def _get_obs(self):
        '''
        STATE INDEX (no preprocess)
        sensor data from base imu
        sensor: length = 22
            IDX    |DATA
            0-2    |base pos x,y,z
            3-6    |base ori quat
            7/9    |left/right hip joint pos in radius
            8/10   |left/right knee joint pos in radius
            11/12  |left/right wheel velocity
            13-15  |base local linear velocity
            16-18  |base local linear acceleration
            19-21  |base local angular velocity
            22-27  |target line info: x, y, quaternion orientation
        '''
        # sensor data
        self.obs = np.array(self.sim.data.sensordata) 
        qpos = np.array(self.sim.data.qpos.flat.copy())
        self.obs[:7] = qpos[:7]
        self.obs = np.append(self.obs, self.target)
        # self.obs

    # 重置模型
    def reset_model(self):
        self.c_step = 0
        qpos = self.init_qpos
        qvel = self.init_qvel

        # reset inital robot position
        qpos[:7] = [0,0,0.25,1,0,0,0]
        qpos[3:7] = Rotation.from_euler('zyx',[0,0,2*np.pi*random.random()]).as_quat()
        self.target = np.append(qpos[:2],qpos[3:7])
            
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