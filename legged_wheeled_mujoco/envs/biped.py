import numpy as np
import gym
from gym import utils
from gym.envs.mujoco import mujoco_env
import os
import random 
from scipy.spatial.transform import Rotation
#import register

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}

# 定义一个仿真环境
class BipedEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    # 初始化环境参数
    def __init__(
        self,
        xml_file=os.path.join(os.path.join(os.path.dirname(__file__),
                                'asset', "Legged_wheel3.xml")),
        ctrl_cost_weight=0.0001,
        healthy_reward=.1,
        healthy_z_range=0.05,
        reset_noise_scale=0.1,
    ):
        utils.EzPickle.__init__(**locals())

        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self.c_step = 0

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
        xy_target = [self.data.get_joint_qpos("end_x").copy(), self.data.get_joint_qpos("end_y").copy()]  # 获取目标位置
        self.xy_target = xy_target
        xy_position = self.get_body_com("base_link")[:2].copy()  # 更新前位置
        xy_distance = xy_target - xy_position
        distance = (xy_distance[0]**2 + xy_distance[1]**2)**0.5
        return distance
    
    # 执行仿真中的一步
    def step(self, action):
        self.c_step+=1
        
        d_before = self.get_xydistance()
        self.do_simulation(action, self.frame_skip)
        d_after = self.get_xydistance()
        
        ctrl_cost = self.control_cost(action)  # 控制损失
        
        sensor = self.sim.data.sensordata # imu sensor on base [angular vel *3, linear vel*3, linear acc*3]
        unstable_punishment = 0.001*(abs(sensor[0])+abs(sensor[1])+abs(sensor[5])+abs(sensor[8]))
        q_diff = Rotation.from_quat(sensor[9:])
        euler_diff = abs(q_diff.as_euler('xyz',degrees=False))
        hip_diff = abs(self.sim.data.get_joint_qpos('right_hip')-self.sim.data.get_joint_qpos('left_hip'))
        knee_diff = abs(self.sim.data.get_joint_qpos('right_knee')-self.sim.data.get_joint_qpos('left_knee'))
        pose_punishments = euler_diff[0]+euler_diff[1]+hip_diff+knee_diff
        # if the robot is moving forward
        forward_check = sensor[3]
        
        # health reward: maintain standing pose
        # control cost: acceleration of motors
        cost = ctrl_cost
        punishment = (unstable_punishment+pose_punishments*0.1)*0.1
        
        approching_reward = 5*(d_before-d_after)# if forward_check>0 else 0 # -5*abs(d_before-d_after)

        done = self.done
        reward =  approching_reward + self.healthy_reward - cost - punishment

        # 判断是否到达终点
        if done == False:
            if d_after < 0.3:
                done = True
                reward = 100
        else:
            # if self.c_step < 100:
            reward -= 50
        observation = self._get_obs()
        info = {}
        # print(reward)
        return observation, reward, done, info

    # 获取当前状态的观察值
    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy() #身体部位
        velocity = self.sim.data.qvel.flat.copy() #速度
        velocity = velocity[2:]
        sensor = self.sim.data.sensordata # imu sensor on base [angular vel *3, linear vel*3, linear acc*3]

        observations = np.concatenate((position, velocity, sensor)) #目标位置
        self.c_step = 0

        return observations

    # 重置模型
    def reset_model(self):
        # self.sim.data.geom_xpos[self.model.geom_name2id("end")][0] = 38*random.random()-19# += self._reset_noise_scale * random.random()
        # self.sim.data.geom_xpos[self.model.geom_name2id("end")][1] = 38*random.random()-19# += self._reset_noise_scale * random.random()
        # self.xy_target = self.data.get_geom_xpos("end")[:2].copy()
        qpos = self.init_qpos
        qvel = self.init_qvel
        # reset target
        qpos[0] = 20*random.random()-10
        qpos[1] = 20*random.random()-10
        # qpos[2] = 20*random.random()-10
        # qpos[3] = 20*random.random()-10
        # qpos[5:9] = Rotation.from_euler('zyx',[0, 0, 2*np.pi*random.random()]).as_quat()
        # while (qpos[2]==0 and qpos[3]==0) or (abs(qpos[0]-qpos[2])<0.1 or abs(qpos[1]-qpos[3])<0.1):
        #     qpos[2] = 20*random.random()-10
        #     qpos[3] = 20*random.random()-10
            
        self.set_state(qpos, qvel)
        observation = self._get_obs()
        return observation

    # 可视化查看器
    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)


# Stair-env
class StairEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    # 初始化环境参数
    def __init__(
        self,
        xml_file=os.path.join(os.path.join(os.path.dirname(__file__),
                                'asset', "Legged_wheel1.xml")),
        ctrl_cost_weight=0.005,
        healthy_reward=1.0,
        healthy_z_range=0.25,
        reset_noise_scale=0.1,
    ):
        utils.EzPickle.__init__(**locals())

        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    @property # 计算健康奖励
    def healthy_reward(self):
        return (
            float(self.is_healthy) * self._healthy_reward
        )

    # 计算控制成本
    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
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
    
    # 执行仿真中的一步
    def step(self, action):
        xy_target = self.data.get_geom_xpos("end")[:2].copy()  # 获取目标位置
        self.xy_target = xy_target
        xy_position_before = self.get_body_com("base_link")[:2].copy()  # 更新前位置
        xy_distance_before = abs(xy_target - xy_position_before)
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("base_link")[:2].copy()   # 更新后位置
        xy_distance_after = abs(xy_target - xy_position_after)
        xy_velocity = -(xy_distance_after - xy_distance_before) / self.dt  # 速度
        x_velocity, y_velocity = xy_velocity
        x_velocity = x_velocity if x_velocity<900 else 0
        y_velocity = y_velocity if y_velocity<900 else 0
        ctrl_cost = self.control_cost(action)  # 控制损失
        quad_impact_cost = 0.5e-6 * np.square(self.sim.data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)

        reward = 10*x_velocity + 10*y_velocity + self.healthy_reward

        if self.done:
            done = True
            reward -= 100
        # 判断是否到达终点
        else:
            if np.sum(xy_distance_after) < 0.1:
                done = True
                reward = reward + 100
        observation = self._get_obs()
        info = {}
        return observation, reward, done, info

    # 获取当前状态的观察值
    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy() #身体部位
        velocity = self.sim.data.qvel.flat.copy() #速度
        sensor = self.sim.sensordata

        observations = np.concatenate((position, velocity, self.xy_target)) #目标位置

        return observations

    # 重置模型
    def reset_model(self):
        self.sim.data.geom_xpos[self.model.geom_name2id("end")][0] += self._reset_noise_scale * random.random()
        self.sim.data.geom_xpos[self.model.geom_name2id("end")][1] += self._reset_noise_scale * random.random()
        observation = self._get_obs()
        return observation

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

    