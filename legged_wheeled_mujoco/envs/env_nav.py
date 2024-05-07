import numpy as np
from typing import Dict, Tuple, Union
import gymnasium as gym
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box, MultiDiscrete
import os, sys
import cv2
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import random 
import math
from time import sleep
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import envs.register

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 15.0,
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
        # xml_file='/scratch/zl4930/wlr/envs/asset/nav_model.xml',
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        max_step = 500,
        healthy_reward=0.1,
        healthy_z_range=0.20,
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
        self.bev = None

        self.is_model = 0 
        self.roll_model = None
        self.walk_model = None
        self.jump_model = None
        if self.is_model:
            self.roll_model = 1
            self.walk_model = 1
            self.jump_model = 1

        self.obs_pos_dict = np.array([[x,y] for y in range(6,-7,-6) for x in range(-6,7,6)])
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

        self.observation_space = Box(
            low=0, high=255, shape=(200,200,3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([4,3])

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
        rob_pos = self.curr_obs[2:4].copy()
        xy_distance = target_pos - rob_pos
        distance = (xy_distance[0]**2 + xy_distance[1]**2)**0.5
        return distance
    
    # 执行仿真中的一步
    def step(self, action):
        bev = self.bev
        reward = -10
        done = False
        info = {}
        
        success = 0
        self.c_step+=1
        dir = action[0]
        skill = action[1]

        # check if out of boundary
        if dir == 0 and self.robot_idx-5 < 0:       # go up
            info.update({'is_success':False})
            return bev, reward, True, False, info
        if dir == 1 and self.robot_idx+5 >24:       # go down
            info.update({'is_success':False})
            return bev, reward, True, False, info
        if dir == 2 and self.robot_idx%5 == 0:      # go left
            info.update({'is_success':False})
            return bev, reward, True, False, info
        if dir == 3 and (self.robot_idx+1)%5 == 0:    # go right
            info.update({'is_success':False})
            return bev, reward, True, False, info
            


        # load locomotion model
        if skill == 0:          # roll
            model = self.roll_model if self.is_model else None
        elif skill == 1:        # walk
            model = self.walk_model if self.is_model else None
        elif skill == 2:        # jump
            model = self.jump_model if self.is_model else None

        # set up for locomotion
        if dir == 0:        # go up
            quat = quat = Rotation.from_euler('zyx',[0, 0, 1*np.pi/2]).as_quat()
            local_target_idx = self.robot_idx-5
        if dir == 1:        # go down
            quat = quat = Rotation.from_euler('zyx',[0, 0, -1*np.pi/2]).as_quat()
            local_target_idx = self.robot_idx+5
        if dir == 2:        # go left
            quat = quat = Rotation.from_euler('zyx',[0, 0, 0*np.pi/2]).as_quat()
            local_target_idx = self.robot_idx-1
        if dir == 3:        # go right
            quat = quat = Rotation.from_euler('zyx',[0, 0, 2*np.pi/2]).as_quat()
            local_target_idx = self.robot_idx+1
        qpos = self.init_qpos
        qvel = self.init_qvel
        # set robot pos, quat
        qpos[2:4] = self.pos_dict[self.robot_idx]
        qpos[5:9] = quat
        self.set_state(qpos, qvel)
        bev = self._get_obs()

        # locomotion 
        center_of_obs = np.array([0,2,4,10,12,14,20,22,24])
        if self.is_model:
            while True:
                # do simulation
                self.do_simulation(action, self.frame_skip)
                bev = self._get_obs()
                break
            return 0
        else:
            if self.robot_idx in center_of_obs:
                i = np.where(center_of_obs == self.robot_idx)
                obsacle = self.obstacle_id[i]
                if obsacle<3:                                   # obstacle for walk
                    if skill == 1 and random.random()<0.95:      # walk
                        success = 1
                    elif skill == 2 and random.random()<0.5:    # jump
                        success = 1 
                elif obsacle>5:                                 # obstacle for jump
                    if skill == 2 and random.random()<0.95:      # jump
                        success = 1
                    elif skill == 1 and random.random()<0.6:    # walk
                        success = 1 
                    elif skill == 0 and random.random()<0.5:    # roll
                        success = 1 
                else:                                           # obstacle for roll
                    if skill == 0 and random.random()<0.95:      # roll
                        success = 1
                    elif skill == 1 and random.random()<0.6:    # walk
                        success = 1 
                    elif skill == 2 and random.random()<0.5:    # roll
                        success = 1 
            elif self.target_idx in center_of_obs:
                i = np.where(center_of_obs == self.target_idx)
                obsacle = self.obstacle_id[i]
                if obsacle<3:                                   # obstacle for walk
                    if skill == 1 and random.random()<0.95:      # walk
                        success = 1
                    elif skill == 2 and random.random()<0.5:    # jump
                        success = 1 
                elif obsacle>5:                                 # obstacle for jump
                    if skill == 2 and random.random()<0.95:      # jump
                        success = 1
                    elif skill == 1 and random.random()<0.6:    # walk
                        success = 1 
                    elif skill == 0 and random.random()<0.5:    # roll
                        success = 1 
                else:                                           # obstacle for roll
                    if skill == 0 and random.random()<0.95:      # roll
                        success = 1
                    elif skill == 1 and random.random()<0.6:    # walk
                        success = 1 
                    elif skill == 2 and random.random()<0.5:    # roll
                        success = 1 
            else: 
                if skill == 0 and random.random()<0.8:          # roll
                    success = 1
                elif skill == 1 and random.random()<0.95:        # walk
                    success = 1 
                elif skill == 2 and random.random()<0.95:        # roll
                    success = 1 


        # update bev
        if success:
            reward=1
            if self.c_step>=20:                     # max step
                done = True
                reward = -1
                info.update({'is_success':False})
            if self.robot_idx == self.target_idx:   # reach target
                done = True
                reward = 10
                info = {'is_success':True}
            self.robot_idx = local_target_idx
            qpos[2:4] = self.pos_dict[self.robot_idx]
            qpos[5:9] = quat
            self.set_state(qpos, qvel)
            bev = self._get_obs()
            return bev, reward, done, False, info

        # collision
        done = True
        info.update({'is_success':False})
        reward = -5
        return bev, reward, done, False, info

    # 获取当前状态的观察值
    def _get_obs(self, render=True):
        '''
        STATE INDEX 
        img from camera : (240, 240)
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
        obs[-56:] /= 10

        self.obs = obs

        if render:
            bev = self.mujoco_renderer.render(render_mode='rgb_array',camera_name='top')
            self.bev = cv2.resize(bev, (200,200))
            # plt.imshow(self.bev)
            return self.bev

        return 0
        

    # 重置模型
    def reset_model(self):
        self.c_step = 0
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.action = np.zeros((6,))

        if self.random_reset:
            # reset obstacle
            self.obstacle_id = np.array(list(range(9)))
            random.shuffle(self.obstacle_id)
            for i, obs_id in enumerate(self.obstacle_id):
                self.data.mocap_pos[obs_id][0:2] = self.obs_pos_dict[i]
                # quat = Rotation.from_euler('zyx',[0, 0, random.randint(0,3)*np.pi/2]).as_quat()
                # self.data.mocap_quat[i] = quat
            # reset target
            self.target_idx = random.randint(0,24)
            self.target = self.pos_dict[self.target_idx]
            qpos[0:2] = self.target
            # reset inital robot position
            self.robot_idx = random.randint(0,24)
            while self.robot_idx==self.target_idx:
                self.robot_idx = random.randint(0,24)
            qpos[2:4] = self.pos_dict[self.robot_idx]
            qpos[5:9] = Rotation.from_euler('zyx',[0, 0, random.randint(0,3)*np.pi/2]).as_quat()
                
            self.set_state(qpos, qvel)
            bev = self._get_obs()
            return bev
        
        self.target = np.array([6,0])
        qpos[0:2] = self.target
        qpos[2:4] = np.array([-6,0])
        self.set_state(qpos, qvel)
        bev = self._get_obs()
        return bev


if __name__ == '__main__':
    
    env = gym.make('Nav-v1',render_mode='human')
    state = env.reset()
    x = env.render()
    plt.imshow(x/255)
    state = env.step(np.zeros((6,)))
    plt.imshow(x/255)
    print(state)

    