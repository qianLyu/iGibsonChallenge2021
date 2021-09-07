from gibson2.utils.utils import quatToXYZW
from gibson2.envs.env_base import BaseEnv
from gibson2.tasks.room_rearrangement_task import RoomRearrangementTask
from gibson2.tasks.point_nav_fixed_task import PointNavFixedTask
from gibson2.tasks.point_nav_random_task import PointNavRandomTask
from gibson2.tasks.interactive_nav_random_task import InteractiveNavRandomTask
from gibson2.tasks.dynamic_nav_random_task import DynamicNavRandomTask
from gibson2.tasks.reaching_random_task import ReachingRandomTask
from gibson2.sensors.scan_sensor import ScanSensor
from gibson2.sensors.vision_sensor import VisionSensor
from gibson2.robots.robot_base import BaseRobot
from gibson2.external.pybullet_tools.utils import stable_z_on_aabb
#from gibson2.sensors.bump_sensor import BumpSensor

from transforms3d.euler import euler2quat
from collections import OrderedDict
import argparse
import gym
import numpy as np
import pybullet as p
import time
import logging

import json
from habitat_cont_2.habitat_cont.rl.resnet_policy import (
    PointNavResNetPolicy,
)
from collections import OrderedDict, defaultdict

import random
import torch
from gym import spaces
from gym.spaces.dict_space import Dict as SpaceDict
from PIL import Image

import time
import cv2
import os
import subprocess
import yaml
DEVICE = torch.device("cuda")

import matplotlib.image as mp


# WEIGHTS_PATH = 'habitat_cont_2/habitat_cont/gaussian_noslide_30deg_63_skyfail.json'
WEIGHTS_PATH = '/nethome/qluo49/iGibsonChallenge2021/ckpt.36.json'

def load_model(weights_path, dim_actions): # DON'T CHANGE
    depth_256_space = SpaceDict({
        'depth': spaces.Box(low=0., high=1., shape=(240,320,1)),
        'pointgoal_with_gps_compass': spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(2,),
            dtype=np.float32,
        )
    })

    action_space = spaces.Box(
        np.array([float('-inf'),float('-inf')]), np.array([float('inf'),float('inf')])
    )
    action_distribution = 'gaussian'

    # action_space = spaces.Discrete(4)
    # action_distribution = 'categorical'

    model = PointNavResNetPolicy(
        observation_space=depth_256_space,
        action_space=action_space,
        hidden_size=512,
        rnn_type='LSTM',
        num_recurrent_layers=2,
        backbone='resnet18',
        normalize_visual_inputs=False,
        action_distribution=action_distribution,
        dim_actions=dim_actions
    )
    model.to(torch.device(DEVICE))

    state_dict = OrderedDict()
    with open(weights_path, 'r') as f:
        state_dict = json.load(f)   
    # state_dict = torch.load(weights_path, map_location=DEVICE) 
    model.load_state_dict(
        {
            k[len("actor_critic.") :]: torch.tensor(v)
            for k, v in state_dict.items()
            if k.startswith("actor_critic.")
        }
    )

    return model

def to_tensor(v): # DON'T CHANGE
    if torch.is_tensor(v):
        return v
    elif isinstance(v, np.ndarray):
        return torch.from_numpy(v)
    else:
        return torch.tensor(v, dtype=torch.float)

def observations_to_image(observation):

    egocentric_view = []
    if "rgb" in observation:
        observation_size = observation["rgb"].shape[0]
        egocentric_view.append(observation["rgb"][:, :, :3])

    # if "depth" in observation:
    #     observation_size = observation["depth"].shape[0]
    #     depth_map = (observation["depth"].squeeze() * 255).astype(np.uint8)
    #     depth_map = np.stack([depth_map for _ in range(3)], axis=2)
    #     egocentric_view.append(depth_map)

    egocentric_view = np.concatenate(egocentric_view, axis=1)

    frame = egocentric_view

    return frame

ACTION_DIM = 2
LINEAR_VEL_DIM = 0
ANGULAR_VEL_DIM = 1

num_processes = 1

index = 0

class TrainedAgent:
    def __init__(self):
        self.model = load_model(
            weights_path = WEIGHTS_PATH,
            dim_actions = 2 #4
        )
        self.model = self.model.eval()
        self.state = OrderedDict()
        self.test_recurrent_hidden_states = torch.zeros(
            self.model.net.num_recurrent_layers,
            num_processes,
            512,
            device=DEVICE,
        )
        self.prev_actions = torch.zeros(num_processes, 2, device=DEVICE)
        self.not_done_masks = torch.zeros(num_processes, 1, device=DEVICE)
        self.index = index

    def reset(self):
        pass

    def act(self, observations):
        self.state = OrderedDict()
        self.state['depth'] = observations['depth']
        self.state['pointgoal_with_gps_compass'] = observations['task_obs'][:2]
        self.state = [self.state]

        self.index += 1
        # frame = observations_to_image(observations)
        # root = f'/nethome/qluo49/iGibsonChallenge2021/pictures/{self.index}.png'
        # mp.imsave(root,frame)

        batch = defaultdict(list)
        #print(state)
        for obs in self.state:
            for sensor in obs: 
                batch[sensor].append(to_tensor(obs[sensor]))

        for sensor in batch:
            batch[sensor] = torch.stack(batch[sensor], dim=0).to(
                device=DEVICE, dtype=torch.float
            )

        # if self.index % 10 != 0:
        #     action = self.prev_actions

        # else:
        with torch.no_grad():
            # Get new action and LSTM hidden state
            _, action, _, self.test_recurrent_hidden_states = self.model.act(
                batch,
                self.test_recurrent_hidden_states,
                self.prev_actions,
                self.not_done_masks,
                deterministic=True,
            )

        self.prev_actions.copy_(action)

        # gaussian action space
        move_amount = torch.clip(action[0][0], min=-1, max=1).item()
        turn_amount = torch.clip(action[0][1], min=-1, max=1).item() 
        move_amount = (move_amount+1.)/4.
        # move_amount = -torch.tanh(action[0][0]).item()
        # turn_amount = torch.tanh(action[0][1]).item()
        # move_amount = (move_amount+1.)/2.
        #action1 = np.array([ 0.25 * move_amount, 0.16 * turn_amount])
        action1 = np.array([ move_amount, turn_amount])

        # # continuous action space
        # action = action.squeeze()
        # action_index = action.item()
        # move_amount, turn_amount = 0,0
        # max_linear_speed = 1.0 #0.5 #1.0
        # max_angular_speed = 1.0 #1/9 #1.0
        # if action_index == 0: # STOP
        #     move_amount, turn_amount = 0,0
        #     #print('[STOP HAS BEEN CALLED]')
        # elif action_index == 1: # Move FWD
        #     move_amount = max_linear_speed
        # elif action_index == 2: # LEFT
        #     turn_amount = max_angular_speed
        # else: # RIGHT
        #     turn_amount = - max_angular_speed

        action1 = np.array([ move_amount, turn_amount])

        self.not_done_masks = torch.ones(num_processes, 1, device=DEVICE)
        return action1


# class ForwardOnlyAgent(RandomAgent):
#     def act(self, observations):
#         action = np.zeros(ACTION_DIM)
#         action[LINEAR_VEL_DIM] = 1.0
#         action[ANGULAR_VEL_DIM] = 0.0
#         return action


if __name__ == "__main__":
    obs = {
        'depth': np.ones((240, 320, 1)),
        'rgb': np.ones((240, 320, 3)),
        'sensor': np.ones((2,))
    }

    # agent = RandomAgent()
    # action = agent.act(obs)
    # print('action', action)

    # agent = ForwardOnlyAgent()
    # action = agent.act(obs)
    # print('action', action)