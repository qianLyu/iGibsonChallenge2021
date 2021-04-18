import json
from habitat_cont.rl.resnet_policy import (
    PointNavResNetPolicy,
)

from collections import OrderedDict, defaultdict

import argparse

import random
import numpy as np
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
LOG_FILENAME = "exp.navigation.log"
MAX_DEPTH = 10


def load_model(weights_path, dim_actions): # DON'T CHANGE
    depth_256_space = SpaceDict({
        'depth': spaces.Box(low=0., high=1., shape=(256,256,1)),
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

    model = PointNavResNetPolicy(
        observation_space=depth_256_space,
        action_space=action_space,
        hidden_size=512,
        rnn_type='LSTM',
        num_recurrent_layers=2,
        backbone='resnet50',
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--sensors", type=str, required=True)
    parser.add_argument("--goal", type=str, required=False, default='0.2,0.0')
    parser.add_argument("--waypoint-path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sensors_dict = {
        **env._reality.sensor_suite.observation_spaces.spaces
    }

    sensors_dict[env.pointgoal_key] = spaces.Box(
        low=np.finfo(np.float32).min,
        high=np.finfo(np.float32).max,
        shape=(2,),
        dtype=np.float32,
    )

    model = load_model(
        weights_path=JSON_WEIGHTS,
        dim_actions=2
    )
    model = model.eval()

    num_processes = 1

    # Will have to change this:
    goal_location = np.array([float(goal_list_x[i]), float(goal_list_y[i]) ], dtype=np.float32)

    test_recurrent_hidden_states = torch.zeros(
        model.net.num_recurrent_layers,
        num_processes,
        512,
        device=DEVICE,
    )

    prev_actions = torch.zeros(num_processes, 2, device=DEVICE)
    not_done_masks = torch.zeros(num_processes, 1, device=DEVICE)

    observations = env.reset(goal_location)
    STOP_CALLED = False
    while not STOP_CALLED:
        # Will have to change these:
        observations = [observations]
        dist_and_heading_to_goal = observations[0][env.pointgoal_key]
        print(
            "Your goal is to get to: {:.3f}, {:.3f}  "
            "rad ({:.2f} degrees)".format(
                dist_and_heading_to_goal[0],
                dist_and_heading_to_goal[1],
                dist_and_heading_to_goal[1] * 180/np.pi
            )
        )

        batch = defaultdict(list)

        for obs in observations:
            for sensor in obs: 
                '''
                'sensor' is one of: 'depth', 'pointgoal_with_gps_compass'
                (see lines 31-32)
                'depth' needs to be a 256x256 mono-channel image with values
                between 0-1
                'pointgoal_with_gps_compass' is the same as line 126
                (see lines 127-134 for details)
                '''
                batch[sensor].append(to_tensor(obs[sensor]))

        for sensor in batch:
            batch[sensor] = torch.stack(batch[sensor], dim=0).to(
                device=DEVICE, dtype=torch.float
            )

        with torch.no_grad():
            # Get new action and LSTM hidden state
            _, actions, _, test_recurrent_hidden_states = model.act(
                batch,
                test_recurrent_hidden_states,
                prev_actions,
                not_done_masks,
                deterministic=True,
            )

        '''
        ^The policy is polled above. It is designed to be polled at 1 Hz.
        '''
        # Linear velocity; max_linear_speed must be 0.25m/s
        linear_velocity = -torch.tanh(actions[0]).item()
        linear_velocity = (move_amount+1.)/2.*max_linear_speed
        # Angular velocity; max_angular_speed must be 30deg/s CONVERTED TO RADIANS/SEC
        angular_velocity = torch.tanh(actions[1]).item()
        angular_velocity *= max_angular_speed

        if (
            linear_velocity/max_linear_speed < 0.1
            and abs(angular_velocity/max_angular_speed) < 0.05
        ):
            log_mesg('[STOP HAS BEEN CALLED]')
            STOP_CALLED = True


        prev_actions.copy_(actions)
        not_done_masks = torch.ones(num_processes, 1, device=DEVICE)

        # Will have to change this, probably like 'obs, reward, done, infos'
        # Also make sure actions are something that iGibson can read
        observations = env.step(actions)


if __name__ == "__main__":
    main()

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model-path", type=str, required=True)
#     parser.add_argument("--sensors", type=str, required=True)
#     parser.add_argument("--goal", type=str, required=False, default='0.2,0.0')
#     parser.add_argument("--waypoint-path", type=str, required=True)
#     parser.add_argument("--depth-model", type=str, required=False, default="")
#     parser.add_argument("--depth-only", action="store_true")
#     parser.add_argument("--seed", type=int, default=42)
#     args = parser.parse_args()


#     torch.manual_seed(args.seed)
#     np.random.seed(args.seed)
#     random.seed(args.seed)
#     # print('sys', sys.path)
#     env = NavEnv(
#         forward_step=0.25, angle_step=30, is_blind=(args.sensors == ""),
#         sensors=args.sensors.split(','),
#     )

#     if args.depth_model != "":
#         d_model = torch.load(args.depth_model, map_location=device)['model']
#         d_model = d_model.eval()

#     sensors_dict = {
#         **env._reality.sensor_suite.observation_spaces.spaces
#     }

#     if args.depth_only:
#         del sensors_dict['rgb']
#         print('Deleting Sensor from model: rgb')

#     sensors_dict[env.pointgoal_key] = spaces.Box(
#         low=np.finfo(np.float32).min,
#         high=np.finfo(np.float32).max,
#         shape=(2,),
#         dtype=np.float32,
#     )

#     print(args.model_path)

#     model = load_model()
#     model = model.eval()

#     num_processes = 1

#     i = 0
#     while i < len(goal_list_x):
#         goal_location = np.array([float(goal_list_x[i]), float(goal_list_y[i]) ], dtype=np.float32)
#         print("Starting new episode")
#         print("Run #: ", i)
#         print("Goal location: {}".format(goal_location))

#         test_recurrent_hidden_states = torch.zeros(
#             model.net.num_recurrent_layers,
#             num_processes,
#             512,
#             device=DEVICE,
#         )

#         prev_actions = torch.zeros(num_processes, 2, device=DEVICE)
#         not_done_masks = torch.zeros(num_processes, 1, device=DEVICE)

#         observations = env.reset(goal_location)

#         timestep = -1
#         reality_action_count = 0
#         collision_count = 0

#         while True:
#             timestep += 1
#             observations = [observations]

#             goal = observations[0][env.pointgoal_key]
#             print(
#                 "Your goal is to get to: {:.3f}, {:.3f}  "
#                 "rad ({:.2f} degrees)".format(
#                     goal[0], goal[1], (goal[1] / np.pi) * 180
#                 )
#             )

#             batch = defaultdict(list)

#             for obs in observations:
#                 for sensor in obs:
#                     batch[sensor].append(to_tensor(obs[sensor]))

#             for sensor in batch:
#                 batch[sensor] = torch.stack(batch[sensor], dim=0).to(
#                     device=DEVICE, dtype=torch.float
#                 )

#             rgb_obs = batch["rgb"][0].numpy()
#             st = time.time()
#             with torch.no_grad():
#                 _, actions, _, test_recurrent_hidden_states = model.act(
#                     batch,
#                     test_recurrent_hidden_states,
#                     prev_actions,
#                     not_done_masks,
#                     deterministic=True,
#                 )
#             print('Time elapsed predicting action: ', time.time()-st)
#             prev_actions.copy_(actions)
#             not_done_masks = torch.ones(num_processes, 1, device=DEVICE)

#             observations = env.step(actions)
#             if env._get_collision_state():
#                 collision_count +=1
#             print("# Collisions: ", collision_count)
#             if collision_count > 40:
#                 print('Max collisions reached. Exiting.')
#                 exit()
#             if reality_action_count > 200:
#                 print('Max actions reached. Exiting.')
#                 exit()
#             reality_action_count +=1
#             print("# Actions: ", reality_action_count)
#         i+=1


# if __name__ == "__main__":
#     main()
