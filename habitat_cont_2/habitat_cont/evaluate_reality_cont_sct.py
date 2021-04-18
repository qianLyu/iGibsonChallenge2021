import json
from habitat_baselines.rl.ddppo.policy.resnet_policy import (
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

import habitat
from habitat.sims import make_sim
# from habitat_baselines.config.default import get_config

import time
import cv2
import os
import subprocess

import yaml

DEVICE = torch.device("cpu")
SIMULATOR_REALITY_ACTIONS = {0: "stop", 1: "forward", 2: "left", 3: "right"}
LOG_FILENAME = "exp.navigation.log"
MAX_DEPTH = 10


# No sliding: d4 vs d6 200M
WEIGHTS_PATH = '/home/locobot/sct_experiments/paper_weights/d6_decay160M_30deg_noslide_101.json'
# WEIGHTS_PATH = '/home/locobot/sct_experiments/paper_weights/d4_decay160M_30deg_noslide_89.json'

# No sliding: d4 vs d6 150M
# WEIGHTS_PATH = '/home/locobot/sct_experiments/paper_weights/d4_decay160M_30deg_noslide_76.json'
# WEIGHTS_PATH = '/home/locobot/sct_experiments/paper_weights/d6_decay160M_30deg_noslide_77.json'

# WEIGHTS_PATH = '/home/locobot/sct_experiments/paper_weights/d4_decay_30deg_79.json'
# WEIGHTS_PATH = '/home/locobot/sct_experiments/paper_weights/d6_decay_30deg_62.json'

# Dont use
# WEIGHTS_PATH = '/home/locobot/sct_experiments/paper_weights/d15_decay_30deg_63.json'


WAYPOINTS_YAML = 'evaluation/waypoints.yaml'

GAUSSIAN, DISCRETE_4, DISCRETE_6 = False, False, False
DISCRETE_4 = 'd4' in WEIGHTS_PATH
DISCRETE_6 = 'd6' in WEIGHTS_PATH
GAUSSIAN   = 'gaussian' in WEIGHTS_PATH

class NavEnv:
    def __init__(self, forward_step, angle_step, is_blind=False, sensors=["RGB_SENSOR"]):
        config = habitat.get_config()

        log_mesg(
            "env: forward_step: {}, angle_step: {}".format(
                forward_step, angle_step
            )
        )

        config.defrost()
        config.PYROBOT.SENSORS = sensors
        config.PYROBOT.RGB_SENSOR.WIDTH = 256
        config.PYROBOT.RGB_SENSOR.HEIGHT = 256
        config.PYROBOT.DEPTH_SENSOR.WIDTH = 256
        config.PYROBOT.DEPTH_SENSOR.HEIGHT = 256
        config.PYROBOT.DEPTH_SENSOR.MAX_DEPTH = 10
        config.PYROBOT.DEPTH_SENSOR.MIN_DEPTH = 0.3
        config.freeze()

        self._reality = make_sim(id_sim="PyRobot-v0", config=config.PYROBOT)
        self._angle = (angle_step / 180) * np.pi
        self._pointgoal_key = "pointgoal_with_gps_compass"
        self.is_blind = is_blind

        if not is_blind:
            sensors_dict = {
                **self._reality.sensor_suite.observation_spaces.spaces
            }
        else:
            sensors_dict = {}

        sensors_dict[self._pointgoal_key] = spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(2,),
            dtype=np.float32,
        )
        self.observation_space = SpaceDict(sensors_dict)

        self.action_space = spaces.Discrete(4)

        self._actions = {
            "forward": [forward_step, 0, 0],
            "left": [0, 0, self._angle],
            "right": [0, 0, -self._angle],
            "stop": [0, 0, 0],
        }
        self._process = None
        self._last_time = -1

    def _pointgoal(self, agent_state, goal):
        agent_x, agent_y, agent_rotation = agent_state
        agent_coordinates = np.array([agent_x, agent_y])
        rho = np.linalg.norm(agent_coordinates - goal)
        theta = (
            np.arctan2(
                goal[1] - agent_coordinates[1], goal[0] - agent_coordinates[0]
            )
            - agent_rotation
        )
        theta = theta % (2 * np.pi)
        if theta >= np.pi:
            theta = -((2 * np.pi) - theta)
        return rho, theta

    @property
    def pointgoal_key(self):
        return self._pointgoal_key

    def reset(self, goal_location):
        self._goal_location = np.array(goal_location)
        observations = self._reality.reset()

        base_state = self._get_base_state()

        # assert np.all(base_state == 0) == True, (
        #     "Please restart the roslaunch command. "
        #     "Current base_state is {}".format(base_state)
        # )

        observations[self._pointgoal_key] = self._pointgoal(
            base_state, self._goal_location
        )

        return observations

    def _get_collision_state(self):
        return self._reality.base.get_collision_state()

    def _get_base_state(self):
        base_state = self._reality.base.get_state("odom")
        base_state = np.array(base_state, dtype=np.float32)
        log_mesg("base_state: {:.3f} {:.3f} {:.3f}".format(*base_state))
        return base_state

    def _go_to_absolute(self, xyt_position):
        return self._reality.base.go_to_absolute(xyt_position, use_map=True)

    @property
    def reality(self):
        return self._reality

    def step(self, actions):
        print('[RAW actions]: ', actions)
        max_linear_speed = 0.25#*3
        max_angular_speed = np.pi/180*30#*3

        actions = actions.squeeze()

        if GAUSSIAN:
            move_amount = -torch.tanh(actions[0]).item()
            turn_amount = torch.tanh(actions[1]).item()
            move_amount = (move_amount+1.)/2.*max_linear_speed
            turn_amount *= max_angular_speed
            log_mesg('[commands]: {} {}'.format(move_amount, turn_amount))

            if move_amount/0.25 < 0.1 and abs(turn_amount/max_angular_speed) < 0.05:
                log_mesg('[STOP HAS BEEN CALLED]')
                exit()
        elif DISCRETE_4:
            action_index = actions.item()
            move_amount, turn_amount = 0,0
            if action_index == 0: # STOP
                log_mesg('[STOP HAS BEEN CALLED]')
                exit()
            elif action_index == 1: # Move FWD
                move_amount = max_linear_speed
            elif action_index == 2: # LEFT
                turn_amount = max_angular_speed
            else: # RIGHT
                turn_amount = -max_angular_speed
        elif DISCRETE_6:
            action_index = actions.item()
            move_amount, turn_amount = 0,0
            if action_index == 0: # STOP
                move_amount = max_linear_speed
                turn_amount = -max_angular_speed
            elif action_index == 1: # Move FWD
                move_amount = 0
                turn_amount = -max_angular_speed
            elif action_index == 2: # LEFT
                move_amount = max_linear_speed
                turn_amount = 0
            elif action_index == 3: # LEFT
                log_mesg('[STOP HAS BEEN CALLED]')
                exit()
            elif action_index == 4: # LEFT
                move_amount = max_linear_speed
                turn_amount = max_angular_speed
            elif action_index == 5: # LEFT
                move_amount = 0
                turn_amount = max_angular_speed

        self._reality._robot.base.set_vel(fwd_speed=move_amount, turn_speed=turn_amount, exe_time=1.)
        while time.time() - self._last_time < 1.:
            pass
        self._last_time = time.time()        
        observations = self._reality._sensor_suite.get_observations(
            self._reality.get_robot_observations()
        )

        base_state = self._get_base_state()

        observations[self._pointgoal_key] = self._pointgoal(
            base_state, self._goal_location
        )

        return observations


def log_mesg(mesg):
    print(mesg)
    with open(LOG_FILENAME, "a") as f:
        f.write(mesg + "\n")

def load_model():
    depth_256_space = SpaceDict({
        'depth': spaces.Box(low=0., high=1., shape=(256,256,1)),
        'pointgoal_with_gps_compass': spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(2,),
            dtype=np.float32,
        )
    })

    if GAUSSIAN:
        action_space = spaces.Box(
            np.array([float('-inf'),float('-inf')]), np.array([float('inf'),float('inf')])
        )
        action_distribution = 'gaussian'
        dim_actions = 2
    elif DISCRETE_4:
        action_space = spaces.Discrete(4)
        action_distribution = 'categorical'
        dim_actions = 4
    elif DISCRETE_6:
        action_space = spaces.Discrete(6)
        action_distribution = 'categorical'
        dim_actions = 6

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
    model.to(torch.device("cpu"))

    data_dict = OrderedDict()
    with open(WEIGHTS_PATH, 'r') as f:
        data_dict = json.load(f)    
    model.load_state_dict(
        {
            k[len("actor_critic.") :]: torch.tensor(v)
            for k, v in data_dict.items()
            if k.startswith("actor_critic.")
        }
    )

    return model

def to_tensor(v):
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
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument(
        "--normalize-visual-inputs", type=int, required=True, choices=[0, 1]
    )
    parser.add_argument(
        "--backbone",
        type=str,
        required=True,
        choices=["resnet50", "se_resneXt50"],
    )
    parser.add_argument("--num-recurrent-layers", type=int, required=True)
    parser.add_argument("--goal", type=str, required=False, default='0.2,0.0')
    parser.add_argument("--waypoint-path", type=str, required=True)
    parser.add_argument("--depth-model", type=str, required=False, default="")
    parser.add_argument("--depth-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # print("TESTING:", torch.__version__)
    # import pdb; pdb.set_trace()

    vtorch = "1.2.0+cpu"
    assert torch.__version__ == vtorch, "Please use torch {}".format(vtorch)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    # print('sys', sys.path)
    env = NavEnv(
        forward_step=0.25, angle_step=30, is_blind=(args.sensors == ""),
        sensors=args.sensors.split(','),
    )
    # with open(args.waypoint_path,'r') as f:
    #     goal_list_x, goal_list_y = zip(*[l.split() for l in f])

    with open(WAYPOINTS_YAML) as f:
        waypoints = yaml.load(f)
    goal_list_x, goal_list_y =  zip(*[waypoints[args.waypoint_path].split()])

    device = torch.device("cpu")

    if args.depth_model != "":
        d_model = torch.load(args.depth_model, map_location=device)['model']
        d_model = d_model.eval()

    sensors_dict = {
        **env._reality.sensor_suite.observation_spaces.spaces
    }

    if args.depth_only:
        del sensors_dict['rgb']
        print('Deleting Sensor from model: rgb')

    sensors_dict[env.pointgoal_key] = spaces.Box(
        low=np.finfo(np.float32).min,
        high=np.finfo(np.float32).max,
        shape=(2,),
        dtype=np.float32,
    )

    print(args.model_path)

    model = load_model()
    model = model.eval()

    num_processes = 1

    i = 0
    while i < len(goal_list_x):
        goal_location = np.array([float(goal_list_x[i]), float(goal_list_y[i]) ], dtype=np.float32)
        log_mesg("Starting new episode")
        print("Run #: ", i)
        log_mesg("Goal location: {}".format(goal_location))

        test_recurrent_hidden_states = torch.zeros(
            model.net.num_recurrent_layers,
            num_processes,
            args.hidden_size,
            device=DEVICE,
        )

        if GAUSSIAN:
            prev_actions = torch.zeros(num_processes, 2, device=DEVICE)
        else:
            prev_actions = torch.zeros(num_processes, 1, device=DEVICE)
        not_done_masks = torch.zeros(num_processes, 1, device=DEVICE)

        observations = env.reset(goal_location)

        timestep = -1
        reality_action_count = 0
        collision_count = 0

        while True:
            timestep += 1
            observations = [observations]

            goal = observations[0][env.pointgoal_key]
            log_mesg(
                "Your goal is to get to: {:.3f}, {:.3f}  "
                "rad ({:.2f} degrees)".format(
                    goal[0], goal[1], (goal[1] / np.pi) * 180
                )
            )

            batch = defaultdict(list)

            for obs in observations:
                for sensor in obs:
                    batch[sensor].append(to_tensor(obs[sensor]))

            for sensor in batch:
                batch[sensor] = torch.stack(batch[sensor], dim=0).to(
                    device=DEVICE, dtype=torch.float
                )

            # if args.depth_model != "":
            #     print('USING FAKE DEPTH')
            #     st = time.time()
            #     with torch.no_grad():
            #         rgb_stretch = batch['rgb'].permute(0, 3, 1, 2) / 255.0

            #         # FASTDEPTH expects a NCHW order
            #         depth_stretch = d_model(rgb_stretch)
            #         depth_stretch = torch.clamp(depth_stretch / MAX_DEPTH, 0, 1.0)
            #         batch['depth'] = depth_stretch.permute(0, 2, 3, 1)
            #     print('Time elapsed creating fake depth image: ', time.time()-st)

                # torch.save(batch, "episode/timestep_{}.pt".format(timestep))

            # depth_obs = batch["depth"][0].numpy()
            #depth_obs = np.squeeze(depth_obs)
            #depth_img = Image.fromarray((depth_obs * 255).astype(np.uint8), mode="L")
            #depth_img.save("real_imgs/depth/depth_" + str(img_ctr) + ".jpg", "JPEG")

            rgb_obs = batch["rgb"][0].numpy()
            #rgb_img = Image.fromarray((rgb_obs).astype(np.uint8), mode="RGB")
            #rgb_img.save("real_imgs/rgb/rgb_" + str(img_ctr) + ".jpg", "JPEG")
            #img_ctr +=1 
 
            st = time.time()
            with torch.no_grad():
                _, actions, _, test_recurrent_hidden_states = model.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=True,
                )
            print('Time elapsed predicting action: ', time.time()-st)
            prev_actions.copy_(actions)
            not_done_masks = torch.ones(num_processes, 1, device=DEVICE)

            observations = env.step(actions)
            if env._get_collision_state():
                collision_count +=1
            print("# Collisions: ", collision_count)
            if collision_count > 40:
                print('Max collisions reached. Exiting.')
                exit()
            if reality_action_count > 200:
                print('Max actions reached. Exiting.')
                exit()
            # simulation_action = actions[0].item()
            # reality_action = SIMULATOR_REALITY_ACTIONS[simulation_action]
            # print("reality_action:", reality_action)
            reality_action_count +=1
            print("# Actions: ", reality_action_count)
            # if reality_action_count > 201 or collision_count > 40:
            #     reality_action = "stop"
            #     # episode failed, reset robot
            # if reality_action != "stop":
            #     observations = env.step(reality_action)
            #     not_done_masks = torch.ones(num_processes, 1, device=DEVICE)
            #     print("# Actions: ", reality_action_count)
            #     if env._get_collision_state():
            #         collision_count +=1
            #     print("# Collisions: ", collision_count)
            # else:
            #     print("STOP called, episode over.")
            #     print("Distance to goal: {:.3f}m".format(goal[0]))
            #     cur_x, cur_y, cur_t = env._get_base_state()
            #     env._go_to_absolute([cur_x, cur_y, 0])
            #     #env._go_to_absolute([float(goal_list_x[i]), float(goal_list_y[i]), 0])
            #     #input("press enter to continue")
            #     if goal[0] > 0.2:
            #         # episode failed, reset robot
            #         print("Episode failed. Resetting to next episode start location: ", goal_list_x[i] + ", " + goal_list_y[i])
            #         env._go_to_absolute([float(goal_list_x[i]), float(goal_list_y[i]), 0])
            #     break
        i+=1


if __name__ == "__main__":
    main()