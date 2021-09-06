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
# from gibson2.sensors.bump_sensor import BumpSensor

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
import matplotlib.image as mp

import time
import cv2
import os
import subprocess
import yaml
from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize


def Pic2Video(imgPath, videoPath):
 
    images = os.listdir(imgPath)
    fps = 10  
 
    fourcc = VideoWriter_fourcc(*"XVID")
 
    image = Image.open(imgPath + images[0])
    videoWriter = cv2.VideoWriter(videoPath, fourcc, fps, image.size)
    for im_name in range(len(images)):
        frame = cv2.imread(imgPath + f'{im_name+1}.png')  
        videoWriter.write(frame)

    videoWriter.release()
    cv2.destroyAllWindows()
DEVICE = torch.device("cuda")

WEIGHTS_PATH = '/nethome/qluo49/iGibsonChallenge2021/ckpt.36.json'

def load_model(weights_path, dim_actions): # DON'T CHANGE
    depth_256_space = SpaceDict({
        'depth': spaces.Box(low=0., high=1., shape=(180,320,1)),
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
        hidden_size=512, #512,
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

# WEIGHTS_PATH = '/coc/pskynet3/nyokoyama3/aug26/hablabhotswap/gaussian_noslide_30deg_63_skyfail.json'

# def load_model(weights_path, dim_actions): # DON'T CHANGE
#     depth_256_space = SpaceDict({
#         'depth': spaces.Box(low=0., high=1., shape=(256,256,1)),
#         'pointgoal_with_gps_compass': spaces.Box(
#             low=np.finfo(np.float32).min,
#             high=np.finfo(np.float32).max,
#             shape=(2,),
#             dtype=np.float32,
#         )
#     })

#     action_space = spaces.Box(
#         np.array([float('-inf'),float('-inf')]), np.array([float('inf'),float('inf')])
#     )
#     action_distribution = 'gaussian'

#     model = PointNavResNetPolicy(
#         observation_space=depth_256_space,
#         action_space=action_space,
#         hidden_size=512,
#         rnn_type='LSTM',
#         num_recurrent_layers=2,
#         backbone='resnet50',
#         normalize_visual_inputs=False,
#         action_distribution=action_distribution,
#         dim_actions=dim_actions
#     )
#     model.to(torch.device(DEVICE))

#     state_dict = OrderedDict()
#     with open(weights_path, 'r') as f:
#         state_dict = json.load(f)   
#     # state_dict = torch.load(weights_path, map_location=DEVICE) 
#     model.load_state_dict(
#         {
#             k[len("actor_critic.") :]: torch.tensor(v)
#             for k, v in state_dict.items()
#             if k.startswith("actor_critic.")
#         }
#     )

#     return model

# WEIGHTS_PATH = 'habitat_cont_2/habitat_cont/gaussian_noslide_30deg_63_skyfail.json'

# WAYPOINTS_YAML = 'evaluation/waypoints.yaml'

# GAUSSIAN, DISCRETE_4, DISCRETE_6 = False, False, False
# def load_model():
#     depth_256_space = SpaceDict({
#         'depth': spaces.Box(low=0., high=1., shape=(256,256,1)),
#         'pointgoal_with_gps_compass': spaces.Box(
#             low=np.finfo(np.float32).min,
#             high=np.finfo(np.float32).max,
#             shape=(2,),
#             dtype=np.float32,
#         )
#     })

#     DISCRETE_4 = True

#     if GAUSSIAN:
#         action_space = spaces.Box(
#             np.array([float('-inf'),float('-inf')]), np.array([float('inf'),float('inf')])
#         )
#         action_distribution = 'gaussian'
#         dim_actions = 2
#     elif DISCRETE_4:
#         action_space = spaces.Discrete(4)
#         action_distribution = 'categorical'
#         dim_actions = 4
#     elif DISCRETE_6:
#         action_space = spaces.Discrete(6)
#         action_distribution = 'categorical'
#         dim_actions = 6

#     model = PointNavResNetPolicy(
#         observation_space=depth_256_space,
#         action_space=action_space,
#         hidden_size=512,
#         rnn_type='LSTM',
#         num_recurrent_layers=2,
#         backbone='resnet50',
#         normalize_visual_inputs=False,
#         action_distribution=action_distribution,
#         dim_actions=dim_actions
#     )
#     model.to(torch.device("cuda"))

#     data_dict = OrderedDict()
#     with open(WEIGHTS_PATH, 'r') as f:
#         data_dict = json.load(f)    
#     model.load_state_dict(
#         {
#             k[len("actor_critic.") :]: torch.tensor(v)
#             for k, v in data_dict.items()
#             if k.startswith("actor_critic.")
#         }
#     )

#     return model

def to_tensor(v): # DON'T CHANGE
    if torch.is_tensor(v):
        return v
    elif isinstance(v, np.ndarray):
        return torch.from_numpy(v)
    else:
        return torch.tensor(v, dtype=torch.float)

class iGibsonEnv(BaseEnv):
    """
    iGibson Environment (OpenAI Gym interface)
    """

    def __init__(
        self,
        config_file,
        scene_id=None,
        mode='headless',
        action_timestep=1 / 10.0,
        physics_timestep=1 / 240.0,
        device_idx=0,
        render_to_tensor=False,
        automatic_reset=False,
    ):
        """
        :param config_file: config_file path
        :param scene_id: override scene_id in config file
        :param mode: headless, gui, iggui
        :param action_timestep: environment executes action per action_timestep second
        :param physics_timestep: physics timestep for pybullet
        :param device_idx: which GPU to run the simulation and rendering on
        :param render_to_tensor: whether to render directly to pytorch tensors
        :param automatic_reset: whether to automatic reset after an episode finishes
        """
        super(iGibsonEnv, self).__init__(config_file=config_file,
                                         scene_id=scene_id,
                                         mode=mode,
                                         action_timestep=action_timestep,
                                         physics_timestep=physics_timestep,
                                         device_idx=device_idx,
                                         render_to_tensor=render_to_tensor)
        self.automatic_reset = automatic_reset

    def load_task_setup(self):
        """
        Load task setup
        """
        self.initial_pos_z_offset = self.config.get(
            'initial_pos_z_offset', 0.1)
        # s = 0.5 * G * (t ** 2)
        drop_distance = 0.5 * 9.8 * (self.action_timestep ** 2)
        # assert drop_distance < self.initial_pos_z_offset, \
        #     'initial_pos_z_offset is too small for collision checking'

        # ignore the agent's collision with these body ids
        self.collision_ignore_body_b_ids = set(
            self.config.get('collision_ignore_body_b_ids', []))
        # ignore the agent's collision with these link ids of itself
        self.collision_ignore_link_a_ids = set(
            self.config.get('collision_ignore_link_a_ids', []))

        # discount factor
        self.discount_factor = self.config.get('discount_factor', 0.99)

        # domain randomization frequency
        self.texture_randomization_freq = self.config.get(
            'texture_randomization_freq', None)
        self.object_randomization_freq = self.config.get(
            'object_randomization_freq', None)

        # task
        if self.config['task'] == 'point_nav_fixed':
            self.task = PointNavFixedTask(self)
        elif self.config['task'] == 'point_nav_random':
            self.task = PointNavRandomTask(self)
        elif self.config['task'] == 'interactive_nav_random':
            self.task = InteractiveNavRandomTask(self)
        elif self.config['task'] == 'dynamic_nav_random':
            self.task = DynamicNavRandomTask(self)
        elif self.config['task'] == 'reaching_random':
            self.task = ReachingRandomTask(self)
        elif self.config['task'] == 'room_rearrangement':
            self.task = RoomRearrangementTask(self)
        else:
            self.task = None

    def build_obs_space(self, shape, low, high):
        """
        Helper function that builds individual observation spaces
        """
        return gym.spaces.Box(
            low=low,
            high=high,
            shape=shape,
            dtype=np.float32)

    def load_observation_space(self):
        """
        Load observation space
        """
        self.output = self.config['output']
        self.image_width = self.config.get('image_width', 128)
        self.image_height = self.config.get('image_height', 128)
        observation_space = OrderedDict()
        sensors = OrderedDict()
        vision_modalities = []
        scan_modalities = []

        if 'task_obs' in self.output:
            observation_space['task_obs'] = self.build_obs_space(
                shape=(self.task.task_obs_dim,), low=-np.inf, high=-np.inf)
        if 'rgb' in self.output:
            observation_space['rgb'] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 3),
                low=0.0, high=1.0)
            vision_modalities.append('rgb')
        if 'depth' in self.output:
            observation_space['depth'] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 1),
                low=0.0, high=1.0)
            vision_modalities.append('depth')
        if 'pc' in self.output:
            observation_space['pc'] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 3),
                low=-np.inf, high=np.inf)
            vision_modalities.append('pc')
        if 'optical_flow' in self.output:
            observation_space['optical_flow'] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 2),
                low=-np.inf, high=np.inf)
            vision_modalities.append('optical_flow')
        if 'scene_flow' in self.output:
            observation_space['scene_flow'] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 3),
                low=-np.inf, high=np.inf)
            vision_modalities.append('scene_flow')
        if 'normal' in self.output:
            observation_space['normal'] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 3),
                low=-np.inf, high=np.inf)
            vision_modalities.append('normal')
        if 'seg' in self.output:
            observation_space['seg'] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 1),
                low=0.0, high=1.0)
            vision_modalities.append('seg')
        if 'rgb_filled' in self.output:  # use filler
            observation_space['rgb_filled'] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 3),
                low=0.0, high=1.0)
            vision_modalities.append('rgb_filled')
        if 'scan' in self.output:
            self.n_horizontal_rays = self.config.get('n_horizontal_rays', 128)
            self.n_vertical_beams = self.config.get('n_vertical_beams', 1)
            assert self.n_vertical_beams == 1, 'scan can only handle one vertical beam for now'
            observation_space['scan'] = self.build_obs_space(
                shape=(self.n_horizontal_rays * self.n_vertical_beams, 1),
                low=0.0, high=1.0)
            scan_modalities.append('scan')
        if 'occupancy_grid' in self.output:
            self.grid_resolution = self.config.get('grid_resolution', 128)
            self.occupancy_grid_space = gym.spaces.Box(low=0.0,
                                                       high=1.0,
                                                       shape=(self.grid_resolution,
                                                              self.grid_resolution, 1))
            observation_space['occupancy_grid'] = self.occupancy_grid_space
            scan_modalities.append('occupancy_grid')

        if 'bump' in self.output:
            observation_space['bump'] = gym.spaces.Box(low=0.0,
                                                       high=1.0,
                                                       shape=(1,))
            sensors['bump'] = BumpSensor(self)

        if len(vision_modalities) > 0:
            sensors['vision'] = VisionSensor(self, vision_modalities)

        if len(scan_modalities) > 0:
            sensors['scan_occ'] = ScanSensor(self, scan_modalities)

        self.observation_space = gym.spaces.Dict(observation_space)
        self.sensors = sensors

    def load_action_space(self):
        """
        Load action space
        """
        self.action_space = self.robots[0].action_space

    def load_miscellaneous_variables(self):
        """
        Load miscellaneous variables for book keeping
        """
        self.current_step = 0
        self.collision_step = 0
        self.current_episode = 0
        self.collision_links = []

    def load(self):
        """
        Load environment
        """
        super(iGibsonEnv, self).load()
        self.load_task_setup()
        self.load_observation_space()
        self.load_action_space()
        self.load_miscellaneous_variables()

    def get_state(self, collision_links=[]):
        """
        Get the current observation

        :param collision_links: collisions from last physics timestep
        :return: observation as a dictionary
        """
        state = OrderedDict()
        if 'task_obs' in self.output:
            state['task_obs'] = self.task.get_task_obs(self)
        if 'vision' in self.sensors:
            vision_obs = self.sensors['vision'].get_obs(self)
            for modality in vision_obs:
                state[modality] = vision_obs[modality]
        if 'scan_occ' in self.sensors:
            scan_obs = self.sensors['scan_occ'].get_obs(self)
            for modality in scan_obs:
                state[modality] = scan_obs[modality]
        if 'bump' in self.sensors:
            state['bump'] = self.sensors['bump'].get_obs(self)

        return state

    def run_simulation(self):
        """
        Run simulation for one action timestep (same as one render timestep in Simulator class)

        :return: collision_links: collisions from last physics timestep
        """
        self.simulator_step()
        collision_links = list(p.getContactPoints(
            bodyA=self.robots[0].robot_ids[0]))
        return self.filter_collision_links(collision_links)

    def filter_collision_links(self, collision_links):
        """
        Filter out collisions that should be ignored

        :param collision_links: original collisions, a list of collisions
        :return: filtered collisions
        """
        new_collision_links = []
        for item in collision_links:
            # ignore collision with body b
            if item[2] in self.collision_ignore_body_b_ids:
                continue

            # ignore collision with robot link a
            if item[3] in self.collision_ignore_link_a_ids:
                continue

            # ignore self collision with robot link a (body b is also robot itself)
            if item[2] == self.robots[0].robot_ids[0] and item[4] in self.collision_ignore_link_a_ids:
                continue
            new_collision_links.append(item)
        return new_collision_links

    def populate_info(self, info):
        """
        Populate info dictionary with any useful information
        """
        info['episode_length'] = self.current_step
        info['collision_step'] = self.collision_step

    def step(self, action):
        """
        Apply robot's action.
        Returns the next state, reward, done and info,
        following OpenAI Gym's convention

        :param action: robot actions
        :return: state: next observation
        :return: reward: reward of this time step
        :return: done: whether the episode is terminated
        :return: info: info dictionary with any useful information
        """
        self.current_step += 1
        if action is not None:
            self.robots[0].apply_action(action)
        collision_links = self.run_simulation()
        self.collision_links = collision_links
        self.collision_step += int(len(collision_links) > 0)

        state = self.get_state(collision_links)
        info = {}
        reward, info = self.task.get_reward(
            self, collision_links, action, info)
        done, info = self.task.get_termination(
            self, collision_links, action, info)
        self.task.step(self)
        self.populate_info(info)

        if done and self.automatic_reset:
            info['last_observation'] = state
            state = self.reset()

        return state, reward, done, info

    def check_collision(self, body_id):
        """
        Check with the given body_id has any collision after one simulator step

        :param body_id: pybullet body id
        :return: whether the given body_id has no collision
        """
        self.simulator_step()
        collisions = list(p.getContactPoints(bodyA=body_id))

        if logging.root.level <= logging.DEBUG:  # Only going into this if it is for logging --> efficiency
            for item in collisions:
                logging.debug('bodyA:{}, bodyB:{}, linkA:{}, linkB:{}'.format(
                    item[1], item[2], item[3], item[4]))

        return len(collisions) == 0

    def set_pos_orn_with_z_offset(self, obj, pos, orn=None, offset=None):
        """
        Reset position and orientation for the robot or the object

        :param obj: an instance of robot or object
        :param pos: position
        :param orn: orientation
        :param offset: z offset
        """
        if orn is None:
            orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])

        if offset is None:
            offset = self.initial_pos_z_offset

        is_robot = isinstance(obj, BaseRobot)
        body_id = obj.robot_ids[0] if is_robot else obj.body_id
        # first set the correct orientation
        obj.set_position_orientation(pos, quatToXYZW(euler2quat(*orn), 'wxyz'))
        # compute stable z based on this orientation
        stable_z = stable_z_on_aabb(body_id, [pos, pos])
        # change the z-value of position with stable_z + additional offset
        # in case the surface is not perfect smooth (has bumps)
        obj.set_position([pos[0], pos[1], stable_z + offset])

    def test_valid_position(self, obj, pos, orn=None):
        """
        Test if the robot or the object can be placed with no collision

        :param obj: an instance of robot or object
        :param pos: position
        :param orn: orientation
        :return: validity
        """
        is_robot = isinstance(obj, BaseRobot)

        self.set_pos_orn_with_z_offset(obj, pos, orn)

        if is_robot:
            obj.robot_specific_reset()
            obj.keep_still()

        body_id = obj.robot_ids[0] if is_robot else obj.body_id
        has_collision = self.check_collision(body_id)
        return has_collision

    def land(self, obj, pos, orn):
        """
        Land the robot or the object onto the floor, given a valid position and orientation

        :param obj: an instance of robot or object
        :param pos: position
        :param orn: orientation
        """
        is_robot = isinstance(obj, BaseRobot)

        self.set_pos_orn_with_z_offset(obj, pos, orn)

        if is_robot:
            obj.robot_specific_reset()
            obj.keep_still()

        body_id = obj.robot_ids[0] if is_robot else obj.body_id

        land_success = False
        # land for maximum 1 second, should fall down ~5 meters
        max_simulator_step = int(1.0 / self.action_timestep)
        for _ in range(max_simulator_step):
            self.simulator_step()
            if len(p.getContactPoints(bodyA=body_id)) > 0:
                land_success = True
                break

        if not land_success:
            print("WARNING: Failed to land")

        if is_robot:
            obj.robot_specific_reset()
            obj.keep_still()

    def reset_variables(self):
        """
        Reset bookkeeping variables for the next new episode
        """
        self.current_episode += 1
        self.current_step = 0
        self.collision_step = 0
        self.collision_links = []

    def randomize_domain(self):
        """
        Domain randomization
        Object randomization loads new object models with the same poses
        Texture randomization loads new materials and textures for the same object models
        """
        if self.object_randomization_freq is not None:
            if self.current_episode % self.object_randomization_freq == 0:
                self.reload_model_object_randomization()
        if self.texture_randomization_freq is not None:
            if self.current_episode % self.texture_randomization_freq == 0:
                self.simulator.scene.randomize_texture()

    def reset(self):
        """
        Reset episode
        """
        self.randomize_domain()
        # move robot away from the scene
        self.robots[0].set_position([100.0, 100.0, 100.0])
        self.task.reset_scene(self)
        self.task.reset_agent(self)
        self.simulator.sync()
        state = self.get_state()
        self.reset_variables()

        return state


ACTION_DIM = 2
LINEAR_VEL_DIM = 0
ANGULAR_VEL_DIM = 1

num_processes = 1

index = 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        '-c',
        help='which config file to use [default: use yaml files in examples/configs]')
    parser.add_argument('--mode',
                        '-m',
                        choices=['headless', 'gui', 'iggui'],
                        default='headless',
                        help='which mode for simulation (default: headless)')
    args = parser.parse_args()

    model = load_model(
        weights_path = WEIGHTS_PATH,
        dim_actions = 2
    )
    # model = load_model()
    model = model.eval()

    env = iGibsonEnv(config_file=args.config,
                     mode=args.mode,
                     device_idx=int(os.environ['CUDA_VISIBLE_DEVICES']),
                     action_timestep=1.0 / 10.0,
                     physics_timestep=1.0 / 240.0)

    step_time_list = []
    # for episode in range(100):
    #     print('Episode: {}'.format(episode))
    #     start = time.time()
    #     env.reset()
    #     for _ in range(100):  # 10 seconds
    #         action = env.action_space.sample()
    #         state, reward, done, _ = env.step(action)
    #         #print(action, state)
    #         print('reward', reward)
    #         if done:
    #             break
    #     print('Episode finished after {} timesteps, took {} seconds.'.format(
    #         env.current_step, time.time() - start))
    # env.close()

    num_processes = 1

    # Will have to change this:
    #goal_location = np.array([float(goal_list_x[i]), float(goal_list_y[i]) ], dtype=np.float32)

    res = []

    for episode in range(10):
        index = 0
        # path = f'/nethome/qluo49/iGibsonChallenge2021/picture_{episode}/'
        # if not os.path.exists(path):
        #     os.makedirs(path)
        test_recurrent_hidden_states = torch.zeros(
            model.net.num_recurrent_layers,
            num_processes,
            512,
            device=DEVICE,
        )

        prev_actions = torch.zeros(num_processes, 2, device=DEVICE)
        not_done_masks = torch.zeros(num_processes, 1, device=DEVICE)

        print('Episode: {}'.format(episode))
        start = time.time()
        state1 = env.reset()
        for _ in range(500):  # 10 seconds
            state = OrderedDict()
            state['depth'] = state1['depth']
            state['pointgoal_with_gps_compass'] = state1['task_obs'][:2]
            state = [state]

            index += 1
            # frame = observations_to_image(state1)
            # root = f'/nethome/qluo49/iGibsonChallenge2021/picture_{episode}/{index}.png'
            # mp.imsave(root,frame)

            batch = defaultdict(list)
            #print(state)
            for obs in state:
                for sensor in obs: 
                    batch[sensor].append(to_tensor(obs[sensor]))


            for sensor in batch:
                batch[sensor] = torch.stack(batch[sensor], dim=0).to(
                    device=DEVICE, dtype=torch.float
                )
            with torch.no_grad():
                # Get new action and LSTM hidden state
                _, action, _, test_recurrent_hidden_states = model.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=True,
                )

            prev_actions.copy_(action)

            # action1 = np.array(action.cpu()[0])[0]
            # if action1 == 0:
            #     action1 = np.array([4])
            # elif action1 == 1:
            #     action1 = np.array([0])
            # elif action1 == 2:
            #     action1 = np.array([3])
            # elif action1 == 3:
            #     action1 = np.array([2])
            # state1, reward, done, _ = env.step(action1)

            move_amount = torch.clip(action[0][0], min=-1, max=1).item()
            turn_amount = torch.clip(action[0][1], min=-1, max=1).item()
            move_amount = (move_amount+1.)/2.
            action1 = np.array([ move_amount, turn_amount])

            # move_amount = -torch.tanh(action[0][0]).item()
            # turn_amount = torch.tanh(action[0][1]).item()
            # move_amount = (move_amount+1.)/2.
            # action1 = np.array([ 0.25 * move_amount, 0.16 * turn_amount])
            # print(action1)
            state1, reward, done, _ = env.step(action1)

            not_done_masks = torch.ones(num_processes, 1, device=DEVICE)
            if done:
                break
        # res.append(state1['task_obs'][:2])
        # print('reward', reward)
        # print('dis', state1['task_obs'][:2])
        # print('Episode finished after {} timesteps, took {} seconds.'.format(
        #     env.current_step, time.time() - start))
        
        # videoPath = f'/nethome/qluo49/iGibsonChallenge2021/videos/{res[-1][0]}.avi' 
        # Pic2Video(path, videoPath) 
    # for m in range(30):
    #     print(m, res[m])
    env.close()