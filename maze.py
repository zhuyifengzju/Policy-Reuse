"""Discrete MDP maze environment."""
# coding: utf-8
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

from gym.core import Env
from gym.spaces import Discrete, Box
from gym.utils import seeding

import pygame

import numpy as np
import math
import os

import typing
from enum import Enum
import math

from PIL import Image

class Color(Enum):
    WHITE=(255, 255, 255)
    BLACK=(0, 0, 0)
    RED=(255, 0, 0)
    BLUE=(0, 0, 255)

DiscreteActionMap = [(1 ,  0),
                     (0 ,  1),
                     (-1,  0),
                     (0 , -1)]
class Agent():
    def __init__(self,
                 x=1.0,
                 y=1.0,
                 scale=1,
                 x_upper_limit=10,
                 y_upper_limit=10):
        self.x = x
        self.y = y
        self.x_lower_limit = 0
        self.y_lower_limit = 0
        self.x_upper_limit = x_upper_limit
        self.y_upper_limit = y_upper_limit
        
        self.scale = scale
        self.Rect = pygame.Rect(math.floor(x) * scale,
                                math.floor(y) * scale,
                                scale,
                                scale)

    def move(self,
             mv_x=0,
             mv_y=0,
             obs=None):
        flag = False
        valid = True

        if ((mv_x, mv_y) not in DiscreteActionMap):
            valid = False
        else:
            temp_x = self.x + mv_x
            temp_y = self.y + mv_y

            if obs is not None:
                if [math.floor(temp_x), math.floor(temp_y)] in obs:
                    flag = True
                    valid = False
                    temp_x = self.x
                    temp_y = self.y
            self.x = temp_x
            self.y = temp_y

            self.x = max(min(self.x, self.x_upper_limit-1), self.x_lower_limit)
            self.y = max(min(self.y, self.y_upper_limit-1), self.y_lower_limit)

            """Add noise to final position"""
            noise_x, noise_y = np.random.uniform(-0.2, 0.2),np.random.uniform(-0.2, 0.2)
            print(f'Noise {noise_x}, {noise_y}')
            print(f'Agent before noise {self.x}, {self.y}')
            print(f'{math.floor(noise_x + self.x), math.floor(noise_y + self.y)}')
            if [math.floor(noise_x + self.x), math.floor(noise_y + self.y)] not in obs:
                self.x = self.x + noise_x
                self.y = self.y + noise_y
            print(f'Agent after noise {self.x}, {self.y}')
            
            self.Rect.left = math.floor(self.x) * self.scale
            self.Rect.top = math.floor(self.y) * self.scale
        return flag, valid

class Maze(Env):
    """
    In this environment, an agent is only allowed to move right, down, left, up. In default, the action is labeled as 0, 1, 2, 3
    """
    def __init__(self,
                 height=21,
                 width=24,
                 scale=8,
                 channel=3,
                 evaluate=False,
                 mask_fn='identity',
                 use_image_input = False,
                 use_discrete_state = False,
                 use_image_action=False,
                 use_grey_image=False,
                 random_init_pos=False,
                 max_steps=600):
        """
        height: vertical direction
        width: horizontal direction
        """
        self.height = height
        self.width = width
        self.scale = scale
        self.channel = channel
        self.evaluate = evaluate
        self.use_image_input = use_image_input
        self.use_discrete_state = use_discrete_state
        self.use_image_action = use_image_action
        self.use_grey_image = use_grey_image
        self.random_init_pos = random_init_pos
        self.max_steps = max_steps

        if mask_fn == 'identity':
            self.mask_fn = self.mask_identity_fn
        elif mask_fn == 'masking':
            self.mask_fn = self.mask_move_action_space_fn
            
        self.obs_coords = read_map()
        self.obs = [pygame.Rect(obs_x*scale,obs_y*scale,scale,scale) for (obs_x, obs_y) in self.obs_coords]
        # Agent
        self.agent_init_x = 1.0
        self.agent_init_y = 1.0
        self.agent = Agent(x=self.agent_init_x,
                           y=self.agent_init_y,
                           scale=self.scale,
                           x_upper_limit=self.width,
                           y_upper_limit=self.height,
        )
        # Goal
        # Read from config
        self.goalx, self.goaly = read_goal()
        self.goal = pygame.Rect(self.goalx * scale,
                                self.goaly * scale,
                                scale,
                                scale)
        # Initialize pygame
        try:
            self.rendering = pygame.display.set_mode((self.width * self.scale,
                                                     self.height * self.scale),
                                                    0,
                                                    32)
            pygame.display.set_caption("Maze")
        except:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            self.rendering = pygame.display.set_mode((self.width * self.scale,
                                                     self.height * self.scale),
                                                    0,
                                                    32)
            pygame.display.set_caption("Maze")

        # Define observation and action space

        if use_image_action is True:
            self.action_space = Discrete(self.height * self.width)
        else:
            self.action_space = Discrete(len(DiscreteActionMap))

        self.observation_space = Box(low=0,
                                     high=255,
                                     shape=(self.width * self.scale,
                                            self.height * self.scale,
                                            self.channel))
        
    def _update(self):
        """Update environment."""

        # Background is White
        self.rendering.fill(Color.WHITE.value)

        # Obstacle are Black
        [pygame.draw.rect(self.rendering, Color.BLACK.value, obs)
         for obs in self.obs]
        # Goal is Red
        pygame.draw.rect(self.rendering,
                         Color.RED.value,
                         self.goal)
        # Agent is Blue
        pygame.draw.rect(self.rendering,
                         Color.BLUE.value,
                         self.agent.Rect)
        
    def render(self):
        pygame.display.flip()
        
    def reset(self):
        # Reset initial states

        # reset start state
        # Generate random position
        if self.random_init_pos:
            x = np.random.randint(self.width)
            y = np.random.randint(self.height)
            while ([x, y] in self.obs_coords):
                x = np.random.randint(self.width)
                y = np.random.randint(self.height)

            self.agent_init_x = x
            self.agent_init_y = y
        
        self.agent = Agent(x=self.agent_init_x,
                           y=self.agent_init_y,
                           scale=self.scale,
                           x_upper_limit=self.width,
                           y_upper_limit=self.height,
        )

        # reset goal state

        self.reach_goal = False
        self.steps = 0
        self._update()
        return self.get_states()
    
    def step(self, action):

        self.steps += 1
        obs_collide_flag = True
        
        if self.use_image_action is True:
            action = (math.floor(action / self.width) - self.agent.x, action % self.width - self.agent.y)
        else:
            action = DiscreteActionMap[action]

        obs_collide_flag, valid_action_flag = self.agent.move(*action,
                                           obs=self.obs_coords)
        self._update()

        if (self.agent.x == self.goalx and self.agent.y == self.goaly):
            self.reach_goal = True
        next_state = self.get_states()
        if self.use_discrete_state is True:
            next_state = int(next_state)
        reward = self._reward()
        if (self.use_image_action is True
            and valid_action_flag is False and self.evaluate is False):
            reward = -1
        is_terminal = self._is_terminal()
        debug_info = {'obs_collide': obs_collide_flag,
                      'reach_goal': self.reach_goal,
                      'valid_action':valid_action_flag}
        return next_state, reward, is_terminal, debug_info

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reward(self):
        if (self.reach_goal is True):
            return 1
        elif self.steps <= self.max_steps:
            return 0
        
    def _is_terminal(self):
        if (self.reach_goal is True or self.steps > self.max_steps):
            return True
        else:
            return False
    
    def get_raw_img(self):
        pygame.display.flip()
        img = pygame.surfarray.array3d(self.rendering)

        if self.use_grey_image is True:
            img = Image.fromarray(img).convert('L')
            img = np.array(img)[..., np.newaxis]

        return img

    def get_states(self):
        """Array returned in uint8."""
        if self.use_image_input is True:
            states = self.mask_fn(self.get_raw_img())
            states = np.array(states, dtype='uint8')
        else:
            states = np.array([self.agent.x,
                              self.agent.y])            
            if self.use_discrete_state is True:
                states = np.floor(states)
        return states
        
    def close(self):
        pygame.quit()
    

    def mask_identity_fn(self, arr):
            return arr

    def mask_move_action_space_fn(self, arr):
        """Add one channel masking to the move action space"""
        valid_action_space = np.zeros(shape=arr.shape[:2]+(1,))
        x, y = self.agent.x, self.agent.y
        x0 = max(self.agent.x_lower_limit, x - 1)
        y0 = max(self.agent.y_lower_limit, y - 1)
        x1 = min(self.agent.x_upper_limit-1, x + 1)
        y1 = min(self.agent.y_upper_limit-1, y + 1)

        if [x0, y] in self.obs_coords:
            x0 = x
        if [x1, y] in self.obs_coords:
            x1 = x
        if [x, y0] in self.obs_coords:
            y0 = y
        if [x, y1] in self.obs_coords:
            y1 = y 
        
        valid_action_space[x * self.scale:(x+1) * self.scale,
                           y0 * self.scale:(y1+1) * self.scale, 0] = 255
        valid_action_space[x0 * self.scale:(x1+1) * self.scale,
                           y * self.scale:(y+1) * self.scale, 0] = 255

        return np.concatenate((arr, valid_action_space), axis=-1)


def read_map():
    from functools import partial

    """Read map"""
    filename = 'maze.cfg'

    def list_duplicates(item, seq):
        start = -1
        locs = []
        while True:
            try:
                loc = seq.index(item, start+1)
            except ValueError:
                break
            else:
                locs.append(loc)
                start = loc
        return locs

    find_duplicates = partial(list_duplicates, '#')
    obs = []
    with open(filename, 'r') as f:
        lines = [line.rstrip('\n') for line in f]
        i = 0
        for line in lines:
            x = find_duplicates(list(line))
            obs = obs + [[loc_x, i] for loc_x in x]
            i = i + 1

    return obs

def read_goal():
    """TODO: Read from a config file"""

    return 18, 1
