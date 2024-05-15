import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation
import random


class OT2Env(gym.Env):
    def __init__(self, render=False, max_steps=1000):
        super(OT2Env, self).__init__()
        self.render = render
        self.max_steps = max_steps

        # Create simulation env
        self.sim = Simulation(num_agents=1, render=render)

        self.action_space = spaces.Box(low=np.array([-1, -1, -1]), high=np.array([1, 1, 1]), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        self.steps = 0
        self.done = False
        self.prev_eucl = 0
        self.stuck = 0
        self.pip_pos = 0
        self.goal_position = 0


    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.goal_position = (round(random.uniform(-0.187, 0.253), 3), round(random.uniform(-0.1705, 0.2195), 3), round(random.uniform(0.1694, 0.2894), 3))


        self.observation = self.sim.reset(num_agents=1)

        data_keys = self.observation.keys()
        self.robotid =  next(iter(data_keys))

        pip_pos = np.array(self.observation[self.robotid]["pipette_position"], dtype=np.float32)
        
        self.observation = np.append(pip_pos, self.goal_position)
        self.observation = np.array(self.observation, dtype=np.float32)

        self.steps = 0
        self.finished = 0
        return self.observation, {}

    def step(self, action, location=None):
        action = np.append(action, 0)


        if location != None:
            self.goal_position = location
        self.observation = self.sim.run([action])
        pip_pos = np.array(self.observation[self.robotid]["pipette_position"], dtype=np.float32)
        self.pip_pos = pip_pos
        self.observation = np.append(pip_pos, self.goal_position)
        self.observation = np.array(self.observation, dtype=np.float32)


        eucl_distance = -np.linalg.norm(pip_pos - self.goal_position)

        if abs(eucl_distance) < 0.1:
            plus_reward = abs(eucl_distance)
            self.reward =+ (10 / plus_reward - 1) / 500
        else:
            self.reward =+ -abs(eucl_distance)
 
        if abs(eucl_distance - self.prev_eucl) <= 0.001:
            self.stuck += 1
        else:
            self.stuck = 0
        
        if self.stuck > 15:
            self.reward -= 5

        self.prev_eucl = eucl_distance



        tolerance = 1e-3
        if (abs(pip_pos[0] - self.goal_position[0]) < tolerance and
            abs(pip_pos[1] - self.goal_position[1]) < tolerance and
            abs(pip_pos[2] - self.goal_position[2]) < tolerance):
            self.reward += (abs((self.steps - 1000) * 2) + 500) * 1000
            terminated = True
            self.finished += 1
            info = eucl_distance, self.steps
            self.steps = 0
            

        else:
            terminated = False
            info = {}

        if self.steps > self.max_steps:
            truncated = True
            info = eucl_distance, self.steps
            if self.finished == 0:
                self.reward += self.reward * 10
            else:
                self.reward += self.finished * 1000
        else:
            truncated = False
    
        self.steps += 1

        return self.observation, self.reward, terminated, truncated, info
    
    def render(self, mode="human"):
        pass

    def drop(self):
        self.sim.run([[0, 0, 0, 1]])
        for x in range(20):
            self.sim.run([[0,0,0,0]])

    def rendering(self):
        for x in range(40):
            self.sim.run([[0, 0, 0, 0]])

    def get_img(self):
        image_path = self.sim.get_plate_image()
        return(image_path)

    def close(self):
        self.sim.close()

