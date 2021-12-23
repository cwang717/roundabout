import gym
import numpy as np
from collections import OrderedDict
from .env import Env

class RLRoundabout(gym.Env):
    def __init__(self, env_config):
        envParams, initParams, RLParams = env_config["envParams"], env_config["initParams"], env_config["RLParams"] 
        self.env = Env(envParams)
        self.initParams = initParams
        self.desired_steps = RLParams['desired_steps']
        self.warmup_steps = RLParams['warmup_steps']
        self.Q_upperbound = RLParams['Q_upperbound']
        self.Q_lowerbound = RLParams['Q_lowerbound']
        self.Q_randomness = RLParams['Q_randomness']
        self.action_space = gym.spaces.Box(low = 0.0, high = 1.0, shape = (4,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict({"new_vehicles": gym.spaces.MultiBinary([4,]),
                                                  "approaching_vehicles": gym.spaces.Box(low=0, high=100, shape = (4,), dtype=np.int64),
                                                  "slot_occupied_rate": gym.spaces.Box(low=0, high=1, shape = (1,), dtype=np.float32),
                                                  "current_queue": gym.spaces.Box(low = 0, high = 30, shape = (4,), dtype=np.int64)})
        
        self.env.initialize(self.initParams)
    
    def step(self, action):
        
        self.env.P = action
        self.env.Q = np.random.uniform(self.Q_lowerbound, self.Q_upperbound, 4) + np.random.normal(0, self.Q_randomness, 4)
        self.env.step()

        state = dict()
        state["approaching_vehicles"] = np.array(list(map(len, self.env.approaching_vehicles)), dtype=np.int64)
        state["current_queue"] = np.array(self.env.queue_length, dtype=np.int64)
        state["new_vehicles"] = np.array(self.env.new_vehicles)
        state["slot_occupied_rate"] = np.array([1 - sum(list(map(lambda slot: slot.virtual_vehicle is None, self.env.slots)))/self.env.numSlots], dtype=np.float32)
        state = OrderedDict(state)
        
        reward = 1 - 0.001 * sum(np.square(self.env.queue_length))
        done = (self.env.num_step >= self.desired_steps)
        if any(state["current_queue"] >= 30):
            done = True
            reward = 0

        info = {}        
        
        return state, reward, done, info
    
    def reset(self): 
        self.env.initialize(self.initParams)

        # warmup steps
        for _ in range(self.warmup_steps):
            state, _, _, _ = self.step(action = np.ones(4))
        
        return state