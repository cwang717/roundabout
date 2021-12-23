import ray
import numpy as np
# from ray.rllib.agents import ppo
from ray.tune import run_experiments
from roundabout.gym_env import RLRoundabout

envParams = {
    "a": 3, 
    "v": 8,
    "C": 3,
    "veh_length": 5,
    "headway": 3,
    "Q": np.array([35, 35, 35, 35])*6/8,
    "step_size": 0.05,
    "boundary": 250,
    "eta": np.array([[0.0, 0.0, 5/7, 2/7],
                    [2/7, 0.0, 0.0, 5/7],
                    [5/7, 2/7, 0.0, 0.0],
                    [0.0, 5/7, 2/7, 0.0]]),
    "fifo": True
}

initParams = {
    "l_0": np.array([0, 0, 0, 0]),  
    #"l_0": np.array([14, 15, 13, 14]),  
}


RLParams = {'desired_steps': 2400,
            'warmup_steps': 800,
            'Q_upperbound': 35,
            'Q_lowerbound': 20,
            'Q_randomness': 3}

ray.init(num_cpus = 8)
# trainer = ppo.PPOTrainer(env=RLRoundabout, config={
#     "framework": "torch",
#     "num_workers": 9,
#     "checkpoint_freq": 20,
#     "checkpoint_at_end": True,
#     "env_config": {"envParams": envParams,
#                 "initParams": initParams,
#                 "RLParams": RLParams},
# })

exp_config = {
    "run": "PPO",
    "env": RLRoundabout,
    "config": {
        "framework": "torch",
        "num_workers": 1,
        "env_config": {"envParams": envParams,
                       "initParams": initParams,
                       "RLParams": RLParams},
    },
    "num_samples": 2,
    "checkpoint_freq": 20,
    "checkpoint_at_end": True,
    "max_failures": 999,
}

run_experiments({
    "RLroundabout": exp_config
    })
