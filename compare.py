from roundabout.utils import test_plot
from roundabout.utils import test_ani, delta_plot
from roundabout.env import Env
import numpy as np
from roundabout.utils import test_ani
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 18})
np.seterr(all='raise')

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

env = Env(envParams)
env.initialize(initParams)



test_ani(env, 2400, "compare-68")
env.save_records("compare_records-68")