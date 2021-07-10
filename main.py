from roundabout.utils import test_plot
from roundabout.utils import test_ani, delta_plot
from roundabout.env import Env
import numpy as np
from roundabout.utils import test_ani
import matplotlib
matplotlib.use('Agg')
np.seterr(all='raise')

envParams = {
    "a": 3, 
    "v": 8,
    "C": 3,
    "veh_length": 5,
    "headway": 3,
    "Q": np.array([30, 25, 35, 20]),
    "step_size": 0.05,
    "boundary": 250,
}
initParams = {
    #"l_0": np.array([10, 0, 0, 0]),  
    "l_0": np.array([14, 15, 13, 14]),  
}

env = Env(envParams)
env.initialize(initParams)



test_ani(env, 6000)