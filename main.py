import numpy as np

envParams = {"a": 3, 
             "v": 8,
             "C": 3,
             "veh_length": 5,
             "headway": 3}

initParams = {"l_0": np.array([4, 5, 3, 4]),
              "Q": np.array([18, 25, 20, 20])  
}

V0 = np.ones(4)*v      # the average speeds of vehicles generated on each approaches
V0_std = np.ones(4)    # std of the speeds of vehicles generated on each approaches

