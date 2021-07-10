import numpy as np
from .utils import direction_xy, d2xy

class Vehicle:
    """
    Define general vehicle class.
    
    Params:
    - destination: destination approach ID;
    - approach_seq: the approach sequence that the vehicle would passby in the roundabout. Empty when created. But the sequence would be generated in terms of the origin approach  (not included in the sequence) and destination approach
    - slotID: default value = 12 when created, which means that the vehicle is not bonded with any slots. The meaningful value would be assigned when the vehicle is assinged to a certain slot.
    """
    
    def __init__(self, 
                 x, 
                 y, 
                 v,
                 d):
        self.x = x
        self.y = y
        self.v = v
        self.approach_seq = []
        self.slot = None 
        self.d = d

    def generate_approach_seq(self, ori_approach, des_approach):
        if des_approach > ori_approach:
            self.approach_seq = [i for i in range(ori_approach + 1, des_approach + 1)]
        else:
            self.approach_seq = [np.mod(i, 4) for i in range(ori_approach+1, des_approach + 5)]

    def update(self, a, step_size, approach, boundary, R):
        if self.d == -1:
            self.v = self.slot.virtual_vehicle.omega * R
            theta = self.slot.virtual_vehicle.theta
            out_theta = np.mod(self.approach_seq[0]*np.pi/2 - np.pi/6, np.pi*2)
            if theta > out_theta and theta < out_theta + np.pi/12:
                self.approach_seq.remove(self.approach_seq[0])
            self.x, self.y = (R*np.cos(theta), R*np.sin(theta))
        else:
            self.d += self.v*step_size + a*step_size**2/2
            self.v += a*step_size
            self.x, self.y = direction_xy[approach](d2xy(self.d, boundary, R))

    def leave(self, step_size, approach, boundary, R):
        self.d -= self.v*step_size
        self.x, self.y = direction_xy[approach](d2xy(self.d, boundary, R, True))

class VirtualVihicle:
    def __init__(self, 
                 veh,
                 theta,
                 omega
                 ):

        self.veh = veh
        self.theta = theta
        self.omega = omega
        self.angular_a = np.NaN
    
    def update(self, step_size):
        self.theta = np.mod(self.theta + self.omega*step_size + self.angular_a*step_size**2/2, np.pi*2)
        self.omega += self.angular_a*step_size