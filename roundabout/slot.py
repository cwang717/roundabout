from .utils import next_a
import numpy as np
from numpy.lib import emath

class Slot:
    """
    Define the slot class.
    
    """
    def __init__(self,
                 slot_id,
                 R, 
                 theta):
        self.id = slot_id
        self.theta = np.mod(theta, 2*np.pi)
        self.R = R

        self.x = R*np.cos(self.theta)
        self.y = R*np.sin(self.theta)
        self.next_approach = (int(self.theta // (np.pi/2)) + 1) % 4
        self.virtual_vehicle = None
        self.veh = []
        self.empty = True

    def rotate(self, angle):
        "return if pass an approach"
        self.theta = np.mod(self.theta + angle, 2*np.pi)
        self.x = self.R*np.cos(self.theta)
        self.y = self.R*np.sin(self.theta)
        initial_next = self.next_approach
        self.next_approach = (int(self.theta // (np.pi/2)) + 1) % 4
        if self.virtual_vehicle is not None and len(self.veh) == 0:
            if next_a(self.next_approach) not in self.virtual_vehicle.veh.approach_seq or \
               next_a(self.next_approach) == self.virtual_vehicle.veh.approach_seq[-1]:
                self.empty = True
        return not initial_next == self.next_approach

    
