import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from numpy import lib

direction_xy = [lambda x: x,
                lambda x: [-x[1], x[0]],
                lambda x: [-x[0], -x[1]],
                lambda x: [x[1], -x[0]]]

def delta_plot(env):
    x_lb, x_ub = -200, 200
    y_lb, y_ub = -200, 200

    fig = plt.figure()
    plt.plot(np.linspace(1, env.num_step, env.num_step), np.array(env.records)[:, 0])
    
    plt.savefig("records.png")

def test_plot(env):
    
    x_lb, x_ub = -200, 200
    y_lb, y_ub = -200, 200

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(x_lb, x_ub), ylim=(y_lb, y_ub))

    env.test_plot(ax)

    plt.savefig("test.png")

def test_ani(env, steps):
    fig =plt.figure(figsize=(15, 15))

    env.ani_save(fig, steps)

def next_a(approach):
    return np.mod(approach + 1, 4)

def d2xy(d, boundary, R, leave = False):
    if d > boundary - np.sqrt(3)*R: 
        alpha = (d - boundary + np.sqrt(3)*R)/R
        if leave:
            return (np.sqrt(3)*R - R*np.sin(alpha), - R + R*np.cos(alpha))
        else:
            return (np.sqrt(3)*R - R*np.sin(alpha), R - R*np.cos(alpha))
    else:
        return (boundary - d, 0)

def adjust_a(veh, desired_d, desired_t, desired_v, a_ub):
    v0 = veh.v
    s = desired_d - veh.d
    vf = desired_v
    tf = desired_t

    t2 = np.abs(v0 - vf)/a_ub
    t1 = tf - t2
    if t1 < 0:
        return a_ub if v0<=vf else -a_ub
    critical_distance = v0*t1 + 1/2*(v0+vf)*t2
#     if np.abs(critical_distance - s) < 0.3:
#         return 0
    return a_ub if s>critical_distance else -a_ub

def isFeasible(veh, a, v, boundary, R):
    if np.abs(v-veh.v)/a > 5:
        return False

    if (veh.v + v)/a < 5:
        lb = veh.v**2/2/a + v**2/2/a
    else:
        tempv = (veh.v + v - 5*a)/2
        lb = (veh.v**2 + v**2 - 2*tempv**2)/2/a
    tempv = (veh.v + v + 5*a)/2
    ub = (2*tempv**2 - veh.v**2 - v**2)/2/a

    s = boundary - np.sqrt(3)*R - veh.d
    return s > lb and s < ub

class IDMController():
    """Intelligent Driver Model (IDM) controller.

    For more information on this controller, see:
    Treiber, Martin, Ansgar Hennecke, and Dirk Helbing. "Congested traffic
    states in empirical observations and microscopic simulations." Physical
    review E 62.2 (2000): 1805.

    Usage
    -----
    See BaseController for usage example.

    Attributes
    ----------
    veh_id : str
        Vehicle ID for SUMO identification
    car_following_params : flow.core.param.SumoCarFollowingParams
        see parent class
    v0 : float
        desirable velocity, in m/s (default: 30)
    T : float
        safe time headway, in s (default: 1)
    a : float
        max acceleration, in m/s2 (default: 1)
    b : float
        comfortable deceleration, in m/s2 (default: 1.5)
    delta : float
        acceleration exponent (default: 4)
    s0 : float
        linear jam distance, in m (default: 2)
    noise : float
        std dev of normal perturbation to the acceleration (default: 0)
    fail_safe : str
        type of flow-imposed failsafe the vehicle should posses, defaults
        to no failsafe (None)
    """

    def __init__(self,
                 stop_line,
                 v0=30,
                 T=1,
                 a=3,
                 b=3,
                 delta=4,
                 s0=4,
                 time_delay=0.0,
                 noise=0,
                 fail_safe=None,
                 display_warnings=True,
                 car_following_params=None):
        self.v0 = v0
        self.T = T
        self.a = a
        self.b = b
        self.delta = delta
        self.s0 = s0
        self.stop_line = stop_line

    def get_accel(self, veh, lead_id):
        """See parent class."""
        v = veh.v

        if lead_id is None:
            h = self.stop_line - veh.d
        else:
            h = min(lead_id.d, self.stop_line) - veh.d

        # in order to deal with ZeroDivisionError
        if abs(h) < 1e-3:
            h = 1e-3

        lead_vel = lead_id.v if lead_id is not None else 1e-3
        s_star = self.s0 + max(
            0, v * self.T + v * (v - lead_vel) /
            (2 * np.sqrt(self.a * self.b)))

        result_1 = self.a * (1 - (v / self.v0)**self.delta - (s_star / h)**2)
        
        h = self.stop_line - veh.d
        if abs(h) < 1e-3:
            h = 1e-3
        lead_vel = 1e-3
        s_star = self.s0 + max(
            0, v * self.T + v * (v - lead_vel) /
            (2 * np.sqrt(self.a * self.b)))
        result_2 = self.a * (1 - (v / self.v0)**self.delta - (s_star / h)**2)

        result = min(result_1, result_2)
        if v < 1e-3 and result < 0:
            return 0
        else:
            return result