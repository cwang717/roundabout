import numpy as np
from .utils import IDMController, adjust_a, direction_xy, isFeasible, next_a
from .vehicle import Vehicle, VirtualVihicle
from .slot import Slot
from functools import reduce
from operator import add
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


class Env():

    def __init__(self, envParams):
        self.slots = []
        self.approaching_vehicles = [[], [], [], []]
        self.adjusting_vehicles = [[], [], [], []]
        self.platooning_vehicles = [[], [], [], []]
        self.leaving_vehicles = [[], [], [], []]
        self.virtual_vehicles = []

        self.headway = envParams["headway"]
        self.veh_length = envParams["veh_length"]
        self.slot_length = self.headway + self.veh_length
        self.numSlots = envParams["C"]*4
        self.R = self.numSlots * self.slot_length / (2*np.pi)
        self.platoon_line = np.sqrt(3)*self.R
        self.v = envParams["v"]
        self.a = envParams["a"]
        self.adjust_d = self.v**2/2/self.a
        self.stop_line = np.sqrt(3) * self.R + self.adjust_d
        self.Q = envParams["Q"]
        self.step_size = envParams["step_size"]
        self.approaching_steps = self.slot_length/self.v/self.step_size
        self.boundary = envParams["boundary"]
        self.idm = IDMController(self.boundary - self.stop_line)
        self.num_step = 0
        self.queue_length = [0, 0, 0, 0]
        self.k_omega = 10
        self.k_theta = 10
        self.records = {
            "theta": np.zeros((1, self.numSlots)), 
            "omega": np.zeros((1, self.numSlots))
        }
        self.eta = envParams["eta"]
        s1 = np.array([[1, 1, 1, 1],
               [0, 1, 0, 0],
               [0, 1, 1, 0],
               [0, 1, 1, 1]])
        s2 = np.array([[1, 0, 1, 1],
                    [1, 1, 1, 1],
                    [0, 0, 1, 0],
                    [0, 0, 1, 1]])
        s3 = np.array([[1, 0, 0, 1],
                    [1, 1, 0, 1],
                    [1, 1, 1, 1],
                    [0, 0, 0, 1]])
        s4 = np.array([[1, 0, 0, 0],
                    [1, 1, 0, 0],
                    [1, 1, 1, 0],
                    [1, 1, 1, 1]])
        A1 = sum(np.transpose(s1*self.eta))
        A2 = sum(np.transpose(s2*self.eta))
        A3 = sum(np.transpose(s3*self.eta))
        A4 = sum(np.transpose(s4*self.eta))
        A = np.concatenate((A1, A2, A3, A4)).reshape(4, 4)
        self.P = np.array([self.Q[0]/(60-A.dot(self.Q)[3] + self.Q.dot(self.eta)[0]),
                           self.Q[1]/(60-A.dot(self.Q)[0] + self.Q.dot(self.eta)[1]),
                           self.Q[2]/(60-A.dot(self.Q)[1] + self.Q.dot(self.eta)[2]),
                           self.Q[3]/(60-A.dot(self.Q)[2] + self.Q.dot(self.eta)[3])])

    def initialize(self, initParams):
        # initial slots
        for i in range(self.numSlots):
            theta = 4/3*np.pi - i/self.numSlots*np.pi*2
            self.slots.append(Slot(i, self.R ,theta))

        # initial queues
        for i in range(4):
            for j in range(initParams["l_0"][i]):
                x, y = direction_xy[i]((self.stop_line + (j+1)*self.slot_length, 0))
                d = self.boundary - (self.stop_line + (j+1)*self.slot_length)
                des = np.random.choice([0, 1, 2, 3], p = self.eta[i])
                self.approaching_vehicles[i].append(Vehicle(x, y, 0, d))
                self.approaching_vehicles[i][-1].generate_approach_seq(i, des)


    def step(self):
        self.num_step += 1

        # Q->approaching queues 
        if np.mod(self.num_step, self.approaching_steps) == 0:
            for i in range(4):
                if np.random.poisson(self.Q[i]/60*self.approaching_steps*self.step_size):
                    x, y = direction_xy[i]([self.boundary, 0])
                    if len(self.approaching_vehicles[i]) > 0 and self.approaching_vehicles[i][-1].d < self.slot_length:
                        x, y = direction_xy[i]([self.approaching_vehicles[i][-1].d - self.slot_length, 0])
                    des = np.random.choice(4)
                    self.approaching_vehicles[i].append(Vehicle(x, y, 0, 0))
                    self.approaching_vehicles[i][-1].generate_approach_seq(i, des)

        # move vehicles
        # vehicles in approaching_vehicles follow idm (except the first one)
        for i in range(4):
            for j in range(len(self.approaching_vehicles[i])):
                veh = self.approaching_vehicles[i][j]
                preceding_veh = None
                if j == 0:
                    if len(self.adjusting_vehicles[i]) > 0:
                        # preceding_veh = self.adjusting_vehicles[i][-1]
                        pass
                else:
                    preceding_veh = self.approaching_vehicles[i][j-1]

                a = self.idm.get_accel(veh, preceding_veh)
                veh.update(a, self.step_size, i, self.boundary, self.R)
                
        # vehicles in adjusting_vehicles follow function
        for i in range(4):
            for veh in self.adjusting_vehicles[i]:
                desired_t = np.mod(np.pi/2 * i - np.pi/6 - veh.slot.theta, 2*np.pi) * self.R / self.v
                for j in range(20):
                    a = adjust_a(veh, self.boundary - np.sqrt(3)*self.R, desired_t-j*self.step_size/20, self.v, self.a)
                    veh.update(a, self.step_size/20, i, self.boundary, self.R)
                    # if veh.d > self.boundary - np.sqrt(3)*self.R:
                    target_theta = np.mod(np.pi/2 * i - np.pi/6, 2*np.pi)
                    if veh.slot.virtual_vehicle is None and veh.slot.theta > target_theta and veh.slot.theta < target_theta + np.pi/12 :
                        self.adjusting_vehicles[i].remove(veh)
                        self.platooning_vehicles[i].append(veh)
                        self.virtual_vehicles.append(
                            VirtualVihicle(veh, 
                                           np.mod(np.pi/2 * i + np.pi/6 - \
                                                  (self.boundary - np.sqrt(3)*self.R + np.pi/3*self.R - veh.d)/self.R, 2*np.pi), 
                                           self.v/self.R))
                        veh.slot.virtual_vehicle = self.virtual_vehicles[-1]
                        veh.slot.veh.remove(veh)
                        break

        # virtual_vehicles give a
        sample = {
            "theta": np.zeros(self.numSlots),
            "omega": np.zeros(self.numSlots)
        }
        for v_veh in self.virtual_vehicles:
            delta = v_veh.theta - v_veh.veh.slot.theta
            if delta > np.pi:
                delta = delta - 2*np.pi
            elif delta < -np.pi:
                delta = delta + 2*np.pi
            else:
                pass
            sample["theta"][self.slots.index(v_veh.veh.slot)] = delta
            sample["omega"][self.slots.index(v_veh.veh.slot)] = v_veh.omega - self.v/self.R

            # v_veh.angular_a = -self.k_omega * (v_veh.omega - self.v/self.R) - self.k_theta * delta
            # v_veh.update(self.step_size)
            
            # delta = v_veh.theta - v_veh.veh.slot.theta
            # if delta > np.pi:
            #     delta = delta - 2*np.pi
            # elif delta < -np.pi:
            #     delta = delta + 2*np.pi
            # else:
            #     pass
            # angles.append(delta)
        self.records["theta"] = np.append(self.records["theta"], [sample["theta"]], axis = 0)
        self.records["omega"] = np.append(self.records["omega"], [sample["omega"]], axis = 0)

        angles = []
        for v_veh in self.virtual_vehicles:
            i = self.slots.index(v_veh.veh.slot)
            pre_i = np.mod(i-1, self.numSlots)
            follow_i = np.mod(i+1, self.numSlots)
            v_veh.angular_a = -self.k_omega * sample["omega"][i]\
                              -self.k_theta * (2*sample["theta"][i] - sample["theta"][pre_i] - sample["theta"][follow_i])
            v_veh.angular_a = min(max(v_veh.angular_a, -self.a), self.a)
            v_veh.update(self.step_size)
            
            delta = v_veh.theta - v_veh.veh.slot.theta
            if delta > np.pi:
                delta = delta - 2*np.pi
            elif delta < -np.pi:
                delta = delta + 2*np.pi
            else:
                pass
            angles.append(delta)

        # vehicles in platooning_vehicles: before merge/after merge 
        for i in range(4):
            for veh in self.platooning_vehicles[i]:
                veh.update(veh.slot.virtual_vehicle.angular_a*self.R, self.step_size, i, self.boundary, self.R)
                if veh.d > self.boundary - np.sqrt(3)*self.R + np.pi/3*self.R:
                    # merge
                    veh.d = -1
                if len(veh.approach_seq) == 0:
                    self.leaving_vehicles[veh.slot.next_approach].append(veh)
                    self.platooning_vehicles[i].remove(veh)
                    self.virtual_vehicles.remove(veh.slot.virtual_vehicle)
                    veh.slot.virtual_vehicle = None
                    veh.slot = None

                    veh.d = self.boundary - np.sqrt(3)*self.R + np.pi/3*self.R
                    veh.v = self.v

        # leaving_vehicles
        for i in range(4):
            for veh in self.leaving_vehicles[i]:
                veh.leave(self.step_size, i, self.boundary, self.R)
                if veh.d < 0:
                    self.leaving_vehicles[i].remove(veh)

        # update slot and vehicles sets
        if len(self.virtual_vehicles) == 0:
            step_angle = self.step_size * self.v /self.R
        else:
            step_angle = np.mean(angles)
        for slot in self.slots:
            pass_an_approach = slot.rotate(step_angle)
            if pass_an_approach and slot.empty and (len(slot.veh) == 0 or next_a(slot.next_approach) == slot.veh[-1].approach_seq[-1]):
                approach = next_a(slot.next_approach)
                if len(self.approaching_vehicles[approach]) > 0 and isFeasible(self.approaching_vehicles[approach][0], self.a, self.v, self.boundary, self.R):
                    if np.random.random() < self.P[approach]:
                        self.adjusting_vehicles[approach].append(self.approaching_vehicles[approach][0])
                        self.approaching_vehicles[approach].remove(self.adjusting_vehicles[approach][-1])
                        self.adjusting_vehicles[approach][-1].slot = slot
                        slot.empty = False
                        slot.veh.append(self.adjusting_vehicles[approach][-1])

        # update queue length
        for i in range(4):
            mask = [ veh.v < 3 and veh.d > 75 for veh in self.approaching_vehicles[i]]
            self.queue_length[i] = sum(mask)

    def ani_save(self, fig, steps):
        
        ax = fig.add_subplot(111, autoscale_on=False, xlim=(-self.boundary/2, self.boundary), ylim=(-self.boundary/2, self.boundary))

        self.animate_init(ax)

        ani = animation.FuncAnimation(fig, self.animate, steps, blit=True)

        ani.save('simulation-%d.mp4' % steps, fps=int(1/self.step_size))

    def animate_init(self, ax):
        # plot the roadways in black
        ax.plot(np.linspace(-self.boundary, -np.sqrt(3)*self.R, 100), np.zeros(100), markersize=0, c='k')
        ax.plot(np.linspace(np.sqrt(3)*self.R, self.boundary, 100), np.zeros(100), markersize=0, c='k')
        ax.plot(np.zeros(100), np.linspace(-self.boundary, -np.sqrt(3)*self.R, 100), markersize=0, c='k')
        ax.plot(np.zeros(100), np.linspace(np.sqrt(3)*self.R, self.boundary, 100), markersize=0, c='k')
        
        ax.plot(np.cos(np.linspace(0, 2*np.pi, 100))*self.R, np.sin(np.linspace(0, 2*np.pi, 100))*self.R, markersize=0, c='k')
        ax.plot((np.sqrt(3)+np.cos(np.linspace(7/6*np.pi, 3/2*np.pi, 100)))*self.R, (1+np.sin(np.linspace(7/6*np.pi, 3/2*np.pi, 100)))*self.R, markersize=0, c='k')
        ax.plot((np.sqrt(3)+np.cos(np.linspace(7/6*np.pi, 3/2*np.pi, 100)))*self.R, -(1+np.sin(np.linspace(7/6*np.pi, 3/2*np.pi, 100)))*self.R, markersize=0, c='k')
        ax.plot(-(np.sqrt(3)+np.cos(np.linspace(7/6*np.pi, 3/2*np.pi, 100)))*self.R, (1+np.sin(np.linspace(7/6*np.pi, 3/2*np.pi, 100)))*self.R, markersize=0, c='k')
        ax.plot(-(np.sqrt(3)+np.cos(np.linspace(7/6*np.pi, 3/2*np.pi, 100)))*self.R, -(1+np.sin(np.linspace(7/6*np.pi, 3/2*np.pi, 100)))*self.R, markersize=0, c='k')
        ax.plot((1+np.sin(np.linspace(7/6*np.pi, 3/2*np.pi, 100)))*self.R, (np.sqrt(3)+np.cos(np.linspace(7/6*np.pi, 3/2*np.pi, 100)))*self.R, markersize=0, c='k')
        ax.plot((1+np.sin(np.linspace(7/6*np.pi, 3/2*np.pi, 100)))*self.R, -(np.sqrt(3)+np.cos(np.linspace(7/6*np.pi, 3/2*np.pi, 100)))*self.R, markersize=0, c='k')
        ax.plot(-(1+np.sin(np.linspace(7/6*np.pi, 3/2*np.pi, 100)))*self.R, (np.sqrt(3)+np.cos(np.linspace(7/6*np.pi, 3/2*np.pi, 100)))*self.R, markersize=0, c='k')
        ax.plot(-(1+np.sin(np.linspace(7/6*np.pi, 3/2*np.pi, 100)))*self.R, -(np.sqrt(3)+np.cos(np.linspace(7/6*np.pi, 3/2*np.pi, 100)))*self.R, markersize=0, c='k')

        # # plot stoplines in red
        # ax.plot([-self.stop_line, -self.stop_line], [0, -2], markersize=0, c='r')
        # ax.plot([self.stop_line, self.stop_line], [0, 2], markersize=0, c='r')
        # ax.plot([0, -2], [self.stop_line, self.stop_line], markersize=0, c='r')
        # ax.plot([0, 2], [-self.stop_line, -self.stop_line], markersize=0, c='r')

        sub_ax = inset_axes(ax,
                    width=6,                     # inch
                    height=6,                    # inch
                    bbox_transform=ax.transData, # data coordinates
                    bbox_to_anchor=(50,50),    # data coordinates
                    loc=3)                       # loc=lower left corner
        sub_ax.set_xlim(-3*self.R, 3*self.R)
        sub_ax.set_ylim(-3*self.R, 3*self.R)
        
        sub_ax.plot(np.linspace(-self.boundary, -np.sqrt(3)*self.R, 100), np.zeros(100), markersize=0, c='k')
        sub_ax.plot(np.linspace(np.sqrt(3)*self.R, self.boundary, 100), np.zeros(100), markersize=0, c='k')
        sub_ax.plot(np.zeros(100), np.linspace(-self.boundary, -np.sqrt(3)*self.R, 100), markersize=0, c='k')
        sub_ax.plot(np.zeros(100), np.linspace(np.sqrt(3)*self.R, self.boundary, 100), markersize=0, c='k')
        
        sub_ax.plot(np.cos(np.linspace(0, 2*np.pi, 100))*self.R, np.sin(np.linspace(0, 2*np.pi, 100))*self.R, markersize=0, c='k')
        sub_ax.plot((np.sqrt(3)+np.cos(np.linspace(7/6*np.pi, 3/2*np.pi, 100)))*self.R, (1+np.sin(np.linspace(7/6*np.pi, 3/2*np.pi, 100)))*self.R, markersize=0, c='k')
        sub_ax.plot((np.sqrt(3)+np.cos(np.linspace(7/6*np.pi, 3/2*np.pi, 100)))*self.R, -(1+np.sin(np.linspace(7/6*np.pi, 3/2*np.pi, 100)))*self.R, markersize=0, c='k')
        sub_ax.plot(-(np.sqrt(3)+np.cos(np.linspace(7/6*np.pi, 3/2*np.pi, 100)))*self.R, (1+np.sin(np.linspace(7/6*np.pi, 3/2*np.pi, 100)))*self.R, markersize=0, c='k')
        sub_ax.plot(-(np.sqrt(3)+np.cos(np.linspace(7/6*np.pi, 3/2*np.pi, 100)))*self.R, -(1+np.sin(np.linspace(7/6*np.pi, 3/2*np.pi, 100)))*self.R, markersize=0, c='k')
        sub_ax.plot((1+np.sin(np.linspace(7/6*np.pi, 3/2*np.pi, 100)))*self.R, (np.sqrt(3)+np.cos(np.linspace(7/6*np.pi, 3/2*np.pi, 100)))*self.R, markersize=0, c='k')
        sub_ax.plot((1+np.sin(np.linspace(7/6*np.pi, 3/2*np.pi, 100)))*self.R, -(np.sqrt(3)+np.cos(np.linspace(7/6*np.pi, 3/2*np.pi, 100)))*self.R, markersize=0, c='k')
        sub_ax.plot(-(1+np.sin(np.linspace(7/6*np.pi, 3/2*np.pi, 100)))*self.R, (np.sqrt(3)+np.cos(np.linspace(7/6*np.pi, 3/2*np.pi, 100)))*self.R, markersize=0, c='k')
        sub_ax.plot(-(1+np.sin(np.linspace(7/6*np.pi, 3/2*np.pi, 100)))*self.R, -(np.sqrt(3)+np.cos(np.linspace(7/6*np.pi, 3/2*np.pi, 100)))*self.R, markersize=0, c='k')

        self.slot_marker = []
        self.slot_text = []
        self.sub_slot_marker = []
        self.sub_slot_text = []
        for slot in self.slots:
            temp, = ax.plot(np.array([self.R-1, self.R+1])*np.cos(slot.theta), 
                            np.array([self.R-1, self.R+1])*np.sin(slot.theta), markersize=0, c='g')
            self.slot_marker.append(temp)
            # temp = ax.text((self.R+2)*np.cos(slot.theta), 
            #                (self.R+2)*np.sin(slot.theta), "%d" % slot.id, ha='center', va='center')
            # self.slot_text.append(temp)

            temp, = sub_ax.plot(np.array([self.R-1, self.R+1])*np.cos(slot.theta), 
                            np.array([self.R-1, self.R+1])*np.sin(slot.theta), markersize=0, c='g')
            self.sub_slot_marker.append(temp)
            temp = sub_ax.text((self.R+2)*np.cos(slot.theta), 
                           (self.R+2)*np.sin(slot.theta), "%d" % slot.id, ha='center', va='center')
            self.sub_slot_text.append(temp)

        self.vehicle_dots = ax.scatter(np.ones(100)*(self.boundary + 1), np.ones(100)*(self.boundary + 1))
        self.vehicle_exit_dots = ax.scatter(np.ones(100)*(self.boundary + 1), np.ones(100)*(self.boundary + 1), marker='o')

        self.sub_vehicle_dots = sub_ax.scatter(np.ones(100)*(self.boundary + 1), np.ones(100)*(self.boundary + 1))
        self.sub_vehicle_exit_dots = sub_ax.scatter(np.ones(100)*(self.boundary + 1), np.ones(100)*(self.boundary + 1), marker='o')

        # information texts handlers
        self.time_template = 'steps = %d'
        self.time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
        self.queue_template = 'queue 0: %d\nqueue 1: %d\nqueue 2: %d\nqueue 3: %s'
        self.queue_text = ax.text(0.05, 0.8, '', transform=ax.transAxes)
        self.slots_template = 'slot exit: %d %d %d %d %d %d %d %d %d %d %d %d\n next approach: %d %d %d %d %d %d %d %d %d %d %d %d'
        self.slots_text = ax.text(0.05, 0.7, '', transform=ax.transAxes)

        # self.time_text.set_text(self.time_template % (0))
        # self.slot_marker_1, self.slot_marker_2, self.slot_marker_3, self.slot_marker_4,\
        # self.slot_marker_5, self.slot_marker_6, self.slot_marker_7, self.slot_marker_8,\
        # self.slot_marker_9, self.slot_marker_10, self.slot_marker_11, self.slot_marker_12 = self.slot_marker

        # self.slot_text_1, self.slot_text_2, self.slot_text_3, self.slot_text_4,\
        # self.slot_text_5, self.slot_text_6, self.slot_text_7, self.slot_text_8,\
        # self.slot_text_9, self.slot_text_10, self.slot_text_11, self.slot_text_12 = self.slot_text

        self.adjusting_handler = []
        for i in range(50):
            temp = ax.text(self.boundary+1, 0, '')
            self.adjusting_handler.append(temp)

        rec_ax = inset_axes(ax,
                    width=6,                     # inch
                    height=1,                    # inch
                    bbox_transform=ax.transData, # data coordinates
                    bbox_to_anchor=(50,-110),    # data coordinates
                    loc=3)                       # loc=lower left corner
        self.rec_steps = int(self.numSlots*self.slot_length/self.v/self.step_size)
        rec_ax.set_xlim(1-self.rec_steps, 0)
        rec_ax.set_ylim(-0.5, 0.5)
        rec_ax.set_ylabel("Headway variance (m)")
        rec_ax.set_xlabel("Steps (0 is current step)")
        rec_steps = min(self.rec_steps, self.num_step)
        self.rec_handler = rec_ax.plot([1, 2], np.zeros((2, 12)))

        rec2_ax = inset_axes(ax,
                    width=6,                     # inch
                    height=1,                    # inch
                    bbox_transform=ax.transData, # data coordinates
                    bbox_to_anchor=(50,-50),    # data coordinates
                    loc=3)                       # loc=lower left corner
        rec2_ax.set_xlim(1-self.rec_steps, 0)
        rec2_ax.set_ylim(-1, 1)
        rec2_ax.set_ylabel("Speed variance (m/s)")
        rec2_ax.set_xlabel("Steps (0 is current step)")
        self.rec2_handler = rec2_ax.plot([1, 2], np.zeros((2, 12)))

    def animate(self, step):
        self.step()
        self.time_text.set_text(self.time_template % self.num_step)
        self.queue_text.set_text(self.queue_template % tuple(self.queue_length))

        for slot in self.slots:
            i = self.slots.index(slot)
            self.slot_marker[i].set_data(np.array([self.R-1, self.R+1])*np.cos(slot.theta), 
                                         np.array([self.R-1, self.R+1])*np.sin(slot.theta))
            self.sub_slot_marker[i].set_data(np.array([self.R-1, self.R+1])*np.cos(slot.theta), 
                                         np.array([self.R-1, self.R+1])*np.sin(slot.theta))
            self.sub_slot_text[i].set_position(((self.R+2)*np.cos(slot.theta), 
                                            (self.R+2)*np.sin(slot.theta)))

        # self.slot_marker_1, self.slot_marker_2, self.slot_marker_3, self.slot_marker_4,\
        # self.slot_marker_5, self.slot_marker_6, self.slot_marker_7, self.slot_marker_8,\
        # self.slot_marker_9, self.slot_marker_10, self.slot_marker_11, self.slot_marker_12 = self.slot_marker

        # self.sub_slot_marker_1, self.sub_slot_marker_2, self.sub_slot_marker_3, self.sub_slot_marker_4,\
        # self.sub_slot_marker_5, self.sub_slot_marker_6, self.sub_slot_marker_7, self.sub_slot_marker_8,\
        # self.sub_slot_marker_9, self.sub_slot_marker_10, self.sub_slot_marker_11, self.sub_slot_marker_12 = self.sub_slot_marker

        # self.sub_slot_text_1, self.sub_slot_text_2, self.sub_slot_text_3, self.sub_slot_text_4,\
        # self.sub_slot_text_5, self.sub_slot_text_6, self.sub_slot_text_7, self.sub_slot_text_8,\
        # self.sub_slot_text_9, self.sub_slot_text_10, self.sub_slot_text_11, self.sub_slot_text_12 = self.sub_slot_text
        
        
        # plot vehicles (no exiting vehicles)
        vehicles_x = np.array([])
        vehicles_y = np.array([])
        for vehicle_group in [self.approaching_vehicles, self.adjusting_vehicles, self.platooning_vehicles]:
            vehicles_x = np.concatenate([vehicles_x, [item.x for item in reduce(add, vehicle_group)]])
            vehicles_y = np.concatenate([vehicles_y, [item.y for item in reduce(add, vehicle_group)]])
        if len(vehicles_x) == 0:
            vehicles_x = np.ones(100)*(self.boundary + 1)
            vehicles_y = np.ones(100)*(self.boundary + 1)
        self.vehicle_dots.set_offsets(np.array([[item[0], item[1]] for item in zip(vehicles_x, vehicles_y)]))
        self.sub_vehicle_dots.set_offsets(np.array([[item[0], item[1]] for item in zip(vehicles_x, vehicles_y)]))
        

        adjusting_info = [("%d"%self.slots.index(veh.slot), (veh.x, veh.y)) \
                            for group in self.adjusting_vehicles\
                            for veh in group if veh.slot is not None]
        for i in range(min(len(adjusting_info), 50)):
            self.adjusting_handler[i].set_position(adjusting_info[i][1])
            self.adjusting_handler[i].set_text(adjusting_info[i][0])
        for i in range(min(len(adjusting_info), 50), 50):
            self.adjusting_handler[i].set_position((self.boundary + 1, 0))
            self.adjusting_handler[i].set_text("")


        # plot exiting vehicles with different markers
        vehicle_exit_x = np.array([item.x for item in reduce(add, self.leaving_vehicles)])
        vehicle_exit_y = np.array([item.y for item in reduce(add, self.leaving_vehicles)])
        if len(vehicle_exit_x) == 0:
            vehicle_exit_x = np.ones(100)*(self.boundary + 1)
            vehicle_exit_y = np.ones(100)*(self.boundary + 1)
        self.vehicle_exit_dots.set_offsets(np.array([[item[0], item[1]] for item in zip(vehicle_exit_x, vehicle_exit_y)]))
        self.sub_vehicle_exit_dots.set_offsets(np.array([[item[0], item[1]] for item in zip(vehicle_exit_x, vehicle_exit_y)]))

        
        rec_steps = min(self.rec_steps, self.num_step)
        if self.num_step > 1:
            for i in range(self.numSlots):
                self.rec_handler[i].set_data(np.linspace(1-rec_steps, 0, rec_steps), self.records["theta"][-rec_steps:, i]*self.R)
                self.rec2_handler[i].set_data(np.linspace(1-rec_steps, 0, rec_steps), self.records["omega"][-rec_steps:, i]*self.R)
        # self.rec_handler_1, self.rec_handler_2, self.rec_handler_3, self.rec_handler_4,\
        # self.rec_handler_5, self.rec_handler_6, self.rec_handler_7, self.rec_handler_8,\
        # self.rec_handler_9, self.rec_handler_10, self.rec_handler_11, self.rec_handler_12 = self.rec_handler

        return [item for group in [self.slot_marker, self.sub_slot_marker, self.sub_slot_text,
                                   self.rec_handler, self.adjusting_handler, [
                                       self.vehicle_dots, self.vehicle_exit_dots, 
                                       self.sub_vehicle_dots, self.sub_vehicle_exit_dots, self.queue_text, self.time_text
                                   ]] for item in group]
        # return self.slot_marker_1, self.slot_marker_2, self.slot_marker_3, self.slot_marker_4,\
            #    self.slot_marker_5, self.slot_marker_6, self.slot_marker_7, self.slot_marker_8,\
            #    self.slot_marker_9, self.slot_marker_10, self.slot_marker_11, self.slot_marker_12, \
            #    self.sub_slot_marker_1, self.sub_slot_marker_2, self.sub_slot_marker_3, self.sub_slot_marker_4,\
            #    self.sub_slot_marker_5, self.sub_slot_marker_6, self.sub_slot_marker_7, self.sub_slot_marker_8,\
            #    self.sub_slot_marker_9, self.sub_slot_marker_10, self.sub_slot_marker_11, self.sub_slot_marker_12,\
            #    self.sub_slot_text_1, self.sub_slot_text_2, self.sub_slot_text_3, self.sub_slot_text_4,\
            #    self.sub_slot_text_5, self.sub_slot_text_6, self.sub_slot_text_7, self.sub_slot_text_8,\
            #    self.sub_slot_text_9, self.sub_slot_text_10, self.sub_slot_text_11, self.sub_slot_text_12,\
            #    self.rec_handler_1, self.rec_handler_2, self.rec_handler_3, self.rec_handler_4,\
            #    self.rec_handler_5, self.rec_handler_6, self.rec_handler_7, self.rec_handler_8,\
            #    self.rec_handler_9, self.rec_handler_10, self.rec_handler_11, self.rec_handler_12,\
            #    self.vehicle_dots, self.vehicle_exit_dots, self.sub_vehicle_dots, self.sub_vehicle_exit_dots, *(self.adjusting_handler)

    def test_plot(self, ax):
        
        # plot the roadways in black
        ax.plot(np.linspace(-self.boundary, -np.sqrt(3)*self.R, 100), np.zeros(100), markersize=0, c='k')
        ax.plot(np.linspace(np.sqrt(3)*self.R, self.boundary, 100), np.zeros(100), markersize=0, c='k')
        ax.plot(np.zeros(100), np.linspace(-self.boundary, -np.sqrt(3)*self.R, 100), markersize=0, c='k')
        ax.plot(np.zeros(100), np.linspace(np.sqrt(3)*self.R, self.boundary, 100), markersize=0, c='k')
        
        ax.plot(np.cos(np.linspace(0, 2*np.pi, 100))*self.R, np.sin(np.linspace(0, 2*np.pi, 100))*self.R, markersize=0, c='k')
        ax.plot((np.sqrt(3)+np.cos(np.linspace(7/6*np.pi, 3/2*np.pi, 100)))*self.R, (1+np.sin(np.linspace(7/6*np.pi, 3/2*np.pi, 100)))*self.R, markersize=0, c='k')
        ax.plot((np.sqrt(3)+np.cos(np.linspace(7/6*np.pi, 3/2*np.pi, 100)))*self.R, -(1+np.sin(np.linspace(7/6*np.pi, 3/2*np.pi, 100)))*self.R, markersize=0, c='k')
        ax.plot(-(np.sqrt(3)+np.cos(np.linspace(7/6*np.pi, 3/2*np.pi, 100)))*self.R, (1+np.sin(np.linspace(7/6*np.pi, 3/2*np.pi, 100)))*self.R, markersize=0, c='k')
        ax.plot(-(np.sqrt(3)+np.cos(np.linspace(7/6*np.pi, 3/2*np.pi, 100)))*self.R, -(1+np.sin(np.linspace(7/6*np.pi, 3/2*np.pi, 100)))*self.R, markersize=0, c='k')
        ax.plot((1+np.sin(np.linspace(7/6*np.pi, 3/2*np.pi, 100)))*self.R, (np.sqrt(3)+np.cos(np.linspace(7/6*np.pi, 3/2*np.pi, 100)))*self.R, markersize=0, c='k')
        ax.plot((1+np.sin(np.linspace(7/6*np.pi, 3/2*np.pi, 100)))*self.R, -(np.sqrt(3)+np.cos(np.linspace(7/6*np.pi, 3/2*np.pi, 100)))*self.R, markersize=0, c='k')
        ax.plot(-(1+np.sin(np.linspace(7/6*np.pi, 3/2*np.pi, 100)))*self.R, (np.sqrt(3)+np.cos(np.linspace(7/6*np.pi, 3/2*np.pi, 100)))*self.R, markersize=0, c='k')
        ax.plot(-(1+np.sin(np.linspace(7/6*np.pi, 3/2*np.pi, 100)))*self.R, -(np.sqrt(3)+np.cos(np.linspace(7/6*np.pi, 3/2*np.pi, 100)))*self.R, markersize=0, c='k')

        # plot stoplines in red
        ax.plot([-self.stop_line, -self.stop_line], [0, -2], markersize=0, c='r')
        ax.plot([self.stop_line, self.stop_line], [0, 2], markersize=0, c='r')
        ax.plot([0, -2], [self.stop_line, self.stop_line], markersize=0, c='r')
        ax.plot([0, 2], [-self.stop_line, -self.stop_line], markersize=0, c='r')

        slot_marker = []

        for slot in self.slots:
            temp, = ax.plot(np.array([self.R-1, self.R+1])*np.cos(slot.theta), 
                            np.array([self.R-1, self.R+1])*np.sin(slot.theta), markersize=0, c='g')
            slot_marker.append(temp)
            ax.text(self.R*np.cos(slot.theta), 
                    self.R*np.sin(slot.theta), "%s:%s" % (str(slot.id), str(slot.next_approach)))

        # plot vehicles (no exiting vehicles)
        vehicle_dots = ax.scatter(np.ones(100)*(self.boundary + 1), np.ones(100)*(self.boundary + 1))
        vehicle_exit_dots = ax.scatter(np.ones(100)*(self.boundary + 1), np.ones(100)*(self.boundary + 1), marker='o')
        vehicles_x = np.array([])
        vehicles_y = np.array([])
        for vehicle_group in [self.approaching_vehicles, self.adjusting_vehicles, self.platooning_vehicles]:
            vehicles_x = np.concatenate([vehicles_x, [item.x for item in reduce(add, vehicle_group)]])
            vehicles_y = np.concatenate([vehicles_y, [item.y for item in reduce(add, vehicle_group)]])
        if len(vehicles_x) == 0:
            vehicles_x = np.ones(100)*(self.boundary + 1)
            vehicles_y = np.ones(100)*(self.boundary + 1)
        vehicle_dots.set_offsets(np.array([[item[0], item[1]] for item in zip(vehicles_x, vehicles_y)]))
        

        # plot exiting vehicles with different markers
        vehicle_exit_x = np.array([item.x for item in reduce(add, self.leaving_vehicles)])
        vehicle_exit_y = np.array([item.y for item in reduce(add, self.leaving_vehicles)])
        if len(vehicle_exit_x) == 0:
            vehicle_exit_x = np.ones(100)*(self.boundary + 1)
            vehicle_exit_y = np.ones(100)*(self.boundary + 1)
        vehicle_exit_dots.set_offsets(np.array([[item[0], item[1]] for item in zip(vehicle_exit_x, vehicle_exit_y)]))