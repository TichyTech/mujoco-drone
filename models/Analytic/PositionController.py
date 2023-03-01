import numpy as np


class PositionController:

    def __init__(self, num_drones):
        self.num_drones = num_drones

        self.error_i = np.zeros((3, self.num_drones))
        self.error_prev = np.zeros((3, self.num_drones))
        self.P = np.array([[0.4], [0.4], [0.6]])
        self.I = np.array([[0.06], [0.06], [0.1]])
        self.D = np.array([[0.7], [0.7], [0.2]])

        self.motor_force = 0.6
        self.first_step = True
        self.dt = 0.01

    def compute_control(self, ref, xyz):
        """given a reference xyz, adjust tilts and thrust accordingly"""
        e = ref[None].T - xyz  # compute control error from xyz
        if self.first_step:  # disable derivate on first step
            self.error_prev = e
            self.first_step = False
        self.error_d = (e - self.error_prev) / self.dt
        self.error_prev = e
        self.error_i = self.error_i + self.dt * e
        self.error_i = np.clip(self.error_i, -1, 1)
        tilts_zacc = self.P * e + self.I * self.error_i + self.D * self.error_d
        tilts_zacc[:2] = np.clip(tilts_zacc[:2], -0.5, 0.5)
        tilts_zacc[2] = np.clip(tilts_zacc[2], -2, 2)
        return tilts_zacc
