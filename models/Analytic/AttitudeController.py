import numpy as np
from scipy.spatial.transform import Rotation as R


class AttittudeController:

    def __init__(self, num_drones, masses, forces):
        self.num_drones = num_drones

        self.error_i = np.zeros((3, self.num_drones))
        self.error_prev = np.zeros((3, self.num_drones))
        self.P = np.array([[2], [2], [3]])
        self.I = np.array([[0.0], [0.0], [0]])
        self.D = np.array([[0.2], [0.2], [0.25]])
        self.mixer = np.array([[1, 1, -1, -1],
                               [-1, 1, 1, -1],
                               [1, -1, 1, -1]]).T

        self.motor_forces = np.array(forces)[None]
        self.masses = np.array(masses)[None]
        self.first_step = True
        self.dt = 0.02

    def tilts2rpy(self, pos_action, heading_ref):
        tilt_x, tilt_y = pos_action[:2]
        z_acc = pos_action[2] + 9.81
        heading_vec = np.array([np.cos(heading_ref), np.sin(heading_ref), np.zeros((self.num_drones))])  # heading vector
        thrust_vec = np.array([np.tan(tilt_x), np.tan(tilt_y), np.ones(self.num_drones)])
        refs = np.zeros((4, self.num_drones))
        for idx in range(self.num_drones):
            Rd = np.empty((3, 3))  # desired rotation matrix
            Rd[:, 2] = thrust_vec[:, idx] / np.linalg.norm(thrust_vec[:, idx])  # desired z axis
            Rd[:, 1] = np.cross(Rd[:, 2], heading_vec[:, idx])  # desired y axis
            Rd[:, 0] = np.cross(Rd[:, 1], Rd[:, 2])  # desired x axis
            rpy = R.from_matrix(Rd).as_euler('ZYX')[::-1]  # roll, pitch, yaw
            ref = np.append(rpy, np.linalg.norm(thrust_vec[:, idx] * z_acc[idx]))  # roll, pitch, yaw, z_acc
            refs[:, idx] = ref
        return refs

    def compute_control(self, rpya_ref, cur_rpy):
        ref_rpy = rpya_ref[:3]
        ref_accel = rpya_ref[3][None]
        error_rpy = ref_rpy - cur_rpy
        if self.first_step:  # disable derivate on first step
            self.error_prev = error_rpy
            self.first_step = False
        self.error_d = (error_rpy - self.error_prev) / self.dt
        self.error_prev = error_rpy
        self.error_i = self.error_i + self.dt * error_rpy
        self.error_i = np.clip(self.error_i, -1, 1)
        action = self.P*error_rpy + self.I*self.error_i + self.D*self.error_d
        forces = self.mixer @ action + 0.25 * ref_accel * self.masses
        ctrl = np.clip(forces / self.motor_forces, 0, 1)
        # self.errs.append(error_rpy)
        # self.ctrls.append(ctrl * self.motor_force * 100)
        return ctrl.T
