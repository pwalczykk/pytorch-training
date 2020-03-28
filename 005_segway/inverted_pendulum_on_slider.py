#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import cv2


class InvertedPendulum:
    def __init__(self, length, pendulum_mass, slider_mass,
                 angular_damping=0.005, initial_position_angular_deg=0.0,
                 linear_damping=0.005, initial_position_linear=0.0):
        self.length = length
        self.pendulum_mass = pendulum_mass
        self.slider_mass = slider_mass
        self.gravity = 9.81

        self.inertia = self.pendulum_mass * np.power(self.length, 2) # [kg * m^2]

        self.step_time = 0.01
        self.step_counter = 0

        self.linear_position = initial_position_linear
        self.linear_velocity = 0.0
        self.linear_damping = linear_damping

        self.angular_position = np.deg2rad(initial_position_angular_deg)
        self.angular_velocity = 0.0
        self.angular_damping = angular_damping

        self.angular_positions_list = [0]

    def step(self, verbose=False, draw=False):

        self.step_counter += 1

        # Slider force from pendulum
        force_parallel = np.cos(self.angular_position) * self.pendulum_mass * self.gravity # [N]
        linear_force = np.sin(self.angular_position) * force_parallel

        # Slider move
        linear_acceleration = linear_force / (self.slider_mass + self.pendulum_mass)
        self.linear_velocity += linear_acceleration * self.step_time
        self.linear_velocity *= (1 - self.linear_damping)
        self.linear_position += self.linear_velocity * self.step_time

        # Pendulum force from slider
        inertia_acceleration = np.cos(self.angular_position) * linear_acceleration / self.length
        # Pendulum force from gravity
        force_perpendicular = np.sin(self.angular_position) * self.pendulum_mass * self.gravity # [N]
        momentum = force_perpendicular * self.length # [N * m]

        # Pendulum move
        angular_acceleration = momentum / self.inertia + inertia_acceleration# [1 / s^2]
        self.angular_velocity += angular_acceleration * self.step_time
        self.angular_velocity *= (1 - self.angular_damping)
        self.angular_position += self.angular_velocity * self.step_time
        self.angular_positions_list.append(self.angular_position)

        if verbose:
            print(np.rad2deg(self.angular_position))

        if draw:
            self.draw_current_state()

    def draw_current_state(self):

        # TODO: Bug with axes, works only with square images
        res_x = 1024
        res_y = 1024

        mid_x = res_x/2 - 1
        mid_y = res_y/2 - 1

        length_pix = 100

        scale = self.length / length_pix

        pendulum_fix = (
            int(mid_x + self.linear_position / scale),
            int(mid_y),
        )

        pendulum_body = (
            int(pendulum_fix[0] - np.sin(self.angular_position) * length_pix),
            int(pendulum_fix[1] - np.cos(self.angular_position) * length_pix)
        )

        image = np.zeros((res_x, res_y, 1), np.uint8)

        cv2.line(image, pendulum_fix, pendulum_body, 255, 3)
        cv2.circle(image, pendulum_body, 9, 255, 3)

        cv2.imshow("win", image)
        cv2.waitKey(int(self.step_time * 1000))

    def plot_history(self):
        plt.plot(range(len(self.angular_positions_list)),
                 self.angular_positions_list)
        plt.show()


def main():

    segway = InvertedPendulum(
        length=1,
        pendulum_mass=1000,
        slider_mass=1,
        initial_position_angular_deg=20,
    )

    for i in range(1000):
        segway.step(draw=True)


if __name__ == '__main__':
    main()
