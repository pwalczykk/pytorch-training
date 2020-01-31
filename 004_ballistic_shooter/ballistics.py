#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from typing import NamedTuple, List


class Trajectory(NamedTuple):
    x: List[float]
    z: List[float]


class Ballistics:
    def __init__(self):
        self.trajectory = Trajectory([], [])

        self.pos_x = 0.0
        self.pos_z = 0.0
        
        self.vel_x = 0.0
        self.vel_z = 0.0

        self.step_counter = 0
        self.time_step = 0.1
        
        self.mass = 1.0

        self.air_drag = lambda x: -0.001*x*np.abs(x)
        
        self.g = -9.81
        
    def step(self, verbose=False):

        self.step_counter += 1

        vel_mag = np.sqrt(self.vel_x**2 + self.vel_z**2)
        air_drag_mag = self.air_drag(vel_mag)
        air_drag_x = self.vel_x / vel_mag * air_drag_mag
        air_drag_z = self.vel_z / vel_mag * air_drag_mag

        acc_x = air_drag_x / self.mass
        acc_z = air_drag_z / self.mass + self.g

        self.vel_x = self.vel_x + acc_x * self.time_step
        self.vel_z = self.vel_z + acc_z * self.time_step
        
        self.pos_x = self.pos_x + self.vel_x * self.time_step
        self.pos_z = self.pos_z + self.vel_z * self.time_step

        self.trajectory.x.append(self.pos_x)
        self.trajectory.z.append(self.pos_z)

        if verbose:
            print(f"PosX: {self.pos_x} | PosZ {self.pos_z} "
                  f"| VelX: {self.vel_x} | VelZ: {self.vel_z} "
                  f"| AccX: {acc_x} | AccZ {acc_z} "
                  f"| Time: {self.step_counter * self.time_step}")

    def shoot(self, velocity, angle_deg):

        angle = np.deg2rad(angle_deg)
        self.vel_x = velocity * np.cos(angle)
        self.vel_z = velocity * np.sin(angle)


def main():
    engine = Ballistics()
    engine.shoot(velocity=1000, angle_deg=30)

    for i in range(int(1e6)):
        engine.step(verbose=True)
        if engine.pos_z < 0:
            break

    plt.plot(engine.trajectory.x, engine.trajectory.z)
    plt.show()


if __name__ == '__main__':
    main()
