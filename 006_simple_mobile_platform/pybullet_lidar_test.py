#!/usr/bin/env python

import numpy as np
import tf.transformations

from pprint import pprint

from pybullet_lidar import PyBulletLidar2D


class TestPyBulletLidar2D(object):
    def test_generating_scan_angles(self):

        lidar_2d = PyBulletLidar2D(
            body="dummy",
            link="dummy",
            topic="dummy",
            angle_min=-10,
            angle_max=20,
            number_of_rays=151,
            mock_ros_comm=True,
        )

        lidar_2d._compute_scan_angles()

        assert isinstance(lidar_2d._scan_angles, list)
        assert len(lidar_2d._scan_angles) == 151
        assert np.isclose(lidar_2d._scan_angles[0], -10.0)
        assert np.isclose(lidar_2d._scan_angles[75], 5.0)
        assert np.isclose(lidar_2d._scan_angles[150], 20.0)

    @staticmethod
    def _assert_array_1d_equal(desired, computed):
        equal = True
        for i in range(len(desired)):
            equal &= np.isclose(desired[i], computed[i])

        assert equal, "\nDESIRED:\n{}\nCOMPUTED:\n{}\n".format(desired, computed)

    @staticmethod
    def _assert_array_2d_equal(desired, computed):
        equal = True
        for i in range(len(desired)):
            for j in range(len(desired[i])):
                equal &= np.isclose(desired[i][j], computed[i][j])

        assert equal, "\nDESIRED:\n{}\nCOMPUTED:\n{}\n".format(desired, computed)

    def test_genertaing_lidar_rays(self):

        angle_min = -np.pi/4
        angle_max = np.pi/4
        range_min = 1
        range_max = 10

        lidar_2d = PyBulletLidar2D(
            body="dummy",
            link="dummy",
            topic="dummy",
            angle_min=angle_min,
            angle_max=angle_max,
            range_min=1,
            range_max=10,
            number_of_rays=3,
            mock_ros_comm=True,
        )

        lidar_2d._update_laser_rays(
            mock_tf=((0, 0, 0), (0, 0, 0, 1))
        )

        assert isinstance(lidar_2d._laser_rays, list)
        assert len(lidar_2d._laser_rays) == 3
        for i in range(len(lidar_2d._laser_rays)):
            assert isinstance(lidar_2d._laser_rays[i], list)
            assert len(lidar_2d._laser_rays[i]) == 2

        # Left ray start
        ray_0_start_desired = np.array([
            [np.cos(angle_min), -np.sin(angle_min), 0, np.cos(angle_min) * range_min],
            [np.sin(angle_min), np.cos(angle_min), 0, np.sin(angle_min) * range_min],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        self._assert_array_2d_equal(ray_0_start_desired, lidar_2d._laser_rays[0][0])

        # Central ray end
        ray_1_start_desired = np.array([
            [np.cos(0), -np.sin(0), 0, np.cos(0) * range_max],
            [np.sin(0), np.cos(0), 0, np.sin(0) * range_max],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        self._assert_array_2d_equal(ray_1_start_desired, lidar_2d._laser_rays[1][1])

        # Right ray end
        ray_2_end_desired = np.array([
            [np.cos(angle_max), -np.sin(angle_max), 0, np.cos(angle_max) * range_max],
            [np.sin(angle_max), np.cos(angle_max), 0, np.sin(angle_max) * range_max],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        self._assert_array_2d_equal(ray_2_end_desired, lidar_2d._laser_rays[2][1])

    def test_genertaing_lidar_rays_in_world_frame(self):
        """
        Translate lidar by vector (3,2,1), then rotate it by 90 deg in Z axis
        Measure central ending point position.
        :return:
        """
        orient_rpy = (0, 0, np.pi/2)
        pose_xyz = (3, 2, 1)

        angle_min = -np.pi / 4
        angle_max = np.pi / 4
        range_min = 1
        range_max = 10

        lidar_2d = PyBulletLidar2D(
            body="dummy",
            link="dummy",
            topic="dummy",
            angle_min=angle_min,
            angle_max=angle_max,
            range_min=range_min,
            range_max=range_max,
            number_of_rays=3,
            mock_ros_comm=True,
        )

        lidar_2d._update_laser_rays(
            mock_tf=(
                pose_xyz,
                tf.transformations.quaternion_from_euler(orient_rpy[0], orient_rpy[1], orient_rpy[2])
            )
        )

        # Central ray end
        ray_1_start_desired = np.array([
            [0., -1., 0., 3.],
            [1., 0., 0., 12.],
            [0., 0., 1., 1.],
            [0., 0., 0., 1.],
        ])
        self._assert_array_2d_equal(ray_1_start_desired, lidar_2d._laser_rays[1][1])

    def test_position_from_matrix(self):

        ray_1_start_desired = np.array([
            [0., -1., 0., -69.],
            [1., 0., 0., 13.],
            [0., 0., 1., 42.],
            [0., 0., 0., 1.],
        ])

        position_ray_1_start_desired = PyBulletLidar2D._position_from_matrix(ray_1_start_desired)

        self._assert_array_1d_equal(position_ray_1_start_desired, [-69.0, 13.0, 42.0])
