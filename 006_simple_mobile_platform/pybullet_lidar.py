#!/usr/bin/env python

import pybullet

import numpy as np

import rospy
import tf
import tf.transformations
from sensor_msgs.msg import LaserScan


class PyBulletLidar2D(object):

    def __init__(self,
                 frame,
                 topic,
                 angle_min=-np.deg2rad(135),
                 angle_max=np.deg2rad(135),
                 range_min=0.1,
                 range_max=100,
                 number_of_rays=270,
                 refresh_rate=15,
                 mock_ros_comm=False
                 ):

        assert range_min < range_max, \
            "LiDAR's range_min ({}) must be smaller than it's range_max ({})"\
                .format(range_min, range_max)

        assert angle_min < angle_max, \
            "LiDAR's angle_min ({}) must be smaller than it's angle_max ({})"\
                .format(angle_min, angle_max)

        assert 0 < number_of_rays, \
            "LiDAR's number_of_rays ({}) must be positive integer"\
                .format(number_of_rays)

        assert number_of_rays < pybullet.MAX_RAY_INTERSECTION_BATCH_SIZE, \
            "LiDAR's number_of_rays({}) exceeds PyBullet's MAX_RAY_INTERSECTION_BATCH_SIZE ({})" \
                .format(number_of_rays, pybullet.MAX_RAY_INTERSECTION_BATCH_SIZE)

        assert 0 < refresh_rate, \
            "LiDAR's refresh_rate ({}) must be positive integer"\
                .format(refresh_rate)

        self._frame = frame
        self._topic = topic
        self._range_min = float(range_min)
        self._range_max = float(range_max)
        self._angle_min = float(angle_min)
        self._angle_max = float(angle_max)
        self._number_of_rays = number_of_rays
        self._refresh_rate = refresh_rate

        self._scan_angles = []
        self._laser_rays = []

        self._laser_scan_publisher = None
        self._tf_listener = None

        self._matrix_lidar_in_world_frame = None

        self._mock_ros_comm = mock_ros_comm
        if not self._mock_ros_comm:
            self._laser_scan_publisher = rospy.Publisher(self._topic, LaserScan, queue_size=10)
            self._tf_listener = tf.TransformListener()
            self._tf_publisher = tf.TransformBroadcaster()
            self._msg = LaserScan()
            self._msg.header.frame_id = self._frame
            self._msg.range_min = self._range_min
            self._msg.range_max = self._range_max
            self._msg.angle_min = self._angle_min
            self._msg.angle_max = self._angle_max
            self._msg.angle_increment = (self._angle_max - self._angle_min) / self._number_of_rays
            self._msg.scan_time = 1.0 / self._refresh_rate
            self._msg.time_increment = self._msg.scan_time / self._number_of_rays

            rospy.loginfo("ROS LIDAR INITIALIZED")

    @staticmethod
    def _matrix_from_transform(transform):
        assert len(transform) == 2
        assert len(transform[0]) == 3
        assert len(transform[1]) == 4
        translation_matrix = tf.transformations.translation_matrix(transform[0])
        rotation_matrix = tf.transformations.quaternion_matrix(transform[1])
        return np.matmul(translation_matrix, rotation_matrix)

    @staticmethod
    def _matrix_from_xyz_rpy(xyz, rpy):
        assert len(xyz) == 3
        assert len(rpy) == 3
        rotation_matrix = tf.transformations.euler_matrix(rpy[0], rpy[1], rpy[2])
        translation_matrix = tf.transformations.translation_matrix(xyz)
        return np.matmul(rotation_matrix, translation_matrix)

    @staticmethod
    def _position_from_matrix(matrix):
        assert len(matrix) == 4
        assert len(matrix[0]) == 4
        return [matrix[0][3], matrix[1][3], matrix[2][3]]

    @staticmethod
    def _orientation_quat_from_matrix(matrix):
        assert len(matrix) == 4
        assert len(matrix[0]) == 4
        return tf.transformations.quaternion_from_matrix(matrix)

    def _compute_scan_angles(self):

        assert isinstance(self._number_of_rays, int)
        assert self._number_of_rays > 1

        self._scan_angles = []
        for x in range(self._number_of_rays):
            self._scan_angles.append(
                self._angle_min + (self._angle_max - self._angle_min) * x / (self._number_of_rays - 1)
            )

    def _compute_laser_rays(self):

        self._laser_rays = []
        i = 0
        for angle in self._scan_angles:
            i += 1
            matrix_starting_point_in_lidar_frame = \
                self._matrix_from_xyz_rpy(
                    xyz=[self._range_min, 0, 0],
                    rpy=[0, 0, angle]
                )
            matrix_ending_point_in_lidar_frame = \
                self._matrix_from_xyz_rpy(
                    xyz=[self._range_max, 0, 0],
                    rpy=[0, 0, angle]
                )

            matrix_starting_point_in_world_frame = \
                np.dot(self._matrix_lidar_in_world_frame, matrix_starting_point_in_lidar_frame)
            matrix_ending_point_in_world_frame = \
                np.dot(self._matrix_lidar_in_world_frame, matrix_ending_point_in_lidar_frame)

            # self._tf_publisher.sendTransform(
            #     translation=self._position_from_matrix(matrix_starting_point_in_world_frame),
            #     rotation=self._orientation_quat_from_matrix(matrix_starting_point_in_world_frame),
            #     time=rospy.Time.now(),
            #     child="t_{}".format(i),
            #     parent="world",
            # )

            self._laser_rays.append([
                matrix_starting_point_in_world_frame,
                matrix_ending_point_in_world_frame,
            ])

    def _compute_lidar_in_world_frame(self, mock_tf=None):

        if mock_tf:
            tf_lidar_in_world_frame = mock_tf
        else:
            assert not self._mock_ros_comm
            tf_lidar_in_world_frame = self._tf_listener.lookupTransform("/world", self._frame, rospy.Time(0))

        self._matrix_lidar_in_world_frame = self._matrix_from_transform(tf_lidar_in_world_frame)

    def _update_laser_rays(self, mock_tf=None):

        self._compute_lidar_in_world_frame(mock_tf)
        self._compute_scan_angles()
        self._compute_laser_rays()

    def _do_ray_tracing_and_publish(self):

        ray_from_position_list = [self._position_from_matrix(x[0]) for x in self._laser_rays]
        ray_to_position_list = [self._position_from_matrix(x[1]) for x in self._laser_rays]

        # for i in range(len(ray_from_position_list)):
        #     rospy.loginfo("RAY {}: from {} to {}".format(i, ray_from_position_list[i], ray_to_position_list[i]))

        ray_tracing_output = pybullet.rayTestBatch(
            rayFromPositions=ray_from_position_list,
            rayToPositions=ray_to_position_list)

        self._msg.header.stamp = rospy.Time.now()
        self._msg.ranges = []

        for ray in ray_tracing_output:
            self._msg.ranges.append(
                self._range_min + (self._range_max - self._range_min) * ray[2] - 0.01
            )
        self._laser_scan_publisher.publish(self._msg)

    def update(self):

        assert self._mock_ros_comm is False

        try:
            self._update_laser_rays()
            self._do_ray_tracing_and_publish()
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn("Failed to update _matrix_lidar_in_world_frame:\n{}".format(e))


if __name__ == '__main__':

    rospy.init_node("pybullet_lidar")
    rospy.sleep(0.1)

    pybullet.connect(pybullet.SHARED_MEMORY)

    lidar_handler = PyBulletLidar2D(
        frame="lidar",
        topic="lidar",
    )

    while not rospy.is_shutdown():
        lidar_handler.update()
        rospy.sleep(1.0/240)
