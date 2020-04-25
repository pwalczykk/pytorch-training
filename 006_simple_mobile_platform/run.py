#!/usr/bin/env python2

import os
import subprocess
import time


def wait_for_roscore():
    while True:
        try:
            subprocess.check_call('rostopic list'.split())
            break
        except subprocess.CalledProcessError:
            time.sleep(0.05)
            pass


def wait_for_rosparam(param):
    while True:
        try:
            subprocess.check_call('rosparam get {}'.format(param).split())
            break
        except subprocess.CalledProcessError:
            time.sleep(0.05)
            pass


def wait_for_rostopic(topic):
    while True:
        try:
            # TODO this could wait for the first message on the topic
            subprocess.check_call('rostopic info {}'.format(topic).split())
            break
        except subprocess.CalledProcessError:
            time.sleep(0.05)
            pass


if __name__ == '__main__':

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # noinspection PyListCreation
    process_list = []

    process_list.append(subprocess.Popen('roscore'.split()))
    wait_for_roscore()
    process_list.append(subprocess.Popen('python2 simulator.py'.split()))
    wait_for_rosparam("/robot_description")
    process_list.append(subprocess.Popen('python2 pybullet_lidar.py'.split()))
    wait_for_rostopic("/lidar")
    process_list.append(subprocess.Popen('rosrun robot_state_publisher robot_state_publisher'.split()))
    process_list.append(subprocess.Popen('rosrun rviz rviz -d config.rviz'.split()))

    while True:
        time.sleep(1.0)

