#/usr/bin/env bash

roscore &
sleep 3

rosparam set robot_description "$(rosrun xacro xacro ~/code/pybullet-training/simple_mobile_platform.xacro)"
sleep 1

rosrun robot_state_publisher robot_state_publisher &
rosrun joint_state_publisher joint_state_publisher &
rosrun rviz rviz

killall rosmaster