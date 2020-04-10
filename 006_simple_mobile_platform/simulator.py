import rospy
import tf
from sensor_msgs.msg import JointState

import pybullet as p
import pybullet_data as p_data
import time

import os
import subprocess

from pprint import pprint


class PhysicsEngine(object):
    def __init__(self):
        p.connect(p.GUI_SERVER)
        p.setGravity(0, 0, -9.81)
        self.simulation_step = 1.0/240.0
        p.setAdditionalSearchPath(p_data.getDataPath())
        self.planeId = p.loadURDF("plane.urdf")

    def simulate_step(self):
        p.stepSimulation()
        time.sleep(self.simulation_step)

    def disconnect(self):
        p.disconnect()


class ROSPublisher(object):
    def __init__(self):
        self.joint_pub = rospy.Publisher("joint_states", JointState, queue_size=10)
        self.joints_msg = JointState()
        
        self.tf_broadcaster = tf.TransformBroadcaster()

    def publish(self, name, position, velocity, effort, pose):
        self.joints_msg.header.frame_id = "base_link"
        self.joints_msg.header.stamp = rospy.Time.now()
        self.joints_msg.name = name
        self.joints_msg.position = position
        self.joints_msg.velocity = velocity
        self.joints_msg.effort = effort

        self.joint_pub.publish(self.joints_msg)

        self.tf_broadcaster.sendTransform(
            translation=pose[0],
            rotation=pose[1],
            time=rospy.Time.now(),
            child="base_link",
            parent="world",
        )


class SceneObject(object):
    def __init__(self, xacro_path, position, orientation):
        urdf_path = xacro_path + ".urdf"
        subprocess.check_call("rosrun xacro xacro.py {} > {}".format(xacro_path, urdf_path), shell=True)
        self.object_id = p.loadURDF(urdf_path, position, orientation)


class MobileRobot(object):
    def __init__(self, xacro_path):
        urdf_path = xacro_path + ".urdf"
        subprocess.check_call("rosrun xacro xacro.py {} > {}".format(xacro_path, urdf_path), shell=True)

        with open(urdf_path, 'r') as urdf:
            rospy.set_param("/robot_description", urdf.read())

        self.robot_id = p.loadURDF(urdf_path, [1, 2, 1], [0, 0, 0, 1])

        self.joint_states_pub = ROSPublisher()

        self.num_joints = p.getNumJoints(self.robot_id)
        self.joints_info = [p.getJointInfo(self.robot_id, i) for i in range(self.num_joints)]
        self.joints_states = None
        self.robot_pose = None

    def drive(self, propulsion_wheel_l_target_vel, propulsion_wheel_r_target_vel, max_force):
        target_velocities = [0]*self.num_joints
        forces = [0]*self.num_joints
        for i in range(self.num_joints):
            if p.getJointInfo(self.robot_id, i)[1] == 'propulsion_wheel_l_joint':
                target_velocities[i] = propulsion_wheel_l_target_vel
                forces[i] = max_force
            if p.getJointInfo(self.robot_id, i)[1] == 'propulsion_wheel_r_joint':
                target_velocities[i] = propulsion_wheel_r_target_vel
                forces[i] = max_force

        p.setJointMotorControlArray(
            bodyUniqueId=self.robot_id,
            jointIndices=range(self.num_joints),
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=target_velocities,
            forces=forces,
        )

    def update_joints_state(self):
        self.robot_pose = p.getBasePositionAndOrientation(self.robot_id)
        self.joints_states = p.getJointStates(self.robot_id, range(self.num_joints))

    def publish_joints_state(self):
        self.joint_states_pub.publish(
            name=[info[1] for info in self.joints_info],
            position=[state[0] for state in self.joints_states],
            velocity=[state[1] for state in self.joints_states],
            effort=[state[3] for state in self.joints_states],
            pose=self.robot_pose,
        )

    def print_robot_info(self):
        pprint("===================================")
        pprint("ROBOT JOINT INFO:")
        pprint(self.joints_info)

    def print_robot_status(self):
        pprint("===================================")
        pprint("ROBOT POSITION:")
        pprint(self.robot_pose)
        pprint("ROBOT JOINT STATES:")
        pprint(self.joints_states)


def main():
    rospy.init_node("simulator_bridge")
    engine = PhysicsEngine()

    robot_path = os.getcwd()+"/simple_mobile_platform.xacro"
    robot = MobileRobot(robot_path)

    box_path = os.getcwd()+"/simple_box.xacro"
    box1 = SceneObject(box_path, [5, 5, 1], [0, 0, 0, 1])
    box2 = SceneObject(box_path, [-5, 5, 1], [0, 0, 0, 1])
    box3 = SceneObject(box_path, [5, -5, 1], [0, 0, 0, 1])
    box4 = SceneObject(box_path, [-5, -5, 1], [0, 0, 0, 1])

    # robot.print_robot_info()

    counter = 0
    while not rospy.is_shutdown():
        robot.drive(
            propulsion_wheel_l_target_vel=-3.0,
            propulsion_wheel_r_target_vel=-2.0,
            max_force=5000,
        )

        robot.update_joints_state()

        engine.simulate_step()

        counter += 1
        if counter % 8 == 0:
            robot.publish_joints_state()
        # if counter % 240 == 0:
        #     robot.print_robot_status()

    engine.disconnect()


if __name__ == '__main__':
    main()
