#!/usr/bin/env python3

import pybullet
import pybullet_data
import time

physicsClient = pybullet.connect(pybullet.GUI)
pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
pybullet.setGravity(0, 0, -9.81)
planeId = pybullet.loadURDF("plane.urdf")
cubeStartPosition = [0, 0, 1]
cubeStartOrientation = pybullet.getQuaternionFromEuler([0, 0, 0])
boxId = pybullet.loadURDF("r2d2.urdf", cubeStartPosition, cubeStartOrientation)
for i in range (10000):
    pybullet.stepSimulation()
    time.sleep(1.0/240.0)
cubePos, cubeOrn = pybullet.getBasePositionAndOrientation(boxId)
print(cubePos, cubeOrn)
pybullet.disconnect()
