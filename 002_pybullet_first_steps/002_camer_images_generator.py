#!/usr/bin/env python3

import pybullet as pb
import pybullet_data as pbd

import os
import glob
import random
import time
from PIL import Image
from datetime import datetime
import numpy as np

SIMULATION_TIME_STEP = 1.0/240.0


class PhysicsSimulator:
    def __init__(self):
        # Start physics engine with GUI and connect to it
        self._physic_client = pb.connect(pb.GUI)  # type: int

        self._load_plane()

        self._loaded_objects = []  # type: [int]

        pb.setGravity(0, 0, -9.8)

    def _load_plane(self):
        # Load a URDF plane
        pb.setAdditionalSearchPath(pbd.getDataPath())
        self.plane_id = pb.loadURDF('plane.urdf')  # type: int

    def create_object(self):
        random_id = random.randint(0, 999)

        visual_id = pb.createVisualShape(
            shapeType=pb.GEOM_MESH,
            fileName=f'random_urdfs/{random_id:03}/{random_id:03}.obj',
            rgbaColor=None,
            meshScale=[0.1, 0.1, 0.1])

        collision_id = pb.createCollisionShape(
            shapeType=pb.GEOM_MESH,
            fileName=f'random_urdfs/{random_id:03}/{random_id:03}_coll.obj',
            meshScale=[0.1, 0.1, 0.1])

        link_id = pb.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=visual_id,
            baseVisualShapeIndex=collision_id,
            basePosition=[0, 0, 3],
            baseOrientation=pb.getQuaternionFromEuler([0, 0, 0]))

        texture_paths = glob.glob(os.path.join('dtd', '**', '*.jpg'), recursive=True)
        random_texture_path = texture_paths[random.randint(0, len(texture_paths) - 1)]
        texture_id = pb.loadTexture(random_texture_path)
        pb.changeVisualShape(link_id, -1, textureUniqueId=texture_id)

        self._loaded_objects.append(link_id)

    def delete_object(self, number=None):
        if not number:
            number = random.randint(0, len(self._loaded_objects) - 1)
        pb.removeBody(self._loaded_objects[number])
        del self._loaded_objects[number]

    @staticmethod
    def snap_from_camera(camera_eye_position, camera_target_position, save_to_file=None):
        view_matrix = pb.computeViewMatrix(
            cameraEyePosition=camera_eye_position,
            cameraTargetPosition=camera_target_position,
            cameraUpVector=[0, 1, 0])

        projection_matrix = pb.computeProjectionMatrixFOV(
            fov=30.0,
            aspect=1.5,
            nearVal=3,
            farVal=5.1)

        width, height, rgb_img, depth_img, seg_img = pb.getCameraImage(
            width=320,
            height=240,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix)  # type: int, int, np.ndarray, np.ndarray, np.ndarray

        if save_to_file:
            rgb_img = rgb_img
            depth_img = depth_img*255
            seg_img = seg_img*255/seg_img.max()
            Image.fromarray(rgb_img).convert("RGB").save(f"results/{save_to_file}-rgb.png")
            Image.fromarray(depth_img).convert("L").save(f"results/{save_to_file}-depth.png")
            Image.fromarray(seg_img).convert("L").save(f"results/{save_to_file}-seg.png")

    @staticmethod
    def run_simulation_steps(steps=240):
        for i in range(steps):
            pb.stepSimulation()
            time.sleep(SIMULATION_TIME_STEP)


def main():
    physics_sim = PhysicsSimulator()

    for i in range(3):
        physics_sim.create_object()
        physics_sim.run_simulation_steps()

    for i in range(100):
        date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        physics_sim.snap_from_camera([0.1, 0, 5], [0.1, 0, 0], date_str+"-cam0")
        physics_sim.snap_from_camera([-0.1, 0, 5], [-0.1, 0, 0], date_str+"-cam1")
        physics_sim.delete_object()
        physics_sim.create_object()
        physics_sim.run_simulation_steps()


if __name__ == '__main__':
    main()
