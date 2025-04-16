#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

try:
    sys.path.append(
        glob.glob(
            '/home/zhizheng/ucla_metadrive/carla_0.9.15/PythonAPI/carla/dist/carla-*%d.%d-%s.egg'
            % (sys.version_info.major, sys.version_info.minor,
               'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import math
import pickle
import random
import time

import carla
from tqdm import tqdm


def main(result_dict, eval_folder, port):
    eval_list = []
    success_list = []
    collision_list = []
    floating_list = []
    # client = carla.Client('localhost', 2000)
    client = carla.Client('localhost', port)
    client.set_timeout(20.0)
    print("CONNECTED!")

    # Once we have a client we can retrieve the world that is currently
    # running.

    for carla_map in result_dict.keys():
        world = client.load_world(carla_map)
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1 / 30.
        settings.no_rendering_mode = True
        world.apply_settings(settings)
        world.set_weather(getattr(carla.WeatherParameters, "ClearNoon"))
        print(f"WORLD {carla_map} LOADED!")
        result_list = result_dict[carla_map]
        # The world contains the list blueprints that we can use for adding new
        # actors into the simulation.
        blueprint_library = world.get_blueprint_library()

        # Now let's filter all the blueprints of type 'vehicle' and choose one
        # at random.
        bp = blueprint_library.find('walker.pedestrian.0001')

        # A blueprint contains the list of attributes that define a vehicle's
        # instance, we can read them and modify some of them. For instance,
        # let's randomize its color.
        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)

        # camera_bp = blueprint_library.find('sensor.camera.rgb')
        # camera_bp.set_attribute('image_size_x', '1280')
        # camera_bp.set_attribute('image_size_y', '720')
        # camera_bp.set_attribute('fov', '47.1')
        # # camera_bp.set_attribute('sensor_tick', '1/30.')

        # # BEV camera

        # bev_bp = blueprint_library.find('sensor.camera.rgb')
        # bev_bp.set_attribute('image_size_x', '1000')
        # bev_bp.set_attribute('image_size_y', '1000')
        # bev_bp.set_attribute('fov', '60')
        # bev_transform = carla.Transform(carla.Location(x=10, z=10),
        #                                 carla.Rotation(pitch=-90))

        print("BEGIN EVALUATION...")
        for result in tqdm(result_list):
            eval_dict = {}
            eval_dict["image"] = result["image"]
            eval_dict["pred_id"] = result["pred_id"]
            eval_dict["status"] = np.zeros(
                60,
            )  # 0 means good, 1 means in collision, 2 means floating or intersecting with the ground
            sensor_pose = result["sensor_pose"].tolist()

            camera_transform = carla.Transform(
                carla.Location(x=sensor_pose[0],
                               y=sensor_pose[1],
                               z=sensor_pose[2]),
                carla.Rotation(roll=0, pitch=0, yaw=sensor_pose[3]))
            # camera = world.spawn_actor(camera_bp, camera_transform)
            # image_queue = queue.Queue()
            # camera.listen(image_queue.put)
            # bev = world.spawn_actor(bev_bp, bev_transform, attach_to=camera)
            # bev_queue = queue.Queue()
            # bev.listen(bev_queue.put)
            collision = False
            floating = False
            timestamp = 0
            while timestamp < 60:
                trans = result["global_trans"][timestamp]
                rot = result["global_orient"][timestamp]

                cam_to_carla = np.array([[0, 0, 1], [1, 0, 0], [0, -1, 0]],
                                        dtype=np.float32)

                trans = cam_to_carla @ trans[:, np.newaxis]
                t = trans[:, 0]
                r = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]],
                             dtype=np.float32) @ rot

                yaw = np.arctan2(-r[2, 0], r[0, 0]) * 180 / math.pi
                yaw = -yaw + 180
                # transform = carla.Transform(carla.Location(x=t[0], y=t[1], z=t[2]),
                #                             carla.Rotation(roll=0, pitch=0, yaw=0))
                transform = carla.Transform(camera_transform.location,
                                            camera_transform.rotation)
                angle = sensor_pose[3] * math.pi / 180

                transform.location.x += t[0] * math.cos(
                    angle) - t[1] * math.sin(angle)
                transform.location.y += t[0] * math.sin(
                    angle) + t[1] * math.cos(angle)
                transform.location.z += t[2] + 0.16  # hardcoded

                transform.rotation.yaw += yaw

                ped = world.try_spawn_actor(bp, transform)

                world.tick()
                # rgb_image = image_queue.get()
                # rgb_image.save_to_disk(
                #     f'data/carla/{eval_folder}/{result["image"]}/{result["pred_id"]}/{timestamp}.png'
                # )

                # bev_image = bev_queue.get()
                # bev_image.save_to_disk(
                #     f'data/carla/{eval_folder}/{result["image"]}/{result["pred_id"]}_bev/{timestamp}.png'
                # )
                if ped is None:
                    eval_dict["status"][timestamp] = 1
                    collision = True
                else:
                    real_transform = ped.get_transform()
                    if abs(transform.location.z -
                           real_transform.location.z) > 0.2:
                        eval_dict["status"][timestamp] = 2
                        floating = True
                    ped.destroy()

                timestamp += 1

            collision_list.append(collision)
            floating_list.append(floating)
            success_list.append(not (collision or floating))
            # camera.destroy()
            # bev.destroy()
            eval_list.append(eval_dict)
            collision_rate = sum(collision_list) / len(collision_list)
            print(f"collision rate: {collision_rate}")
            floating_rate = sum(floating_list) / len(floating_list)
            print(f"floating rate: {floating_rate}")
            success_rate = sum(success_list) / len(success_list)
            print(f"success rate: {success_rate}")

    with open(f"data/carla/{eval_folder}/eval.pkl", "wb") as f:
        pickle.dump(eval_list, f)


if __name__ == '__main__':
    result_path = sys.argv[1]
    eval_folder = sys.argv[2]
    port = int(sys.argv[3])
    with open(f"{result_path}/vis_test/epoch_0/result.pkl", "rb") as f:
        result_list = pickle.load(f)

    # result_dict
    result_dict = {}
    for result in result_list:
        if result["map_info"] not in result_dict:
            result_dict[result["map_info"]] = []
        result_dict[result["map_info"]].append(result)

    start_time = time.time()
    main(result_dict, eval_folder, port)
    print("--- %s seconds ---" % (time.time() - start_time))
