#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import queue
import string
import sys

import numpy as np

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
import re

import carla


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [
        x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)
    ]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def main(data_name, port):
    client = carla.Client('localhost', port)
    client.set_timeout(20.0)

    label_list = []
    if os.path.exists(f"data/{data_name}/label.pkl"):
        with open(f"data/{data_name}/label.pkl", "rb") as f:
            label_list = pickle.load(f)

    idx = 0
    for t in ['10HD']:

        # Once we have a client we can retrieve the world that is currently
        # running.
        print(f"Loading town {t}...")
        # world = client.load_world('Town01')
        world = client.load_world('Town10HD')
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1 / 30.
        world.apply_settings(settings)
        weather = find_weather_presets()
        world.set_weather(getattr(carla.WeatherParameters, "ClearNoon"))

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

        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '1280')
        camera_bp.set_attribute('image_size_y', '720')
        camera_bp.set_attribute('fov', '47.1')

        # depth camera

        depth_bp = blueprint_library.find('sensor.camera.depth')
        depth_bp.set_attribute('image_size_x', '1280')
        depth_bp.set_attribute('image_size_y', '720')
        depth_bp.set_attribute('fov', '47.1')

        # semantic camera

        semantic_bp = blueprint_library.find(
            'sensor.camera.semantic_segmentation')
        semantic_bp.set_attribute('image_size_x', '1280')
        semantic_bp.set_attribute('image_size_y', '720')
        semantic_bp.set_attribute('fov', '47.1')

        # Now we need to give an initial transform to the vehicle. We choose a
        # random transform from the list of recommended spawn points of the map.
        transform = carla.Transform()

        for i in range(50):
            print(f"    Picking location {i + 1}...")
            # bp = blueprint_library.find(
            #     f'walker.pedestrian.00{random.randint(1, 20):02d}')
            ped = None
            while ped is None:
                location = world.get_random_location_from_navigation()
                # so here we are truly random sample from locations
                transform.location = location
                # So let's tell the world to spawn the pedestrian.
                ped = world.try_spawn_actor(bp, transform)

            # transform we give here is now relative to the pedestrian.
            camera_transform = carla.Transform(carla.Location(x=-10, z=0))
            camera = world.spawn_actor(camera_bp,
                                       camera_transform,
                                       attach_to=ped)
            image_queue = queue.Queue()
            camera.listen(image_queue.put)
            world.tick()
            transform = ped.get_transform()
            rgb_image = image_queue.get()
            rgb_image.save_to_disk(f'data/{data_name}/image/test_ped.png')

            ped.destroy()
            camera.destroy()

            loc_id = t + '_' + ''.join(random.choices(string.ascii_letters,
                                                      k=4))

            for j in range(5):
                print(f"        Processing camera view {j + 1}...")
                # world.set_weather(random.choice(weather)[0])

                # angle = j * 90
                angle = random.random() * 360
                idx += 1
                # id = str(idx).zfill(6)
                id = loc_id + '_' + ''.join(
                    random.choices(string.ascii_letters, k=4))

                sensor_transform = carla.Transform(transform.location,
                                                   transform.rotation)

                sensor_transform.rotation.yaw += (angle - 180)
                dist = random.random() * 10 + 5
                sensor_transform.location.x += dist * math.cos(
                    angle * math.pi / 180)
                sensor_transform.location.y += dist * math.sin(
                    angle * math.pi / 180)

                camera = world.spawn_actor(camera_bp, sensor_transform)
                image_queue = queue.Queue()
                camera.listen(image_queue.put)

                depth = world.spawn_actor(depth_bp, sensor_transform)
                depth_queue = queue.Queue()
                depth.listen(depth_queue.put)

                semantic = world.spawn_actor(
                    semantic_bp,
                    sensor_transform,
                )
                semantic_queue = queue.Queue()
                semantic.listen(semantic_queue.put)

                world.tick()

                rgb_image = image_queue.get()
                rgb_image.save_to_disk(f'data/{data_name}/image/{id}.png')

                depth_image = depth_queue.get()
                depth_image.save_to_disk(f'data/{data_name}/depth/{id}.png')

                depth_image.save_to_disk(f'data/{data_name}/depth/{id}_vis.png',
                                         carla.ColorConverter.Depth)

                semantic_image = semantic_queue.get()
                semantic_image.save_to_disk(
                    f'data/{data_name}/semantic/{id}.png')

                semantic_image.save_to_disk(
                    f'data/{data_name}/semantic/{id}_vis.png',
                    carla.ColorConverter.CityScapesPalette)

                width = 1280 * dist / 1468.6
                shift = random.random() * width / 2 + width / 4.

                sensor_to_ped_trans = np.array([[dist], [shift - width / 2]])
                angle = sensor_transform.rotation.yaw * np.pi / 180.
                sensor_to_ped_rot = np.array([[np.cos(angle), -np.sin(angle)],
                                              [np.sin(angle),
                                               np.cos(angle)]])
                sensor_to_ped = sensor_to_ped_rot @ sensor_to_ped_trans

                ped_transform = carla.Transform(sensor_transform.location,
                                                sensor_transform.rotation)
                ped_transform.location.x += float(sensor_to_ped[0])
                ped_transform.location.y += float(sensor_to_ped[1])
                # ped_transform.rotation.yaw += random.random() * 180
                ped = world.try_spawn_actor(bp, ped_transform)
                if ped is not None:
                    world.tick()
                    rgb_ped_image = image_queue.get()
                    rgb_ped_image.save_to_disk(
                        f'data/{data_name}/image/{id}_ped.png')
                    ped.destroy()

                    ped = None
                    while ped is None:
                        goal_angle = random.random() * np.pi * 2
                        goal_dist = random.random() * 2 + 1
                        ped_goal_transform = carla.Transform(
                            ped_transform.location, ped_transform.rotation)
                        ped_goal_transform.location.x += np.cos(
                            goal_angle) * goal_dist
                        ped_goal_transform.location.y += np.sin(
                            goal_angle) * goal_dist
                        ped = world.try_spawn_actor(bp, ped_goal_transform)

                    world.tick()
                    rgb_ped_image = image_queue.get()
                    rgb_ped_image.save_to_disk(
                        f'data/{data_name}/image/{id}_ped_goal.png')
                    ped.destroy()

                    label_dict = {}
                    label_dict["map_info"] = f"Town{t}"
                    label_dict["actor_info"] = "walker.pedestrian.0001"
                    label_dict["image"] = f'{id}.png'
                    label_dict["global_trans"] = np.array(
                        [shift - width / 2, 0.16, dist],  # hardcoded
                        dtype=np.float32)

                    label_dict["global_trans_goal"] = np.array(
                        [
                            shift - width / 2 +
                            np.sin(goal_angle - angle) * goal_dist, 0.16,
                            dist + np.cos(goal_angle - angle) * goal_dist
                        ],  # hardcoded
                        dtype=np.float32)

                    label_dict["betas"] = np.array(
                        [
                            -0.16181, 1.47, 0.72749, 0.3814, 0.36384, 0.15637,
                            -0.14787, 0.94152, 0.65114, 0.76266
                        ],  # hardcoded
                        dtype=np.float32)

                    label_dict["sensor_pose"] = np.array([
                        sensor_transform.location.x,
                        sensor_transform.location.y,
                        sensor_transform.location.z,
                        sensor_transform.rotation.yaw
                    ],
                                                         dtype=np.float32)

                    label_list.append(label_dict)
                camera.destroy()
                depth.destroy()
                semantic.destroy()
    with open(f"data/{data_name}/label.pkl", "wb") as f:
        pickle.dump(label_list, f)


if __name__ == '__main__':
    data_name = sys.argv[1]
    port = int(sys.argv[2])
    main(data_name, port)
