# Copyright (c) 2022 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

"""Example casting a single ray using the ray cast service.
"""
# pylint: disable=missing-function-docstring
# pylint: disable=consider-using-f-string
import argparse

import bosdyn.client

from bosdyn.api import ray_cast_pb2
from bosdyn.client.ray_cast import RayCastClient
from bosdyn.client.util import add_base_arguments, setup_logging
from bosdyn.client.math_helpers import Vec3


def ray_intersection_type_strings():
    names = ray_cast_pb2.RayIntersection.Type.keys()
    return names[1:]


def ray_intersection_type_strings_to_enum(strings):
    retval = []
    type_dict = dict(ray_cast_pb2.RayIntersection.Type.items())
    for enum_string in strings:
        retval.append(type_dict[enum_string])
    return retval


def main():
    setup_logging()

    sdk = bosdyn.client.create_standard_sdk('cast-single-ray')
    robot = sdk.create_robot('192.168.80.3')
    bosdyn.client.util.authenticate(robot)

    rc_client = robot.ensure_client(RayCastClient.default_service_name)

    raycast_types = ray_intersection_type_strings_to_enum(options.type)
    # ray_origin = Vec3(*options.ray_origin)
    ray_origin = Vec3(0.5, 0, 0)
    # ray_direction = Vec3(*options.ray_direction)
    ray_direction = Vec3(1, 0, -1)
    # ray_frame_name = options.frame_name
    ray_frame_name = "body"
    # min_distance = options.min_distance
    min_distance = 0.0

    print("Raycasting from position: {}".format(ray_origin))
    print("Raycasting in direction: {}".format(ray_direction))

    response = rc_client.raycast(ray_origin, ray_direction, raycast_types,
                                 min_distance=min_distance, frame_name=ray_frame_name)

    print('Raycast returned {} hits.'.format(len(response.hits)))
    for idx, hit in enumerate(response.hits):
        print('Hit {}:'.format(idx))
        hit_position = Vec3.from_proto(hit.hit_position_in_hit_frame)
        print('\tPosition: {}'.format(hit_position))
        hit_type_str = ray_cast_pb2.RayIntersection.Type.keys()[hit.type]
        print('\tType: {}'.format(hit_type_str))
        print('\tDistance: {}'.format(hit.distance_meters))


if __name__ == '__main__':
    main()
