# Copyright (c) 2022 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

import argparse
import numpy as np
import os
import sys

from bosdyn.api.graph_nav import map_pb2
from bosdyn.client.frame_helpers import *
from bosdyn.client.math_helpers import *

"""
This example shows how to load and view a graph nav map.

"""


def write_ply(data, output):
    """
    Writes an ASCII PLY file to the output file path.
    """
    print('Saving to {}'.format(output))
    with open(output, 'w') as f:
        num_points = data.shape[0]
        f.write(
            'ply\nformat ascii 1.0\nelement vertex {}\nproperty float x\nproperty float y\nproperty float '
            'z\nend_header\n'.format(
                num_points))

        for i in range(0, num_points):
            (x, y, z) = data[i, :]
            f.write('{} {} {}\n'.format(x, y, z))


def load_map(path):
    """
    Load a map from the given file path.
    :param path: Path to the root directory of the map.
    :return: the graph, waypoints, waypoint snapshots and edge snapshots.
    """
    with open(os.path.join(path, "graph"), "rb") as graph_file:
        # Load the graph file and deserialize it. The graph file is a protobuf containing only the waypoints and the
        # edges between them.
        data = graph_file.read()
        current_graph = map_pb2.Graph()
        current_graph.ParseFromString(data)

        # Set up maps from waypoint ID to waypoints, edges, snapshots, etc.
        current_waypoints = {}
        current_waypoint_snapshots = {}
        current_edge_snapshots = {}
        current_anchors = {}
        current_anchored_world_objects = {}

        # Load the anchored world objects first so we can look in each waypoint snapshot as we load it.
        for anchored_world_object in current_graph.anchoring.objects:
            current_anchored_world_objects[anchored_world_object.id] = (anchored_world_object,)
        # For each waypoint, load any snapshot associated with it.
        for waypoint in current_graph.waypoints:
            current_waypoints[waypoint.id] = waypoint

            if len(waypoint.snapshot_id) == 0:
                continue
            # Load the snapshot. Note that snapshots contain all of the raw data in a waypoint and may be large.
            file_name = os.path.join(path, "waypoint_snapshots", waypoint.snapshot_id)
            if not os.path.exists(file_name):
                continue
            with open(file_name, "rb") as snapshot_file:
                waypoint_snapshot = map_pb2.WaypointSnapshot()
                waypoint_snapshot.ParseFromString(snapshot_file.read())
                current_waypoint_snapshots[waypoint_snapshot.id] = waypoint_snapshot

                for fiducial in waypoint_snapshot.objects:
                    if not fiducial.HasField("apriltag_properties"):
                        continue

                    str_id = str(fiducial.apriltag_properties.tag_id)
                    if (str_id in current_anchored_world_objects and
                            len(current_anchored_world_objects[str_id]) == 1):
                        # Replace the placeholder tuple with a tuple of (wo, waypoint, fiducial).
                        anchored_wo = current_anchored_world_objects[str_id][0]
                        current_anchored_world_objects[str_id] = (anchored_wo, waypoint, fiducial)

        # Similarly, edges have snapshot data.
        for edge in current_graph.edges:
            if len(edge.snapshot_id) == 0:
                continue
            file_name = os.path.join(path, "edge_snapshots", edge.snapshot_id)
            if not os.path.exists(file_name):
                continue
            with open(file_name, "rb") as snapshot_file:
                edge_snapshot = map_pb2.EdgeSnapshot()
                edge_snapshot.ParseFromString(snapshot_file.read())
                current_edge_snapshots[edge_snapshot.id] = edge_snapshot
        for anchor in current_graph.anchoring.anchors:
            current_anchors[anchor.id] = anchor
        print("Loaded graph with {} waypoints, {} edges, {} anchors, and {} anchored world objects".
              format(len(current_graph.waypoints), len(current_graph.edges),
                     len(current_graph.anchoring.anchors), len(current_graph.anchoring.objects)))
        return (current_graph, current_waypoints, current_waypoint_snapshots,
                current_edge_snapshots, current_anchors, current_anchored_world_objects)


def create_graph_objects(current_graph, current_waypoint_snapshots, current_waypoints):
    """

    Args:
        current_graph:
        current_waypoint_snapshots:
        current_waypoints:

    Returns:

    """

    # Now, perform a breadth first search of the graph starting from an arbitrary waypoint. Graph nav graphs
    # have no global reference frame. The only thing we can say about waypoints is that they have relative
    # transformations to their neighbors via edges. So the goal is to get the whole graph into a global reference
    # frame centered on some waypoint as the origin.
    queue = [(current_graph.waypoints[0], np.eye(4))]
    visited = {}
    # Get the camera in the ballpark of the right position by centering it on the average position of a waypoint.
    avg_pos = np.array([0.0, 0.0, 0.0])

    data = None

    # Breadth first search.
    while len(queue) > 0:
        # Visit a waypoint.
        curr_element = queue[0]
        queue.pop(0)
        curr_waypoint = curr_element[0]
        if curr_waypoint.id in visited:
            continue
        visited[curr_waypoint.id] = True

        # extract point cloud for each waypoint and transform it to world frame
        snapshot = current_waypoint_snapshots[curr_waypoint.snapshot_id]
        cloud = snapshot.point_cloud
        point_cloud_data = np.frombuffer(cloud.data, dtype=np.float32).reshape(int(cloud.num_points), 3)

        odom_tform_cloud = get_a_tform_b(cloud.source.transforms_snapshot, ODOM_FRAME_NAME,
                                         cloud.source.frame_name_sensor)
        waypoint_tform_odom = SE3Pose.from_obj(curr_waypoint.waypoint_tform_ko)
        waypoint_tform_cloud = waypoint_tform_odom * odom_tform_cloud

        world_tform_current_waypoint = curr_element[1]

        world_tform_current_cloud = SE3Pose.from_matrix(world_tform_current_waypoint) * waypoint_tform_cloud
        cloud_data = world_tform_current_cloud.transform_cloud(point_cloud_data)

        if data is None:
            data = cloud_data
        else:
            data = np.concatenate((data, cloud_data))

        # Now, for each edge, walk along the edge and concatenate the transform to the neighbor.
        for edge in current_graph.edges:
            # If the edge is directed away from us...
            if edge.id.from_waypoint == curr_waypoint.id and edge.id.to_waypoint not in visited:
                current_waypoint_tform_to_waypoint = SE3Pose.from_obj(
                    edge.from_tform_to).to_matrix()

                world_tform_to_wp = np.dot(world_tform_current_waypoint, current_waypoint_tform_to_waypoint)

                # Add the neighbor to the queue.
                queue.append((current_waypoints[edge.id.to_waypoint], world_tform_to_wp))
                avg_pos += world_tform_to_wp[:3, 3]
            # If the edge is directed toward us...
            elif edge.id.to_waypoint == curr_waypoint.id and edge.id.from_waypoint not in visited:
                current_waypoint_tform_from_waypoint = (SE3Pose.from_obj(
                    edge.from_tform_to).inverse()).to_matrix()
                world_tform_from_wp = np.dot(world_tform_current_waypoint, current_waypoint_tform_from_waypoint)

                # Add the neighbor to the queue.
                queue.append((current_waypoints[edge.id.from_waypoint], world_tform_from_wp))
                avg_pos += world_tform_from_wp[:3, 3]

    return data


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)

    options = parser.parse_args(argv)

    options.path = os.path.join(os.getcwd(), "robotics_lab.walk")
    options.output = os.path.join(os.getcwd(), "cloud.ply")

    # Load the map from the given file.
    (current_graph, current_waypoints, current_waypoint_snapshots, current_edge_snapshots,
     current_anchors, current_anchored_world_objects) = load_map(options.path)

    cloud_data = create_graph_objects(current_graph, current_waypoint_snapshots, current_waypoints)

    write_ply(cloud_data, options.output)


if __name__ == '__main__':
    main(sys.argv[1:])
