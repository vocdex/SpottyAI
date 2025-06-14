#                                                                               
# Copyright (c) 2025, FAU, Shukrullo Nazirjonov
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#                                                                               
# Copyright (c) 2022 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

"""Command line interface for graph nav with options to download/upload a map and to navigate a map. """

import math
import os
import time
from typing import List, Tuple

import bosdyn.client.robot_command as robot_command
import numpy as np
from bosdyn.api import robot_state_pb2
from bosdyn.api.graph_nav import graph_nav_pb2, map_pb2, nav_pb2
from bosdyn.client.exceptions import ResponseError
from bosdyn.client.frame_helpers import get_odom_tform_body
from bosdyn.client.graph_nav import GraphNavClient
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive, ResourceAlreadyClaimedError
from bosdyn.client.math_helpers import Quat, SE3Pose
from bosdyn.client.power import PowerClient, power_on, safe_power_off
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.client.robot_state import RobotStateClient

from spotty.mapping.navigator import (
    ClosestWaypointStrategy,
    FarthestWaypointStrategy,
    NavigationCommand,
    NavigationResult,
    WaypointNavigator,
)
from spotty.utils import graph_nav_utils


class GraphNavInterface(object):
    """GraphNav service command line interface."""

    def __init__(self, robot, upload_path):
        self._robot = robot

        # Force trigger timesync.
        self._robot.time_sync.wait_for_sync()

        # Create the lease client with keep-alive, then acquire the lease.
        self._lease_client = self._robot.ensure_client(LeaseClient.default_service_name)
        self._lease_wallet = self._lease_client.lease_wallet
        try:
            self._lease = self._lease_client.acquire()
        except ResourceAlreadyClaimedError as err:
            print("The robot's lease is currently in use. Check for a tablet connection or try again in a few seconds.")
            os._exit(1)
        self._lease_keepalive = LeaseKeepAlive(self._lease_client)

        # Create robot state and command clients.
        self._robot_command_client = self._robot.ensure_client(RobotCommandClient.default_service_name)
        self._robot_state_client = self._robot.ensure_client(RobotStateClient.default_service_name)

        # Create the client for the Graph Nav main service.
        self._graph_nav_client = self._robot.ensure_client(GraphNavClient.default_service_name)

        # Create a power client for the robot.
        self._power_client = self._robot.ensure_client(PowerClient.default_service_name)

        # Boolean indicating the robot's power state.
        power_state = self._robot_state_client.get_robot_state().power_state
        self._started_powered_on = power_state.motor_power_state == power_state.STATE_ON
        self._powered_on = self._started_powered_on

        # Number of attempts to wait before trying to re-power on.
        self._max_attempts_to_wait = 50

        # Store the most recent knowledge of the state of the robot based on rpc calls.
        self._current_graph = None
        self._current_edges = dict()  # maps to_waypoint to list(from_waypoint)
        self._current_waypoint_snapshots = dict()  # maps id to waypoint snapshot
        self._current_edge_snapshots = dict()  # maps id to edge snapshot
        self._current_annotation_name_to_wp_id = dict()

        # Filepath for uploading a saved graph's and snapshots too.
        if upload_path[-1] == "/":
            self._upload_filepath = upload_path[:-1]
        else:
            self._upload_filepath = upload_path

        self._command_dictionary = {
            "1": self._get_localization_state,
            "2": self._set_initial_localization_fiducial,
            "3": self._set_initial_localization_waypoint,
            "4": self._list_graph_waypoint_and_edge_ids,
            "5": self._upload_graph_and_snapshots,
            "6": self._navigate_to,
            "7": self._navigate_route,
            "8": self._navigate_to_anchor,
            "9": self._clear_graph,
            "10": self._navigate_to_by_location,
            "11": self._initialize_map,
        }

    def _get_localization_state(self, *args):
        """Get the current localization and state of the robot."""
        state = self._graph_nav_client.get_localization_state()
        print("Got localization: \n%s" % str(state.localization))
        odom_tform_body = get_odom_tform_body(state.robot_kinematics.transforms_snapshot)
        print("Got robot state in kinematic odometry frame: \n%s" % str(odom_tform_body))

    def _set_initial_localization_fiducial(self, *args):
        """Trigger localization when near a fiducial."""
        robot_state = self._robot_state_client.get_robot_state()
        current_odom_tform_body = get_odom_tform_body(robot_state.kinematic_state.transforms_snapshot).to_proto()
        # Create an empty instance for initial localization since we are asking it to localize
        # based on the nearest fiducial.
        localization = nav_pb2.Localization()
        self._graph_nav_client.set_localization(
            initial_guess_localization=localization, ko_tform_body=current_odom_tform_body
        )

    def _set_initial_localization_waypoint(self, *args):
        """Trigger localization to a waypoint."""
        # Take the first argument as the localization waypoint.
        if len(args) < 1:
            # If no waypoint id is given as input, then return without initializing.
            print("No waypoint specified to initialize to.")
            return
        destination_waypoint = graph_nav_utils.find_unique_waypoint_id(
            args[0][0], self._current_graph, self._current_annotation_name_to_wp_id
        )
        if not destination_waypoint:
            # Failed to find the unique waypoint id.
            return

        robot_state = self._robot_state_client.get_robot_state()
        current_odom_tform_body = get_odom_tform_body(robot_state.kinematic_state.transforms_snapshot).to_proto()
        # Create an initial localization to the specified waypoint as the identity.
        localization = nav_pb2.Localization()
        localization.waypoint_id = destination_waypoint
        localization.waypoint_tform_body.rotation.w = 1.0
        self._graph_nav_client.set_localization(
            initial_guess_localization=localization,
            # It's hard to get the pose perfect, search +/-20 deg and +/-20cm (0.2m).
            max_distance=0.2,
            max_yaw=20.0 * math.pi / 180.0,
            fiducial_init=graph_nav_pb2.SetLocalizationRequest.FIDUCIAL_INIT_NO_FIDUCIAL,
            ko_tform_body=current_odom_tform_body,
        )

    def _list_graph_waypoint_and_edge_ids(self, *args):
        """List the waypoint ids and edge ids of the graph currently on the robot."""

        # Download current graph
        graph = self._graph_nav_client.download_graph()
        if graph is None:
            print("Empty graph.")
            return
        self._current_graph = graph

        localization_id = self._graph_nav_client.get_localization_state().localization.waypoint_id

        # Update and print waypoints and edges
        self._current_annotation_name_to_wp_id, self._current_edges = graph_nav_utils.update_waypoints_and_edges(
            graph, localization_id
        )

    def _upload_graph_and_snapshots(self, *args):
        """Upload the graph and snapshots to the robot."""
        print("Loading the graph from disk into local storage...")
        with open(self._upload_filepath + "/graph", "rb") as graph_file:
            # Load the graph from disk.
            data = graph_file.read()
            self._current_graph = map_pb2.Graph()
            self._current_graph.ParseFromString(data)
            print(
                "Loaded graph has {} waypoints and {} edges".format(
                    len(self._current_graph.waypoints), len(self._current_graph.edges)
                )
            )
        for waypoint in self._current_graph.waypoints:
            # Load the waypoint snapshots from disk.
            with open(
                self._upload_filepath + "/waypoint_snapshots/{}".format(waypoint.snapshot_id), "rb"
            ) as snapshot_file:
                waypoint_snapshot = map_pb2.WaypointSnapshot()
                waypoint_snapshot.ParseFromString(snapshot_file.read())
                self._current_waypoint_snapshots[waypoint_snapshot.id] = waypoint_snapshot
        for edge in self._current_graph.edges:
            if len(edge.snapshot_id) == 0:
                continue
            # Load the edge snapshots from disk.
            with open(self._upload_filepath + "/edge_snapshots/{}".format(edge.snapshot_id), "rb") as snapshot_file:
                edge_snapshot = map_pb2.EdgeSnapshot()
                edge_snapshot.ParseFromString(snapshot_file.read())
                self._current_edge_snapshots[edge_snapshot.id] = edge_snapshot
        # Upload the graph to the robot.
        print("Uploading the graph and snapshots to the robot...")
        true_if_empty = not len(self._current_graph.anchoring.anchors)
        response = self._graph_nav_client.upload_graph(
            lease=self._lease.lease_proto, graph=self._current_graph, generate_new_anchoring=true_if_empty
        )
        # Upload the snapshots to the robot.
        for snapshot_id in response.unknown_waypoint_snapshot_ids:
            waypoint_snapshot = self._current_waypoint_snapshots[snapshot_id]
            self._graph_nav_client.upload_waypoint_snapshot(waypoint_snapshot)
            print("Uploaded {}".format(waypoint_snapshot.id))
        for snapshot_id in response.unknown_edge_snapshot_ids:
            edge_snapshot = self._current_edge_snapshots[snapshot_id]
            self._graph_nav_client.upload_edge_snapshot(edge_snapshot)
            print("Uploaded {}".format(edge_snapshot.id))

        # The upload is complete! Check that the robot is localized to the graph,
        # and if it is not, prompt the user to localize the robot before attempting
        # any navigation commands.
        localization_state = self._graph_nav_client.get_localization_state()
        if not localization_state.localization.waypoint_id:
            # The robot is not localized to the newly uploaded graph.
            print("\n")
            print(
                "Upload complete! The robot is currently not localized to the map; please localize",
                "the robot using commands (2) or (3) before attempting a navigation command.",
            )

    def _navigate_to_anchor(self, *args):
        """Navigate to a pose in seed frame, using anchors."""
        # The following options are accepted for arguments: [x, y], [x, y, yaw], [x, y, z, yaw],
        # [x, y, z, qw, qx, qy, qz].
        # When a value for z is not specified, we use the current z height.
        # When only yaw is specified, the quaternion is constructed from the yaw.
        # When yaw is not specified, an identity quaternion is used.

        if len(args) < 1 or len(args[0]) not in [2, 3, 4, 7]:
            print("Invalid arguments supplied.")
            return

        seed_T_goal = SE3Pose(float(args[0][0]), float(args[0][1]), 0.0, Quat())

        if len(args[0]) in [4, 7]:
            seed_T_goal.z = float(args[0][2])
        else:
            localization_state = self._graph_nav_client.get_localization_state()
            if not localization_state.localization.waypoint_id:
                print("Robot not localized")
                return
            seed_T_goal.z = localization_state.localization.seed_tform_body.position.z

        if len(args[0]) == 3:
            seed_T_goal.rot = Quat.from_yaw(float(args[0][2]))
        elif len(args[0]) == 4:
            seed_T_goal.rot = Quat.from_yaw(float(args[0][3]))
        elif len(args[0]) == 7:
            seed_T_goal.rot = Quat(w=float(args[0][3]), x=float(args[0][4]), y=float(args[0][5]), z=float(args[0][6]))

        self._lease = self._lease_wallet.get_lease()
        if not self.toggle_power(should_power_on=True):
            print("Failed to power on the robot, and cannot complete navigate to request.")
            return

        # Stop the lease keepalive and create a new sublease for graph nav.
        self._lease = self._lease_wallet.advance()
        sublease = self._lease.create_sublease()
        self._lease_keepalive.shutdown()
        nav_to_cmd_id = None
        # Navigate to the destination.
        is_finished = False
        while not is_finished:
            # Issue the navigation command about twice a second such that it is easy to terminate the
            # navigation command (with estop or killing the program).
            try:
                nav_to_cmd_id = self._graph_nav_client.navigate_to_anchor(
                    seed_T_goal.to_proto(), 1.0, leases=[sublease.lease_proto], command_id=nav_to_cmd_id
                )
            except ResponseError as e:
                print("Error while navigating {}".format(e))
                break
            time.sleep(0.5)  # Sleep for half a second to allow for command execution.
            # Poll the robot for feedback to determine if the navigation command is complete. Then sit
            # the robot down once it is finished.
            is_finished = self._check_success(nav_to_cmd_id)

        self._lease = self._lease_wallet.advance()
        self._lease_keepalive = LeaseKeepAlive(self._lease_client)

        # Update the lease and power off the robot if appropriate.
        if self._powered_on and not self._started_powered_on:
            # Sit the robot down + power off after the navigation command is complete.
            self.toggle_power(should_power_on=False)

    def _navigate_to(self, *args):
        """Navigate to a specific waypoint."""
        # Take the first argument as the destination waypoint.
        if len(args) < 1:
            # If no waypoint id is given as input, then return without requesting navigation.
            print("No waypoint provided as a destination for navigate to.")
            return

        self._lease = self._lease_wallet.get_lease()
        destination_waypoint = graph_nav_utils.find_unique_waypoint_id(
            args[0][0], self._current_graph, self._current_annotation_name_to_wp_id
        )
        if not destination_waypoint:
            # Failed to find the appropriate unique waypoint id for the navigation command
            print("Failed to find the appropriate unique waypoint id for the navigation command.")
            return
        if not self.toggle_power(should_power_on=True):
            print("Failed to power on the robot, and cannot complete navigate to request.")
            return

        # Stop the lease keep-alive and create a new sublease for graph nav.
        self._lease = self._lease_wallet.advance()
        sublease = self._lease.create_sublease()
        self._lease_keepalive.shutdown()
        nav_to_cmd_id = None
        # Navigate to the destination waypoint.
        is_finished = False
        while not is_finished:
            # Issue the navigation command about twice a second such that it is easy to terminate the
            # navigation command (with estop or killing the program).
            try:
                nav_to_cmd_id = self._graph_nav_client.navigate_to(
                    destination_waypoint, 1.0, leases=[sublease.lease_proto], command_id=nav_to_cmd_id
                )
            except ResponseError as e:
                print("Error while navigating {}".format(e))
                break
            time.sleep(0.5)  # Sleep for half a second to allow for command execution.
            # Poll the robot for feedback to determine if the navigation command is complete. Then sit
            # the robot down once it is finished.
            is_finished = self._check_success(nav_to_cmd_id)

        self._lease = self._lease_wallet.advance()
        self._lease_keepalive = LeaseKeepAlive(self._lease_client)
        # Remove power-off behavior for now
        # # Update the lease and power off the robot if appropriate.
        # if self._powered_on and not self._started_powered_on:
        #     # Sit the robot down + power off after the navigation command is complete.
        #     self.toggle_power(should_power_on=False)

        return is_finished

    def _navigate_to_by_location(self, *args):
        """Navigate to the most central waypoint with a given annotation."""
        if len(args) < 1:
            print("No waypoint annotation provided as a destination.")
            return

        annotation_to_find = args[0][0]

        # Ensure graph is loaded
        if self._current_graph is None:
            self._list_graph_waypoint_and_edge_ids()

        # Collect all waypoints with matching annotation
        matching_waypoints = [
            waypoint for waypoint in self._current_graph.waypoints if waypoint.annotations.name == annotation_to_find
        ]

        if not matching_waypoints:
            print(f"No waypoint found with annotation: {annotation_to_find}")
            return

        # Select the most central waypoint by finding the one with minimum total distance to other waypoints
        def calculate_centrality(waypoint):
            # Extract 3D position coordinates
            x = waypoint.waypoint_tform_ko.position.x
            y = waypoint.waypoint_tform_ko.position.y
            z = waypoint.waypoint_tform_ko.position.z

            # Calculate total Euclidean distance to all other waypoints
            distances = [
                np.linalg.norm(
                    np.array([x, y, z])
                    - np.array(
                        [w.waypoint_tform_ko.position.x, w.waypoint_tform_ko.position.y, w.waypoint_tform_ko.position.z]
                    )
                )
                for w in matching_waypoints
            ]

            # Return sum of distances (lower sum indicates more central location)
            return sum(distances)

        destination_waypoint = min(matching_waypoints, key=calculate_centrality).id
        if not destination_waypoint:
            # Failed to find the appropriate unique waypoint id for the navigation command.
            print("Failed to find the appropriate unique waypoint id for the navigation command.")
            print("Using the first matching waypoint instead.")
            destination_waypoint = matching_waypoints[0].id
        matching_waypoints = [waypoint.id for waypoint in matching_waypoints]
        # print(f"The following waypoints matched your query: {matching_waypoints}")
        print(f"Destination waypoint: {destination_waypoint}")

        # Lease and power management
        self._lease = self._lease_wallet.get_lease()
        if not self.toggle_power(should_power_on=True):
            print("Failed to power on the robot, and cannot complete navigate to request.")
            return

        # Stop the lease keep-alive and create a new sublease for graph nav
        self._lease = self._lease_wallet.advance()
        sublease = self._lease.create_sublease()
        self._lease_keepalive.shutdown()
        nav_to_cmd_id = None

        # Navigate to the destination waypoint
        is_finished = False
        while not is_finished:
            try:
                nav_to_cmd_id = self._graph_nav_client.navigate_to(
                    destination_waypoint, 1.0, leases=[sublease.lease_proto], command_id=nav_to_cmd_id
                )
            except ResponseError as e:
                print(f"Error while navigating {e}")
                break

            time.sleep(0.5)  # Sleep for half a second to allow for command execution
            # Poll the robot for feedback to determine if the navigation command is complete
            is_finished = self._check_success(nav_to_cmd_id)

        # Update lease and power management
        self._lease = self._lease_wallet.advance()
        self._lease_keepalive = LeaseKeepAlive(self._lease_client)

        # Remove power-off behavior for now
        # # Power off if appropriate
        # if self._powered_on and not self._started_powered_on:
        #     self.toggle_power(should_power_on=False)

        return is_finished, destination_waypoint  # return waypoint id for annotation in main interface

    def _navigate_route(self, *args):
        """Navigate through a specific route of waypoints."""
        if len(args) < 1 or len(args[0]) < 1:
            # If no waypoint ids are given as input, then return without requesting navigation.
            print("No waypoints provided for navigate route.")
            return
        waypoint_ids = args[0]
        for i in range(len(waypoint_ids)):
            waypoint_ids[i] = graph_nav_utils.find_unique_waypoint_id(
                waypoint_ids[i], self._current_graph, self._current_annotation_name_to_wp_id
            )
            if not waypoint_ids[i]:
                # Failed to find the unique waypoint id.
                return

        edge_ids_list = []
        all_edges_found = True
        # Attempt to find edges in the current graph that match the ordered waypoint pairs.
        # These are necessary to create a valid route.
        for i in range(len(waypoint_ids) - 1):
            start_wp = waypoint_ids[i]
            end_wp = waypoint_ids[i + 1]
            edge_id = self._match_edge(self._current_edges, start_wp, end_wp)
            if edge_id is not None:
                edge_ids_list.append(edge_id)
            else:
                all_edges_found = False
                print("Failed to find an edge between waypoints: ", start_wp, " and ", end_wp)
                print("List the graph's waypoints and edges to ensure pairs of waypoints has an edge.")
                break

        self._lease = self._lease_wallet.get_lease()
        if all_edges_found:
            if not self.toggle_power(should_power_on=True):
                print("Failed to power on the robot, and cannot complete navigate route request.")
                return

            # Stop the lease keep-alive and create a new sublease for graph nav.
            self._lease = self._lease_wallet.advance()
            sublease = self._lease.create_sublease()
            self._lease_keepalive.shutdown()

            # Navigate a specific route.
            route = self._graph_nav_client.build_route(waypoint_ids, edge_ids_list)
            is_finished = False
            while not is_finished:
                # Issue the route command about twice a second such that it is easy to terminate the
                # navigation command (with estop or killing the program).
                nav_route_command_id = self._graph_nav_client.navigate_route(
                    route, cmd_duration=1.0, leases=[sublease.lease_proto]
                )
                time.sleep(0.5)  # Sleep for half a second to allow for command execution.
                # Poll the robot for feedback to determine if the route is complete. Then sit
                # the robot down once it is finished.
                is_finished = self._check_success(nav_route_command_id)

            self._lease = self._lease_wallet.advance()
            self._lease_keepalive = LeaseKeepAlive(self._lease_client)

            # Update the lease and power off the robot if appropriate.
            if self._powered_on and not self._started_powered_on:
                # Sit the robot down + power off after the navigation command is complete.
                self.toggle_power(should_power_on=False)

    def _find_nearest_farthest_waypoints(
        self, current_waypoint_id: str, matching_waypoints: List[str]
    ) -> Tuple[str, str]:
        """
        Find the nearest and farthest waypoints from current position among matching waypoints.
        Uses Dijkstra's algorithm to compute shortest paths in the graph.

        Args:
            current_waypoint_id (str): Current waypoint ID where the robot is located
            matching_waypoints (List[str]): List of waypoint IDs that match the query criteria

        Returns:
            Tuple[str, str]: A tuple containing (closest_waypoint_id, farthest_waypoint_id)
                            Returns (None, None) if current_waypoint_id is invalid or no path exists
        """
        if not current_waypoint_id or not matching_waypoints:
            return None, None

        if current_waypoint_id not in self._current_edges:
            return None, None

        # Initialize distances dictionary with infinity for all waypoints
        distances = {waypoint_id: float("inf") for waypoint_id in self._current_edges.keys()}
        distances[current_waypoint_id] = 0

        # Dictionary to keep track of visited nodes
        visited = set()

        # Priority queue to store (distance, waypoint_id) pairs
        import heapq

        pq = [(0, current_waypoint_id)]

        while pq:
            current_distance, current_id = heapq.heappop(pq)

            # Skip if we've already processed this waypoint
            if current_id in visited:
                continue

            visited.add(current_id)

            # Check neighbors through edges
            for neighbor_id in self._current_edges[current_id]:
                if neighbor_id not in visited:
                    # Get edge between current and neighbor
                    edge = self._match_edge(self._current_edges, current_id, neighbor_id)
                    if not edge:
                        continue

                    # Calculate new distance (assuming uniform edge weights of 1)
                    # You could modify this to use actual edge distances if available
                    new_distance = distances[current_id] + 1

                    if new_distance < distances[neighbor_id]:
                        distances[neighbor_id] = new_distance
                        heapq.heappush(pq, (new_distance, neighbor_id))

        # Filter distances for only matching waypoints and find min/max
        matching_distances = {wp: distances[wp] for wp in matching_waypoints if wp in distances}

        if not matching_distances:
            return None, None

        # Find closest and farthest among reachable matching waypoints
        closest_wp = min(matching_distances.items(), key=lambda x: x[1])[0]
        farthest_wp = max(matching_distances.items(), key=lambda x: x[1])[0]

        return closest_wp, farthest_wp

    def navigate_to_closest_waypoint(self, matching_waypoints: List[str]) -> NavigationResult:
        """
        Navigate to the closest waypoint from the current position among matching waypoints.

        Args:
            matching_waypoints: List of waypoint IDs that match the query criteria

        Returns:
            NavigationResult indicating success/failure and details
        """
        navigator = WaypointNavigator(self)
        command = NavigationCommand(ClosestWaypointStrategy(), matching_waypoints)
        return navigator.execute_navigation(command)

    def navigate_to_farthest_waypoint(self, matching_waypoints: List[str]) -> NavigationResult:
        """
        Navigate to the farthest waypoint from the current position among matching waypoints.

        Args:
            matching_waypoints: List of waypoint IDs that match the query criteria

        Returns:
            NavigationResult indicating success/failure and details
        """
        navigator = WaypointNavigator(self)
        command = NavigationCommand(FarthestWaypointStrategy(), matching_waypoints)
        return navigator.execute_navigation(command)

    def sit(self):
        """Command the robot to sit."""
        try:
            cmd = RobotCommandBuilder.synchro_sit_command()
            # Wait for the sit command to complete
            robot_command.blocking_sit(self._robot_command_client, timeout_sec=5)
            # Power off the robot after sitting
            self.toggle_power(should_power_on=False)
            return True
        except Exception as e:
            print(f"Error in sit command: {e}")
            return False

    def stand(self):
        """Command the robot to stand."""
        # If not powered on, power on the robot first
        if not self._powered_on:
            self.toggle_power(should_power_on=True)
        try:
            robot_command.blocking_stand(self._robot_command_client, timeout_sec=5)
            return True
        except Exception as e:
            print(f"Error in stand command: {e}")
            return False

    def _initialize_map(self, maybe_clear=False):
        """Upload the map, localize to nearest fiducial, and list waypoints."""
        print("Starting map initialization process...")
        # Step 0: Clear the graph if requested
        if maybe_clear:
            print("\n0. Clearing the graph...")
            self._clear_graph()
        # Step 1: Upload the graph and snapshots
        print("\n1. Uploading graph and snapshots...")
        self._upload_graph_and_snapshots()

        # Step 2: Initialize localization to nearest fiducial
        print("\n2. Initializing localization to nearest fiducial...")
        self._set_initial_localization_fiducial()

        # Wait a bit for localization to complete
        time.sleep(2.0)

        # Step 3: List waypoints and edges
        print("\n3. Listing graph waypoints and edges...")
        self._list_graph_waypoint_and_edge_ids()

        print("\nMap initialization complete!")

    def _clear_graph(self, *args):
        """Clear the state of the map on the robot, removing all waypoints and edges."""
        return self._graph_nav_client.clear_graph(lease=self._lease.lease_proto)

    def toggle_power(self, should_power_on):
        """Power the robot on/off dependent on the current power state."""
        is_powered_on = self.check_is_powered_on()
        if not is_powered_on and should_power_on:
            # Power on the robot up before navigating when it is in a powered-off state.
            power_on(self._power_client)
            motors_on = False
            while not motors_on:
                future = self._robot_state_client.get_robot_state_async()
                state_response = future.result(timeout=10)  # 10 second timeout for waiting for the state response.
                if state_response.power_state.motor_power_state == robot_state_pb2.PowerState.STATE_ON:
                    motors_on = True
                else:
                    # Motors are not yet fully powered on.
                    time.sleep(0.25)
        elif is_powered_on and not should_power_on:
            # Safe power off (robot will sit then power down) when it is in a
            # powered-on state.
            safe_power_off(self._robot_command_client, self._robot_state_client)
        else:
            # Return the current power state without change.
            return is_powered_on
        # Update the locally stored power state.
        self.check_is_powered_on()
        return self._powered_on

    def check_is_powered_on(self):
        """Determine if the robot is powered on or off."""
        power_state = self._robot_state_client.get_robot_state().power_state
        self._powered_on = power_state.motor_power_state == power_state.STATE_ON
        return self._powered_on

    def _check_success(self, command_id=-1):
        """Use a navigation command id to get feedback from the robot and sit when command succeeds."""
        if command_id == -1:
            # No command, so we have no status to check.
            return False
        status = self._graph_nav_client.navigation_feedback(command_id)
        if status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_REACHED_GOAL:
            # Successfully completed the navigation commands!
            return True
        elif status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_LOST:
            print("Robot got lost when navigating the route, the robot will now sit down.")
            return True
        elif status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_STUCK:
            print("Robot got stuck when navigating the route, the robot will now sit down.")
            return True
        elif status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_ROBOT_IMPAIRED:
            print("Robot is impaired.")
            return True
        else:
            # Navigation command is not complete yet.
            return False

    def _match_edge(self, current_edges, waypoint1, waypoint2):
        """Find an edge in the graph that is between two waypoint ids."""
        # Return the correct edge id as soon as it's found.
        for edge_to_id in current_edges:
            for edge_from_id in current_edges[edge_to_id]:
                if (waypoint1 == edge_to_id) and (waypoint2 == edge_from_id):
                    # This edge matches the pair of waypoints! Add it the edge list and continue.
                    return map_pb2.Edge.Id(from_waypoint=waypoint2, to_waypoint=waypoint1)
                elif (waypoint2 == edge_to_id) and (waypoint1 == edge_from_id):
                    # This edge matches the pair of waypoints! Add it the edge list and continue.
                    return map_pb2.Edge.Id(from_waypoint=waypoint1, to_waypoint=waypoint2)
        return None

    def return_lease(self):
        """Shutdown lease keep-alive and return lease."""
        self._lease_keepalive.shutdown()
        self._lease_client.return_lease(self._lease)

    def _on_quit(self):
        """Cleanup on quit from the command line interface."""
        # Sit the robot down + power off after the navigation command is complete.
        if self._powered_on and not self._started_powered_on:
            self._robot_command_client.robot_command(
                RobotCommandBuilder.safe_power_off_command(), end_time_secs=time.time()
            )
        self.return_lease()

    def run(self):
        """Main loop for the command line interface."""
        while True:
            print(
                """
            Options:
            (1) Get localization state.
            (2) Initialize localization to the nearest fiducial (must be in sight of a fiducial).
            (3) Initialize localization to a specific waypoint (must be exactly at the waypoint)."""
                """
            (4) List the waypoint ids and edge ids of the map on the robot.
            (5) Upload the graph and its snapshots.
            (6) Navigate to. The destination waypoint id is the second argument.
            (7) Navigate route. The (in-order) waypoint ids of the route are the arguments.
            (8) Navigate to in seed frame. The following options are accepted for arguments: [x, y],
                [x, y, yaw], [x, y, z, yaw], [x, y, z, qw, qx, qy, qz]. (Don't type the braces).
                When a value for z is not specified, we use the current z height.
                When only yaw is specified, the quaternion is constructed from the yaw.
                When yaw is not specified, an identity quaternion is used.
            (9) Clear the current graph.
            (10) Navigate to the most central waypoint with a given annotation.
            (11) Initialize map (upload, localize to fiducial, and list waypoints).

            (q) Exit.
            """
            )
            try:
                inputs = input(">")
            except NameError:
                pass
            req_type = str.split(inputs)[0]

            if req_type == "q":
                self._on_quit()
                break

            if req_type not in self._command_dictionary:
                print("Request not in the known command dictionary.")
                continue
            try:
                cmd_func = self._command_dictionary[req_type]
                cmd_func(str.split(inputs)[1:])
            except Exception as e:
                print(e)
