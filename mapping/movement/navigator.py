import time
from typing import Any, Optional
import bosdyn.client
from bosdyn.client.robot import Robot
from bosdyn.client.graph_nav import GraphNavClient
from bosdyn.api.graph_nav import map_pb2
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.frame_helpers import BODY_FRAME_NAME
from bosdyn.client.math_helpers import SE2Pose
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient

class SpotNavigator:
    def __init__(self, robot: Robot):
        """Initialize the SpotNavigator with a robot instance."""
        self.robot = robot
        self.graph_nav_client = robot.ensure_client(GraphNavClient.default_service_name)
        self.lease_client = robot.ensure_client(LeaseClient.default_service_name)
        self.command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        
        # Constants for velocity control
        self.__VELOCITY_BASE_SPEED = 0.5
        self.__VELOCITY_BASE_ANGULAR = 0.8
        self.__VELOCITY_CMD_DURATION = 0.6
        
        self.waypoints = {}
        self.load_waypoints()

    def load_waypoints(self):
        """Load waypoints and their labels from the map."""
        try:
            graph = self.graph_nav_client.download_graph()
            for waypoint_id, waypoint in graph.waypoints.items():
                annotations = waypoint.annotations
                self.waypoints[waypoint_id] = {
                    "place": annotations.name,
                    "objects": annotations.object_labels,
                }
            print(f"Loaded {len(self.waypoints)} waypoints.")
        except Exception as e:
            print(f"Failed to load waypoints: {e}")

    def find_waypoint_by_label(self, label, label_type="place"):
        """Find a waypoint by its label (place or object)."""
        for waypoint_id, info in self.waypoints.items():
            if label_type == "place" and info["place"] == label:
                return waypoint_id
            elif label_type == "object" and label in info["objects"]:
                return waypoint_id
        return None

    def navigate_to(self, location: str):
        """Navigate to a specific location by place label."""
        waypoint_id = self.find_waypoint_by_label(location, label_type="place")
        if not waypoint_id:
            print(f"Location '{location}' not found in waypoints.")
            return {"error": "Location not found"}

        try:
            with LeaseKeepAlive(self.lease_client, must_acquire=True, return_at_exit=True):
                self.graph_nav_client.navigate_to(destination_waypoint_id=waypoint_id, travel_params=None)
                print(f"Navigating to {location} (Waypoint ID: {waypoint_id})")
                return {"status": "success", "action": "navigate", "location": location, "waypoint_id": waypoint_id}
        except Exception as e:
            print(f"Failed to navigate to {location}: {e}")
            return {"error": "Navigation failed", "details": str(e)}

    def navigate_to_waypoint(self, waypoint_id: str):
        """Navigate directly to a specific waypoint ID."""
        if waypoint_id not in self.waypoints:
            print(f"Waypoint ID '{waypoint_id}' not found.")
            return {"error": "Waypoint not found"}

        try:
            with LeaseKeepAlive(self.lease_client, must_acquire=True, return_at_exit=True):
                self.graph_nav_client.navigate_to(destination_waypoint_id=waypoint_id, travel_params=None)
                print(f"Navigating to Waypoint ID: {waypoint_id}")
                return {"status": "success", "action": "navigate", "waypoint_id": waypoint_id}
        except Exception as e:
            print(f"Failed to navigate to waypoint {waypoint_id}: {e}")
            return {"error": "Navigation failed", "details": str(e)}

    def find_object(self, object_name: str):
        """Find a waypoint containing a specific object label."""
        waypoint_id = self.find_waypoint_by_label(object_name, label_type="object")
        if not waypoint_id:
            print(f"Object '{object_name}' not found in waypoints.")
            return {"error": "Object not found"}

        print(f"Object '{object_name}' found at waypoint ID: {waypoint_id}")
        return {"status": "success", "action": "find_object", "object": object_name, "waypoint_id": waypoint_id}

    def sit(self) -> dict:
        """Sit the robot down."""
        try:
            with LeaseKeepAlive(self.lease_client, must_acquire=True, return_at_exit=True):
                command = RobotCommandBuilder.synchro_sit_command()
                self.command_client.robot_command(command)
            return {"status": "success", "action": "sit"}
        except Exception as e:
            return {"error": "Sit command failed", "details": str(e)}

    def stand(self) -> dict:
        """Stand the robot up."""
        try:
            with LeaseKeepAlive(self.lease_client, must_acquire=True, return_at_exit=True):
                command = RobotCommandBuilder.synchro_stand_command()
                self.command_client.robot_command(command)
            return {"status": "success", "action": "stand"}
        except Exception as e:
            return {"error": "Stand command failed", "details": str(e)}

    def forward(self, distance: Optional[float] = None) -> dict:
        """
        Move the robot forward.
        
        :param distance: Optional distance to move. If None, move at velocity.
        """
        if distance is not None:
            return self.move("forward", distance)
        return self.__execute_velocity(v_x=self.__VELOCITY_BASE_SPEED)

    def backward(self, distance: Optional[float] = None) -> dict:
        """
        Move the robot backward.
        
        :param distance: Optional distance to move. If None, move at velocity.
        """
        if distance is not None:
            return self.move("backward", distance)
        return self.__execute_velocity(v_x=-self.__VELOCITY_BASE_SPEED)

    def left(self) -> dict:
        """Move the robot left."""
        return self.__execute_velocity(v_y=self.__VELOCITY_BASE_SPEED)

    def right(self) -> dict:
        """Move the robot right."""
        return self.__execute_velocity(v_y=-self.__VELOCITY_BASE_SPEED)

    def rotate_left(self, angle: Optional[float] = None) -> dict:
        """
        Rotate the robot left.
        
        :param angle: Optional angle to rotate. If None, rotate at velocity.
        """
        if angle is not None:
            return self.rotate("anticlockwise", angle)
        return self.__execute_velocity(v_rot=self.__VELOCITY_BASE_ANGULAR)

    def rotate_right(self, angle: Optional[float] = None) -> dict:
        """
        Rotate the robot right.
        
        :param angle: Optional angle to rotate. If None, rotate at velocity.
        """
        if angle is not None:
            return self.rotate("clockwise", angle)
        return self.__execute_velocity(v_rot=-self.__VELOCITY_BASE_ANGULAR)

    def move(self, direction: str, distance: float):
        """Move the robot a certain distance forward or backward."""
        try:
            with LeaseKeepAlive(self.lease_client, must_acquire=True, return_at_exit=True):
                if direction == "forward":
                    self.robot.logger.info(f"Moving forward by {distance} meters.")
                    self.graph_nav_client.navigate_to_anchor(distance_forward=distance)
                elif direction == "backward":
                    self.robot.logger.info(f"Moving backward by {distance} meters.")
                    self.graph_nav_client.navigate_to_anchor(distance_forward=-distance)
                else:
                    raise ValueError(f"Invalid direction: {direction}")
            print(f"Moved {direction} by {distance} meters.")
            return {"status": "success", "action": "move", "direction": direction, "distance": distance}
        except Exception as e:
            print(f"Failed to move {direction} by {distance} meters: {e}")
            return {"error": "Movement failed", "details": str(e)}

    def rotate(self, direction: str, angle: float):
        """Rotate the robot clockwise or anticlockwise by a certain angle."""
        try:
            with LeaseKeepAlive(self.lease_client, must_acquire=True, return_at_exit=True):
                rotation = angle if direction == "clockwise" else -angle
                self.graph_nav_client.navigate_to_anchor(angle_rotate=rotation)
            print(f"Rotated {direction} by {angle} degrees.")
            return {"status": "success", "action": "rotate", "direction": direction, "angle": angle}
        except Exception as e:
            print(f"Failed to rotate {direction} by {angle} degrees: {e}")
            return {"error": "Rotation failed", "details": str(e)}

    def move_to_destination(self, x: float, y: float, angle: float) -> dict:
        """
        Move the robot to the specified destination using SE2 coordinates.

        Parameters:
        x : float
            X coordinate of the destination
        y : float
            Y coordinate of the destination
        angle : float
            Rotation angle at the destination
        """
        try:
            with LeaseKeepAlive(self.lease_client, must_acquire=True, return_at_exit=True):
                destination = SE2Pose(x, y, angle)
                command = RobotCommandBuilder.synchro_se2_trajectory_point_command(
                    destination.x,
                    destination.y,
                    destination.angle,
                    BODY_FRAME_NAME,
                )
                self.command_client.robot_command(command)
            return {"status": "success", "action": "move_to_destination", "x": x, "y": y, "angle": angle}
        except Exception as e:
            return {"error": "Move to destination failed", "details": str(e)}

    def __execute_velocity(
        self,
        v_x: float = 0.0,
        v_y: float = 0.0,
        v_rot: float = 0.0,
    ) -> dict:
        """
        Execute the specified velocity.
        """
        try:
            with LeaseKeepAlive(self.lease_client, must_acquire=True, return_at_exit=True):
                command = RobotCommandBuilder.synchro_velocity_command(v_x=v_x, v_y=v_y, v_rot=v_rot)
                self.command_client.robot_command(command, end_time=time.time() + self.__VELOCITY_CMD_DURATION)
            return {
                "status": "success", 
                "action": "velocity_command", 
                "v_x": v_x, 
                "v_y": v_y, 
                "v_rot": v_rot
            }
        except Exception as e:
            return {"error": "Velocity command failed", "details": str(e)}