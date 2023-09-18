import logging
import math
import os
import subprocess
import sys
import open3d as o3d
import time
from abc import ABC, abstractmethod

import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.util
import bosdyn.geometry
import cv2
import numpy as np
from bosdyn.client.frame_helpers import get_vision_tform_body, get_odom_tform_body, VISION_FRAME_NAME, get_a_tform_b, \
    BODY_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME
from bosdyn.client.image import ImageClient, build_image_request, depth_image_to_pointcloud
from bosdyn.client.local_grid import LocalGridClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.time_sync import TimedOutError

from .grid_utils import get_terrain_markers

VALUE_FOR_Q_KEYSTROKE = 113  # quit
VALUE_FOR_ESC_KEYSTROKE = 27  # quit
VALUE_FOR_P_KEYSTROKE = 112  # pause


def ping_robot(hostname):
    try:
        with open(os.devnull, "wb") as devnull:
            resp = subprocess.check_call(["ping", "-c", "1", hostname], stdout=devnull, stderr=subprocess.STDOUT, )
            if resp != 0:
                print(
                    "ERROR: Cannot detect a Spot with IP: {}.\nMake sure Spot is powered on and on the same network".format(
                        hostname))
                sys.exit()
    except:
        print("ERROR: Cannot detect a Spot with IP: {}.\nMake sure Spot is powered on and on the same network".format(
            hostname))
        sys.exit()


def fail_safe(robot):
    """
    Power the robot off. By specifying "cut_immediately=False", a safe power off command
    is issued to the robot. This will attempt to sit the robot before powering off.
    """
    try:
        robot.power_off(cut_immediately=False, timeout_sec=20)
        assert not robot.is_powered_on(), "Robot power off failed."
        robot.logger.info("Robot safely powered off.")
    except:
        pass


def quat_to_euler(quat):
    """Convert a quaternion to xyz Euler angles."""
    q = [quat.x, quat.y, quat.z, quat.w]
    roll = math.atan2(2 * q[3] * q[0] + q[1] * q[2], 1 - 2 * q[0] ** 2 + 2 * q[1] ** 2)
    pitch = math.atan2(2 * q[1] * q[3] - 2 * q[0] * q[2], 1 - 2 * q[1] ** 2 - 2 * q[2] ** 2)
    yaw = math.atan2(2 * q[2] * q[3] + 2 * q[0] * q[1], 1 - 2 * q[1] ** 2 - 2 * q[2] ** 2)
    return bosdyn.geometry.EulerZXY(yaw=yaw, roll=roll, pitch=pitch)


class SpotRobotWrapper(ABC):
    """Callbacks for an instance of a Spot robot"""

    # 0.6 s is the standard duration for cmds in boston dynamics Spot examples
    VELOCITY_CMD_DURATION = 0.6  # [seconds]
    TRAJECTORY_CMD_TIMEOUT = 20.0  # [seconds]
    WARN_BATTERY_LEVEL = 30  # [%]
    MIN_BATTERY_LEVEL = 15  # [%]

    def __init__(self, config):
        self._LOGGER = logging.getLogger(__name__)

        self.config = config

        self.hostname = "192.168.80.3"
        self.username = "user"
        self.password = "c037gcf6n93f"

        # Ensure interface can ping Spot
        ping_robot(self.hostname)

        # Set up SDK
        bosdyn.client.util.setup_logging(config.verbose)
        self.sdk = bosdyn.client.create_standard_sdk("SpotWrapper")

        # Create instance of a robot
        try:
            self.robot = self.sdk.create_robot(self.hostname)
            self.robot.authenticate(self.username, self.password, timeout=20)
            self.robot.sync_with_directory()
            self.robot.time_sync.wait_for_sync()
        except bosdyn.client.RpcError as err:
            self.robot.logger.error("Failed to communicate with robot: %s", err)

        # define clients

        # Client to request images from the robot
        self.image_client = self.robot.ensure_client(ImageClient.default_service_name)

        # Client to request robot state
        self.state_client = self.robot.ensure_client(RobotStateClient.default_service_name)

        # Client to request robot command
        self.motion_client = self.robot.ensure_client(RobotCommandClient.default_service_name)

        # Client to request local occupancy grid
        self.grid_client = self.robot.ensure_client(LocalGridClient.default_service_name)

        # Only one client at a time can operate a robot.
        self.lease_client = self.robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)

        # you can define which cameras you want to use (see config)
        # it is recommended to use all depth cameras to get a 360 degree point cloud
        image_source_visual = config.image_visual_sources
        image_source_depth = config.image_depth_sources

        self.image_requests = [build_image_request(source, quality_percent=config.jpeg_quality_percent) for source in
                               [string + '_fisheye_image' for string in image_source_visual]]

        self.depth_image_requests = [build_image_request(source, quality_percent=20) for source
                                     in [string + '_depth' for string in image_source_depth]]

        # list of image responses
        self.images_visual = None

        # counter to reset image client if RPC timeout
        self.image_reset_counter = 0

        self.point_cloud = None
        self.pcd = o3d.geometry.PointCloud()
        self.o3d_visualizer = None

        self.local_grid_types = self.grid_client.get_local_grid_types()

        # flag if motors are on
        self.motors_on = config.motors_on

        if self.motors_on:
            # Verify the robot is not e-stopped and that an external application has registered and holds
            # an e-stop endpoint.
            assert not self.robot.is_estopped(), "Robot is estopped. Please use an external E-Stop client, " \
                                                 "such as the estop SDK example, to configure E-Stop."

    def reset_image_client(self):
        """Recreate the ImageClient from the robot object."""
        del self.robot.service_clients_by_name["image"]
        del self.robot.channels_by_authority["api.spot.robot"]
        return self.robot.ensure_client("image")

    def get_images(self):
        try:
            # try to retrieve the image
            images_future = self.image_client.get_image_async(self.image_requests)

            while not images_future.done():
                keystroke = cv2.waitKey(1)
                # print(keystroke)
                if keystroke == VALUE_FOR_ESC_KEYSTROKE or keystroke == VALUE_FOR_Q_KEYSTROKE:
                    sys.exit(1)
                if keystroke == VALUE_FOR_P_KEYSTROKE:
                    pass  # TODO: find use case to print image

            # if future is available -> retrieve
            self.images_visual = images_future.result()


        except TimedOutError as time_err:
            # Attempt to handle bad coms:
            # Continue the live image stream, try recreating the image client after having an RPC timeout 5 times.
            if self.image_reset_counter == 5:
                self._LOGGER.info("Resetting image client after 5+ timeout errors.")
                self.image_client = self.reset_image_client()
                self.image_reset_counter = 0
            else:
                self.image_reset_counter += 1

        except Exception as err:
            self._LOGGER.warning(err)
            raise err

    def show_images(self):
        if self.images_visual is None:
            return

        for image_visual in self.images_visual:
            # Visual is a JPEG
            cv_visual = cv2.imdecode(np.frombuffer(image_visual.shot.image.data, dtype=np.uint8), -1)

            # Convert the visual image from a single channel to RGB so we can add color
            visual_rgb = cv_visual if len(cv_visual.shape) == 3 else cv2.cvtColor(cv_visual, cv2.COLOR_GRAY2RGB)

            # Add the two images together.
            cv2.imshow(image_visual.shot.frame_name_image_sensor, visual_rgb)

    def get_local_grid(self):
        # (currently not used)
        pass

    def show_local_grid(self):
        # (currently not used)
        pass

    def get_point_cloud(self):
        try:
            # get depth image
            images_future = self.image_client.get_image_async(self.depth_image_requests)

            while not images_future.done():
                keystroke = cv2.waitKey(1)
                # print(keystroke)
                if keystroke == VALUE_FOR_ESC_KEYSTROKE or keystroke == VALUE_FOR_Q_KEYSTROKE:
                    sys.exit(1)
                if keystroke == VALUE_FOR_P_KEYSTROKE:
                    pass  # TODO: find use case to print image

            # if future is available -> retrieve
            images_depth = images_future.result()

            # retrieve point cloud from depth images
            self.point_cloud = np.zeros((0, 3))
            for image_depth in images_depth:
                # transformation matrix from camera frame to body frame
                body_T_image_sensor = get_a_tform_b(image_depth.shot.transforms_snapshot, GRAV_ALIGNED_BODY_FRAME_NAME,
                                                    image_depth.shot.frame_name_image_sensor)

                # get point cloud from depth image in camera frame
                camera_point_cloud = depth_image_to_pointcloud(image_depth)

                # transform point cloud from camera frame to body frame
                self.point_cloud = np.vstack(
                    (self.point_cloud, body_T_image_sensor.transform_cloud(camera_point_cloud)))

        except Exception as err:
            self._LOGGER.warning(err)
            raise err

    def show_point_cloud(self):
        if self.point_cloud is None:
            return

        self.pcd.points = o3d.utility.Vector3dVector(self.point_cloud[::10])

        if self.o3d_visualizer is None:
            self.o3d_visualizer = o3d.visualization.Visualizer()
            self.o3d_visualizer.create_window()
            self.o3d_visualizer.add_geometry(self.pcd)
            # TODO: add more geometries if needed

        self.o3d_visualizer.update_geometry(self.pcd)
        self.o3d_visualizer.poll_events()
        self.o3d_visualizer.update_renderer()

    def get_robot_state(self):
        """get robot state - kinematic state and robot state"""
        robot_state = self.state_client.get_robot_state()
        
        # vision_tform_body:
        # SE3Pose representing transform of Spot's Body frame
        # relative to the inertial Vision frame (includes vision information)
        vision_tform_body = get_vision_tform_body(robot_state.kinematic_state.transforms_snapshot)

        # odom_tform_body:
        # SE3Pose representing transform of Spot's Body frame
        # relative to the odometry frame (includes only odometry information)
        odom_tform_body = get_odom_tform_body(robot_state.kinematic_state.transforms_snapshot)

        # TODO: you might want to change this according to your needs
        kinematic_state = vision_tform_body

        return kinematic_state, robot_state

    def is_battery_ok(self) -> bool:
        """
        reads battery state. Warns and shut down robot if battery level is too low
        return True if all OK, else False
        """
        robot_state = self.get_robot_state()[1]
        battery_level = robot_state.power_state.locomotion_charge_percentage.value

        if battery_level <= self.WARN_BATTERY_LEVEL:
            self.robot.logger.warning(
                f"Battery level is at {battery_level}%! Robot will shut down at {self.MIN_BATTERY_LEVEL}%!")
        if battery_level <= self.MIN_BATTERY_LEVEL:
            self.robot.logger.warning(f"Battery level is too low to operate! Robot will shut down!")
            fail_safe(self.robot)
            return False

        return True

    def get_local_occupancy(self):
        local_grid_proto = self.grid_client.get_local_grids(["terrain"])
        markers = get_terrain_markers(local_grid_proto)
        return markers

    def self_right(self):
        # stand up
        if not self.config.motors_on:
            self.robot.logger.error("You need to turn on the motors first")
            raise ValueError("config.motors_on is False! -> can not power on robot.")

        """Callback that sends self-right cmd"""
        command = RobotCommandBuilder.selfright_command()
        ret = self.motion_client.robot_command(command)
        self.robot.logger.info("Robot self right cmd sent. {}".format(ret))

    def stand_up(self):
        # stand up
        if not self.config.motors_on:
            self.robot.logger.error("You need to turn on the motors first")
            raise ValueError("config.motors_on is False! -> can not power on robot.")

        command = RobotCommandBuilder.stand_command()
        ret = self.motion_client.robot_command(command)
        self.robot.logger.info("Robot stand cmd sent. {}".format(ret))

    def velocity_command(self, twist):
        """Callback that sends instantaneous velocity [m/s] commands to Spot"""

        v_x = twist.velocity.linear.x
        v_y = twist.velocity.linear.y
        v_rot = twist.velocity.angular.z

        cmd = RobotCommandBuilder.velocity_command(v_x=v_x, v_y=v_y, v_rot=v_rot)

        self.motion_client.robot_command(cmd, end_time_secs=time.time() + self.VELOCITY_CMD_DURATION)

        # self.robot.logger.info("Robot velocity cmd sent: v_x=${},v_y=${},v_rot${}".format(v_x, v_y, v_rot))

    def pose_command(self, pose):
        """
        Callback that sends a pose command to Spot
        Note that the pose commands are always relative to the vision frame, which is an inertial world frame
        """
        p_x = pose.position.x
        p_y = pose.position.y
        theta = quat_to_euler(pose.orientation).yaw

        # Note that the pose commands are always relative to the vision frame, which is an inertial world frame
        frame = VISION_FRAME_NAME

        cmd = RobotCommandBuilder.trajectory_command(goal_x=p_x, goal_y=p_y, goal_heading=theta, frame_name=frame)

        self.motion_client.robot_command(lease=None, command=cmd,
                                         end_time_secs=time.time() + self.TRAJECTORY_CMD_TIMEOUT)

        # self.robot.logger.info("Robot pose cmd sent: p_x=${},p_y=${},theta${}".format(p_x, p_y, theta))

    def trajectory_command(self, trajectory):
        """
        Callback that sends a trajectory command to Spot
        Note that the pose commands are always relative to the vision frame, which is an inertial world frame
        """
        for pose in trajectory.waypoints.poses:
            self.pose_command(pose)

    @abstractmethod
    def init_robot(self):
        # overwrite this abstract class method with your own initialization
        pass

    @abstractmethod
    def loop_robot(self):
        # overwrite this abstract class method with your own loop iteration
        pass

    def run_robot(self):
        """
        This method is the main loop of the robot. It will acquire a lease, power on the robot, and then
        initialize your code. Then it will run the loop_stuff() method in a loop until the program is terminated.
        """
        try:
            with bosdyn.client.lease.LeaseKeepAlive(self.lease_client, must_acquire=True, return_at_exit=True):
                self.robot.logger.info("Acquired lease")

                # power on motors if not already on
                if self.motors_on:
                    self.robot.logger.info("Powering on robot... This may take a several seconds.")
                    self.robot.power_on(timeout_sec=20)
                    assert self.robot.is_powered_on(), "Robot power on failed."
                    self.robot.logger.info("Robot powered on.")
                else:
                    self.robot.logger.info("Not powering on robot, continuing")

                # This method should initialize all your stuff which runs in the loop_stuff() method
                # e.g. initialize states, sensors, stand_up, etc...
                self.init_robot()

                # Loop until Ctrl + C is pressed
                try:
                    self.robot.logger.info("Starting spot loop...")
                    while True:
                        if not self.is_battery_ok():
                            break

                        # This method should be contained every thing that should run in a loop
                        # e.g. obtain sensor readings, perform computations, and command actions ...
                        self.loop_robot()

                except KeyboardInterrupt:
                    self.robot.logger.info("... stopping spot loop.")

        except Exception as err:
            # try to sit down spot and shut down motors
            self.robot.logger.error("An error occurred!")
            self.robot.logger.error(err)

        finally:
            # stop robot -> sit down
            fail_safe(self.robot)
