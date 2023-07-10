import os
import subprocess
import sys

from abc import ABC, abstractmethod
import time
import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.util

from bosdyn.client.image import ImageClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.client.local_grid import LocalGridClient
from bosdyn.client.robot_state import RobotStateClient

from bosdyn.client.frame_helpers import (
    get_a_tform_b,
    get_vision_tform_body,
    get_odom_tform_body,
    BODY_FRAME_NAME,
    GRAV_ALIGNED_BODY_FRAME_NAME,
    VISION_FRAME_NAME,
    ODOM_FRAME_NAME,
)


from .grid_utils import get_terrain_markers


def ping_robot(hostname):
    try:
        with open(os.devnull, "wb") as devnull:
            resp = subprocess.check_call(
                ["ping", "-c", "1", hostname],
                stdout=devnull,
                stderr=subprocess.STDOUT,
            )
            if resp != 0:
                print(
                    "ERROR: Cannot detect a Spot with IP: {}.\nMake sure Spot is powered on and on the same network".format(
                        hostname
                    )
                )
                sys.exit()
    except:
        print(
            "ERROR: Cannot detect a Spot with IP: {}.\nMake sure Spot is powered on and on the same network".format(
                hostname
            )
        )
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


class SpotRobotWrapper(ABC):
    """Callbacks for an instance of a Spot robot"""

    # 0.6 s is the standard duration for cmds in boston dynamics Spot examples
    VELOCITY_CMD_DURATION = 0.6  # [seconds]
    TRAJECTORY_CMD_TIMEOUT = 20.0  # [seconds]
    WARN_BATTERY_LEVEL = 30  # [%]
    MIN_BATTERY_LEVEL = 15  # [%]

    def __init__(self, config):
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
        self.image_client = self.robot.ensure_client(ImageClient.default_service_name)
        self.image_source_names = [
            src.name
            for src in self.image_client.list_image_sources()
            if "image" in src.name
        ]
        self.depth_image_sources = [
            src.name
            for src in self.image_client.list_image_sources()
            if "depth" in src.name
        ]

        self.state_client = self.robot.ensure_client(
            RobotStateClient.default_service_name
        )

        self.motion_client = self.robot.ensure_client(
            RobotCommandClient.default_service_name
        )

        # Client to request local occupancy grid
        self.grid_client = self.robot.ensure_client(
            LocalGridClient.default_service_name
        )
        self.local_grid_types = self.grid_client.get_local_grid_types()

        # Only one client at a time can operate a robot.
        self.lease_client = self.robot.ensure_client(
            bosdyn.client.lease.LeaseClient.default_service_name
        )

        # Verify the robot is not estopped and that an external application has registered and holds
        # an estop endpoint.
        assert not self.robot.is_estopped(), "Robot is estopped. Please use an external E-Stop client, " \
                                             "such as the estop SDK example, to configure E-Stop."

        # flag if motors are on
        self.motors_on = config.motors_on

    def get_images(self):
        pass  # TODO

    def get_robot_state(self):
        """get robot state - kinematic state and robot state"""
        robot_state = self.state_client.get_robot_state()
        # SE3Pose representing transform of Spot's Body frame relative to the inertial Vision frame
        vision_tform_body = get_vision_tform_body(
            robot_state.kinematic_state.transforms_snapshot
        )
        # odom_tform_body: SE3Pose representing transform of Spot's Body frame relative to the odometry frame
        odom_tform_body = get_odom_tform_body(
            robot_state.kinematic_state.transforms_snapshot
        )

        # TODO define kinematic state as an object with for pose and twist in different frames
        kinematic_state = []

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
                f"Battery level is at {battery_level}%! Robot will shut down at {self.MIN_BATTERY_LEVEL}%!"
            )
        if battery_level <= self.MIN_BATTERY_LEVEL:
            self.robot.logger.warning(
                f"Battery level is too low to operate! Robot will shut down!"
            )
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
            self.robot.logger.error("Yo need to turn on the motors first")
            raise ValueError("config.motors_on is False! -> can not power on robot.")

        """Callback that sends self-right cmd"""
        command = RobotCommandBuilder.selfright_command()
        ret = self.motion_client.robot_command(command)
        self.robot.logger.info("Robot self right cmd sent. {}".format(ret))

    def stand_up(self):
        # stand up
        if not self.config.motors_on:
            self.robot.logger.error("Yo need to turn on the motors first")
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

        self.motion_client.robot_command(
            cmd, end_time_secs=time.time() + self.VELOCITY_CMD_DURATION
        )
        self.robot.logger.info(
            "Robot velocity cmd sent: v_x=${},v_y=${},v_rot${}".format(v_x, v_y, v_rot)
        )

    def pose_command(self, pose):
        pass  # TODO

    def trajectory_command(self, pose):
        pass  # TODO

    @abstractmethod
    def init_stuff(self):
        # overwrite this abstract class method with your own initialization
        pass

    @abstractmethod
    def loop_stuff(self):
        # overwrite this abstract class method with your own loop iteration
        pass

    def run_robot(self):
        try:
            with bosdyn.client.lease.LeaseKeepAlive(
                self.lease_client, must_acquire=True, return_at_exit=True
            ):
                self.robot.logger.info("Acquired lease")

                # power on motors if not already on
                if self.motors_on:
                    self.robot.logger.info(
                        "Powering on robot... This may take a several seconds."
                    )
                    self.robot.power_on(timeout_sec=20)
                    assert self.robot.is_powered_on(), "Robot power on failed."
                    self.robot.logger.info("Robot powered on.")
                else:
                    self.robot.logger.info("Not powering on robot, continuing")

                # This method should initialize all your stuff which runs in the loop_stuff() method
                # e.g. initialize states, sensors, stand_up, etc...
                self.init_stuff()

                # Loop until Ctrl + C is pressed
                try:
                    self.robot.logger.info("Starting spot loop...")
                    while True:
                        if not self.is_battery_ok():
                            break

                        # This method should be contain every thing that should run in a loop
                        # e.g. obtain sensor readings, perform computations, and command actions ...
                        self.loop_stuff()

                except KeyboardInterrupt:
                    self.robot.logger.info("... stopping spot loop.")
        except Exception as err:
            # try to sit down spot and shut down motors
            self.robot.logger.error("An error occurred!")
            self.robot.logger.error(err)

        finally:
            # stop robot -> sit down
            fail_safe(self.robot)
            # If we successfully acquired a lease, return it.
            # self.lease_client.return_lease(self.lease)
