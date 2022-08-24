# This tutorial is an alternation of the hello spot tutorial.
# We focus here only on starting the robot leasing it and starting up the motors.
# The robot them stands up rotates around its axes and sits down again.
# To enable the motors, an estop is required! - see examples/estop

"""Tutorial to show how to use the Boston Dynamics API"""
import argparse
import sys
import time
import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
import bosdyn.geometry

from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, blocking_stand

from dataclasses import dataclass


# we define the ip address, username and password as global variables because we will use always the same robot
@dataclass
class RobotClient:
    def __init__(self):
        pass

    client_name = "StartUpSpot"
    verbose = False
    hostname = "192.168.80.3"
    user = "user"
    password = "c037gcf6n93f"


robot_client = RobotClient()


def start_up(config):
    """A simple example of using the Boston Dynamics API to command a Spot robot."""

    # The Boston Dynamics Python library uses Python's logging module to
    # generate output. Applications using the library can specify how
    # the logging information should be output.
    bosdyn.client.util.setup_logging(robot_client.verbose)

    # The SDK object is the primary entry point to the Boston Dynamics API.
    # create_standard_sdk will initialize an SDK object with typical default
    # parameters. The argument passed in is a string identifying the client.
    sdk = bosdyn.client.create_standard_sdk(robot_client.client_name)

    # A Robot object represents a single robot. Clients using the Boston
    # Dynamics API can manage multiple robots, but this tutorial limits
    # access to just one. The network address of the robot needs to be
    # specified to reach it. This can be done with a DNS name
    # (e.g. spot.intranet.example.com) or an IP literal (e.g. 10.0.63.1)
    robot = sdk.create_robot(robot_client.hostname)

    # Clients need to authenticate to a robot before being able to use it.
    robot.authenticate(robot_client.user, robot_client.password, timeout=20)

    # Establish time sync with the robot. This kicks off a background thread to establish time sync.
    # Time sync is required to issue commands to the robot. After starting time sync thread, block
    # until sync is established.
    robot.time_sync.wait_for_sync()

    # Verify the robot is not estopped and that an external application has registered and holds
    # an estop endpoint.
    assert not robot.is_estopped(), "Robot is estopped. Please use an external E-Stop client, " \
                                    "such as the estop SDK example, to configure E-Stop."

    # Only one client at a time can operate a robot. Clients acquire a lease to
    # indicate that they want to control a robot. Acquiring may fail if another
    # client is currently controlling the robot. When the client is done
    # controlling the robot, it should return the lease so other clients can
    # control it. The LeaseKeepAlive object takes care of acquiring and returning
    # the lease for us.
    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        # Now, we are ready to power on the robot. This call will block until the power
        # is on. Commands would fail if this did not happen. We can also check that the robot is
        # powered at any point.
        robot.logger.info("Powering on robot... This may take several seconds.")
        robot.power_on(timeout_sec=20)
        assert robot.is_powered_on(), "Robot power on failed."
        robot.logger.info("Robot powered on.")

        # Tell the robot to stand up. The command service is used to issue commands to a robot.
        # The set of valid commands for a robot depends on hardware configuration. See
        # SpotCommandHelper for more detailed examples on command building. The robot
        # command service requires timesync between the robot and the client.
        robot.logger.info("Commanding robot to stand...")
        command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        blocking_stand(command_client, timeout_sec=10)
        robot.logger.info("Robot standing.")
        time.sleep(3)

        # Tell the robot to stand in a twisted position.
        #
        # The RobotCommandBuilder constructs command messages, which are then
        # issued to the robot using "robot_command" on the command client.
        #
        # In this example, the RobotCommandBuilder generates a stand command
        # message with a non-default rotation in the footprint frame. The footprint
        # frame is a gravity aligned frame with its origin located at the geometric
        # center of the feet. The X axis of the footprint frame points forward along
        # the robot's length, the Z axis points up aligned with gravity, and the Y
        # axis is the cross-product of the two.
        footprint_R_body = bosdyn.geometry.EulerZXY(yaw=0.4, roll=0.0, pitch=0.0)
        cmd = RobotCommandBuilder.synchro_stand_command(footprint_R_body=footprint_R_body)
        command_client.robot_command(cmd)
        robot.logger.info("Robot standing twisted.")
        time.sleep(3)

        # Now tell the robot to stand taller, using the same approach of constructing
        # a command message with the RobotCommandBuilder and issuing it with
        # robot_command.
        cmd = RobotCommandBuilder.synchro_stand_command(body_height=0.1)
        command_client.robot_command(cmd)
        robot.logger.info("Robot standing tall.")
        time.sleep(3)

        # Power the robot off. By specifying "cut_immediately=False", a safe power off command
        # is issued to the robot. This will attempt to sit the robot before powering off.
        robot.power_off(cut_immediately=False, timeout_sec=20)
        assert not robot.is_powered_on(), "Robot power off failed."
        robot.logger.info("Robot safely powered off.")


def main(argv):
    """Command line interface."""
    parser = argparse.ArgumentParser()
    # bosdyn.client.util.add_base_arguments(parser)
    # parser.add_argument(
    #     '-s', '--save', action='store_true', help=
    #     'Save the image captured by Spot to the working directory. To chose the save location, use --save_path instead.'
    # )
    # parser.add_argument(
    #     '--save-path', default=None, nargs='?', help=
    #     'Save the image captured by Spot to the provided directory. Invalid path saves to working directory.'
    # )
    options = parser.parse_args(argv)
    try:
        start_up(options)
        return True
    except Exception as exc:  # pylint: disable=broad-except
        logger = bosdyn.client.util.get_logger()
        logger.error("Hello, Spot! threw an exception: %r", exc)
        return False


if __name__ == '__main__':
    if not main(sys.argv[1:]):
        sys.exit(1)
