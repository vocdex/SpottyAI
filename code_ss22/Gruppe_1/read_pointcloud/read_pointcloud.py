from __future__ import print_function
import argparse
import sys
import time
import os
import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
import bosdyn.geometry
#import bosdyn.client.point_cloud
from bosdyn.client.point_cloud import PointCloudClient

from bosdyn.client.image import ImageClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, blocking_stand


def read_point_cloud(config):
    """A simple example of using the Boston Dynamics API to command a Spot robot."""

    # The Boston Dynamics Python library uses Python's logging module to
    # generate output. Applications using the library can specify how
    # the logging information should be output.
    bosdyn.client.util.setup_logging(config.verbose)

    # The SDK object is the primary entry point to the Boston Dynamics API.
    # create_standard_sdk will initialize an SDK object with typical default
    # parameters. The argument passed in is a string identifying the client.
    sdk = bosdyn.client.create_standard_sdk('PointcloudSpotClient')

    # A Robot object represents a single robot. Clients using the Boston
    # Dynamics API can manage multiple robots, but this tutorial limits
    # access to just one. The network address of the robot needs to be
    # specified to reach it. This can be done with a DNS name
    # (e.g. spot.intranet.example.com) or an IP literal (e.g. 10.0.63.1)
    robot = sdk.create_robot(config.hostname)

    # Clients need to authenticate to a robot before being able to use it.
    bosdyn.client.util.authenticate(robot)

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









        #TODO:read pointcloud
        #possible Solution:
        # - start recording map with commands from recording_command_line.py example
        # - walk around the room in order to build up the map and improve quality of the map
        # - stop walking
        # - download the current map
        # - extract pointcloud from it using methods form extract_point_cloud.py
        # - do postprocessing to find out which room we're in
        # - if room is ambiguous try walking out of the room and create a larger graph
        # - then fit the graph to the blueprint using graph nav anchoring optimization
        # - position of the first graph node corresponds to a position inside the starting room

        #questions:
        # - pc extratction is only possible when anocring point is present.
        # - how to create one? -> probably run anchoring optimization once (from recording command line)

        # wie steuren in welche richutng die exploration stattdfinden soll
        # - > bsp. menge der neuen infomrationen erhöhen
        # Q learning?
        # Karte rastern und jedes Quardrat mal betreten

        #the following wasn't useful until now

        pc_client = robot.ensure_client(PointCloudClient.default_service_name)
        print("SUCCESS: client created")
        sources = pc_client.list_point_cloud_sources()
        print("SOURCES:")
        print(sources)
        print()
        print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||")
        print()
        pc_response = pc_client.get_point_cloud_from_sources(sources[0].get("name"))
        print("PC RESPONSE:")
        print(pc_response)




        # Capture an image.
        # Spot has five sensors around the body. Each sensor consists of a stereo pair and a
        # fisheye camera. The list_image_sources RPC gives a list of image sources which are
        # available to the API client. Images are captured via calls to the get_image RPC.
        # Images can be requested from multiple image sources in one call.
        image_client = robot.ensure_client(ImageClient.default_service_name)
        sources = image_client.list_image_sources()
        print("SOURCES:")
        print(sources)
        print()
        print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||")
        print()
        image_response = image_client.get_image_from_sources(['frontleft_fisheye_image','back_fisheye_image'])
        print("IMG RESPONSE:")
        print(image_response)
        _maybe_display_image(image_response[0].shot.image)
        _maybe_display_image(image_response[1].shot.image)
        if config.save or config.save_path is not None:
            _maybe_save_image(image_response[0].shot.image, config.save_path)
            _maybe_save_image(image_response[1].shot.image, config.save_path)


        # Power the robot off. By specifying "cut_immediately=False", a safe power off command
        # is issued to the robot. This will attempt to sit the robot before powering off.
        robot.power_off(cut_immediately=False, timeout_sec=20)
        assert not robot.is_powered_on(), "Robot power off failed."
        robot.logger.info("Robot safely powered off.")


def _maybe_display_image(image, display_time=3.0):
    """Try to display image, if client has correct deps."""
    try:
        from PIL import Image
        import io
    except ImportError:
        logger = bosdyn.client.util.get_logger()
        logger.warning("Missing dependencies. Can't display image.")
        return
    try:
        image = Image.open(io.BytesIO(image.data))
        image.show()
        time.sleep(display_time)
    except Exception as exc:
        logger = bosdyn.client.util.get_logger()
        logger.warning("Exception thrown displaying image. %r", exc)


def _maybe_save_image(image, path):
    """Try to save image, if client has correct deps."""
    logger = bosdyn.client.util.get_logger()
    try:
        from PIL import Image
        import io
    except ImportError:
        logger.warning("Missing dependencies. Can't save image.")
        return
    name = "hello-spot-img.jpg"
    if path is not None and os.path.exists(path):
        path = os.path.join(os.getcwd(), path)
        name = os.path.join(path, name)
        logger.info("Saving image to: {}".format(name))
    else:
        logger.info("Saving image to working directory as {}".format(name))
    try:
        image = Image.open(io.BytesIO(image.data))
        image.save(name)
    except Exception as exc:
        logger = bosdyn.client.util.get_logger()
        logger.warning("Exception thrown saving image. %r", exc)


#TODO: save pc correclty
def _maybe_save_pc(pc, path):
    """Try to save pc, if client has correct deps."""
    logger = bosdyn.client.util.get_logger()
    try:
        from PIL import Image
        import io
    except ImportError:
        logger.warning("Missing dependencies. Can't save pc.")
        return
    name = "hello-spot-pc.jpg"
    if path is not None and os.path.exists(path):
        path = os.path.join(os.getcwd(), path)
        name = os.path.join(path, name)
        logger.info("Saving pc to: {}".format(name))
    else:
        logger.info("Saving pc to working directory as {}".format(name))
    try:
        pc = Image.open(io.BytesIO(pc.data))
        pc.save(name)
    except Exception as exc:
        logger = bosdyn.client.util.get_logger()
        logger.warning("Exception thrown saving pc. %r", exc)


def main(argv):
    """Command line interface."""
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument(
        '-s', '--save', action='store_true', help=
        'Save the image captured by Spot to the working directory. To chose the save location, use --save_path instead.'
    )
    parser.add_argument(
        '--save-path', default=None, nargs='?', help=
        'Save the image captured by Spot to the provided directory. Invalid path saves to working directory.'
    )
    options = parser.parse_args(argv)
    try:
        read_point_cloud(options)
        return True
    except Exception as exc:  # pylint: disable=broad-except
        logger = bosdyn.client.util.get_logger()
        logger.error("Hello, Spot! threw an exception: %r", exc)
        return False


if __name__ == '__main__':
    if not main(sys.argv[1:]):
        sys.exit(1)