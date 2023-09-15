import argparse
import sys

from bosdyn.client.image import ImageClient

from example_project.robot import Robot


def main(argv):
    # Parse args
    parser = argparse.ArgumentParser()

    # You can add arguments that can be defined via console arguments or use the default values instead

    parser.add_argument(
        "--dbg-mode", help="activate Debug Mode", type=bool, default=True
    )
    parser.add_argument(
        "--verbose", help="show logging", type=bool, default=False
    )
    parser.add_argument(
        "--motors_on",
        help="if true, will power on motors",
        type=bool,
        default=False,
    )

    # image specific attributes
    parser.add_argument(
        "--image-sources",
        help="Get image from source(s)",
        action="append",
        default=["frontright_fisheye_image", "frontleft_fisheye_image"],
    )
    parser.add_argument(
        "--image-service",
        help="Name of the image service to query.",
        default=ImageClient.default_service_name,
    )
    parser.add_argument(
        "-j",
        "--jpeg-quality-percent",
        help="JPEG quality percentage (0-100)",
        type=int,
        default=100,
    )
    # point cloud specific attributes
    # FIXME
    parser.add_argument(
        "--point_cloud-sources",
        help="Get point cloud from source(s)",
        action="append",
        default=["frontright_fisheye_image", "frontleft_fisheye_image"],
    )
    parser.add_argument(
        "--show_point_cloud",
        help="Show point cloud",
        type=bool,
        default=False,
    )
    # TODO: You can add/change arguments according to your needs

    options = parser.parse_args(argv)

    # Now we create a robot client and pass the parsed arguments as options
    robot_client = Robot(options)

    # this will execute the init_stuff and loop_stuff method, until code is stopped
    robot_client.run_robot()

    return True


if __name__ == "__main__":
    if not main(sys.argv[1:]):
        sys.exit(1)
