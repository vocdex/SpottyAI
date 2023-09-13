import argparse
import sys

from bosdyn.client.image import ImageClient
from gesture_recognition.robot import Robot


def main(argv):
    # Parse args
    parser = argparse.ArgumentParser()

    # You can add arguments that can be defined via console arguments or use the default values instead

    parser.add_argument(
        "--dbg-mode", help="activate Debug Mode", type=bool, default=True
    )
    parser.add_argument(
        "--verbose", help="show logging of LBR", type=bool, default=False
    )
    parser.add_argument(
        "--motors_on",
        help="if true, will power on motors",
        type=bool,
        default=True,
    )
    # human tracking specific attributes
    parser.add_argument(
        "--track-human",
        help="if true, tries to find and track human, if no gesture was recognized",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--track-human--allow-new-stand",
        help="if true, will search human by changing stand, if not in field of view",
        type=bool,
        default=True,
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
    parser.add_argument(
        "--annotate", help="annotate captured image", type=bool, default=True
    )

    # You can add/change the arguments according to your needs

    options = parser.parse_args(argv)

    # Now we create a robot client and pass the parsed arguments as options
    robot_client = Robot(options)

    #
    robot_client.run_robot()
    return True


if __name__ == "__main__":
    if not main(sys.argv[1:]):
        sys.exit(1)
