# Copyright (c) 2022 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

"""Command line interface integrating options to record maps with WASD controls. """
import argparse
import os
import sys

import bosdyn.client.channel

from spotty.mapping.recording_interface import RecordingInterface
from spotty.utils.robot_utils import auto_authenticate


def main(argv):
    """Run the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument(
        "-d",
        "--download-filepath",
        help="Full filepath for where to download graph and snapshots.",
        default=os.getcwd(),
    )
    parser.add_argument("hostname", help="The hostname or IP address of the Spot robot.")

    options = parser.parse_args(argv)

    # Create robot object.
    sdk = bosdyn.client.create_standard_sdk("RecordingClient")
    robot = sdk.create_robot(options.hostname)
    auto_authenticate(robot)

    recording_command_line = RecordingInterface(robot, options.download_filepath)

    try:
        recording_command_line.run()
        return True
    except Exception as exc:  # pylint: disable=broad-except
        print(exc)
        print("Recording command line client threw an error.")
        return False


if __name__ == "__main__":
    if not main(sys.argv[1:]):
        sys.exit(1)
