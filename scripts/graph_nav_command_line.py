# Copyright (c) 2022 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

"""Command line interface for graph nav with options to download/upload a map and to navigate a map. """

import argparse
import os
import sys

import bosdyn.client.util

from spotty.mapping.graph_nav_interface import GraphNavInterface
from spotty.utils.robot_utils import auto_authenticate


def main(argv):
    """Run the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-u", "--upload-filepath", help="Full filepath to graph and snapshots to be uploaded.", required=True
    )
    parser.add_argument("hostname", help="The hostname or IP address of the Spot robot.")

    bosdyn.client.util.add_base_arguments(parser)

    options = parser.parse_args(argv)

    # Setup and authenticate the robot.
    sdk = bosdyn.client.create_standard_sdk("GraphNavClient")
    robot = sdk.create_robot(options.hostname)
    auto_authenticate(robot)

    graph_nav_command_line = GraphNavInterface(robot, options.upload_filepath)
    try:
        graph_nav_command_line.run()
        return True
    except Exception as exc:  # pylint: disable=broad-except
        print(exc)
        print("Graph nav command line client threw an error.")
        graph_nav_command_line.return_lease()
        return False


if __name__ == "__main__":
    exit_code = 0
    if not main(sys.argv[1:]):
        exit_code = 1
    os._exit(exit_code)  # Exit hard, no cleanup that could block.
