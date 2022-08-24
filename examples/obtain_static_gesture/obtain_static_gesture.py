# Copyright (c) 2022 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

"""Simple image display example."""

import argparse
import sys

from bosdyn.api import image_pb2
import bosdyn.client
from bosdyn.client.time_sync import TimedOutError
import bosdyn.client.util
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.api import image_pb2
import cv2
import mediapipe as mp
import gesture_recognition as gc
import numpy as np
from scipy import ndimage

from dataclasses import dataclass


import logging
_LOGGER = logging.getLogger(__name__)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

VALUE_FOR_Q_KEYSTROKE = 113
VALUE_FOR_ESC_KEYSTROKE = 27

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

ROTATION_ANGLE = {
    'back_fisheye_image': 0,
    'frontleft_fisheye_image': -78,
    'frontright_fisheye_image': -102,
    'left_fisheye_image': 0,
    'right_fisheye_image': 180
}


def image_to_opencv(image, auto_rotate=True):
    """Convert an image proto message to an openCV image."""
    num_channels = 1  # Assume a default of 1 byte encodings.
    if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
        dtype = np.uint16
        extension = ".png"
    else:
        dtype = np.uint8
        if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGB_U8:
            num_channels = 3
        elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGBA_U8:
            num_channels = 4
        elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8:
            num_channels = 1
        elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U16:
            num_channels = 1
            dtype = np.uint16
        extension = ".jpg"

    img = np.frombuffer(image.shot.image.data, dtype=dtype)
    if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
        try:
            # Attempt to reshape array into a RGB rows X cols shape.
            img = img.reshape((image.shot.image.rows, image.shot.image.cols, num_channels))
        except ValueError:
            # Unable to reshape the image data, trying a regular decode.
            img = cv2.imdecode(img, -1)
    else:
        img = cv2.imdecode(img, -1)

    if auto_rotate:
        img = ndimage.rotate(img, ROTATION_ANGLE[image.source.name])

    return img, extension


def reset_image_client(robot):
    """Recreate the ImageClient from the robot object."""
    del robot.service_clients_by_name['image']
    del robot.channels_by_authority['api.spot.robot']
    return robot.ensure_client('image')


def start_up(config):
    bosdyn.client.util.setup_logging(robot_client.verbose)
    # Create robot object with an image client.
    sdk = bosdyn.client.create_standard_sdk(robot_client.client_name)
    robot = sdk.create_robot(robot_client.hostname)
    robot.authenticate(robot_client.user, robot_client.password, timeout=20)

    robot.sync_with_directory()
    robot.time_sync.wait_for_sync()

    # define image client with which we capture the image
    image_client = robot.ensure_client(config.image_service)
    requests = [
        build_image_request(source, quality_percent=config.jpeg_quality_percent)
        for source in config.image_sources
    ]

    keystroke = None
    timeout_count_before_reset = 0
    # use mediapipe module for identifying the hand in the captured image
    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.2,
            min_tracking_confidence=0.5) as hands:
        # stop when esc or q is pressed
        while keystroke != VALUE_FOR_Q_KEYSTROKE and keystroke != VALUE_FOR_ESC_KEYSTROKE:
            try:
                # try to retrieve the image
                images_future = image_client.get_image_async(requests, timeout=0.5)
                while not images_future.done():
                    keystroke = cv2.waitKey(25)
                    # print(keystroke)
                    if keystroke == VALUE_FOR_ESC_KEYSTROKE or keystroke == VALUE_FOR_Q_KEYSTROKE:
                        sys.exit(1)
                # if future is available -> retrieve
                images = images_future.result()
            except TimedOutError as time_err:
                if timeout_count_before_reset == 5:
                    # To attempt to handle bad comms and continue the live image stream, try recreating the
                    # image client after having an RPC timeout 5 times.
                    _LOGGER.info("Resetting image client after 5+ timeout errors.")
                    image_client = reset_image_client(robot)
                    timeout_count_before_reset = 0
                else:
                    timeout_count_before_reset += 1
            except Exception as err:
                _LOGGER.warning(err)
                continue
            for i in range(len(images)):
                # process retrieved image and convert it to openCV image for future processing
                image, _ = image_to_opencv(images[i], config.auto_rotate)

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # process image using mediapipe module to retrieve landmarks of hand
                results = hands.process(image)

                # process hand
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        gesture = gc.get_static_gesture(hand_landmarks)

                # The following is optional but is helpful for debugging:
                # draw landmarks of hand into image and display it
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Draw the hand annotations on the image.
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())

                # show image with annotations
                cv2.imshow(images[i].source.name, image)

            keystroke = cv2.waitKey(config.capture_delay)


def main(argv):
    # Parse args
    parser = argparse.ArgumentParser()
    # bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('--image-sources', help='Get image from source(s)', action='append')
    parser.add_argument('--image-service', help='Name of the image service to query.',
                        default=ImageClient.default_service_name)
    parser.add_argument('-j', '--jpeg-quality-percent', help="JPEG quality percentage (0-100)",
                        type=int, default=100)
    parser.add_argument('-c', '--capture-delay', help="Time [ms] to wait before the next capture",
                        type=int, default=100)
    parser.add_argument('--disable-full-screen', help="A single image source gets displayed full screen by default. This flag disables that.", action='store_true')
    parser.add_argument('--auto-rotate', help='rotate right and front images to be upright',
                        action='store_true')
    options = parser.parse_args(argv)

    try:
        start_up(options)
        return True
    except Exception as exc:  # pylint: disable=broad-except
        logger = bosdyn.client.util.get_logger()
        logger.error("Hello, Spot! threw an exception: %r", exc)
        return False


if __name__ == "__main__":
    if not main(sys.argv[1:]):
        sys.exit(1)
