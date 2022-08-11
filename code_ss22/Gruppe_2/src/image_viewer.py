# Copyright (c) 2022 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

"""Simple image display example."""

import argparse
import sys
import gesturerecognition as grc
import time

from MySpot import MySpot

from bosdyn.api import image_pb2
import bosdyn.client
from bosdyn.client.time_sync import TimedOutError
import bosdyn.client.util
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.api import image_pb2
import bosdyn.client.lease
from bosdyn.client.robot_command import RobotCommandClient, RobotCommandBuilder, blocking_stand
import cv2
import numpy as np
from scipy import ndimage

import mediapipe as mp

#Build Keypoints using MP Holistic
mp_holistic = mp.solutions.hands # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

import logging
_LOGGER = logging.getLogger(__name__)

VALUE_FOR_Q_KEYSTROKE = 113
VALUE_FOR_ESC_KEYSTROKE = 27

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
            num_channels = 1
        elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGBA_U8:
            num_channels = 1
        elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8:
            num_channels = 3
        elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U16:
            num_channels = 4
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

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR CONVERSION RGB 2 BGR
    return image, results
   
def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(
      image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(
      image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections
     
def draw_styled_landmarks(image, results):
    # Draw left hand connections
	if results.multi_hand_landmarks:
		for num, hand in enumerate(results.multi_hand_landmarks):
			mp_drawing.draw_landmarks(image, hand, mp_holistic.HAND_CONNECTIONS)
			gesture = grc.gestrec(hand)
			return gesture

def main(argv):
    # Parse args
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('--image-sources', help='Get image from source(s)', action='append')
    parser.add_argument('--image-service', help='Name of the image service to query.',
                        default=ImageClient.default_service_name)
    parser.add_argument('-j', '--jpeg-quality-percent', help="JPEG quality percentage (0-100)",
                        type=int, default=50)
    parser.add_argument('-c', '--capture-delay', help="Time [ms] to wait before the next capture",
                        type=int, default=100)
    parser.add_argument('--disable-full-screen', help="A single image source gets displayed full screen by default. This flag disables that.", action='store_true')
    parser.add_argument('--auto-rotate', help='rotate right and front images to be upright',
                        action='store_true')
    options = parser.parse_args(argv)#
	
    # Create robot object with an image client.
    sdk = bosdyn.client.create_standard_sdk('Command_follow')
    robot = sdk.create_robot(options.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.sync_with_directory()
    robot.time_sync.wait_for_sync()

    image_client = robot.ensure_client(options.image_service)
    requests = [
        build_image_request(source, quality_percent=options.jpeg_quality_percent)
        for source in options.image_sources
    ]

    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
    lease = lease_client.acquire()
    lease_keep_alive = bosdyn.client.lease.LeaseKeepAlive(lease_client)
    
    robot.power_on(timeout_sec=20)
	
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    sit = RobotCommandBuilder.synchro_sit_command()
    walk = RobotCommandBuilder.synchro_velocity_command(0.3, 0, 0)
    blocking_stand(command_client, timeout_sec=10)
	
    sit_counter = 0
    stand_counter = 0
    walk_counter = 0
	
    for image_source in options.image_sources:
        cv2.namedWindow(image_source, cv2.WINDOW_NORMAL)
        if len(options.image_sources) > 1 or options.disable_full_screen:
            cv2.setWindowProperty(image_source, cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_AUTOSIZE)
        else:
            cv2.setWindowProperty(image_source, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    
    keystroke = None
    timeout_count_before_reset = 0
    with mp_holistic.Hands(min_detection_confidence=0.2, min_tracking_confidence=0.5) as holistic:
        while keystroke != VALUE_FOR_Q_KEYSTROKE and keystroke != VALUE_FOR_ESC_KEYSTROKE:
            try:
                images_future = image_client.get_image_async(requests, timeout=0.5)
                while not images_future.done():
                    keystroke = cv2.waitKey(25)
                    if keystroke != -1:
                        print(keystroke)
                    if keystroke == VALUE_FOR_ESC_KEYSTROKE or keystroke == VALUE_FOR_Q_KEYSTROKE:
                        sys.exit(1)
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
                image, _ = image_to_opencv(images[i], options.auto_rotate)
                image_raw = cv2.GaussianBlur(image, (0,0), 3) # Funktioniert gut: 3
                image_raw = cv2.addWeighted(image, 1.5, image_raw, -0.8, 0) # Funktioniert gut: 1.5/-0,8
                bild, erg = mediapipe_detection(image_raw, holistic)
                gest = draw_styled_landmarks(bild, erg)
                if gest == 3:
                    sit_counter = sit_counter+1
                    stand_counter = 0
                if gest == 4:
                    stand_counter = stand_counter + 1
                    sit_counter = 0
                if gest == 2:
                    command_client.robot_command(walk, end_time_secs=time.time() + 0.5)
                if sit_counter > 4:
                    sit_counter = 0
                    command_client.robot_command(sit)
                if stand_counter > 4:
                    stand_counter = 0
                    blocking_stand(command_client, timeout_sec=10)
                cv2.imshow(images[i].source.name, bild)
            keystroke = cv2.waitKey(options.capture_delay)
        command_client.robot_command(sit)
        robot.power_off()

if __name__ == "__main__":
    if not main(sys.argv[1:]):
        sys.exit(1)
