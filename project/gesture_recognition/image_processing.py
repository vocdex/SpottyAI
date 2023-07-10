"""
Handles all image related task: get image from the camera, and run mediapipe on it
setup camera, cv2, mediapipe,...
"""

import logging
import sys

import cv2
import mediapipe as mp
import numpy as np
from bosdyn.api import image_pb2
from bosdyn.client.image import build_image_request
from bosdyn.client.time_sync import TimedOutError
from scipy import ndimage

from . import stitch_front_images

CAMCAL_mtx = np.array(
    [[917.87243629, 0.0, 507.66305637], [0.0, 645.85695, 384.91475472], [0.0, 0.0, 1.0]]
)

CAMCAL_dist = np.array([[0.88962234, -0.72184465, -0.09753065, 0.00564781, 1.28539346]])

CAMCAL_newcameramtx = np.array(
    [
        [1.08690259e03, 0.00000000e00, 5.18230710e02],
        [0.00000000e00, 7.58720581e02, 3.43867982e02],
        [0.00000000e00, 0.00000000e00, 1.00000000e00],
    ]
)

CAMCAL_roi = (90, 63, 882, 613)


class ImageProcessing:
    def __init__(self, robot, config):
        self._LOGGER = logging.getLogger(__name__)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose

        self.VALUE_FOR_Q_KEYSTROKE = 113
        self.VALUE_FOR_ESC_KEYSTROKE = 27
        self.VALUE_FOR_P_KEYSTROKE = 112

        self.ROTATION_ANGLE = {
            "back_fisheye_image": 0,
            "frontleft_fisheye_image": -90,
            "frontright_fisheye_image": -90,
            "left_fisheye_image": 0,
            "right_fisheye_image": 180,
        }

        # define image client with which we capture the image
        self.image_client = robot.ensure_client(config.image_service)
        self.requests = [
            build_image_request(source, quality_percent=config.jpeg_quality_percent)
            for source in config.image_sources
        ]

        self.keystroke = None
        self.timeout_count_before_reset = 0

        self.config = config
        self.robot = robot

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=True,
            min_detection_confidence=0.5,
        )

        self.flag_print = False
        self.print_num = 27

        self.image = None

        # List image sources
        # sources = self.image_client.list_image_sources()
        # print(f"Available image sources: {sources}")

    def image_to_opencv(self, image, auto_rotate=True):
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
            elif (
                image.shot.image.pixel_format
                == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8
            ):
                num_channels = 1
            elif (
                image.shot.image.pixel_format
                == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U16
            ):
                num_channels = 1
                dtype = np.uint16
            extension = ".jpg"

        img = np.frombuffer(image.shot.image.data, dtype=dtype)
        if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
            try:
                # Attempt to reshape array into a RGB rows X cols shape.
                img = img.reshape(
                    (image.shot.image.rows, image.shot.image.cols, num_channels)
                )
            except ValueError:
                # Unable to reshape the image data, trying a regular decode.
                img = cv2.imdecode(img, -1)
        else:
            img = cv2.imdecode(img, -1)

        if auto_rotate:
            img = ndimage.rotate(img, self.ROTATION_ANGLE[image.source.name])

        return img, extension

    def reset_image_client(self):
        """Recreate the ImageClient from the robot object."""
        del self.robot.service_clients_by_name["image"]
        del self.robot.channels_by_authority["api.spot.robot"]
        return self.robot.ensure_client("image")

    def show_image(self):
        """
        show annotated image
        """
        if self.image is not None:
            cv2.imshow("Image with pose landmarks", self.image)

    def annotate_gesture(
        self, gesture: str = "undefined", person_location: str = "undefined", state_msg: str ="undefined"
    ):
        cv2.putText(
            img=self.image,
            text="Current state: " + state_msg,
            org=(50, 50),
            fontFace=cv2.FONT_HERSHEY_TRIPLEX,
            fontScale=1,
            color=(0, 255, 0),
            thickness=1,
        )
        cv2.putText(
            img=self.image,
            text=gesture,
            org=(50, 100),
            fontFace=cv2.FONT_HERSHEY_TRIPLEX,
            fontScale=1,
            color=(0, 255, 0),
            thickness=1,
        )
        cv2.putText(
            img=self.image,
            text="Last human detection: " + person_location,
            org=(50, 150),
            fontFace=cv2.FONT_HERSHEY_TRIPLEX,
            fontScale=1,
            color=(0, 255, 0),
            thickness=1,
        )

    def process_landmarks(self):
        """
        1) retrieve image from image client,
        2) remove camera distortion
        3) use mediapipe to obtain landmarks (in dbg_mode draw landmarks in image)

        return pose_landmarks_world, pose_landmarks
        """

        try:
            # try to retrieve the image
            images_future = self.image_client.get_image_async(self.requests)
            while not images_future.done():
                self.keystroke = cv2.waitKey(10)
                # print(keystroke)
                if (
                    self.keystroke == self.VALUE_FOR_ESC_KEYSTROKE
                    or self.keystroke == self.VALUE_FOR_Q_KEYSTROKE
                ):
                    sys.exit(1)
                if self.keystroke == self.VALUE_FOR_P_KEYSTROKE:
                    self.flag_print = True

            # if future is available -> retrieve
            images = images_future.result()

            self.image = stitch_front_images.stitch_front_images(images)
            # remove camera distortion
            self.image = cv2.undistort(
                self.image, CAMCAL_mtx, CAMCAL_dist, None, CAMCAL_newcameramtx
            )
            # crop image
            # x, y, w, h = CAMCAL_roi
            # image = image[y:y + h, x:x + w]
            # print images
            if self.flag_print:
                self.flag_print = False
                cv2.imwrite(f"img{self.print_num}.jpg", self.image)
                self.print_num += 1

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image_detection = self.image
            image_detection.flags.writeable = False
            image_detection = cv2.cvtColor(image_detection, cv2.COLOR_BGR2RGB)

            # process image using mediapipe module to retrieve landmarks
            try:
                results = self.pose.process(image_detection)

                # The following is optional but is helpful for debugging:
                # draw landmarks into image and display it
                if self.config.annotate:
                    # Draw pose landmarks on the image.
                    # image.flags.writeable = True
                    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    self.mp_drawing.draw_landmarks(
                        self.image,
                        results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style(),
                    )

                return results.pose_world_landmarks, results.pose_landmarks

            except Exception as err:
                self._LOGGER.warning(err)
                raise err

        except TimedOutError as time_err:
            # Attempt to handle bad coms:
            # Continue the live image stream, try recreating the image client after having an RPC timeout 5 times.
            if self.timeout_count_before_reset == 5:
                self._LOGGER.info("Resetting image client after 5+ timeout errors.")
                self.image_client = self.reset_image_client()
                self.timeout_count_before_reset = 0
            else:
                self.timeout_count_before_reset += 1

        except Exception as err:
            self._LOGGER.warning(err)
            raise err
