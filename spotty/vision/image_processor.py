#                                                                               
# Copyright (c) 2025, FAU, Shukrullo Nazirjonov
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#                                                                               
# spotty/vision/image_processor.py
import base64
import logging
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from bosdyn.api import image_pb2
from bosdyn.client.image import ImageClient
from PIL import Image
from scipy import ndimage


class ImageProcessor:
    """Handles image processing for robot vision"""

    def __init__(self, image_client: ImageClient, logger=None, rotation_angles=None):
        self.image_client = image_client
        self.logger = logger or logging.getLogger(__name__)
        self.current_images = {}
        self.rotation_angles = rotation_angles or {
            "back_fisheye_image": 0,
            "frontleft_fisheye_image": -90,
            "frontright_fisheye_image": -90,
            "left_fisheye_image": 0,
            "right_fisheye_image": 180,
        }

    def fetch_images(self, image_sources: List[str]) -> Dict[str, np.ndarray]:
        """Fetch images from specified camera sources"""
        from bosdyn.client.image import build_image_request

        try:
            image_requests = [build_image_request(source, quality_percent=75) for source in image_sources]
            image_responses = self.image_client.get_image(image_requests)

            result = {}
            for image in image_responses:
                img = self._process_image_response(image)
                if img is not None:
                    result[image.source.name] = img

            return result
        except Exception as e:
            self.logger.error(f"Error fetching images: {str(e)}")
            return {}

    def _process_image_response(self, image_response) -> Optional[np.ndarray]:
        """Process a single image response into an OpenCV image"""
        try:
            dtype = np.uint8
            img = np.frombuffer(image_response.shot.image.data, dtype=dtype)

            if image_response.shot.image.format == image_pb2.Image.FORMAT_RAW:
                img = img.reshape((image_response.shot.image.rows, image_response.shot.image.cols, 3))
            else:
                img = cv2.imdecode(img, -1)

            # Apply rotation if needed
            if image_response.source.name in self.rotation_angles:
                img = ndimage.rotate(img, self.rotation_angles[image_response.source.name])

            return img
        except Exception as e:
            self.logger.error(f"Error processing image from {image_response.source.name}: {str(e)}")
            return None

    def encode_to_base64(self, img: np.ndarray) -> str:
        """Convert OpenCV image to base64 string"""
        try:
            success, buffer = cv2.imencode(".jpg", img)
            if success:
                return base64.b64encode(buffer).decode("utf-8")
            return ""
        except Exception as e:
            self.logger.error(f"Error encoding image to base64: {str(e)}")
            return ""
