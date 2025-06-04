#                                                                               
# Copyright (c) 2025, FAU, Shukrullo Nazirjonov
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#                                                                               
"""Given a recorded GraphNav graph, this script updates the waypoint annotations using two methods:
1. Manual Annotation: Update annotations using a dictionary of custom labels.
2. CLIP Annotation: Automatically label waypoints using CLIP model based on waypoint snapshots.
# """

import logging
import multiprocessing as mp
import os
import shutil
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from bosdyn.api import image_pb2
from bosdyn.api.graph_nav import map_pb2
from PIL import Image
from scipy import ndimage
from transformers import CLIPModel, CLIPProcessor

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class AnnotationError(Exception):
    """Custom exception for annotation-related errors."""

    pass


class BaseWaypointAnnotationUpdater:
    """
    Enhanced base class for updating waypoint annotations in a GraphNav graph.
    """

    def __init__(self, graph_file_path: str, logger: logging.Logger):
        """
        Initialize the updater with enhanced error checking.

        :param graph_file_path: Path to the saved graph file
        :param logger: Logger instance to use for logging
        :raises AnnotationError: If graph file is invalid
        """
        self.logger = logger

        if not os.path.exists(graph_file_path):
            raise AnnotationError(f"Graph file not found: {graph_file_path}")

        self.graph_file_path = graph_file_path
        self.graph = self._load_graph()

    def _load_graph(self) -> map_pb2.Graph:
        """
        Load the graph with enhanced error handling.

        :return: Loaded graph object
        :raises AnnotationError: If graph cannot be parsed
        """
        graph = map_pb2.Graph()
        try:
            with open(self.graph_file_path, "rb") as graph_file:
                graph.ParseFromString(graph_file.read())

            if not graph.waypoints:
                self.logger.warning("Loaded graph contains no waypoints.")

            self.logger.info(f"Loaded graph with {len(graph.waypoints)} waypoints.")
            return graph
        except Exception as e:
            raise AnnotationError(f"Failed to load graph: {e}")

    def print_waypoint_annotations(self):
        """Print current waypoint annotations with logging."""
        self.logger.info("Current Waypoint Annotations:")
        for waypoint in self.graph.waypoints:
            self.logger.info(f"Waypoint {waypoint.id}: {waypoint.annotations.name}")

    def save_updated_graph(self, output_dir: Optional[str] = None) -> str:
        """
        Save the updated graph, preserving the original as an unlabeled graph.

        :param output_dir: Optional directory to save the updated graph.
        :return: Path to the saved graph file
        :raises AnnotationError: If saving fails
        """
        if output_dir is None:
            output_dir = os.path.dirname(self.graph_file_path)

        os.makedirs(output_dir, exist_ok=True)

        try:
            # Path for the unlabeled (original) graph
            unlabeled_graph_path = os.path.join(output_dir, "unlabeled_graph")

            # Copy the original graph file as unlabeled_graph
            shutil.copy2(self.graph_file_path, unlabeled_graph_path)
            self.logger.info(f"Original graph saved as {unlabeled_graph_path}")

            # Path for the new labeled graph (default name 'graph')
            labeled_graph_path = os.path.join(output_dir, "graph")

            with open(labeled_graph_path, "wb") as graph_file:
                graph_file.write(self.graph.SerializeToString())

            self.logger.info(f"Updated graph saved to {labeled_graph_path}")
            return labeled_graph_path

        except Exception as e:
            raise AnnotationError(f"Failed to save graph: {e}")


class ManualAnnotationUpdater(BaseWaypointAnnotationUpdater):
    """
    Updater for manually specifying waypoint annotations.
    """

    def update_annotations(
        self,
        custom_labels: Dict[str, str],
    ) -> None:
        """
        Update waypoint annotations using a dictionary of labels.

        :param custom_labels: Dictionary mapping original annotations to new annotations.
        """
        for waypoint in self.graph.waypoints:
            original_annotation = waypoint.annotations.name

            if original_annotation in custom_labels:
                new_annotation = custom_labels[original_annotation]
                waypoint.annotations.name = new_annotation
                self.logger.info(f"Updated {original_annotation} to {new_annotation}")


class ClipAnnotationUpdater(BaseWaypointAnnotationUpdater):
    """
    Enhanced CLIP-based waypoint annotation updater with more configuration options.
    """

    def __init__(
        self,
        graph_file_path: str,
        logger: logging.Logger,
        snapshot_dir: str,
        text_prompts: Optional[List[str]] = None,
        clip_model: str = "openai/clip-vit-base-patch32",
        use_multiprocessing: bool = True,
        load_clip: bool = True,  # New parameter to control CLIP model loading
        confidence_threshold: float = 0.5,
        neighbor_threshold: float = 0.5,
    ):
        """
        Initialize the CLIP annotation updater with more flexibility.

        :param graph_file_path: Path to the graph file
        :param snapshot_dir: Directory containing waypoint snapshots
        :param text_prompts: Custom location descriptions for classification
        :param clip_model: CLIP model to use
        :param use_multiprocessing: Enable parallel processing
        :param load_clip: Whether to load the CLIP model
        """
        super().__init__(graph_file_path, logger)

        if not os.path.exists(snapshot_dir):
            raise AnnotationError(f"Snapshot directory not found: {snapshot_dir}")

        self.snapshot_dir = snapshot_dir
        self.use_multiprocessing = use_multiprocessing
        self.confidence_threshold = confidence_threshold
        self.neighbor_threshold = neighbor_threshold

        # Default prompts with more comprehensive locations
        self.text_prompts = text_prompts or [
            "kitchen",
            "office",
            "hallway",
        ]
        # Rotation angles for different camera sources
        self.ROTATION_ANGLE = {
            "back_fisheye_image": 0,
            "frontleft_fisheye_image": -90,
            "frontright_fisheye_image": -90,
            "left_fisheye_image": 0,
            "right_fisheye_image": 180,
        }

        # Camera sources to process
        self.CAMERA_SOURCES = [
            "back_fisheye_image",
            "frontleft_fisheye_image",
            "frontright_fisheye_image",
            "left_fisheye_image",
            "right_fisheye_image",
        ]
        self.FRONT_CAMERA_SOURCES = ["frontleft_fisheye_image", "frontright_fisheye_image"]

        # Depth camera sources to ignore
        self.DEPTH_CAMERA_SOURCES = ["back_depth", "frontleft_depth", "frontright_depth", "left_depth", "right_depth"]

        # Load CLIP model if requested
        self.model = None
        self.processor = None
        if load_clip:
            try:
                self.logger.info(f"Loading CLIP model: {clip_model}")
                self.model = CLIPModel.from_pretrained(clip_model)
                self.processor = CLIPProcessor.from_pretrained(clip_model)
            except Exception as e:
                raise AnnotationError(f"Failed to load CLIP model: {e}")

    def _get_neighbor_waypoints(self, waypoint_id: str) -> List[str]:
        """Get IDs of neighboring waypoints connected by edges."""
        neighbors = []
        for edge in self.graph.edges:
            if edge.id.from_waypoint == waypoint_id:
                neighbors.append(edge.id.to_waypoint)
            elif edge.id.to_waypoint == waypoint_id:
                neighbors.append(edge.id.from_waypoint)
        return neighbors

    def _validate_snapshot(self, snapshot: map_pb2.WaypointSnapshot, waypoint_id: str) -> None:
        """
        Validate that the snapshot has required images.

        :param snapshot: WaypointSnapshot to validate
        :param waypoint_id: ID of the waypoint (for error messaging)
        :raises AnnotationError: If snapshot validation fails
        """
        if not snapshot.images:
            raise AnnotationError(f"Waypoint {waypoint_id} snapshot contains no images")

        # Check if we have any usable images (non-depth cameras from our configured sources)
        valid_images = [
            img
            for img in snapshot.images
            if (img.source.name not in self.DEPTH_CAMERA_SOURCES and img.source.name in self.CAMERA_SOURCES)
        ]

        if not valid_images:
            raise AnnotationError(
                f"Waypoint {waypoint_id} snapshot contains no valid images. "
                f"Available sources: {[img.source.name for img in snapshot.images]}"
            )

    def _validate_annotation(
        self, waypoint_id: str, predicted_label: str, confidence: float, waypoint_labels: Dict[str, str]
    ) -> str:
        """Validate waypoint annotation based on neighboring waypoints."""
        # If confidence is high enough, keep the prediction
        if confidence >= self.confidence_threshold:
            return predicted_label

        neighbors = self._get_neighbor_waypoints(waypoint_id)
        if not neighbors:
            return predicted_label

        neighbor_labels = [waypoint_labels[n] for n in neighbors if n in waypoint_labels]

        if not neighbor_labels:
            return predicted_label

        label_counts = {}
        for label in neighbor_labels:
            label_counts[label] = label_counts.get(label, 0) + 1

        majority_label = max(label_counts.items(), key=lambda x: x[1])[0]
        majority_ratio = label_counts[majority_label] / len(neighbor_labels)

        if majority_ratio >= self.neighbor_threshold and majority_label != predicted_label:
            self.logger.info(
                f"Correcting waypoint {waypoint_id} from {predicted_label} "
                f"to {majority_label} (confidence: {confidence:.2f}, "
                f"neighbor ratio: {majority_ratio:.2f})"
            )
            return majority_label

        return predicted_label

    def _process_single_waypoint(self, waypoint_data: Tuple[str, str]) -> Tuple[str, str, float]:
        """
        Process a single waypoint and return ID, label, and confidence.

        :raises AnnotationError: If snapshot is missing or invalid
        """
        waypoint_id, snapshot_id = waypoint_data
        snapshot_file_path = os.path.join(self.snapshot_dir, snapshot_id)

        if not os.path.exists(snapshot_file_path):
            raise AnnotationError(f"Snapshot not found: {snapshot_file_path}")

        snapshot = map_pb2.WaypointSnapshot()
        with open(snapshot_file_path, "rb") as snapshot_file:
            snapshot.ParseFromString(snapshot_file.read())

        # Validate snapshot before processing
        self._validate_snapshot(snapshot, waypoint_id)
        text_inputs = self.processor(text=self.text_prompts, return_tensors="pt", padding=True)
        text_features = self.model.get_text_features(**text_inputs)

        best_confidence = 0.0
        best_label = "unknown"

        for image in snapshot.images:
            if image.source.name in self.DEPTH_CAMERA_SOURCES or image.source.name not in self.CAMERA_SOURCES:
                continue

            opencv_image, _ = self.convert_image_from_snapshot(image.shot.image, image.source.name)
            pil_image = Image.fromarray(cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB))

            image_inputs = self.processor(images=pil_image, return_tensors="pt")
            image_features = self.model.get_image_features(**image_inputs)

            similarity = torch.nn.functional.cosine_similarity(image_features, text_features)
            confidence, pred_idx = torch.max(similarity, dim=0)
            confidence = confidence.item()

            if confidence > best_confidence:
                best_confidence = confidence
                best_label = self.text_prompts[pred_idx]

        return waypoint_id, best_label, best_confidence

    def update_annotations(self):
        """
        Update waypoint annotations with neighbor validation.

        :raises AnnotationError: If any snapshot is missing or invalid
        """
        self.logger.info("Starting CLIP-based waypoint annotation...")

        waypoint_data = [(waypoint.id, waypoint.snapshot_id) for waypoint in self.graph.waypoints]

        try:
            if self.use_multiprocessing:
                with mp.Pool(processes=max(1, mp.cpu_count() - 1)) as pool:
                    results = pool.map(self._process_single_waypoint, waypoint_data)
            else:
                results = [self._process_single_waypoint(data) for data in waypoint_data]

            initial_predictions = {waypoint_id: (label, confidence) for waypoint_id, label, confidence in results}

            final_labels = {}
            for waypoint_id, (label, confidence) in initial_predictions.items():
                initial_labels = {wid: pred[0] for wid, pred in initial_predictions.items()}
                final_label = self._validate_annotation(waypoint_id, label, confidence, initial_labels)
                final_labels[waypoint_id] = final_label

            for waypoint in self.graph.waypoints:
                if waypoint.id in final_labels:
                    waypoint.annotations.name = final_labels[waypoint.id]
                    self.logger.info(f"Waypoint {waypoint.id} final label: " f"{final_labels[waypoint.id]}")

            self.logger.info("CLIP-based annotation with neighbor validation complete.")

        except AnnotationError as e:
            self.logger.error(f"Annotation process failed: {e}")
            raise

    def convert_image_from_snapshot(self, image_data, image_source, auto_rotate=True):
        """
        Convert an image from a GraphNav waypoint snapshot to an OpenCV image.

        :param image_data: Image data from WaypointSnapshot
        :param image_source: Name of the camera source
        :param auto_rotate: Whether to automatically rotate images based on camera source
        :return: OpenCV image, file extension
        """
        # Determine pixel format and number of channels
        num_channels = 1  # Default to 1 channel
        dtype = np.uint8  # Default to 8-bit unsigned integer

        # Determine pixel format
        if image_data.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
            dtype = np.uint16
            extension = ".png"
        else:
            if image_data.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGB_U8:
                num_channels = 3
            elif image_data.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGBA_U8:
                num_channels = 4
            elif image_data.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8:
                num_channels = 1
            elif image_data.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U16:
                num_channels = 1
                dtype = np.uint16
            extension = ".jpg"

        # Convert image data to numpy array
        img = np.frombuffer(image_data.data, dtype=dtype)

        # Reshape or decode the image
        if image_data.format == image_pb2.Image.FORMAT_RAW:
            try:
                # Attempt to reshape array into rows x cols x channels
                img = img.reshape((image_data.rows, image_data.cols, num_channels))
            except ValueError:
                # If reshaping fails, use OpenCV decode
                img = cv2.imdecode(img, -1)
        else:
            img = cv2.imdecode(img, -1)

        # Auto-rotate if requested and source name is known
        if auto_rotate:
            try:
                rotation_angle = self.ROTATION_ANGLE.get(image_source, 0)
                img = ndimage.rotate(img, rotation_angle)
            except KeyError:
                print(f"Warning: No rotation defined for source {image_source}")

        return img, extension

    def classify_waypoint_location(self, snapshot):
        """
        Classify the location of a waypoint using CLIP.

        :param snapshot: WaypointSnapshot object
        :return: Most likely location label
        """
        location_votes = {}

        # Prepare text inputs ONCE outside the image loop
        text_inputs = self.processor(text=self.text_prompts, return_tensors="pt", padding=True, truncation=True)
        text_features = self.model.get_text_features(**text_inputs)

        # Process images from the snapshot
        for image in snapshot.images:
            # Skip depth images
            if image.source.name in self.DEPTH_CAMERA_SOURCES:
                continue

            # Skip sources not in our list of interest
            if image.source.name not in self.CAMERA_SOURCES:
                continue

            try:
                # Convert image to OpenCV, then to PIL
                opencv_image, _ = self.convert_image_from_snapshot(image.shot.image, image.source.name)
                pil_image = Image.fromarray(cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB))

                # CLIP classification
                image_inputs = self.processor(images=pil_image, return_tensors="pt", padding=True, truncation=True)
                image_features = self.model.get_image_features(**image_inputs)

                # Compute similarity
                similarity = torch.nn.functional.cosine_similarity(image_features, text_features)
                top_pred = self.text_prompts[similarity.argmax().item()]
                location_votes[top_pred] = location_votes.get(top_pred, 0) + 1

            except Exception as e:
                print(f"Error processing image from {image.source.name}: {e}")

        # Majority voting for waypoint location
        if location_votes:
            final_location = max(location_votes, key=location_votes.get)
            return final_location

        return "unknown"
