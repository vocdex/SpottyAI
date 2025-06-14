#                                                                               
# Copyright (c) 2025, FAU, Shukrullo Nazirjonov
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#                                                                               
"""Visualize GraphNav maps with Dash and enable waypoint label editing.
To run:
1. Install the required packages using the following command:
```bash
pip install dash
pip install dash-table
pip install plotly
```
2. Run the script using the following command:
```bash
python visualize_map.py --map-path /path/to/map --rag-path /path/to/rag_db
"""

import argparse
import base64
import json
import os
import re
from datetime import datetime
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import cv2
import dash
import numpy as np
import plotly.graph_objects as go
from bosdyn.api import image_pb2
from bosdyn.api.graph_nav import map_pb2
from bosdyn.client.math_helpers import SE3Pose
from dash import Dash, Input, Output, State, dash_table, dcc, html
from PIL import Image
from scipy import ndimage


class DashMapVisualizer:
    FRONT_CAMERA_SOURCES = {"left": "frontright_fisheye_image", "right": "frontleft_fisheye_image"}

    ROTATION_ANGLE = {
        "back_fisheye_image": 0,
        "frontleft_fisheye_image": -90,
        "frontright_fisheye_image": -90,
        "left_fisheye_image": 0,
        "right_fisheye_image": 180,
    }

    def __init__(self, map_path: str, rag_path: str, logger=None):
        """Initialize the Dash map visualizer."""
        self.map_path = map_path
        self.rag_db_path = rag_path
        self.logger = logger
        self.snapshot_dir = os.path.join(self.map_path, "waypoint_snapshots")

        # Load data
        self.graph, self.waypoints, self.snapshots, self.anchors, self.anchored_world_objects = self.load_graph()

        # Compute global transforms during initialization
        self.compute_global_transforms()  # BFS-based global transforms
        self.compute_anchored_transforms()  # Seed frame anchoring

        # Create a default grayscale image
        self.default_image = np.zeros((100, 100), dtype=np.uint8)
        self.default_image.fill(128)
        buffered = BytesIO()
        Image.fromarray(self.default_image).save(buffered, format="JPEG")
        self.default_image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Continue with rest of initialization...
        self.waypoint_annotations = self.load_rag_annotations()
        self.waypoint_images = self.load_waypoint_images()

        self.all_labels = self.extract_all_labels()

        # Extract all unique objects for filtering
        self.all_objects = self.extract_all_objects()

        # Initialize Dash app
        self.app = Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()

        # Add path for saving updated graph
        self.graph_file_path = os.path.join(self.map_path, "graph")
        self.label_changes = {}

    def extract_all_labels(self) -> List[str]:
        """Extract all unique labels from existing waypoints."""
        labels = set()
        for waypoint in self.graph.waypoints:
            if waypoint.annotations.name:
                labels.add(waypoint.annotations.name)
        return sorted(list(labels))

    def save_updated_graph(self):
        """Save the graph with updated waypoint labels."""
        try:
            # Apply all pending label changes
            for waypoint in self.graph.waypoints:
                if waypoint.id in self.label_changes:
                    waypoint.annotations.name = self.label_changes[waypoint.id]

            # Serialize and save the updated graph
            with open(self.graph_file_path, "wb") as f:
                f.write(self.graph.SerializeToString())

            # Clear the changes after successful save
            self.label_changes.clear()

            return True, "Graph saved successfully"
        except Exception as e:
            return False, f"Error saving graph: {str(e)}"

    def load_graph(self) -> Tuple[map_pb2.Graph, Dict, Dict, Dict, Dict]:
        """Load GraphNav map data including anchoring information."""
        with open(os.path.join(self.map_path, "graph"), "rb") as f:
            graph = map_pb2.Graph()
            graph.ParseFromString(f.read())

        waypoints = {waypoint.id: waypoint for waypoint in graph.waypoints}
        snapshots = {}
        anchors = {}
        anchored_world_objects = {}

        # Load snapshots
        for waypoint in graph.waypoints:
            if waypoint.snapshot_id:
                snapshot_path = os.path.join(self.snapshot_dir, waypoint.snapshot_id)
                if os.path.exists(snapshot_path):
                    with open(snapshot_path, "rb") as f:
                        snapshot = map_pb2.WaypointSnapshot()
                        snapshot.ParseFromString(f.read())
                        snapshots[snapshot.id] = snapshot

        # Load anchoring information
        for anchor in graph.anchoring.anchors:
            anchors[anchor.id] = anchor

        # Load anchored world objects
        for anchored_wo in graph.anchoring.objects:
            anchored_world_objects[anchored_wo.id] = anchored_wo

        return graph, waypoints, snapshots, anchors, anchored_world_objects

    def load_rag_annotations(self) -> Dict:
        """Load RAG annotations."""
        annotations = {}
        metadata_files = [f for f in os.listdir(self.rag_db_path) if f.startswith("metadata_") and f.endswith(".json")]

        for metadata_file in metadata_files:
            with open(os.path.join(self.rag_db_path, metadata_file)) as f:
                metadata = json.load(f)
                waypoint_id = metadata["waypoint_id"]
                annotations[waypoint_id] = metadata

        return annotations

    @staticmethod
    def _clean_text(text):
        "Remove articles, plurals, and lower-case"
        text = re.sub(r"\b(a|an|the)\b", "", text)
        text = re.sub(r"s\b", "", text)
        text = text.lower()
        return text

    def extract_all_objects(self) -> List[str]:
        """Extract all unique objects from the annotations with debug prints."""
        objects = set()
        print("\nExtracting objects from annotations...")

        for waypoint_id, ann in self.waypoint_annotations.items():
            if "views" in ann:
                for view_type, view_data in ann["views"].items():
                    visible_objects = view_data.get("visible_objects", [])
                    # Clean and normalize object names
                    cleaned_objects = [self._clean_text(obj) for obj in visible_objects]

                    objects.update(cleaned_objects)

        sorted_objects = sorted(list(objects))
        return sorted_objects

    def load_waypoint_images(self) -> Dict[str, Dict[str, str]]:
        """
        Load and encode images for each waypoint.
        Returns:
            Dict with structure: {waypoint_id: {'left': base64_str, 'right': base64_str}}
        """
        print("\nLoading waypoint images...")
        waypoint_images = {}

        try:
            for waypoint in self.graph.waypoints:
                if self.logger:
                    self.logger.info(f"Processing waypoint {waypoint.id}")

                snapshot_path = os.path.join(self.snapshot_dir, waypoint.snapshot_id)

                if not os.path.exists(snapshot_path):
                    if self.logger:
                        self.logger.warning(f"Snapshot not found: {snapshot_path}")
                    continue
                snapshot = map_pb2.WaypointSnapshot()
                with open(snapshot_path, "rb") as f:
                    snapshot.ParseFromString(f.read())

                # Initialize image dict for this waypoint
                waypoint_images[waypoint.id] = {"left": self.default_image_base64, "right": self.default_image_base64}
                for image in snapshot.images:
                    # Process only front camera images
                    if image.source.name == self.FRONT_CAMERA_SOURCES["left"]:
                        opencv_image, _ = self.convert_image_from_snapshot(image.shot.image, image.source.name)
                        if opencv_image is not None:
                            waypoint_images[waypoint.id]["left"] = self._encode_image_to_base64(opencv_image)

                    elif image.source.name == self.FRONT_CAMERA_SOURCES["right"]:
                        opencv_image, _ = self.convert_image_from_snapshot(image.shot.image, image.source.name)
                        if opencv_image is not None:
                            waypoint_images[waypoint.id]["right"] = self._encode_image_to_base64(opencv_image)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error processing waypoint {waypoint.id}: {str(e)}")
        print("\nWaypoint images loaded")
        return waypoint_images

    def convert_image_from_snapshot(self, image_data, image_source, auto_rotate=True):
        """
        Convert an image from a GraphNav waypoint snapshot to an OpenCV image.

        Args:
            image_data: Image data from WaypointSnapshot
            image_source: Name of the camera source
            auto_rotate: Whether to automatically rotate images based on camera source
        Returns:
            tuple: (OpenCV image, file extension)
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

        if auto_rotate:
            try:
                rotation_angle = self.ROTATION_ANGLE.get(image_source, 0)
                img = ndimage.rotate(img, rotation_angle)
            except KeyError:
                print(f"Warning: No rotation defined for source {image_source}")

        return img, extension

    def _encode_image_to_base64(self, cv_image):
        """Convert OpenCV image to base64 string with consistent compression."""
        try:
            if self.logger:
                self.logger.info(f"Input image shape: {cv_image.shape}")

            # Check if image is grayscale (1 channel)
            if len(cv_image.shape) == 2 or (len(cv_image.shape) == 3 and cv_image.shape[2] == 1):
                # For grayscale, directly convert to PIL
                if len(cv_image.shape) == 3:
                    cv_image = cv_image.squeeze()  # Remove single-dimension
                image = Image.fromarray(cv_image, mode="L")
            else:
                # For color images, convert BGR to RGB
                image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

            buffered = BytesIO()
            image.save(buffered, format="JPEG", quality=85)
            jpeg_size = len(buffered.getvalue())
            if self.logger:
                self.logger.info(f"JPEG compressed size: {jpeg_size}")

            encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
            if self.logger:
                self.logger.info(f"Base64 encoded size: {len(encoded)}")

            return encoded

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in _encode_image_to_base64: {str(e)}")
            return None

    def compute_global_transforms(self):
        """Compute global transforms for all waypoints using BFS."""
        # Dictionary to store global transforms for each waypoint
        self.global_transforms = {}

        # Start BFS from the first waypoint
        queue = []
        visited = {}

        # Start with first waypoint and identity matrix
        first_waypoint = self.graph.waypoints[0]
        queue.append((first_waypoint, np.eye(4)))

        while len(queue) > 0:
            curr_element = queue[0]
            queue.pop(0)
            curr_waypoint = curr_element[0]
            world_tform_current = curr_element[1]

            if curr_waypoint.id in visited:
                continue

            visited[curr_waypoint.id] = True
            self.global_transforms[curr_waypoint.id] = world_tform_current

            # Process all edges
            for edge in self.graph.edges:
                # Handle forward edges
                if edge.id.from_waypoint == curr_waypoint.id and edge.id.to_waypoint not in visited:
                    to_waypoint = self.waypoints[edge.id.to_waypoint]
                    current_tform_to = SE3Pose.from_proto(edge.from_tform_to).to_matrix()
                    world_tform_to = np.dot(world_tform_current, current_tform_to)
                    queue.append((to_waypoint, world_tform_to))

                # Handle reverse edges
                elif edge.id.to_waypoint == curr_waypoint.id and edge.id.from_waypoint not in visited:
                    from_waypoint = self.waypoints[edge.id.from_waypoint]
                    current_tform_from = SE3Pose.from_proto(edge.from_tform_to).inverse().to_matrix()
                    world_tform_from = np.dot(world_tform_current, current_tform_from)
                    queue.append((from_waypoint, world_tform_from))

    def compute_anchored_transforms(self):
        """Compute transforms for waypoints using seed frame anchoring."""
        self.anchored_transforms = {}

        # Process each waypoint that has an anchor
        for waypoint_id, anchor in self.anchors.items():
            # Get the transform from seed frame to waypoint
            seed_tform_waypoint = SE3Pose.from_proto(anchor.seed_tform_waypoint).to_matrix()
            self.anchored_transforms[waypoint_id] = seed_tform_waypoint

    def create_graph_figure(self, filtered_waypoints: Optional[List[str]] = None, use_anchoring: bool = False):
        """Create the main graph visualization with separate traces for filtered and non-filtered elements."""
        print("Filtered waypoints:", filtered_waypoints)
        # Choose which transforms to use
        transforms = self.anchored_transforms if use_anchoring else self.global_transforms

        # Initialize lists for filtered and non-filtered waypoints
        filtered_x, filtered_y = [], []
        filtered_hover, filtered_labels = [], []
        other_x, other_y = [], []
        other_hover, other_labels = [], []

        # Initialize lists for filtered and non-filtered edges
        filtered_edge_x, filtered_edge_y = [], []
        other_edge_x, other_edge_y = [], []

        # Process waypoints
        for waypoint in self.graph.waypoints:
            if waypoint.id not in transforms:
                continue

            # Get the transform for this waypoint
            world_transform = transforms[waypoint.id]
            position = world_transform[:3, 3]

            # Create hover text
            hover_text = f"Location: {waypoint.annotations.name}<br>"
            hover_text += f"Waypoint ID: {waypoint.id}<br>"
            hover_text += f"Position: ({position[0]:.2f}, {position[1]:.2f})<br>"
            hover_text += f"Frame: {'Seed' if use_anchoring else 'BFS'}<br>"

            # Add RAG annotations
            if waypoint.id in self.waypoint_annotations:
                ann = self.waypoint_annotations[waypoint.id]
                if "views" in ann:
                    for view_type, view_data in ann["views"].items():
                        objects = view_data.get("visible_objects", [])
                        if objects:
                            hover_text += f"<br>{self._clean_text(view_type)} objects:<br>"
                            hover_text += "<br>".join(f"- {self._clean_text(obj)}" for obj in objects)

            # Sort into filtered or non-filtered lists
            if filtered_waypoints and waypoint.id in filtered_waypoints:
                filtered_x.append(position[0])
                filtered_y.append(position[1])
                filtered_hover.append(hover_text)
                filtered_labels.append(waypoint.annotations.name)
            else:
                other_x.append(position[0])
                other_y.append(position[1])
                other_hover.append(hover_text)
                other_labels.append(waypoint.annotations.name)

        # Process edges
        def add_edge(from_wp_id, to_wp_id):
            is_filtered = filtered_waypoints and (from_wp_id in filtered_waypoints or to_wp_id in filtered_waypoints)

            from_transform = transforms[from_wp_id]
            to_transform = transforms[to_wp_id]
            from_pos = from_transform[:3, 3]
            to_pos = to_transform[:3, 3]

            if is_filtered:
                filtered_edge_x.extend([from_pos[0], to_pos[0], None])
                filtered_edge_y.extend([from_pos[1], to_pos[1], None])
            else:
                other_edge_x.extend([from_pos[0], to_pos[0], None])
                other_edge_y.extend([from_pos[1], to_pos[1], None])

        # Add edges based on mode
        if use_anchoring:
            for edge in self.graph.edges:
                if edge.id.from_waypoint in transforms and edge.id.to_waypoint in transforms:
                    add_edge(edge.id.from_waypoint, edge.id.to_waypoint)
        else:
            for edge in self.graph.edges:
                add_edge(edge.id.from_waypoint, edge.id.to_waypoint)

        fig = go.Figure()

        # Add non-filtered edges (semi-transparent)
        if other_edge_x:
            fig.add_trace(
                go.Scatter(
                    x=other_edge_x,
                    y=other_edge_y,
                    mode="lines",
                    line=dict(color="lightgrey", width=1),
                    opacity=0.2,
                    hoverinfo="none",
                    showlegend=False,
                )
            )

        # Add filtered edges
        if filtered_edge_x:
            fig.add_trace(
                go.Scatter(
                    x=filtered_edge_x,
                    y=filtered_edge_y,
                    mode="lines",
                    line=dict(color="lightgrey", width=1),
                    hoverinfo="none",
                    showlegend=False,
                )
            )

        # Add non-filtered waypoints (semi-transparent)
        if other_x:
            fig.add_trace(
                go.Scatter(
                    x=other_x,
                    y=other_y,
                    mode="markers+text",
                    marker=dict(size=8, color="blue", symbol="circle"),
                    text=other_labels,
                    textposition="top center",
                    hovertext=other_hover,
                    hoverinfo="text",
                    opacity=0.3,
                    showlegend=False,
                    # Add customdata to identify points
                    customdata=[
                        {"id": wp.id} for wp in self.graph.waypoints if wp.id not in (filtered_waypoints or [])
                    ],
                )
            )

        # Add filtered waypoints (highlighted)
        if filtered_x:
            fig.add_trace(
                go.Scatter(
                    x=filtered_x,
                    y=filtered_y,
                    mode="markers+text",
                    marker=dict(size=8, color="red", symbol="circle"),
                    text=filtered_labels,
                    textposition="top center",
                    hovertext=filtered_hover,
                    hoverinfo="text",
                    showlegend=False,
                    # Add customdata to identify points
                    customdata=[{"id": wp.id} for wp in self.graph.waypoints if wp.id in (filtered_waypoints or [])],
                )
            )

        # Add anchored world objects if in anchoring mode
        if use_anchoring:
            obj_x, obj_y = [], []
            obj_texts = []

            for obj_id, anchored_obj in self.anchored_world_objects.items():
                seed_tform_obj = SE3Pose.from_proto(anchored_obj.seed_tform_object).to_matrix()
                position = seed_tform_obj[:3, 3]

                obj_x.append(position[0])
                obj_y.append(position[1])
                obj_texts.append(f"Anchor: {obj_id}")

            if obj_x:  # Only add if there are objects
                fig.add_trace(
                    go.Scatter(
                        x=obj_x,
                        y=obj_y,
                        mode="markers+text",
                        marker=dict(size=10, color="green", symbol="square"),
                        text=obj_texts,
                        textposition="top center",
                        hovertext=obj_texts,
                        hoverinfo="text",
                        name="Anchored Objects",
                    )
                )

        # Update layout
        fig.update_layout(
            showlegend=True if use_anchoring else False,
            hovermode="closest",
            margin=dict(t=0, l=0, r=0, b=0),
            plot_bgcolor="white",
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor="LightGray"),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor="LightGray"),
            title=None,
        )

        # Make axes equal scale
        fig.update_yaxes(scaleanchor="x", scaleratio=1)

        return fig

    def setup_layout(self):
        """Set up the Dash app layout with label dropdown."""
        self.app.layout = html.Div(
            [
                # Top controls row
                html.Div(
                    [
                        dcc.Dropdown(
                            id="object-filter",
                            options=[{"label": obj, "value": obj} for obj in self.all_objects],
                            placeholder="Filter by object...",
                            multi=True,
                            style={"width": "40%", "display": "inline-block"},
                        ),
                        dcc.Checklist(
                            id="use-anchoring",
                            options=[{"label": "Use Seed Frame Anchoring", "value": "anchor"}],
                            value=[],
                            style={"width": "30%", "display": "inline-block"},
                        ),
                        html.Button(
                            "Save Changes",
                            id="save-changes-button",
                            style={
                                "width": "20%",
                                "display": "inline-block",
                                "backgroundColor": "#4CAF50",
                                "color": "white",
                                "padding": "10px",
                                "border": "none",
                                "borderRadius": "10px",
                                "cursor": "pointer",
                                "margin": "10px",
                            },
                        ),
                        html.Div(
                            id="save-status", style={"width": "10%", "display": "inline-block", "paddingLeft": "10px"}
                        ),
                    ]
                ),
                # Main content area
                html.Div(
                    [
                        # Graph and Label Editing Row
                        html.Div(
                            [
                                # Graph section
                                html.Div(
                                    [
                                        dcc.Graph(
                                            id="map-graph",
                                            figure=self.create_graph_figure(),
                                            style={"height": "500px", "width": "100%"},  # Full width of its container
                                            config={"displayModeBar": True, "scrollZoom": True, "doubleClick": "reset"},
                                        )
                                    ],
                                    style={"width": "60%", "display": "inline-block", "verticalAlign": "top"},
                                ),
                                # Label editing section (next to graph)
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.H4(
                                                    "Edit Waypoint Label",
                                                    style={
                                                        "marginBottom": "10px",
                                                        "text-align": "center",
                                                        "fontWeight": "bold",
                                                        "marginTop": "none",
                                                    },
                                                ),
                                                html.Div(
                                                    [
                                                        html.Strong("Selected Waypoint: "),
                                                        html.Span(
                                                            id="selected-waypoint-id", style={"marginLeft": "10px"}
                                                        ),
                                                    ]
                                                ),
                                                html.Div(
                                                    [
                                                        html.Strong("Current Label: "),
                                                        html.Span(id="current-label", style={"marginLeft": "10px"}),
                                                    ],
                                                    style={"marginTop": "10px", "marginBottom": "10px"},
                                                ),
                                                html.Div(
                                                    [
                                                        dcc.Dropdown(
                                                            id="label-dropdown",
                                                            options=[
                                                                {"label": label, "value": label}
                                                                for label in self.all_labels
                                                            ],
                                                            placeholder="Select existing label...",
                                                            style={"marginBottom": "10px"},
                                                        ),
                                                        dcc.Input(
                                                            id="new-label-input",
                                                            type="text",
                                                            placeholder="Or type new label...",
                                                            style={
                                                                "width": "100%",
                                                                "padding": "8px",
                                                                "marginBottom": "10px",
                                                            },
                                                        ),
                                                        html.Button(
                                                            "Update Label",
                                                            id="update-label-button",
                                                            style={
                                                                "backgroundColor": "#008CBA",
                                                                "color": "white",
                                                                "padding": "8px",
                                                                "border": "none",
                                                                "borderRadius": "4px",
                                                                "cursor": "pointer",
                                                                "width": "100%",
                                                            },
                                                        ),
                                                    ]
                                                ),
                                            ],
                                            style={
                                                "padding": "20px",
                                                "backgroundColor": "#f8f9fa",
                                                "borderRadius": "4px",
                                            },
                                        )
                                    ],
                                    style={
                                        "width": "40%",
                                        "display": "inline-block",
                                        "verticalAlign": "top",
                                        "paddingLeft": "20px",
                                    },
                                ),
                            ],
                            style={"display": "flex", "flexDirection": "row", "gap": "20px"},
                        ),  # Flexbox for side-by-side layout
                        # Images and Info Table Section
                        html.Div(
                            [
                                # Images container with flexbox for horizontal layout
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Div(
                                                    "Left",
                                                    style={
                                                        "textAlign": "center",
                                                        "fontWeight": "bold",
                                                        "marginBottom": "5px",
                                                    },
                                                ),
                                                html.Img(
                                                    id="waypoint-image-left",
                                                    style={
                                                        "width": "100%",
                                                        "maxWidth": "400px",
                                                        "height": "auto",
                                                        "margin": "0",  # Reset margin
                                                        "padding": "0",  # Reset padding
                                                    },
                                                ),
                                            ],
                                            style={
                                                "flex": "1",
                                                "padding": "0",  # Reset padding
                                                "margin": "0",  # Reset margin
                                                "textAlign": "center",
                                            },
                                        ),
                                        html.Div(
                                            [
                                                html.Div(
                                                    "Right",
                                                    style={
                                                        "textAlign": "center",
                                                        "fontWeight": "bold",
                                                        "marginBottom": "5px",
                                                    },
                                                ),
                                                html.Img(
                                                    id="waypoint-image-right",
                                                    style={
                                                        "width": "100%",
                                                        "maxWidth": "400px",
                                                        "height": "auto",
                                                        "margin": "0",  # Reset margin
                                                        "padding": "0",  # Reset padding
                                                    },
                                                ),
                                            ],
                                            style={
                                                "flex": "1",
                                                "padding": "0",  # Reset padding
                                                "margin": "0",  # Reset margin
                                                "textAlign": "center",
                                            },
                                        ),
                                    ],
                                    style={
                                        "display": "flex",
                                        "flexDirection": "row",
                                        "justifyContent": "center",
                                        "alignItems": "center",
                                        "marginTop": "20px",
                                        "gap": "5px",  # Reduced gap between images
                                        "padding": "0",  # Reset padding
                                        "margin": "0",  # Reset margin
                                    },
                                ),
                                # Info table
                                dash_table.DataTable(
                                    id="waypoint-info",
                                    columns=[{"name": "Property", "id": "property"}, {"name": "Value", "id": "value"}],
                                    data=[],
                                    style_table={"margin-top": "20px", "width": "100%"},
                                    style_cell={
                                        "textAlign": "left",
                                        "padding": "10px",
                                        "whiteSpace": "normal",
                                        "height": "auto",
                                    },
                                    style_header={
                                        "fontWeight": "bold",
                                        "backgroundColor": "#f8f9fa",
                                        "textAlign": "left",
                                    },
                                    style_data={
                                        "whiteSpace": "normal",
                                        "height": "auto",
                                    },
                                    style_data_conditional=[
                                        {"if": {"column_id": "property"}, "fontWeight": "bold", "width": "30%"}
                                    ],
                                ),
                            ]
                        ),
                    ]
                ),
                # Store components
                dcc.Store(id="selected-waypoint-store"),
                dcc.Store(id="unsaved-changes-store", data={"changes": False}),
                dcc.Store(id="view-state", data={}),
            ]
        )

    def setup_callbacks(self):
        """Set up the Dash callbacks with fixed image display logic."""

        @self.app.callback(
            [
                Output("map-graph", "figure"),
                Output("waypoint-image-left", "src"),
                Output("waypoint-image-right", "src"),
                Output("waypoint-info", "data"),
                Output("selected-waypoint-id", "children"),
                Output("current-label", "children"),
                Output("selected-waypoint-store", "data"),
            ],
            [
                Input("map-graph", "clickData"),
                Input("object-filter", "value"),
                Input("use-anchoring", "value"),
                Input("unsaved-changes-store", "data"),
            ],
            [State("map-graph", "relayoutData")],
        )
        def update_ui(clickData, selected_objects, use_anchoring, changes_data, relayoutData):
            # Filter waypoints based on selected objects
            filtered_waypoints = []
            if selected_objects:
                for waypoint_id, ann in self.waypoint_annotations.items():
                    if "views" in ann:
                        for view_data in ann["views"].values():
                            visible_objects = [self._clean_text(obj) for obj in view_data.get("visible_objects", [])]
                            if any(obj in visible_objects for obj in selected_objects):
                                filtered_waypoints.append(waypoint_id)
                                break
            # Create the map figure
            map_figure = self.create_graph_figure(
                filtered_waypoints=filtered_waypoints, use_anchoring=bool("anchor" in (use_anchoring or []))
            )

            # Preserve the view state
            if relayoutData:
                # Keep relevant layout properties
                view_state = {
                    k: v
                    for k, v in relayoutData.items()
                    if k
                    in [
                        "autosize",
                        "xaxis.range[0]",
                        "xaxis.range[1]",
                        "yaxis.range[0]",
                        "yaxis.range[1]",
                        "xaxis.autorange",
                        "yaxis.autorange",
                    ]
                }
                # Apply the saved view state
                if view_state:
                    map_figure.update_layout(
                        xaxis=dict(
                            range=[view_state.get("xaxis.range[0]", None), view_state.get("xaxis.range[1]", None)]
                            if "xaxis.range[0]" in view_state
                            else None,
                            autorange=view_state.get("xaxis.autorange", None),
                        ),
                        yaxis=dict(
                            range=[view_state.get("yaxis.range[0]", None), view_state.get("yaxis.range[1]", None)]
                            if "yaxis.range[0]" in view_state
                            else None,
                            autorange=view_state.get("yaxis.autorange", None),
                        ),
                    )

            if not clickData:
                return (
                    map_figure,
                    f"data:image/jpeg;base64,{self.default_image_base64}",
                    f"data:image/jpeg;base64,{self.default_image_base64}",
                    [],
                    "No waypoint selected",
                    "No waypoint selected",
                    None,
                )

            # Get waypoint ID from customdata
            point_data = clickData["points"][0]
            if "customdata" in point_data and point_data["customdata"]:
                waypoint_id = point_data["customdata"]["id"]
            else:
                # Fallback to finding the waypoint ID
                clicked_x = point_data["x"]
                clicked_y = point_data["y"]
                transforms = (
                    self.anchored_transforms if bool("anchor" in (use_anchoring or [])) else self.global_transforms
                )

                # Find the closest waypoint to the clicked point
                min_dist = float("inf")
                waypoint_id = None

                for wp in self.graph.waypoints:
                    if wp.id in transforms:
                        pos = transforms[wp.id][:3, 3]
                        dist = ((pos[0] - clicked_x) ** 2 + (pos[1] - clicked_y) ** 2) ** 0.5
                        if dist < min_dist:
                            min_dist = dist
                            waypoint_id = wp.id

            if not waypoint_id:
                return (
                    map_figure,
                    f"data:image/jpeg;base64,{self.default_image_base64}",
                    f"data:image/jpeg;base64,{self.default_image_base64}",
                    [],
                    "No waypoint selected",
                    "No waypoint selected",
                    None,
                )

            waypoint = self.waypoints[waypoint_id]
            current_label = self.label_changes.get(waypoint_id, waypoint.annotations.name)

            # Get images for both views with corrected mapping
            waypoint_images = self.waypoint_images.get(
                waypoint_id, {"left": self.default_image_base64, "right": self.default_image_base64}
            )

            # Use the corrected mapping for display
            image_src_left = f"data:image/jpeg;base64,{waypoint_images['left']}"
            image_src_right = f"data:image/jpeg;base64,{waypoint_images['right']}"

            # Update the info table with the correct view assignments
            info = []
            info.append({"property": "Waypoint ID", "value": waypoint_id})
            info.append({"property": "Location", "value": current_label})

            if waypoint_id in self.waypoint_annotations:
                ann = self.waypoint_annotations[waypoint_id]
                if "views" in ann:
                    for view_type, view_data in ann["views"].items():
                        # Map the view type to the correct side
                        display_type = "Left View" if "right" in view_type.lower() else "Right View"
                        objects = view_data.get("visible_objects", [])
                        if objects:
                            info.append({"property": f"{display_type} Objects", "value": ", ".join(objects)})

            return (
                map_figure,
                image_src_left,
                image_src_right,
                info,
                waypoint_id,
                current_label,
                {"waypoint_id": waypoint_id},
            )

        @self.app.callback(
            [
                Output("unsaved-changes-store", "data"),
                Output("new-label-input", "value"),
                Output("label-dropdown", "value"),
            ],
            [
                Input("update-label-button", "n_clicks"),
                Input("label-dropdown", "value"),
                Input("new-label-input", "value"),
            ],
            [State("selected-waypoint-store", "data"), State("unsaved-changes-store", "data")],
        )
        def manage_label_inputs(update_clicks, dropdown_value, text_input, selected_waypoint, changes_data):
            ctx = dash.callback_context
            trigger_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

            # Handle update button click
            if trigger_id == "update-label-button":
                if update_clicks is None or selected_waypoint is None:
                    return changes_data, "", None

                waypoint_id = selected_waypoint["waypoint_id"]

                # Prefer the text input over dropdown if both are provided
                selected_label = text_input if text_input else dropdown_value

                if selected_label:
                    self.label_changes[waypoint_id] = selected_label

                    # If this is a new label, add it to the dropdown options
                    if selected_label not in self.all_labels:
                        self.all_labels.append(selected_label)
                        self.all_labels.sort()

                    # Mark that we have unsaved changes and include timestamp to force update
                    changes_data["changes"] = True
                    changes_data["last_update"] = datetime.now().isoformat()

                # Clear both input fields
                return changes_data, "", None

            # Handle dropdown selection
            elif trigger_id == "label-dropdown":
                return dash.no_update, "", dropdown_value

            # Handle text input
            elif trigger_id == "new-label-input":
                return dash.no_update, text_input, None

            # Default case
            return dash.no_update, dash.no_update, dash.no_update

        @self.app.callback(
            [Output("save-status", "children"), Output("save-status", "style")],
            [Input("save-changes-button", "n_clicks")],
            [State("unsaved-changes-store", "data")],
        )
        def save_changes(n_clicks, changes_data):
            if n_clicks is None:
                return "", {}

            if not changes_data.get("changes", False):
                return "No changes to save", {"color": "blue"}

            success, message = self.save_updated_graph()

            style = {"color": "green" if success else "red", "fontWeight": "bold"}

            return message, style

    def run(self, debug=True, port=8050):
        """Run the Dash app."""
        self.app.run(debug=debug, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-path", type=str, default="../assets/maps/chair_v3")
    parser.add_argument("--rag-path", type=str, default="../assets/database/chair_v3")
    args = parser.parse_args()
    visualizer = DashMapVisualizer(map_path=args.map_path, rag_path=args.rag_path, logger=None)
    visualizer.run()
