from dash import Dash, dcc, html, Input, Output, dash_table
import plotly.graph_objects as go
import numpy as np
from bosdyn.api.graph_nav import map_pb2
from bosdyn.api import image_pb2
import os
import json
import base64
from typing import Dict, Tuple, List, Optional
from io import BytesIO
import cv2
from scipy import ndimage
from PIL import Image
import re
from bosdyn.client.math_helpers import SE3Pose
import numpy as np


class DashMapVisualizer:
    FRONT_CAMERA_SOURCES = [
        'frontleft_fisheye_image',
        'frontright_fisheye_image'
    ]
    ROTATION_ANGLE = {
        'back_fisheye_image': 0,
        'frontleft_fisheye_image': -90,
        'frontright_fisheye_image': -90,
        'left_fisheye_image': 0,
        'right_fisheye_image': 180
    }

    def __init__(self, graph_path: str, rag_db_path: str, logger=None):
        """Initialize the Dash map visualizer."""
        self.graph_path = graph_path
        self.rag_db_path = rag_db_path
        self.logger = logger
        self.snapshot_dir = os.path.join(self.graph_path, "waypoint_snapshots")
        
        # Load data
        self.graph, self.waypoints, self.snapshots = self.load_graph()
        
        # Compute global transforms during initialization
        self.compute_global_transforms()
        
        # Continue with rest of initialization...
        self.waypoint_annotations = self.load_rag_annotations()
        self.waypoint_images = self.load_waypoint_images()
        
        # Create a default grayscale image
        self.default_image = np.zeros((100, 100), dtype=np.uint8)
        self.default_image.fill(128)
        buffered = BytesIO()
        Image.fromarray(self.default_image).save(buffered, format="JPEG")
        self.default_image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Extract all unique objects for filtering
        self.all_objects = self.extract_all_objects()
        
        # Initialize Dash app
        self.app = Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()
        
    def load_graph(self) -> Tuple[map_pb2.Graph, Dict, Dict]:
        """Load GraphNav map data."""
        with open(os.path.join(self.graph_path, "graph"), "rb") as f:
            graph = map_pb2.Graph()
            graph.ParseFromString(f.read())
            
        waypoints = {waypoint.id: waypoint for waypoint in graph.waypoints}
        snapshots = {}
        
        for waypoint in graph.waypoints:
            if waypoint.snapshot_id:
                snapshot_path = os.path.join(self.snapshot_dir, waypoint.snapshot_id)
                if os.path.exists(snapshot_path):
                    with open(snapshot_path, "rb") as f:
                        snapshot = map_pb2.WaypointSnapshot()
                        snapshot.ParseFromString(f.read())
                        snapshots[snapshot.id] = snapshot
                        
        return graph, waypoints, snapshots
    
    def load_rag_annotations(self) -> Dict:
        """Load RAG annotations."""
        annotations = {}
        metadata_files = [f for f in os.listdir(self.rag_db_path) 
                         if f.startswith("metadata_") and f.endswith(".json")]
        
        for metadata_file in metadata_files:
            with open(os.path.join(self.rag_db_path, metadata_file)) as f:
                metadata = json.load(f)
                waypoint_id = metadata["waypoint_id"]
                annotations[waypoint_id] = metadata
                
        return annotations
    @staticmethod
    def _clean_text(text):
        # Remove articles
        text = re.sub(r'\b(a|an|the)\b', '', text)
        # Remove plurals
        text = re.sub(r's\b', '', text)
        # Convert to lowercase
        text = text.lower()
        return text
    
    def extract_all_objects(self) -> List[str]:
        """Extract all unique objects from the annotations."""
        objects = set()
        for ann in self.waypoint_annotations.values():
            if "views" in ann:
                for view_data in ann["views"].values():
                    visible_objects = view_data.get("visible_objects", [])
                    # Clean and normalize object names
                    visible_objects = [self._clean_text(obj) for obj in visible_objects]
                    
                    objects.update(visible_objects)
        return sorted(objects)
    
    def load_waypoint_images(self) -> Dict[str, str]:
        """Load and encode images for each waypoint."""
        waypoint_images = {}
        
        for waypoint in self.graph.waypoints:
            if self.logger:
                self.logger.info(f"Processing waypoint {waypoint.id}")
            
            snapshot_path = os.path.join(self.snapshot_dir, waypoint.snapshot_id)
            
            if not os.path.exists(snapshot_path):
                if self.logger:
                    self.logger.warning(f"Snapshot not found: {snapshot_path}")
                continue
                
            try:
                snapshot = map_pb2.WaypointSnapshot()
                with open(snapshot_path, 'rb') as f:
                    snapshot.ParseFromString(f.read())
                
                for image in snapshot.images:
                    if image.source.name not in self.FRONT_CAMERA_SOURCES:
                        continue
                        
                    opencv_image, _ = self.convert_image_from_snapshot(
                        image.shot.image,
                        image.source.name
                    )
        
                    if opencv_image is not None:
                        waypoint_images[waypoint.id] = self._encode_image_to_base64(opencv_image)
                        if self.logger:
                            self.logger.info(f"Successfully added image for waypoint {waypoint.id}")
                    break  # Only store one image per waypoint
                        
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error processing waypoint {waypoint.id}: {str(e)}")
                    
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
                image = Image.fromarray(cv_image, mode='L')
            else:
                # For color images, convert BGR to RGB
                image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            
            buffered = BytesIO()
            image.save(buffered, format="JPEG", quality=85)
            jpeg_size = len(buffered.getvalue())
            if self.logger:
                self.logger.info(f"JPEG compressed size: {jpeg_size}")
                
            encoded = base64.b64encode(buffered.getvalue()).decode('utf-8')
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
    

    def create_graph_figure(self, filtered_waypoints: Optional[List[str]] = None):
        """Create the main graph visualization using proper transforms."""
        # Compute global transforms if not already computed
        if not hasattr(self, 'global_transforms'):
            self.compute_global_transforms()
        
        # Extract waypoint positions using global transforms
        x, y = [], []
        hover_texts = []
        edge_x, edge_y = [], []
        
        for waypoint in self.graph.waypoints:
            # Get the global transform for this waypoint
            world_transform = self.global_transforms[waypoint.id]
            # Extract position from transform matrix (last column)
            position = world_transform[:3, 3]
            x.append(position[0])
            y.append(position[1])
            
            # Create detailed hover text
            hover_text = f"Location: {waypoint.annotations.name}<br>"
            hover_text += f"Waypoint ID: {waypoint.id}<br>"
            hover_text += f"Position: ({position[0]:.2f}, {position[1]:.2f})<br>"
            
            # Add annotation information from RAG
            if waypoint.id in self.waypoint_annotations:
                ann = self.waypoint_annotations[waypoint.id]
                if "views" in ann:
                    for view_type, view_data in ann["views"].items():
                        hover_text += f"<br>{self._clean_text(view_type)} objects:<br>"
                        hover_text += "<br>".join(
                            f"- {self._clean_text(obj)}" for obj in view_data.get("visible_objects", [])
                        )
            hover_texts.append(hover_text)
        
        # Add edges using global transforms
        for edge in self.graph.edges:
            from_transform = self.global_transforms[edge.id.from_waypoint]
            to_transform = self.global_transforms[edge.id.to_waypoint]
            
            from_pos = from_transform[:3, 3]
            to_pos = to_transform[:3, 3]
            
            edge_x.extend([from_pos[0], to_pos[0], None])
            edge_y.extend([from_pos[1], to_pos[1], None])
        
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(
            go.Scatter(
                x=edge_x, y=edge_y,
                mode='lines',
                line=dict(color='lightgrey', width=1),
                hoverinfo='none',
                showlegend=False
            )
        )
        
        # Add waypoints
        marker_colors = [
            'red' if filtered_waypoints and waypoint.id in filtered_waypoints else 'blue'
            for waypoint in self.graph.waypoints
        ]
        
        fig.add_trace(
            go.Scatter(
                x=x, y=y,
                mode='markers+text',
                marker=dict(
                    size=8, 
                    color=marker_colors,
                    symbol='circle'
                ),
                text=[wp.annotations.name for wp in self.graph.waypoints],
                textposition="top center",
                hovertext=hover_texts,
                hoverinfo='text',
                showlegend=False
            )
        )
        
        # Update layout
        fig.update_layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(t=0, l=0, r=0, b=0),
            plot_bgcolor='white',
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray'),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='LightGray')
        )
        
        # Make axes equal scale
        fig.update_yaxes(
            scaleanchor="x",
            scaleratio=1
        )
        
        return fig
        
    def setup_layout(self):
        """Set up the Dash app layout."""
        self.app.layout = html.Div([
            html.Div([
                dcc.Dropdown(
                    id='object-filter',
                    options=[{'label': obj, 'value': obj} for obj in self.all_objects],
                    placeholder="Filter by object...",
                    multi=True
                ),
                dcc.Graph(
                    id='map-graph',
                    figure=self.create_graph_figure(),
                    style={'height': '500px', 'width': '70%', 'display': 'inline-block'}
                ),
                html.Div([
                    html.Img(
                        id='waypoint-image-left',
                        style={'width': '100%'}
                    ),
                    html.Img(
                        id='waypoint-image-right',
                        style={'width': '100%'}
                    )
                ], style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'top'})
            ]),
            dash_table.DataTable(
                id='waypoint-info',
                columns=[
                    {'name': 'Property', 'id': 'property'},
                    {'name': 'Value', 'id': 'value'}
                ],
                data=[],
                style_table={'margin-top': '20px', 'width': '70%'}
            )
        ])
        
    def setup_callbacks(self):
        """Set up the Dash callbacks."""
        @self.app.callback(
            [Output('map-graph', 'figure'),
             Output('waypoint-image-left', 'src'),
             Output('waypoint-image-right', 'src'),
             Output('waypoint-info', 'data')],
            [Input('map-graph', 'clickData'),
             Input('object-filter', 'value')]
        )
        def update_ui(clickData, selected_objects):
            # Filter waypoints based on selected objects
            filtered_waypoints = []
            if selected_objects:
                for waypoint_id, ann in self.waypoint_annotations.items():
                    if "views" in ann:
                        for view_data in ann["views"].values():
                            if any(obj in view_data.get("visible_objects", []) for obj in selected_objects):
                                filtered_waypoints.append(waypoint_id)
                                break

            # Update the map figure
            map_figure = self.create_graph_figure(filtered_waypoints)

            # Update the images and info based on clicked waypoint
            if not clickData:
                return map_figure, f"data:image/jpeg;base64,{self.default_image_base64}", f"data:image/jpeg;base64,{self.default_image_base64}", []

            point = clickData['points'][0]
            waypoint_id = self.graph.waypoints[point['pointIndex']].id
            
            # Get images
            image_data_left = self.waypoint_images.get(waypoint_id, self.default_image_base64)
            image_data_right = self.waypoint_images.get(waypoint_id, self.default_image_base64)
            image_src_left = f"data:image/jpeg;base64,{image_data_left}"
            image_src_right = f"data:image/jpeg;base64,{image_data_right}"
            
            # Get waypoint info
            info = []
            waypoint = self.waypoints[waypoint_id]
            info.append({'property': 'Waypoint ID', 'value': waypoint_id})
            info.append({'property': 'Location', 'value': waypoint.annotations.name})
            
            if waypoint_id in self.waypoint_annotations:
                ann = self.waypoint_annotations[waypoint_id]
                if "views" in ann:
                    for view_type, view_data in ann["views"].items():
                        objects = view_data.get("visible_objects", [])
                        if objects:
                            info.append({'property': f"{view_type} Objects", 'value': ", ".join(objects)})
            
            return map_figure, image_src_left, image_src_right, info
            
    def run(self, debug=True, port=8050):
        """Run the Dash app."""
        self.app.run(debug=debug, port=port)

if __name__ == "__main__":
    visualizer = DashMapVisualizer(
        graph_path="visualizer/assets/maps/chair_v3",
        rag_db_path="visualizer/assets/database/chair_v3",
        logger=None
    )
    visualizer.run()