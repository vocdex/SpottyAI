import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from bosdyn.api.graph_nav import map_pb2
from typing import Dict, List, Optional, Tuple
import os
import json

class MapVisualizer:
    def __init__(
        self,
        graph_path: str,
        rag_db_path: str,
        logger=None
    ):
        """
        Initialize the map visualizer.
        
        Args:
            graph_path: Path to GraphNav map directory
            rag_db_path: Path to RAG vector store directory
            logger: Optional logger instance
        """
        self.graph_path = graph_path
        self.rag_db_path = rag_db_path
        self.logger = logger
        
        # Load graph and RAG data
        self.graph, self.waypoints, self.snapshots = self.load_graph()
        self.waypoint_annotations = self.load_rag_annotations()
        
    def load_graph(self) -> Tuple[map_pb2.Graph, Dict, Dict]:
        """Load GraphNav map data."""
        with open(os.path.join(self.graph_path, "graph"), "rb") as f:
            graph = map_pb2.Graph()
            graph.ParseFromString(f.read())
            
        waypoints = {}
        snapshots = {}
        
        for waypoint in graph.waypoints:
            waypoints[waypoint.id] = waypoint
            
            if len(waypoint.snapshot_id) > 0:
                snapshot_path = os.path.join(
                    self.graph_path,
                    "waypoint_snapshots",
                    waypoint.snapshot_id
                )
                if os.path.exists(snapshot_path):
                    with open(snapshot_path, "rb") as f:
                        snapshot = map_pb2.WaypointSnapshot()
                        snapshot.ParseFromString(f.read())
                        snapshots[snapshot.id] = snapshot
        
        return graph, waypoints, snapshots
    
    def load_rag_annotations(self) -> Dict:
        """Load RAG annotations from vector store metadata."""
        annotations = {}
        metadata_files = [f for f in os.listdir(self.rag_db_path) 
                         if f.startswith("metadata_") and f.endswith(".json")]
        
        for metadata_file in metadata_files:
            with open(os.path.join(self.rag_db_path, metadata_file)) as f:
                metadata = json.load(f)
                waypoint_id = metadata["waypoint_id"]
                annotations[waypoint_id] = metadata
                
        return annotations

    def filter_by_object(self, object_name: str) -> List[str]:
        """
        Filter waypoints by presence of a specific object.
        
        Args:
            object_name: Name of object to filter by
            
        Returns:
            List of waypoint IDs containing the object
        """
        filtered_waypoints = []
        for wp_id, ann in self.waypoint_annotations.items():
            for view_data in ann.get('views', {}).values():
                if object_name.lower() in [obj.lower() for obj in view_data.get('visible_objects', [])]:
                    filtered_waypoints.append(wp_id)
        return filtered_waypoints

    def color_by_location(self, location_colors: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """
        Generate colors for waypoints based on their location type.
        
        Args:
            location_colors: Optional dictionary mapping location types to colors.
                           If not provided, will generate colors automatically.
                           
        Returns:
            Dictionary mapping waypoint IDs to colors
        """
        # Get unique location types
        location_types = set()
        for waypoint in self.graph.waypoints:
            if waypoint.annotations.name:
                # Extract general location type (e.g., 'Kitchen Area' -> 'Kitchen')
                location_type = waypoint.annotations.name.split()[0]
                location_types.add(location_type)
        
        # Generate colors if not provided
        if location_colors is None:
            import plotly.express as px
            colors = px.colors.qualitative.Set3
            location_colors = {
                loc_type: colors[i % len(colors)]
                for i, loc_type in enumerate(location_types)
            }
            
        # Map waypoints to colors
        waypoint_colors = {}
        for waypoint in self.graph.waypoints:
            location_type = waypoint.annotations.name.split()[0] if waypoint.annotations.name else 'Unknown'
            waypoint_colors[waypoint.id] = location_colors.get(location_type, '#808080')  # Grey for unknown
            
        return waypoint_colors

    def create_visualization(self, 
                           filtered_waypoints: Optional[List[str]] = None,
                           waypoint_colors: Optional[Dict[str, str]] = None):
        """
        Create interactive Plotly visualization.
        
        Args:
            filtered_waypoints: Optional list of waypoint IDs to highlight
            waypoint_colors: Optional dictionary mapping waypoint IDs to colors
        """
        # Create figure with secondary y-axis
        fig = make_subplots(rows=1, cols=2,
                           column_widths=[0.7, 0.3],
                           specs=[[{"type": "scatter3d"}, {"type": "table"}]])
        
        # Extract waypoint positions and annotations
        x, y, z = [], [], []
        waypoint_ids = []
        annotations = []
        location_names = []
        edge_x, edge_y, edge_z = [], [], []
        
        for waypoint in self.graph.waypoints:
            pos = waypoint.waypoint_tform_ko.position
            x.append(pos.x)
            y.append(pos.y) 
            z.append(pos.z)
            waypoint_ids.append(waypoint.id)
            location_names.append(waypoint.annotations.name if waypoint.annotations.name else "Unknown")
            
            # Add hover text with annotations if available
            hover_text = f"Location: {waypoint.annotations.name}<br>"
            hover_text += f"Waypoint ID: {waypoint.id}<br>"
            
            if waypoint.id in self.waypoint_annotations:
                ann = self.waypoint_annotations[waypoint.id]
                if "views" in ann:
                    for view_type, view_data in ann["views"].items():
                        hover_text += f"<br>{view_type} objects:<br>"
                        hover_text += "<br>".join(
                            f"- {obj}" for obj in view_data.get("visible_objects", [])
                        )
            
            annotations.append(hover_text)
        
        # Add edges
        for edge in self.graph.edges:
            from_wp = self.waypoints[edge.id.from_waypoint]
            to_wp = self.waypoints[edge.id.to_waypoint]
            
            from_pos = from_wp.waypoint_tform_ko.position
            to_pos = to_wp.waypoint_tform_ko.position
            
            edge_x.extend([from_pos.x, to_pos.x, None])
            edge_y.extend([from_pos.y, to_pos.y, None])
            edge_z.extend([from_pos.z, to_pos.z, None])
        
        # Add edges as lines
        fig.add_trace(
            go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                mode='lines',
                line=dict(color='lightgrey', width=1),
                hoverinfo='none',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Prepare waypoint colors and sizes
        if waypoint_colors is None:
            waypoint_colors = {wp_id: 'blue' for wp_id in waypoint_ids}
            
        # Process colors and apply opacity through color alpha
        from matplotlib.colors import to_rgba, to_hex
        
        def adjust_color_opacity(color, opacity):
            """Convert color to rgba and adjust opacity."""
            rgba = to_rgba(color)
            return f'rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},{opacity})'
        
        marker_colors = []
        marker_sizes = []
        
        for wp_id in waypoint_ids:
            base_color = waypoint_colors[wp_id]
            is_highlighted = filtered_waypoints is None or wp_id in filtered_waypoints
            
            # Adjust opacity through color
            color = adjust_color_opacity(base_color, 1.0 if is_highlighted else 0.3)
            size = 8 if is_highlighted else 5
            
            marker_colors.append(color)
            marker_sizes.append(size)
        
        # Add waypoints as markers
        fig.add_trace(
            go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers+text',
                marker=dict(
                    size=marker_sizes,
                    color=marker_colors,
                    symbol='circle',
                    opacity=1.0
                ),
                text=location_names,  # Use location names instead of waypoint IDs
                textposition='top center',  # Position text above markers
                hovertext=annotations,
                hoverinfo='text',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Add annotation table (initially empty)
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['Property', 'Value'],
                    fill_color='paleturquoise',
                    align='left'
                ),
                cells=dict(
                    values=[[], []],
                    align='left'
                )
            ),
            row=1, col=2
        )
        
        # Get unique objects across all waypoints
        unique_objects = set()
        for ann in self.waypoint_annotations.values():
            for view_data in ann.get('views', {}).values():
                objects = [obj.lower() for obj in view_data.get('visible_objects', [])]
                unique_objects.update(objects)
        
        # Create dropdown for object filtering
        object_buttons = []
        for obj_name in sorted(unique_objects):
            filtered_ids = self.filter_by_object(obj_name)
            if self.logger:
                self.logger.info(f"Creating filter for {obj_name}: found in {len(filtered_ids)} waypoints")
            
            # Build visibility array - True for filtered waypoints, False for others
            visibility = []
            for wp_id in waypoint_ids:
                is_visible = wp_id in filtered_ids
                visibility.append(is_visible)
                
            object_buttons.append(dict(
                args=[{
                    'marker.size': [8 if wp_id in filtered_ids else 5 for wp_id in waypoint_ids],
                    'marker.color': [waypoint_colors[wp_id] for wp_id in waypoint_ids],
                    'visible': [True],  # Keep trace visible
                    'marker.opacity': [1.0 if wp_id in filtered_ids else 0 for wp_id in waypoint_ids]
                }],
                label=f"{obj_name.title()} ({len(filtered_ids)})",
                method='restyle'
            ))
        
        # Add "Show All" option
        object_buttons.insert(0, dict(
            args=[{
                'marker.size': [8] * len(waypoint_ids),
                'marker.color': [waypoint_colors[wp_id] for wp_id in waypoint_ids],
                'visible': [True],
                'marker.opacity': [1.0] * len(waypoint_ids)
            }],
            label='Show All',
            method='restyle'
        ))

        # Get unique location types
        location_types = {waypoint.annotations.name.split()[0].lower() 
                         for waypoint in self.graph.waypoints 
                         if waypoint.annotations.name}

        # Default color scheme
        default_colors = {
            'kitchen': '#FF0000',
            'office': '#00FF00',
            'hallway': '#0000FF',
            'lab': '#FFFF00',
            'conference': '#FF00FF'
        }
        
        # Create color scheme buttons
        color_buttons = [
            dict(
                args=[{
                    'marker.color': [adjust_color_opacity(default_colors.get(
                        self.waypoints[wp_id].annotations.name.split()[0].lower(), '#808080'
                    ), 1.0) for wp_id in waypoint_ids]
                }],
                label='Default Colors',
                method='restyle'
            ),
            dict(
                args=[{
                    'marker.color': ['#0000FF'] * len(waypoint_ids)
                }],
                label='Single Color',
                method='restyle'
            )
        ]

        # Update layout with UI controls
        fig.update_layout(
            title='GraphNav Map with RAG Annotations',
            scene=dict(
                aspectmode='data',
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            updatemenus=[
                # Object filter dropdown
                dict(
                    buttons=object_buttons,
                    direction='down',
                    showactive=True,
                    x=0.1,
                    y=1.1,
                    xanchor='left',
                    yanchor='top',
                    name='Object Filter',
                    bgcolor='#FFFFFF',
                    type='dropdown'
                ),
                # Color scheme dropdown
                dict(
                    buttons=color_buttons,
                    direction='down',
                    showactive=True,
                    x=0.4,
                    y=1.1,
                    xanchor='left',
                    yanchor='top',
                    name='Color Scheme',
                    bgcolor='#FFFFFF',
                    type='dropdown'
                )
            ],
            # Add dropdown labels
            annotations=[
                dict(text="Filter by Object:", x=0.1, y=1.15, 
                     xref="paper", yref="paper", showarrow=False),
                dict(text="Color Scheme:", x=0.4, y=1.15,
                     xref="paper", yref="paper", showarrow=False)
            ],
            showlegend=False,
            height=800
        )

        def update_table(trace, points, selector):
            """Callback to update annotation table on waypoint click."""
            if not points.point_inds:
                return
            
            point_idx = points.point_inds[0]
            waypoint_id = waypoint_ids[point_idx]
            
            if waypoint_id in self.waypoint_annotations:
                ann = self.waypoint_annotations[waypoint_id]
                properties = []
                values = []
                
                # Basic properties
                properties.extend(['Waypoint ID', 'Location'])
                values.extend([waypoint_id, ann.get('location', 'N/A')])
                
                # Add view-specific data
                for view_type, view_data in ann.get('views', {}).items():
                    properties.append(f'\n{view_type} Objects')
                    values.append('\n' + '\n'.join(
                        f"- {obj}" for obj in view_data.get('visible_objects', [])
                    ))
                
                with fig.batch_update():
                    fig.data[-1].header.values = ['Property', 'Value']
                    fig.data[-1].cells.values = [properties, values]
        
        # Add click callback
        fig.data[1].on_click(update_table)
        
        return fig
    
    def show(self, 
             filter_object: Optional[str] = None,
             location_colors: Optional[Dict[str, str]] = None):
        """
        Create and display the visualization.
        
        Args:
            filter_object: Optional object name to filter waypoints by
            location_colors: Optional dictionary mapping location types to colors
        """
        filtered_waypoints = None
        if filter_object:
            filtered_waypoints = self.filter_by_object(filter_object)
            
        waypoint_colors = None
        if location_colors is not None:
            waypoint_colors = self.color_by_location(location_colors)
            
        fig = self.create_visualization(
            filtered_waypoints=filtered_waypoints,
            waypoint_colors=waypoint_colors
        )
        fig.show()

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("MapVisualizer")
    
    visualizer = MapVisualizer(
        graph_path="assets/maps/chair_v3",
        rag_db_path="assets/database/chair_v3",
        logger=logger
    )
    location_colors = {
        'kitchen': '#FF0000',
        'office': '#00FF00',
        'hallway': '#0000FF',
        'lab': '#FFFF00',
    }
    visualizer.show(location_colors=location_colors)