"""Given a recorded GraphNav graph, this script updates the annotations of the waypoints based on a custom mapping.
Custom labels are defined in a dictionary where the original annotation is the key and the new annotation is the value.
Refer to the example_usage function for an example of how to use this script.
"""
import os
import argparse
from typing import Dict
from bosdyn.api.graph_nav import map_pb2

class WaypointAnnotationUpdater:
    def __init__(self, graph_file_path: str):
        """
        Initialize the updater with a graph file.
        
        :param graph_file_path: Path to the saved graph file
        """
        self.graph_file_path = graph_file_path
        self.graph = self._load_graph()

    def _load_graph(self) -> map_pb2.Graph:
        """
        Load the graph from the file.
        
        :return: Loaded graph object
        """
        graph = map_pb2.Graph()
        with open(self.graph_file_path, 'rb') as graph_file:
            graph.ParseFromString(graph_file.read())
        print(f"Loaded graph with {len(graph.waypoints)} waypoints.")
        return graph

    def print_waypoint_annotations(self):
        """
        Print current waypoint annotations before updating.
        """
        print("Current Waypoint Annotations:")
        for i, waypoint in enumerate(self.graph.waypoints):
            print(f"Waypoint {i}: {waypoint.annotations.name}")

    def update_annotations(
        self, 
        custom_labels: Dict[str, str]
    ) -> None:
        """
        Update waypoint annotations using a dictionary of labels.
        
        :param custom_labels: Dictionary mapping original annotations to new annotations.
        """
        for waypoint in self.graph.waypoints:
            original_annotation = waypoint.annotations.name
            
            # Update annotation if there's a matching label
            if original_annotation in custom_labels:
                new_annotation = custom_labels[original_annotation]
                waypoint.annotations.name = new_annotation
                print(f"Updated {original_annotation} to {new_annotation}")

    def save_updated_graph(
        self, 
        output_dir: str = None, 
        output_filename: str = None
    ) -> str:
        """
        Save the updated graph to a new file.
        
        :param output_dir: Optional directory to save the updated graph. 
        :param output_filename: Optional custom filename.
        :return: Path to the saved graph file
        """
        # Determine the output directory
        if output_dir is None:
            output_dir = os.path.dirname(self.graph_file_path)
        
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine the output filename
        if not output_filename:
            base_filename = os.path.basename(self.graph_file_path)
            output_filename = f"updated_{base_filename}"
        
        # Create full output path
        full_output_path = os.path.join(output_dir, output_filename)
        
        # Save the updated graph
        with open(full_output_path, 'wb') as graph_file:
            graph_file.write(self.graph.SerializeToString())
        
        print(f"Updated graph saved to {full_output_path}")
        return full_output_path

def example_usage(args):
    """
    Example usage of WaypointAnnotationUpdater with custom labels.
    
    :param args: Parsed command-line arguments
    """
    custom_labels = {
    # Table Tennis Waypoints
    "waypoint_11": "table_tennis", 
    "waypoint_12": "table_tennis", 
    "waypoint_13": "table_tennis", 
    "waypoint_14": "table_tennis", 
    "waypoint_15": "table_tennis", 
    "waypoint_16": "table_tennis", 
    "waypoint_17": "table_tennis", 
    "waypoint_18": "table_tennis", 
    "waypoint_19": "table_tennis", 
    "waypoint_20": "table_tennis", 
    "waypoint_21": "table_tennis", 
    "waypoint_22": "table_tennis", 
    "waypoint_23": "table_tennis", 
    "waypoint_24": "table_tennis", 
    "waypoint_25": "table_tennis", 
    "waypoint_26": "table_tennis", 

    # Office Waypoints
    "waypoint_0": "office", 
    "waypoint_1": "office", 
    "waypoint_2": "office", 
    "waypoint_3": "office", 
    "waypoint_4": "office", 
    "waypoint_5": "office", 
    "waypoint_6": "office", 
    "waypoint_7": "office", 
    "waypoint_102": "office", 
    "waypoint_103": "office", 
    "waypoint_104": "office", 

    # Hallway Waypoints
    "waypoint_33": "hallway", 
    "waypoint_34": "hallway", 
    "waypoint_35": "hallway", 
    "waypoint_36": "hallway", 
    "waypoint_95": "hallway", 
    "waypoint_96": "hallway", 

    # Kitchen Waypoints
    "waypoint_75": "kitchen", 
    "waypoint_76": "kitchen", 
    "waypoint_77": "kitchen", 
    "waypoint_78": "kitchen", 
    "waypoint_79": "kitchen"
    }


    # Validate input arguments
    if not args.graph_file_path:
        raise ValueError("Graph file path is required")
    
    if not os.path.exists(args.graph_file_path):
        raise FileNotFoundError(f"Graph file not found: {args.graph_file_path}")

    # Initialize the updater
    updater = WaypointAnnotationUpdater(args.graph_file_path)

    # Print current annotations
    updater.print_waypoint_annotations()

    # Update annotations with custom labels    
    updater.update_annotations(custom_labels)

    # Save updated graph
    updater.save_updated_graph(
        output_dir=args.output_dir, 
        output_filename=args.output_filename
    )

def main():
    """
    Main function to parse arguments and call example usage.
    """
    parser = argparse.ArgumentParser(description="Update waypoint annotations in a graph file.")
    parser.add_argument("--graph_file_path", type=str, required=True, 
                        help="Path to the graph file that is to be updated")
    parser.add_argument("--output_dir", type=str, required=False,
                        help="Directory to save the updated graph")
    parser.add_argument("--output_filename", type=str, required=False,
                        help="Filename for the updated graph")
    
    args = parser.parse_args()
    
    try:
        example_usage(args)
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()