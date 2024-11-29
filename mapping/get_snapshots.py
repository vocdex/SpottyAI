"""Experimental code to load a graph with waypoint images and do some processing. Not ready for usage."""
import os
from bosdyn.api.graph_nav import map_pb2
from google.protobuf import json_format
from google.protobuf import text_format
import numpy as np
import cv2
"""Path to Waypoint Message Structure: https://dev.bostondynamics.com/protos/bosdyn/api/proto_reference#waypoint"""
# Path to the downloaded map files
graph_path = "/Users/shuk/Desktop/spot/practical-seminar-mobile-robotics/mapping/downloaded_graph_images/graph"
waypoint_snapshot_dir = "/Users/shuk/Desktop/spot/practical-seminar-mobile-robotics/mapping/downloaded_graph_images/waypoint_snapshots"
snapshot_id = "snapshot_dapper-hawk-7STA4IsiCEApEPC8jGphFA=="

# Load the graph
graph = map_pb2.Graph()
with open(graph_path, 'rb') as graph_file:
    graph.ParseFromString(graph_file.read())

# List all waypoints
waypoints = graph.waypoints
print(f"Total waypoints: {len(waypoints)}")
for waypoint in waypoints:
    # print(type(waypoint))
    # print(f"Waypoint ID: {waypoint.id}, Annotation: {waypoint.annotations.name}")
    # print(f"Snapshot ID: {waypoint.snapshot_id}")
    # Waypoint Fields: id, snapshot_id, waypoint_tform_ko, annotations

    waypoint_num = int(waypoint.annotations.name.split("_")[1])
   


def load_waypoint_snapshot(snapshot_id, snapshot_dir):
    snapshot_path = os.path.join(snapshot_dir, snapshot_id)
    print(snapshot_path)
    snapshot = map_pb2.WaypointSnapshot()
    with open(snapshot_path, 'rb') as snapshot_file:
        snapshot.ParseFromString(snapshot_file.read())
    return snapshot

def convert_depth_image_to_opencv(image_proto):
    """
    Convert depth image data from protobuf format to OpenCV format.
    
    :param image_proto: Protobuf message containing the image data.
    :return: Depth image in OpenCV format.
    """
    # Extract raw image data
    raw_data = image_proto.data

    # Get image dimensions from the protobuf message
    height = image_proto.rows
    width = image_proto.cols

    # Calculate the expected size
    expected_size = height * width * 2  # 2 bytes per pixel for uint16

    if len(raw_data) != expected_size:
        print(f"Mismatch: Raw data size is {len(raw_data)}, expected {expected_size}")
        # Fix mismatch if necessary (e.g., resize or crop)
        adjusted_size = min(len(raw_data) // 2, height * width)
        height = adjusted_size // width
        print(f"Adjusting to new height: {height}")

    # Convert raw data to a NumPy array of the appropriate type (uint16 for depth)
    depth_image = np.frombuffer(raw_data, dtype=np.uint16).reshape((height, width))

    # Optionally, normalize the depth image for visualization (convert to 8-bit)
    normalized_image = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return depth_image, normalized_image


def load_local_graph_and_snapshot(graph_file_path, snapshot_dir):
    """
    Load a locally saved graph and its associated waypoint snapshots.

    :param graph_file_path: Path to the saved graph file.
    :param snapshot_dir: Directory containing the waypoint snapshot files.
    :return: Graph and a dictionary of snapshot IDs to snapshot objects.
    """
    # Load the graph
    graph = map_pb2.Graph()
    with open(graph_file_path, 'rb') as graph_file:
        graph.ParseFromString(graph_file.read())
    print(f"Loaded graph with {len(graph.waypoints)} waypoints.")

    # Load all snapshots
    snapshots = {}
    for waypoint in graph.waypoints:
        # Annotation is a human-readable name for the waypoint.
        annotation = waypoint.annotations.name
        print(f"Waypoint: {annotation}")
        # We can load the images for each snapshot using the snapshot ID.
        # Then we classify the images using a pre-trained model.
        # We annotate the waypoint with the classification results. ["kitchen", "living_room", "bedroom", "bathroom"]
        # We can then use the annotated waypoints to plan a path through the house.


        # Snapshot ID is the unique identifier for the snapshot.
        snapshot_id = waypoint.snapshot_id
        snapshot_file_path = os.path.join(snapshot_dir, f"{snapshot_id}")
        if os.path.exists(snapshot_file_path):
            snapshot = map_pb2.WaypointSnapshot()
            with open(snapshot_file_path, 'rb') as snapshot_file:
                snapshot.ParseFromString(snapshot_file.read())
            snapshots[snapshot_id] = snapshot
            # Fields: images, point_cloud, objects,
            # access images
            images = snapshot.images
            for image in images:
                image_data = image.shot 
                print()
                image = image_data.image
                depth_image, normalized_image = convert_depth_image_to_opencv(image)
                # Show normalized image
                cv2.imshow("Normalized Image", normalized_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            print(f"Loaded snapshot: {snapshot_id} with {len(images)} images.")
            # f = open('b.txt', 'w')
            # f.write(text_format.MessageToString(snapshot))
            # f.close()
            # Getting the fields of the snapshot
            indexes = range(len(snapshot.DESCRIPTOR.fields))
            for index in indexes:
                field_name = snapshot.DESCRIPTOR.fields[index].name
                print(field_name)
            
            # Getting the value of the field
            print(getattr(snapshot, "is_point_cloud_processed"))
            field_name = "is_point_cloud_processed"
            # Set the value of the field. Note this doesn't work for repeated fields or nested messages.
            setattr(snapshot, field_name, True)
            print(getattr(snapshot, field_name))
            # Save the snapshot with the updated field
            with open(snapshot_file_path, 'wb') as snapshot_file:
                snapshot_file.write(snapshot.SerializeToString())
            # print(objects)

            # print keys of snapshot

            # breakpoint()
            print(f"Loaded snapshot: {snapshot_id} with {len(images)} images.")
            point_cloud = snapshot.point_cloud.num_points
            print(f"Loaded snapshot: {snapshot_id} with {point_cloud} point_cloud.")
        else:
            print(f"Snapshot file not found for ID: {snapshot_id}")

    return graph, snapshots

load_local_graph_and_snapshot(graph_path, waypoint_snapshot_dir)


# # # Example: Load a waypoint snapshot with images
# waypoint_snapshot = load_waypoint_snapshot(snapshot_id, waypoint_snapshot_dir)
# print(waypoint_snapshot)

# if waypoint_snapshot.image_sources:
#     for image_source in waypoint_snapshot.image_sources:
#         print(f"Found image source: {image_source.name}")
# else:
#     print("No images were recorded in this snapshot.")


