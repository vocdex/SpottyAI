import os
import cv2
import argparse
import numpy as np
from scipy import ndimage
from bosdyn.api import image_pb2
from bosdyn.api.graph_nav import map_pb2
from utils import get_map_paths

# Rotation angles for different camera sources (similar to previous script)
ROTATION_ANGLE = {
    'back_fisheye_image': 0,
    'frontleft_fisheye_image': -90,
    'frontright_fisheye_image': -90,
    'left_fisheye_image': 0,
    'right_fisheye_image': 180
}
camera_sources = ['back_depth', 'back_depth_in_visual_frame', 'back_fisheye_image', 'frontleft_depth', 'frontleft_depth_in_visual_frame', 'frontleft_fisheye_image', 'frontright_depth', 'frontright_depth_in_visual_frame', 'frontright_fisheye_image', 'left_depth', 'left_depth_in_visual_frame', 'left_fisheye_image', 'right_depth', 'right_depth_in_visual_frame', 'right_fisheye_image']
depth_camera_sources = ['back_depth', 'frontleft_depth', 'frontright_depth', 'left_depth', 'right_depth']


def convert_image_from_snapshot(image_data,image_source, auto_rotate=True):
    """
    Convert an image from a GraphNav waypoint snapshot to an OpenCV image.
    
    :param image_data: Image data from WaypointSnapshot
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
            rotation_angle = ROTATION_ANGLE.get(image_source, 0)
            img = ndimage.rotate(img, rotation_angle)
        except KeyError:
            print(f"Warning: No rotation defined for source {image_source}")

    return img, extension

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
    for i,waypoint in enumerate(graph.waypoints):
        print(waypoint.annotations.name)
        
        print(f"Processing waypoint {i+1}/{len(graph.waypoints)}")
        # print("Processing waypoint:", waypoint.id)
        # Snapshot ID is the unique identifier for the snapshot
        snapshot_id = waypoint.snapshot_id
        print("Snapshot ID:", snapshot_id)
        snapshot_file_path = os.path.join(snapshot_dir, f"{snapshot_id}")
        
        if os.path.exists(snapshot_file_path):
            snapshot = map_pb2.WaypointSnapshot()
            with open(snapshot_file_path, 'rb') as snapshot_file:
                snapshot.ParseFromString(snapshot_file.read())
            snapshots[snapshot_id] = snapshot

            # Process and display images from this snapshot
            process_snapshot_images(snapshot)

    return graph, snapshots

def process_snapshot_images(snapshot, display=True):
    """
    Process and optionally display images from a waypoint snapshot.
    
    :param snapshot: WaypointSnapshot object
    :param display: Whether to display images
    :return: List of processed images
    """
    processed_images = []
    
    # Access images in the snapshot
    image_response = snapshot.images
    for image in image_response:
        # Get the image data
        image_source = image.source.name
        # Ignore depth images for now
        if image_source in depth_camera_sources:
            continue
        image_data = image.shot.image
        try:
            # Convert image to OpenCV format
            opencv_image, _ = convert_image_from_snapshot(image_data,image_source)
            processed_images.append(opencv_image)
            
            # Display image if requested
            if display:
                # Use the source name as the window title
                window_name = image_source
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.imshow(window_name, opencv_image)   
                # pause for a second
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        
        except Exception as e:
            print(f"Error processing image: {e}")
    
    return processed_images

# Example usage
def main(args):
    
    graph_file_path, snapshot_dir, _ = get_map_paths(args.map_path)

    graph, snapshots = load_local_graph_and_snapshot(graph_file_path, snapshot_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--map_path', help='Path to the map directory')
    args = parser.parse_args()
    main()
