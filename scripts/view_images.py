"""View images from waypoint snapshots in a GraphNav map.
To run:
python view_images.py --map-path /path/to/map --waypoint-id <waypoint_id>
"""


import os
import cv2
import argparse
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from bosdyn.api import image_pb2
from bosdyn.api.graph_nav import map_pb2
from spotty.utils.common_utils import get_map_paths

# Rotation angles for different camera sources
ROTATION_ANGLE = {
    'back_fisheye_image': 0,
    'frontleft_fisheye_image': -90,
    'frontright_fisheye_image': -90,
    'left_fisheye_image': 0,
    'right_fisheye_image': 180
}

# Camera source categories
DEPTH_CAMERA_SOURCES = ['back_depth', 'frontleft_depth', 'frontright_depth', 'left_depth', 'right_depth']
FRONT_CAMERA_SOURCES = ['frontleft_fisheye_image', 'frontright_fisheye_image']


def convert_image_from_snapshot(image_data, image_source, auto_rotate=True):
    """
    Convert an image from a GraphNav waypoint snapshot to an OpenCV image.

    :param image_data: Image data from WaypointSnapshot
    :param image_source: Name of the image source
    :param auto_rotate: Whether to automatically rotate images based on camera source
    :return: OpenCV image, file extension
    """
    num_channels = 1
    dtype = np.uint8

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

    img = np.frombuffer(image_data.data, dtype=dtype)

    if image_data.format == image_pb2.Image.FORMAT_RAW:
        try:
            img = img.reshape((image_data.rows, image_data.cols, num_channels))
        except ValueError:
            img = cv2.imdecode(img, -1)
    else:
        img = cv2.imdecode(img, -1)

    if auto_rotate:
        rotation_angle = ROTATION_ANGLE.get(image_source, 0)
        img = ndimage.rotate(img, rotation_angle)

    return img, extension


def load_local_graph_and_snapshot(graph_file_path, snapshot_dir):
    """
    Load a locally saved graph and its associated waypoint snapshots.

    :param graph_file_path: Path to the saved graph file.
    :param snapshot_dir: Directory containing the waypoint snapshot files.
    :return: Graph and a dictionary of snapshot IDs to snapshot objects.
    """
    graph = map_pb2.Graph()
    with open(graph_file_path, 'rb') as graph_file:
        graph.ParseFromString(graph_file.read())
    print(f"Loaded graph with {len(graph.waypoints)} waypoints.")

    snapshots = {}
    for waypoint in graph.waypoints:
        snapshot_id = waypoint.snapshot_id
        snapshot_file_path = os.path.join(snapshot_dir, f"{snapshot_id}")

        if os.path.exists(snapshot_file_path):
            snapshot = map_pb2.WaypointSnapshot()
            with open(snapshot_file_path, 'rb') as snapshot_file:
                snapshot.ParseFromString(snapshot_file.read())
            snapshots[waypoint.id] = snapshot

    return graph, snapshots


def process_snapshot_images(snapshot, waypoint_id):
    """
    Process images from a waypoint snapshot and prepare for batch display.

    :param snapshot: WaypointSnapshot object
    :param waypoint_id: The ID of the waypoint
    :return: List of (image, title) tuples
    """
    processed_images = []

    for image in snapshot.images:
        image_source = image.source.name
        if image_source in DEPTH_CAMERA_SOURCES or image_source not in FRONT_CAMERA_SOURCES:
            continue

        image_data = image.shot.image
        try:
            opencv_image, _ = convert_image_from_snapshot(image_data, image_source)
            if image_source == 'frontleft_fisheye_image':
                image_source = "left"
            elif image_source == 'frontright_fisheye_image':
                image_source = "right"
            title = f"{waypoint_id}\n{image_source}"
            processed_images.append((opencv_image, title))
        except Exception as e:
            print(f"Error processing image from {image_source}: {e}")

    return processed_images

def display_images_in_batches(images, grid_size=6):
    """
    Display images in grid batches using matplotlib.

    :param images: List of (image, title) tuples
    :param grid_size: Size of the grid (default is 6x6)
    """
    batch_size = grid_size * grid_size

    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        num_images = len(batch)
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))

        # Handle cases where axes is a single object (e.g., 1x1 grid)
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]

        for j, ax in enumerate(axes):
            if j < num_images:
                img, title = batch[j]
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                ax.set_title(title, fontsize=10)
                ax.axis('off')
            else:
                ax.axis('off')  # Turn off unused axes

        plt.tight_layout()
        plt.show()


def main(args):
    graph_file_path, snapshot_dir, _ = get_map_paths(args.map_path)
    graph, snapshots = load_local_graph_and_snapshot(graph_file_path, snapshot_dir)

    if args.waypoint_id:
        if args.waypoint_id in snapshots:
            images = process_snapshot_images(snapshots[args.waypoint_id], args.waypoint_id)
            display_images_in_batches(images, grid_size=1)
        else:
            print(f"Waypoint ID {args.waypoint_id} not found.")
    else:
        all_images = []
        for waypoint_id, snapshot in snapshots.items():
            all_images.extend(process_snapshot_images(snapshot, waypoint_id))
        display_images_in_batches(all_images)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--map-path', required=True, help='Path to the map directory')
    parser.add_argument('--waypoint-id', help='Specific waypoint ID to display')
    args = parser.parse_args()
    main(args)
