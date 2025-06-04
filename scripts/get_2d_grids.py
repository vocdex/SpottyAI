#                                                                               
# Copyright (c) 2025, FAU, Shukrullo Nazirjonov
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#                                                                               
"""Convert .ply file to 2D occupancy grid map. White pixels are occupied, black pixels are free space."""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from PIL import Image
from scipy.ndimage import binary_dilation, gaussian_filter


def point_cloud_to_occupancy_grid(ply_file, resolution=0.1, height_filter=(0.0, 2.0), apply_filters=True):
    """
    Converts a 3D point cloud into a 2D occupancy grid map.

    Args:
        ply_file (str): Path to the .ply file.
        resolution (float): Grid cell size in meters.
        height_filter (tuple): Min and max Z values for filtering points.
        apply_filters (bool): Whether to apply dilation and smoothing.

    Returns:
        np.ndarray: 2D occupancy grid map.
    """
    # Load point cloud
    pcd = o3d.io.read_point_cloud(ply_file)
    points = np.asarray(pcd.points)

    # Filter by height
    z_min, z_max = height_filter
    filtered_points = points[(points[:, 2] >= z_min) & (points[:, 2] <= z_max)]
    points_2d = filtered_points[:, :2]  # Project to XY-plane

    # Determine grid bounds
    x_min, y_min = points_2d.min(axis=0)
    x_max, y_max = points_2d.max(axis=0)

    grid_width = int((x_max - x_min) / resolution) + 1
    grid_height = int((y_max - y_min) / resolution) + 1

    # Create occupancy grid
    grid = np.zeros((grid_height, grid_width), dtype=np.int8)
    for x, y in points_2d:
        grid_x = int((x - x_min) / resolution)
        grid_y = int((y - y_min) / resolution)
        grid[grid_y, grid_x] = 1  # Mark as occupied

    # Apply filters
    if apply_filters:
        # Dilation to fill gaps
        grid = binary_dilation(grid, structure=np.ones((1, 1))).astype(np.int8)

        # Gaussian smoothing to remove noise
        grid = gaussian_filter(grid.astype(float), sigma=1.0) > 0.1  # Threshold to keep binary

    return grid


def occupancy_grid_to_image(grid, resolution=0.1):
    """
    Converts an occupancy grid map into a grayscale image.

    Args:
        grid (np.ndarray): 2D occupancy grid map.
        resolution (float): Grid cell size in meters.

    Returns:
        PIL.Image: Grayscale image.
    """
    grid = grid.astype(np.uint8) * 255
    img = Image.fromarray(grid, mode="L")
    # Invert image vertically
    img = img.transpose(method=Image.FLIP_TOP_BOTTOM)
    return img


def plot_occupancy_grid(grid, resolution=0.1):
    """
    Plots an occupancy grid map.

    Args:
        grid (np.ndarray): 2D occupancy grid map.
        resolution (float): Grid cell size in meters.
    """
    plt.imshow(grid, cmap="gray", origin="lower")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Occupancy Grid Map")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a 3D point cloud into a 2D occupancy grid map.")
    parser.add_argument("--ply_file", type=str, help="Path to the .ply file")
    parser.add_argument("--output_dir", type=str, default="./images", help="Output directory for the image")
    parser.add_argument("--resolution", type=float, default=0.1, help="Grid cell size in meters")
    parser.add_argument(
        "--height_filter", type=float, nargs=2, default=(0.0, 2.0), help="Min and max Z values for filtering points"
    )
    parser.add_argument("--apply_filters", action="store_true", help="Apply dilation and smoothing filters")
    args = parser.parse_args()

    grid = point_cloud_to_occupancy_grid(args.ply_file, args.resolution, args.height_filter, args.apply_filters)
    plot_occupancy_grid(grid, args.resolution)

    img = occupancy_grid_to_image(grid, args.resolution)
    ply_name = os.path.basename(args.ply_file)
    img_name = os.path.splitext(ply_name)[0] + "_filters_" + str(args.apply_filters) + ".png"
    img_name = os.path.join(args.output_dir, img_name)
    img.save(img_name)
    print(f"Saved occupancy grid image to {img_name}")
