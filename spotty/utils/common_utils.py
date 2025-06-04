#                                                                               
# Copyright (c) 2025, FAU, Shukrullo Nazirjonov
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#                                                                               
import os
from typing import Dict, List


def get_map_paths(map_path: str) -> List[str]:
    # Allow relative paths
    if not os.path.isabs(map_path):
        map_path = os.path.abspath(map_path)
    # Check if the path exists
    if not os.path.exists(map_path):
        raise FileNotFoundError(f"Path {map_path} does not exist.")
    graph_file_path = os.path.join(map_path, "graph")
    snapshot_dir = os.path.join(map_path, "waypoint_snapshots")
    output_graph_path = os.path.join(map_path)
    return graph_file_path, snapshot_dir, output_graph_path


def read_manual_labels(label_file: str) -> Dict[str, str]:
    """The custom label file should be a CSV file with two columns: old_label, new_label, separated by a comma."""
    custom_labels = {}
    with open(label_file, "r") as f:
        for line in f:
            old_label, new_label = line.strip().split(",")
            custom_labels[old_label] = new_label
    return custom_labels


def get_abs_path(relative_path: str):
    """This function is used to convert relative path to absolute path."""
    if not os.path.isabs(relative_path):
        return os.path.abspath(relative_path)
    return relative_path
