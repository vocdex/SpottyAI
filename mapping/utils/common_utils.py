import os
from typing import List, Dict


def get_map_paths(map_path: str) -> List[str]:
    graph_file_path = os.path.join(map_path, 'graph')
    snapshot_dir = os.path.join(map_path, 'waypoint_snapshots')
    output_graph_path = os.path.join(map_path)
    return graph_file_path, snapshot_dir, output_graph_path


def read_manual_labels(label_file: str) -> Dict[str, str]:
    """The custom label file should be a CSV file with two columns: old_label, new_label, separated by a comma."""
    custom_labels = {}
    with open(label_file, 'r') as f:
        for line in f:
            old_label, new_label = line.strip().split(',')
            custom_labels[old_label] = new_label
    return custom_labels