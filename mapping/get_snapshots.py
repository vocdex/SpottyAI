import os
from bosdyn.api.graph_nav import map_pb2
from google.protobuf import json_format

# Path to the downloaded map files
graph_path = "/Users/shuk/Desktop/spot/practical-seminar-mobile-robotics/spot-sdk/python/examples/graph_nav_command_line/downloaded_graph/graph"
waypoint_snapshot_dir = "/Users/shuk/Desktop/spot/practical-seminar-mobile-robotics/spot-sdk/python/examples/graph_nav_command_line/downloaded_graph/waypoint_snapshots"

# Load the graph
graph = map_pb2.Graph()
with open(graph_path, 'rb') as graph_file:
    graph.ParseFromString(graph_file.read())

# List all waypoints
waypoints = graph.waypoints
print(f"Total waypoints: {len(waypoints)}")
# for waypoint in waypoints:
#     print(f"Waypoint ID: {waypoint.id}, Annotation: {waypoint.annotations.name}")
#     # annotation: waypoint_{num}
#     # replace {num} with the following rule: if {num}<10, then annotation = "kitchen", else annotation = "living_room"
#     waypoint_num = int(waypoint.annotations.name.split("_")[1])
#     if waypoint_num < 10:
#         waypoint.annotations.name = "kitchen"
#     else:
#         waypoint.annotations.name = "living_room"
#     print(f"Waypoint ID: {waypoint.id}, Annotation: {waypoint.annotations.name}")

# # List all waypoint snapshots
waypoint_snapshots = {}
for waypoint in waypoints:
    if len(waypoint.snapshot_id) == 0:
        continue
    snapshot_path = os.path.join(waypoint_snapshot_dir, waypoint.snapshot_id)
    with open(snapshot_path, 'rb') as snapshot_file:
        waypoint_snapshot = map_pb2.WaypointSnapshot()
        waypoint_snapshot.ParseFromString(snapshot_file.read())
        print(f"Waypoint Snapshot ID: {waypoint_snapshot.id}")
        # print(json_format.MessageToJson(waypoint_snapshot))
        waypoint_snapshots[waypoint_snapshot.id] = waypoint_snapshot
print(f"Total waypoint snapshots: {len(waypoint_snapshots)}")


def load_waypoint_snapshot(snapshot_id, snapshot_dir):
    snapshot_path = os.path.join(snapshot_dir, snapshot_id)
    snapshot = map_pb2.WaypointSnapshot()
    with open(snapshot_path, 'rb') as snapshot_file:
        snapshot.ParseFromString(snapshot_file.read())
    return snapshot

# Example: Load a waypoint snapshot
waypoint_snapshot_dir = "/Users/shuk/Desktop/spot/practical-seminar-mobile-robotics/spot-sdk/python/examples/graph_nav_command_line/downloaded_graph/waypoint_snapshots"
snapshot_id = "snapshot_alary-larva-a5i98pib8ZXVWGWD8inpnA=="
waypoint_snapshot = load_waypoint_snapshot(snapshot_id, waypoint_snapshot_dir)

if waypoint_snapshot.image_sources:
    for image_source in waypoint_snapshot.image_sources:
        print(f"Found image source: {image_source.name}")
else:
    print("No images were recorded in this snapshot.")