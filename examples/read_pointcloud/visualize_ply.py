import open3d as o3d

def main():
    cloud_1 = o3d.io.read_point_cloud("cloud.ply") # Read the point cloud
    o3d.visualization.draw_geometries([cloud_1]) # Visualize the point cloud

if __name__ == "__main__":
    main()