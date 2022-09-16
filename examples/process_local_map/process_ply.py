from plyfile import PlyData, PlyElement
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

# Global flags
PLOT_3D = False
PLOT_2D = True
PRINT = True

PIXELS_PER_METER = 10


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


def convert_to_img(pcd):
    # covert point cloud to image
    data_arr = np.asarray(pcd.points)[:, :2]

    # get x,y limits to generate image size
    x_off = np.min(data_arr[:, 0])
    y_off = np.min(data_arr[:, 1])
    data_arr -= [x_off, y_off]
    x_max = np.max(data_arr[:, 0])
    y_max = np.max(data_arr[:, 1])

    pc_img = np.ones((int(x_max * PIXELS_PER_METER), int(y_max * PIXELS_PER_METER))) * 255
    for el in data_arr:
        pc_img[int(el[0] * PIXELS_PER_METER) - 1, int(el[1] * PIXELS_PER_METER) - 1] = 0
    pc_img = pc_img.astype(np.uint8)

    if PLOT_2D:
        plt.imshow(pc_img, cmap='gray')
        plt.show()


def get_o3d_FOR(origin=[0, 0, 0], size=0.5):
    """
    Create a FOR that can be added to the open3d point cloud
    """
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=size)
    mesh_frame.translate(origin)
    return mesh_frame


def main():
    ply_path = 'cloud.ply'
    plydata = PlyData.read(ply_path)
    data_arr = plydata.elements[0].data
    data_arr = [list(item) for item in data_arr]
    data_arr = np.array(data_arr, dtype='float')

    # pre-process data
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data_arr)
    # o3d.visualization.draw_geometries([pcd])

    voxel_size = 1/PIXELS_PER_METER

    print("Downsample the point cloud with a voxel of " + str(voxel_size))
    voxel_down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    # o3d.visualization.draw_geometries([voxel_down_pcd])

    print("Statistical outlier removal")
    processed_pcd, idx = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=20,
                                                                   std_ratio=1.0)

    # remove floor
    floor_height = -0.4
    points = processed_pcd.points
    points = np.asarray(points)
    idx_floor = np.where(points[:, 2] > floor_height)[0]

    # display filtering
    # display_inlier_outlier(voxel_down_pcd, idx)
    # display_inlier_outlier(processed_pcd, idx_floor)

    processed_pcd = processed_pcd.select_by_index(idx_floor)

    print('voxelization')
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(processed_pcd,
                                                                voxel_size=voxel_size)

    # Create a coordinate frame in the origin (i.e. where the robot stands)
    FOR = get_o3d_FOR()
    o3d.visualization.draw_geometries([FOR, voxel_grid])

    convert_to_img(processed_pcd)


if __name__ == "__main__":
    main()
