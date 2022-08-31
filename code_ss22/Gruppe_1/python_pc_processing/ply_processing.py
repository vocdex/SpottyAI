'''
Author: Philipp Santer
'''

import numpy as np
from plyfile import PlyData, PlyElement
from pdf2image import convert_from_path
import matplotlib.pyplot as plt
import cv2 as cv
import statistics
from scipy import ndimage
import matlab.engine
import open3d as o3d


#These values for ppm and dpi ensure that the pointcloud and the pdf roughly have the same dimensions per pixel when converted to an image
#To be more generalizing this value should be automatically calculated through setting Pixels per meter to a preset value and adjusting
#DPI according to dimensions of the PDF plan, given by an operator
PIXELS_PER_METER = 48.1 #35.25 #47  # 250/25.2 #1982 / 25.2
DPI = 60 #45
PLOT = False
#PLOT = True
PRINT = False
#PRINT = True


def localize(pdf_path='OG3.pdf', ply_path='preprocessed.ply'):

    #TODO: tune hough line parameters depending on the image size
    #       -> solve corner cases e.g. when no lines are found when the image is too small
    #       -> don't rely on try except in global_localization
    #TODO: consider using a prebuilt matching algorithm which does not rely on brute force
    #TODO: solve the case when the matching_img is larger than the pdf_img
    #TODO: make it that PIXELS_PER_METER depends on input pdf/can be given as an input parameter

    # load floor map
    path = pdf_path #'OG3.pdf'#
    floor_img, floor_lines = get_floor(path, DPI, verbose=False)
    floor_lines = rgb2gray(floor_lines).astype(np.uint16)
    floor_lines[floor_lines > 0] = 255
    if PLOT:
        plt.imshow(floor_img, cmap='gray')
        plt.show()
        plt.imshow(floor_lines, cmap='gray')
        plt.show()


    # load pointcloud data  #5-16 klappen mehr oder weniger
    plydata = PlyData.read(ply_path)  # '2_LRT_9_ground.ply'
    data_arr = plydata.elements[0].data
    data_arr = [list(item) for item in data_arr]
    data_arr = np.array(data_arr, dtype='float')

    if PLOT:
        #PC Visualization
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data_arr)
        #o3d.io.write_point_cloud("./data.ply", pcd)
        #You can also visualize the point cloud using Open3D.

        o3d.visualization.draw_geometries([pcd])



    # covert point cloud to image
    data_arr = data_arr[:,:2]
    if PRINT:
        print(plydata)
        print()
        print(data_arr)

    x_off = np.min(data_arr[:, 0])
    y_off = np.min(data_arr[:, 1])
    data_arr -= [x_off, y_off]
    x_max = np.max(data_arr[:, 0])
    y_max = np.max(data_arr[:, 1])

    pc_img = np.zeros((int(x_max * PIXELS_PER_METER), int(y_max * PIXELS_PER_METER)))
    for el in data_arr:
        pc_img[int(el[0] * PIXELS_PER_METER)-1, int(el[1] * PIXELS_PER_METER)-1] = 255
    pc_img = pc_img.astype(np.uint8)

    # Note: from here on out we are dealing with images -> [y,x] compared to before (ply format) [x,y,z]
    # e.g. origin_pixel is [y,x]
    origin_pixel = (int(-x_off * PIXELS_PER_METER), int(-y_off * PIXELS_PER_METER))
    if PRINT:
        print(f"origin_pixels: {origin_pixel}")
        print()
    if PLOT:
        plt.imshow(pc_img, cmap='gray')
        plt.show()


    # Use HoughTransform to find lines in pc_img
    line_img1, rot_angle, _ = hough_line(pc_img, rho=1, theta=np.pi/180, thres=300, min_len=3.7, max_gap=0.5)
    if PRINT:
        print()
        print(line_img1.shape)
    if PLOT:
        plt.imshow(line_img1)
        plt.show()

    #Use a black image with a mark to track the origin
    marked_image = get_marked_image(pc_img.shape, origin_pixel, 2)


    # line_img_rotated = ndimage.rotate(line_img1, -90 - rot_angle)
    # if PRINT:
    #     print(line_img_rotated.shape)
    # if PLOT:
    #     plt.imshow(line_img_rotated)
    #     plt.show()


    # rotate pc_img to get it in the same orientation as the floor map
    img_rotated = ndimage.rotate(pc_img, -90 - rot_angle)
    marked_rotated = ndimage.rotate(marked_image, -90 - rot_angle)

    origin_pixels = np.where(marked_rotated == 255)
    origin_pixels = (int(np.median(origin_pixels[0])), int(np.median(origin_pixels[1])))

    if PRINT:
        print(f"origin_pixels: {origin_pixels}")
        print()
    if PLOT:
        plt.imshow(img_rotated, cmap='gray')
        plt.show()


    # Apply the HoughTransform a second time
    # some parameter tuning fixed it -> (trying HoughLines again on rotated image does not yield good results)
    line_img2, rot_angle, lines = hough_line(img_rotated, rho=0.1, theta=np.pi/180, thres=180, min_len=3.7, max_gap=0.2)
    if PLOT:
        plt.imshow(line_img2)
        plt.show()

    line_crop = np.zeros(line_img2.shape).astype(np.uint8)
    for i in range(0, len(lines)):
        l = lines[i][0]
        cv.line(line_crop, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)


    # build match_img
    marked_image = get_marked_image(line_crop.shape, origin_pixels, 20, mode='rgb')
    if PLOT:
        plt.imshow(marked_image)
        plt.show()
    match_img, marked_image = build_match_img(line_crop, lines, marked_image)



    # match the output of the second hough transform with the hough transform of the floor map
    res_img = 0
    min_coords = (0,0)
    min_sum = np.infty
    min_rotation = 0
    for rotations in range(4):
        img_, coords_, sum_ = match_imgs(floor_lines, np.rot90(match_img,k=rotations) )
        if sum_ < min_sum:
            res_img = img_
            min_coords = coords_
            min_sum = sum_
            min_rotation = rotations

    relative_min_pixel_count = min_sum / cv.countNonZero(match_img)
    print(f"relative_min_pixel_count: {relative_min_pixel_count}")

    if PLOT:
        plt.imshow(res_img, cmap='gray')
        plt.show()

        # plt.imshow(marked_image)
        # plt.show()

    #Mark the origin on the floor map
    marked_out = np.rot90(marked_image, k=min_rotation)

    origin_pixels = np.where(marked_out == [255, -255, -255])
    origin_pixels = (int(np.median(origin_pixels[0])), int(np.median(origin_pixels[1])))
    origin_pixels_in_floor_map = tuple(map(sum, zip(origin_pixels, (min_coords[1], min_coords[0]))))

    if PRINT:
        print(f"origin_pixels_in_floor_map: {origin_pixels_in_floor_map}")
        print()

    if PLOT:
        plt.imshow(marked_out)
        plt.show()



    Output = np.copy(get_floor_rgb(path, DPI, verbose=False))
    Output[min_coords[1]:min_coords[1]+marked_out.shape[0], min_coords[0]:min_coords[0]+marked_out.shape[1]] += marked_out
    if PLOT:
        plt.imshow(Output)
        plt.show()

    return origin_pixels_in_floor_map, relative_min_pixel_count




def get_marked_image(shape, origin_coords, width, mode='gray'):
    '''

    Args:
        shape: tupel
        origin_coords: tupel

    Returns: a balck image with the origin as a marked spot

    '''
    off = width // 2
    img_out = np.zeros(shape).astype(np.uint8)

    if mode is 'gray':
        img_out[origin_coords[0] - off:origin_coords[0]+off+1, origin_coords[1] - off:origin_coords[1]+off+1] = 255
    elif mode is 'rgb':
        img_out[origin_coords[0] - off:origin_coords[0]+off+1, origin_coords[1] - off:origin_coords[1]+off+1] = [255, -255, -255]

    return img_out




def build_match_img(in_img, lines, marked_img, MARGIN=0):
    '''

    Args:
        in_img: HoughLines img -- np.array()
        lines: lines array from hough Lines
        floor_line_img: np.array()
        MARGIN = 0  # 5Pixel margin added

    Returns: prepared image for matching -- np.array().astype(np.uint16)

    '''

    match_img = rgb2gray(in_img)
    match_img[match_img > 0] = 255
    match_img = match_img.astype(np.uint16)
    if PLOT:
        plt.imshow(match_img, cmap='gray')
        plt.show()

    line_coords = np.squeeze(lines)
    # to avoid cutting of the origin and to speed up the matching algorithm
    # add to coordinates of the origin to line_coords
    ORIGIN_MARGIN = 10

    origin_pixels = np.where(marked_img == 255)
    y_o,x_o = (int(np.median(origin_pixels[0])), int(np.median(origin_pixels[1])))

    line_coords = np.append(line_coords,[[max(0, x_o+ORIGIN_MARGIN), max(0, y_o+ORIGIN_MARGIN),
                                          max(0, x_o-ORIGIN_MARGIN), max(0, y_o-ORIGIN_MARGIN)]], axis=0)

    x_s = np.append(np.array(line_coords[:, 0]), np.array(line_coords[:, 2]))
    y_s = np.append(np.array(line_coords[:, 1]), np.array(line_coords[:, 3]))
    x_range = (np.min(x_s), np.max(x_s))
    y_range = (np.min(y_s), np.max(y_s))

    x_line_max = x_range[1]
    y_line_max = y_range[1]
    x_line_min = x_range[0]
    y_line_min = y_range[0]


    if not ((x_line_min >= MARGIN) and (y_line_min >= MARGIN)
                        and (match_img.shape[1] >= x_line_max + MARGIN)
                                and (match_img.shape[0] >= y_line_max + MARGIN)):
        MARGIN = 0
    match_img = match_img[y_line_min - MARGIN:y_line_max + MARGIN,
                          x_line_min - MARGIN:x_line_max + MARGIN]

    marked_img = marked_img[y_line_min - MARGIN:y_line_max + MARGIN,
                            x_line_min - MARGIN:x_line_max + MARGIN]

    if PLOT:
        plt.imshow(match_img, cmap='gray')
        plt.show()
        plt.imshow(marked_img, cmap='gray')
        plt.show()

    return match_img, marked_img




def match_imgs(floor_line_img, match_img):
    '''

    Args:
        floor_line_img: np.array().astype(np.uint16)
        match_img: np.array().astype(np.uint16)
        x_range: (x_min, x_max)
        y_range: (y_min, y_max)

    Returns: res_img, (x_off, y_off), min_sum

    '''
    floor_img = floor_line_img

    #Check here if dimensions of match and floor are similar if not punish this
    y_diff = floor_line_img.shape[0] - match_img.shape[0]
    x_diff = floor_line_img.shape[1] - match_img.shape[1]
    if y_diff < 0 or x_diff < 0:
        y_diff = abs(y_diff) if y_diff < 0 else 0
        x_diff = abs(x_diff) if x_diff < 0 else 0
        diff = y_diff + x_diff

        #Check if rotating the image would decrese the missmatch regarding the dimensions
        y_diff_rot = floor_line_img.shape[0] - match_img.shape[1]
        x_diff_rot = floor_line_img.shape[1] - match_img.shape[0]
        y_diff_rot = abs(y_diff_rot) if y_diff_rot < 0 else 0
        x_diff_rot = abs(x_diff_rot) if x_diff_rot < 0 else 0
        diff_rot = y_diff_rot + x_diff_rot

        if diff > diff_rot * 1.1:
            return floor_line_img, (), np.infty

    #TODO:solve if input pc is larger than the floor plan:
    # If this is solved, the match_img does not have to be cut that aggresively as it is now
    # -> would lead to fewer iterations for the matching algorithm

    # if (floor_line_img.shape[1] > (x_line_max - x_line_min) + MARGIN) and (
    #         floor_line_img.shape[0] > (y_line_max - y_line_min) + MARGIN):
    # else:
    #     MARGIN = 0
    #     match_img = match_img[y_line_min - MARGIN:y_line_max + MARGIN, x_line_min - MARGIN:x_line_max + MARGIN]
    #     x_diff = floor_line_img.shape[1] - (x_line_max - x_line_min)
    #     y_diff = floor_line_img.shape[0] - (y_line_max - y_line_min)
    #     match_img = match_img[y_diff // 2 + 1:y_diff // 2 - 1, x_diff // 2 + 1:x_diff // 2 - 1]


    x_comp_range = floor_img.shape[1] - match_img.shape[1]
    y_comp_range = floor_img.shape[0] - match_img.shape[0]
    #Comparing floor_img and pc_img

    y_min = 0
    x_min = 0
    min_sum = np.infty
    #TODO: consider a variable step size: first run algorithm with stepsize 20: around the minimum run again with 1-5

    STEPSIZE = 10
    EXECUTION_TIME_LIMITER = 3000 + 1000
    i = 0
    print("Start Matching ...")
    for y_off in range(0, y_comp_range, STEPSIZE):
        for x_off in range(0, x_comp_range, STEPSIZE):
            i += 1
            if(i == EXECUTION_TIME_LIMITER): i = 1/0

            #print(f"off_coords:{y_off},{x_off}")
            floor_cpy = np.copy(floor_img)
            floor_cpy[y_off:y_off+match_img.shape[0], x_off:x_off+match_img.shape[1]] += match_img
            floor_cpy[floor_cpy >= 255] = 255

            curr_sum = np.sum(floor_cpy)
            if(curr_sum<min_sum):
                y_min = y_off
                x_min = x_off
                min_sum = curr_sum

    print(f"Stopped Matching. # of iterations: {i}.")
    floor_sum = np.sum(floor_img)
    match_img_sum = np.sum(match_img)
    diff_sum = min_sum - floor_sum
    normalized = diff_sum/floor_sum * 100
    could_be_matched = match_img_sum - diff_sum
    could_be_matched_percent = could_be_matched / match_img_sum * 100

    print(f"    Pixel sums of...  floor_img: {floor_sum}, matching_img: {match_img_sum}, combined: {min_sum}")
    print(f"    Difference floor->combined: +{diff_sum} / +{normalized:.2f}%")
    print(f"    For matching_img {could_be_matched} of {match_img_sum} pixels OR {could_be_matched_percent:.2f}% could be matched.")

    res_img = np.copy(floor_img)
    res_img[y_min:y_min+match_img.shape[0], x_min:x_min+match_img.shape[1]] += match_img*2
    if PLOT:
        plt.imshow(res_img, cmap='gray')
        plt.show()

    return res_img, (x_min, y_min), min_sum




def get_floor(path, dpi, verbose=False):
    # get pdf & convert to np.array
    pages = convert_from_path(path, dpi, single_file=True)
    floor_img = np.array(pages[0].convert('L'))
    if verbose:
        print("----GET-FLOOR----")
        print(floor_img)
        plt.imshow(floor_img, cmap='gray')
        plt.show()

    floor_img = 255 - floor_img
    floor_img[floor_img > 0] = 255
    if verbose:
        plt.imshow(floor_img, cmap='gray')
        plt.show()

    if dpi>=55:
        # with 60 DPI
        line_img, rot_angle, lines = hough_line(floor_img, rho=0.5, theta=np.pi / 180, thres=100, min_len=2.5,
                                                max_gap=0.2)
    else:
        # with40 or 45DPI
        line_img, rot_angle, lines = hough_line(floor_img, rho=0.5, theta=np.pi / 180, thres=50, min_len=2.5,
                                                max_gap=0.2)

    if verbose:
        plt.imshow(line_img)
        plt.show()

    line_crop_floor = np.zeros(line_img.shape).astype(np.uint8)
    for i in range(0, len(lines)):
        l = lines[i][0]
        cv.line(line_crop_floor, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)

    if verbose:
        plt.imshow(line_crop_floor)
        plt.show()

    return floor_img, line_crop_floor




def get_floor_rgb(path, dpi, verbose=False):
    # get pdf & convert to np.array
    pages = convert_from_path(path, dpi, single_file=True)
    floor_img = np.array(pages[0])
    if verbose:
        print("----GET-FLOOR----")
        print(floor_img)
        plt.imshow(floor_img, cmap='gray')
        plt.show()

    return floor_img




def hough_line(img_gray, rho=1, theta=np.pi/180, thres=120, min_len=3.7, max_gap=0.5):
    '''

    Args:
        img_gray: np.array(dtype=np.uint8)

    Returns: two line images: line_img, line_imgP

    '''

    # Copy edges to the images that will display the results in BGR
    cdstP = cv.cvtColor(img_gray, cv.COLOR_GRAY2BGR)
    min_len_pix = int(min_len * PIXELS_PER_METER) #2.5 -3.0
    max_gap_pix = int(max_gap * PIXELS_PER_METER) #0.5
    # lines is a redundant parameter since linesP would have the same content
    linesP = cv.HoughLinesP(img_gray, rho, theta, threshold=thres, lines=None, minLineLength=min_len_pix, maxLineGap=max_gap_pix)

    rotation_angle_deg = 0.0
    if linesP is not None:
        angles = list()
        for line in linesP:
            p1 = np.array(line[0, 0:2])
            p2 = np.array(line[0, 2:4])

            angles.append(np.arctan2(p1[0] - p2[0], p1[1] - p2[1]))

        rotation_angle = statistics.median(angles)
        rotation_angle_deg = rotation_angle * 180 / np.pi
        if PRINT:
            print(angles)
            print(f"angle in rad: {rotation_angle}")
            print(f"angle in deg: {rotation_angle_deg}")


        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)

    # cv.imshow("Source", img_gray)
    # cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
    # cv.waitKey()

    return cdstP, rotation_angle_deg, linesP




def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def run_MATLAB_preprocess():
    #extract_ground_from_pc.extract_ground()
    # oc = Oct2Py()
    # oc.run('extract_ground_from_pc.m')
    eng = matlab.engine.start_matlab()
    eng.extract_ground_from_pc(nargout=0)
    return

def draw_final_map(pos, path='OG3.pdf'):
    floor_img = get_floor_rgb(path, DPI, verbose=False)
    marked_img = get_marked_image(floor_img.shape, pos, 20, mode='rgb')

    return floor_img + marked_img


if __name__ == "__main__":
    localize()
    #run_MATLAB_preprocess()













#########################################################
#########################END#############################
##########################################################
#%matplotlib inline
# sns.set_context('poster')
# sns.set_style('white')
# sns.set_color_codes()
# plot_kwds = {'alpha' : 0.5, 's' : 80, 'linewidths':0}
#
#
#
# clusterer = hdbscan.HDBSCAN(min_cluster_size=1000, metric='manhattan')
#clusterer.fit(data_arr)
# HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
#     gen_min_span_tree=True, leaf_size=40, memory=Memory(cachedir=None),
#     metric='euclidean', min_cluster_size=5, min_samples=None, p=None)


#print(clusterer.labels_.max())
# print(data_arr)
# print()
# print(np.append(np.array(2.0 * (np.random.rand((2))-0.5)), -2.0))
# print()


# for i in range(len(data_arr)):
#     data_arr[i] = np.append(np.array(1.0 * (np.random.rand((2))-0.5)), 0.0)
# data_arr[:] =


#data_arr[-170000:] = [0.0, 0.0, 0.0]#np.array(1.0 * (np.random.rand((3))-0.5))

#print(np.logical_not(np.array(clusterer.labels_)==0))

# cluster_list = []
# for i in range(clusterer.labels_.max()):
#     cluster_list.append(copy.deepcopy(data_arr))
#     cluster_list[i][np.logical_not(np.array(clusterer.labels_)==i)] = 0.0
#
#
# # Pass numpy array to Open3D.o3d.geometry.PointCloud and visualize
# # xyz = np.random.rand(100, 3)
# for i in range(clusterer.labels_.max()):
#     cluster_list.append(copy.deepcopy(data_arr))
#     cluster_list[i][np.logical_not(np.array(clusterer.labels_)==i)] = 0.0
#
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(cluster_list[i])
#     # o3d.io.write_point_cloud("./data.ply", pcd)
#     # You can also visualize the point cloud using Open3D.
#
#     o3d.visualization.draw_geometries([pcd])





