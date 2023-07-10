# after Tutorial for camera calibration
# https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
# Create images with Spot and checkerboard and then use this script to get camera parameters
# these can be copied from the print output
# calibration images should be in subfolder '/calibration/'

import numpy as np
import cv2 as cv
import glob

# checkerboard size (amount of corners in both directions)
# always one less than black and white checks in the direction
x_size = 6
y_size = 4

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((x_size*y_size, 3), np.float32)
objp[:,:2] = np.mgrid[0:x_size, 0:y_size].T.reshape(-1, 2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('calibration/*.jpg')
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (x_size, y_size), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (x_size, y_size), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(100)
cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# print(f"ret:\n{ret}\n\nmtx:\n{mtx}\n\ndist:\n{dist}\n\nrvecs:\n{rvecs}\n\ntvecs:\n{tvecs}")

# new camera matrix
img = cv.imread('calibration/img0.jpg')
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

print(f"mtx:\n{mtx}\n\ndist:\n{dist}\n\nnewcameramtx:\n{newcameramtx}\n\nroi:\n{roi}")

# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
# crop the image
x, y, w, h = roi
cv.imwrite('calibration/calibresult.png', dst)
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibration/calibresult_crop.png', dst)

