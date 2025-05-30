import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import camera_calibration_show_extrinsics as show
from PIL import Image

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# (8,6) is for the given testing images.
# If you use the another data (e.g. pictures you take by your smartphone),
# you need to set the corresponding numbers.
corner_x = 7
corner_y = 7
objp = np.zeros((corner_x * corner_y, 3), np.float32)
objp[:, :2] = np.mgrid[0:corner_x, 0:corner_y].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane.

# Make a list of calibration images
# images = glob.glob("data/*.jpg")  # use Teacher provided data
images = glob.glob("my_data/*.jpg")     # use our smartphone data

# Step through the list and search for chessboard corners
print("Start finding chessboard corners...")
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray)

    # Find the chessboard corners
    print("find the chessboard corners of", fname)
    ret, corners = cv2.findChessboardCorners(gray, (corner_x, corner_y), None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (corner_x, corner_y), corners, ret)
        plt.imshow(img)

#######################################################################################################
#                                Homework 1 Camera Calibration                                        #
#               You need to implement camera calibration(02-camera p.76-80) here.                     #
#   DO NOT use the function directly, you need to write your own calibration function from scratch.   #
#                                          H I N T                                                    #
#                        1.Use the points in each images to find Hi                                   #
#                        2.Use Hi to find out the intrinsic matrix K                                  #
#                        3.Find out the extrensics matrix of each images.                             #
#######################################################################################################
print("Camera calibration...")
img_size = img[0].shape
# You need to comment these functions and write your calibration function from scratch.
# Notice that rvecs is rotation vector, not the rotation matrix, and tvecs is translation vector.
# In practice, you'll derive extrinsics matrixes directly. The shape must be [pts_num,3,4], and use them to plot.
# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
#     objpoints, imgpoints, img_size, None, None
# )
# Vr = np.array(rvecs)
# Tr = np.array(tvecs)
# extrinsics = np.concatenate((Vr, Tr), axis=1).reshape(-1, 6)
# """
# Write your code here
""" 從頭計算了相機的內參數和外參數，取代了使用 OpenCV 函數直接求解 """
# 1. Use the points in each images to find Hi
H_list = []
for i in range(len(objpoints)):
    obj = objpoints[i]
    img = imgpoints[i].reshape(-1, 2)
    H = []
    for obj, img in zip(obj, img):
        X, Y, _ = obj
        u, v = img
        H.append([X, Y, 1, 0, 0, 0, -u * X, -u * Y, -u])
        H.append([0, 0, 0, X, Y, 1, -v * X, -v * Y, -v])
    H = np.array(H)
    _, _, Vh = np.linalg.svd(H)
    Hi = Vh.T[:, -1].reshape(3, 3)
    Hi = Hi / Hi[2, 2]
    H_list.append(Hi)

# 2. Use Hi to find out the intrinsic matrix A
V = []
for H in H_list:
    h1, h2, h3 = np.insert(H.T, 0, 0, axis=1)
    V.append(
        [
            h1[1] * h2[1],  # h11 * h21
            h1[1] * h2[2] + h1[2] * h2[1],  # h11 * h22 + h12 * h21
            h1[2] * h2[2],  # h12 * h22
            h1[3] * h2[1] + h1[1] * h2[3],  # h13 * h21 + h11 * h23
            h1[3] * h2[2] + h1[2] * h2[3],  # h13 * h22 + h12 * h23
            h1[3] * h2[3],  # h13 * h23
        ]
    )
    V.append(
        [
            h1[1] * h1[1] - h2[1] * h2[1],  # h11 * h11 - h21 * h21
            (h1[1] * h1[2] + h1[2] * h1[1])
            - (
                h2[1] * h2[2] + h2[2] * h2[1]
            ),  # (h11 * h12 + h12 * h11) - (h21 * h22 + h22 * h21)
            h1[2] * h1[2] - h2[2] * h2[2],  # h12 * h12 - h22 * h22
            (h1[3] * h1[1] + h1[1] * h1[3])
            - (
                h2[3] * h2[1] + h2[1] * h2[3]
            ),  # (h13 * h11 + h11 * h13) - (h21 * h23 + h23 * h21)
            (h1[3] * h1[2] + h1[2] * h1[3])
            - (
                h2[3] * h2[2] + h2[2] * h2[3]
            ),  # (h13 * h12 + h12 * h13) - (h22 * h23 + h23 * h22)
            h1[3] * h1[3] - h2[3] * h2[3],  # h13 * h13 - h23 * h23
        ]
    )
V = np.array(V)
_, _, Vt = np.linalg.svd(V)
b = Vt.T[:, -1]
B = np.array(
    [
        [b[0], b[1], b[3]],
        [b[1], b[2], b[4]],
        [b[3], b[4], b[5]],
    ]
)
eigvalsB = np.linalg.eigvals(B)
if not np.all(eigvalsB > 0):
    B = -B
G = np.linalg.cholesky(B)
k_inv = G.T
K = np.linalg.inv(k_inv)  # intrinsic matrix

# 3. Find out the extrinsics matrix of each images.
import scipy

rvecs = []
tvecs = []
for H in H_list:
    h1, h2, h3 = H.T
    s = 1 / np.linalg.norm(np.dot(k_inv, h1))
    r1 = s * np.dot(k_inv, h1)
    r2 = s * np.dot(k_inv, h2)
    r3 = np.cross(r1, r2)
    r = np.array([r1, r2, r3]).T

    tvec = s * np.dot(k_inv, h3)
    rvec = scipy.spatial.transform.Rotation.from_matrix(r).as_rotvec()

    # rvec, _ = cv2.Rodrigues(r)
    # rvec = rvec.squeeze()

    rvecs.append(rvec)
    tvecs.append(tvec)


Vr = np.array(rvecs)
Tr = np.array(tvecs)

mtx = K
extrinsics = np.concatenate((Vr, Tr), axis=1).reshape(-1, 6)

# """
# show the camera extrinsics
print("Show the camera extrinsics")
# plot setting
# You can modify it for better visualization
fig = plt.figure(figsize=(10, 10))
# ax = fig.gca(projection="3d")
ax = fig.add_subplot(projection="3d")
# camera setting
camera_matrix = mtx
cam_width = 0.064 / 0.1
cam_height = 0.032 / 0.1
scale_focal = 1600
# chess board setting
board_width = 8
board_height = 6
square_size = 1
# display
# True -> fix board, moving cameras
# False -> fix camera, moving boards
min_values, max_values = show.draw_camera_boards(
    ax,
    camera_matrix,
    cam_width,
    cam_height,
    scale_focal,
    extrinsics,
    board_width,
    board_height,
    square_size,
    True,
)

X_min = min_values[0]
X_max = max_values[0]
Y_min = min_values[1]
Y_max = max_values[1]
Z_min = min_values[2]
Z_max = max_values[2]
max_range = np.array([X_max - X_min, Y_max - Y_min, Z_max - Z_min]).max() / 2.0

mid_x = (X_max + X_min) * 0.5
mid_y = (Y_max + Y_min) * 0.5
mid_z = (Z_max + Z_min) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, 0)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.set_xlabel("x")
ax.set_ylabel("z")
ax.set_zlabel("-y")
ax.set_title("Extrinsic Parameters Visualization")
plt.show()

# animation for rotating plot
"""
for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(.001)
"""
