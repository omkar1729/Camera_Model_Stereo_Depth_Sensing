import cv2
import numpy as np
from math import cos, sin, radians

objp = np.zeros((6 * 9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
print(objp[:, 1])


d3 = []
ip_left = []
ip_right = []
d3_points = []
for i in range(6):
    for j in range(9):
        d3_points.append([float(j), float(i), 2.0])

d3_points = np.asarray(d3_points, np.float32)

object_points = []
image_points_left = []
image_points_right = []
shape = ()

intrinsic_matrix_left = np.loadtxt('../../parameters/left_camera_intrinsics/intrinsic_l.csv', delimiter=',')
intrinsic_matrix_right = np.loadtxt('../../parameters/right_camera_intrinsics/intrinsic_r.csv', delimiter=',')
distortion_left = np.loadtxt('../../parameters/left_camera_intrinsics/distortion_l.csv', delimiter=',')
distortion_right = np.loadtxt('../../parameters/right_camera_intrinsics/distortion_r.csv', delimiter=',')
# distortion = distortion.reshape((480,640,3))
# print(distortion_left)
for i in range(1):
    left_img = cv2.imread('../../images/task_2/left_' + str(i) + '.png')
    h, w = left_img.shape[:-1]
    shape = (w, h)
    print(shape)
    right_img = cv2.imread('../../images/task_2/right_' + str(i) + '.png')
    # print(left_img.shape[:-1])
    ret_l, corner_left = cv2.findChessboardCorners(left_img, (9, 6))
    ret_r, corner_right = cv2.findChessboardCorners(right_img, (9, 6))

    result_left = cv2.drawChessboardCorners(left_img, (9, 6), corner_left, ret_l)
    result_right = cv2.drawChessboardCorners(right_img, (9, 6), corner_right, ret_r)

    image_points_left.append(corner_left)
    image_points_right.append(corner_right)
    object_points.append(objp)

    cv2.imshow('img_left', result_left)
    cv2.imshow('img_right', result_right)
    cv2.waitKey(1000)

cv2.destroyAllWindows()
#
# print(corner_left.shape)
# print(np.shape(image_points_left))
# print(np.shape(image_points_right))
# print(np.shape(object_points))
# print(np.shape(distortion))
# print(np.shape(intrinsic_matrix_left))

stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
# print(distortion)
retval, matrix1, dist1, matrix2, dist2, R, T, F, E = cv2.stereoCalibrate(object_points,
                                                                         image_points_left, image_points_right,
                                                                         intrinsic_matrix_left, distortion_left,
                                                                         intrinsic_matrix_right, distortion_right,
                                                                         shape,
                                                                         R=None, T=None, E=None, F=None,
                                                                         criteria=stereocalib_criteria,
                                                                         flags=cv2.CALIB_FIX_INTRINSIC)
# print(matrix1,matrix2)
translate = np.zeros([3, 1])
rotate = np.identity(3)
proj1 = np.dot(intrinsic_matrix_left, np.concatenate((rotate, translate), axis=1))
R = np.asarray(R)
T = np.asarray(T)
origin2 = np.dot(R, T)
norm2 = np.dot(R, np.array([0, 0, 1]).reshape(3, 1))
#norm2 = np.dot(T,np.transpose(norm2))
print(R.shape,T.shape)
print(norm2)
proj2 = np.dot(intrinsic_matrix_right, np.concatenate((R, T), axis=1))

# print(image_points_left)

undistorted_left = cv2.undistortPoints(np.reshape(image_points_left, (54, 1, 2)), intrinsic_matrix_left,
                                       distortion_left)
undistorted_right = cv2.undistortPoints(np.reshape(image_points_right, (54, 1, 2)), intrinsic_matrix_right,
                                        distortion_right)
# print(np.shape(undistorted_left))

triangulate = cv2.triangulatePoints(proj1, proj2, undistorted_left, undistorted_right)
print(undistorted_left.shape)
# print(triangulate)
# tasks 5-7

rotation1, rotation2, projection1, projection2, Q, roi1, roi2 = cv2.stereoRectify(cameraMatrix1=intrinsic_matrix_left,
                                                                                  distCoeffs1=distortion_left,
                                                                                  cameraMatrix2=intrinsic_matrix_right,
                                                                                  distCoeffs2=distortion_right,
                                                                                  imageSize=shape,
                                                                                  R=R,
                                                                                  T=T,
                                                                                  flags=cv2.CALIB_ZERO_DISPARITY

                                                                                  )

print(projection1)
print(projection2)
print(rotation1)
# print(triangulate)

undistort_map_left_x, undistort_map_left_y = cv2.initUndistortRectifyMap(intrinsic_matrix_left, distortion_left,
                                                                         rotation1, projection1, shape, cv2.CV_32FC1)
undistort_map_right_x, undistort_map_right_y = cv2.initUndistortRectifyMap(intrinsic_matrix_right, distortion_right,
                                                                           rotation2, projection2, shape, cv2.CV_32FC1)

for i in range(1):
    left_img = cv2.imread('../../images/task_2/left_' + str(i) + '.png')
    right_img = cv2.imread('../../images/task_2/right_' + str(i) + '.png')

    remap_left_img = cv2.remap(left_img, undistort_map_left_x, undistort_map_left_y, cv2.INTER_LINEAR)
    remap_right_img = cv2.remap(right_img, undistort_map_right_x, undistort_map_right_y, cv2.INTER_LINEAR)
    cv2.imshow('left_remap' + str(i), remap_left_img)
    cv2.imshow('right_remap' + str(i), remap_right_img)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

np.savetxt('../../parameters/stereo_calibration/R.csv', R, delimiter=',')
np.savetxt('../../parameters/stereo_calibration/T.csv', T, delimiter=',')
np.savetxt('../../parameters/stereo_calibration/F.csv', F, delimiter=',')
np.savetxt('../../parameters/stereo_calibration/E.csv', E, delimiter=',')
np.savetxt('../../parameters/stereo_rectification/P1.csv', projection1, delimiter=',')
np.savetxt('../../parameters/stereo_rectification/P2.csv', projection2, delimiter=',')
np.savetxt('../../parameters/stereo_rectification/R1.csv', rotation1, delimiter=',')
np.savetxt('../../parameters/stereo_rectification/R2.csv', rotation2, delimiter=',')
np.savetxt('../../parameters/stereo_rectification/Q.csv', Q, delimiter=',')

###############################################################################################################################3


# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# #ax.scatter3D(objp[:,0], objp[:,1],objp[:,2])
# ax.scatter3D(triangulate[0]/triangulate[3],triangulate[1]/triangulate[3],triangulate[2]/triangulate[3])
# # ax.scatter3D(origin2[0],origin2[1],origin2[2],c='green')
# # ax.scatter3D(0.0,0.0,0.0,c='red')
#
# plt.show()
print(image_points_left)
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

p0 = [0.799319, -3.477045e-01, 0.490093]
p1 = [0.852512, 9.113778e-16, -0.522708]
p2 = [0.296422, 9.376042e-01, 0.181748]

origin = [0, 0, 0]
z_axis = [0,0,1]
normz = np.dot(R, np.array([0, 0, 1]).reshape(3, 1))
y_axis = [0,1,0]
normy = np.dot(R, np.array([0, 1, 0]).reshape(3, 1))

x_axis = [1,0,0]
normx = np.dot(R, np.array([1, 0, 0]).reshape(3, 1))
X, Y, Z = zip(origin)
U, V, W = zip(origin2)
x,y,z = zip(normz)
x1,y1,z1 = zip(z_axis)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.quiver(X, Y, Z, x1, y1, z1,arrow_length_ratio=0.01)
ax.quiver(U, V, W, x, y, z, arrow_length_ratio=0.01)

x,y,z = zip(normx)
x1,y1,z1 = zip(x_axis)
ax.quiver(X, Y, Z, x1, y1, z1,arrow_length_ratio=0.01)
ax.quiver(U, V, W, x, y, z, arrow_length_ratio=0.01)

x,y,z = zip(normy)
x1,y1,z1 = zip(y_axis)
ax.quiver(X, Y, Z, x1, y1, z1,arrow_length_ratio=0.01)
ax.quiver(U, V, W, x, y, z, arrow_length_ratio=0.01)


ax.scatter3D(triangulate[0]/triangulate[3],triangulate[1]/triangulate[3],triangulate[2]/triangulate[3])
#ax.scatter3D(d3_points[:,0], d3_points[:,1],d3_points[:,2])
ax.s
plt.show()
