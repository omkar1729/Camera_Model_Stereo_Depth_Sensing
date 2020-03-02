import cv2
import numpy as np
from math import cos, sin, radians

objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
# print(objp)

object_points = []
image_points_left = []
image_points_right = []
shape = ()

intrinsic_matrix_left = np.loadtxt('/home/sarthake/project_2a/parameters/intrinsic.csv', delimiter=',')
intrinsic_matrix_right = np.loadtxt('/home/sarthake/project_2a/parameters/intrinsic.csv', delimiter=',')
distortion = np.loadtxt('/home/sarthake/project_2a/parameters/distortion.csv',delimiter=',')
# distortion = distortion.reshape((480,640,3))
print(distortion)
for i in range(2):
    left_img = cv2.imread('/home/sarthake/project_2a/images/task_2/left_'+str(i)+'.png')
    shape = left_img.shape[:-1]
    print(shape)
    right_img = cv2.imread('/home/sarthake/project_2a/images/task_2/right_'+str(i)+'.png')
    # print(left_img.shape[:-1])
    ret_l, corner_left = cv2.findChessboardCorners(left_img, (9, 6))
    ret_r, corner_right = cv2.findChessboardCorners(right_img, (9, 6))

    image_points_left.append(corner_left)
    image_points_right.append(corner_right)
    object_points.append(objp)

    result_left = cv2.drawChessboardCorners(left_img, (9, 6), corner_left, ret_l)
    result_right = cv2.drawChessboardCorners(right_img, (9, 6), corner_right, ret_r)

    cv2.imshow('img_left', result_left)
    cv2.imshow('img_right', result_right)
    cv2.waitKey(50)

cv2.destroyAllWindows()
# print(object_points)
# print(image_points_left)

# T = np.zeros((3, 1), dtype=np.float64)
# R = np.eye(3, dtype=np.float64)
print(np.shape(image_points_left))
print(np.shape(image_points_right))
print(np.shape(object_points))
print(np.shape(distortion))
print(np.shape(intrinsic_matrix_left))
# constant_flag = 0
#
# constant_flag |= cv2.CALIB_FIX_INTRINSIC
#
# constant_flag |= cv2.CALIB_USE_INTRINSIC_GUESS
#
# constant_flag |= cv2.CALIB_FIX_FOCAL_LENGTH
#
# constant_flag |= cv2.CALIB_ZERO_TANGENT_DIST

stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
print(distortion)
retval, matrix1, dist1, matrix2, dist2, R, T, F, E = cv2.stereoCalibrate(object_points,
                                                                         image_points_left,image_points_right,
                                                                         intrinsic_matrix_left,distortion,
                                                                         intrinsic_matrix_right,distortion,shape,
                                                                         R=None, T=None, E=None, F=None,
                                                                         criteria=stereocalib_criteria,
                                                                         flags=cv2.CALIB_FIX_INTRINSIC)

# print(R)
# print(T)

# print(np.shape(image_points_left))
undistorted_left = cv2.undistortPoints(np.reshape(image_points_left,(108,1,2)), intrinsic_matrix_left, distortion)
undistorted_right = cv2.undistortPoints(np.reshape(image_points_right,(108,1,2)), intrinsic_matrix_right, distortion)
# print(np.shape(undistorted_left))

rotation1, rotation2, projection1, projection2, Q, roi1, roi2 = cv2.stereoRectify(cameraMatrix1=intrinsic_matrix_left,
                      distCoeffs1=distortion,
                      cameraMatrix2=intrinsic_matrix_right,
                      distCoeffs2=distortion,
                      imageSize=shape,
                      R=R,
                      T=T,
                      flags=cv2.CALIB_ZERO_DISPARITY,
                      )

print(projection1)
print(projection2)
print(rotation1)

undistort_map_left_x,undistort_map_left_y = cv2.initUndistortRectifyMap(intrinsic_matrix_left, distortion, rotation1, projection1, shape, cv2.CV_32FC1)
undistort_map_right_x,undistort_map_right_y = cv2.initUndistortRectifyMap(intrinsic_matrix_right, distortion, rotation2, projection2, shape, cv2.CV_32FC1)


for i in range(2):
    left_img = cv2.imread('/home/sarthake/project_2a/images/task_2/left_'+str(i)+'.png')
    right_img = cv2.imread('/home/sarthake/project_2a/images/task_2/right_'+str(i)+'.png')

    remap_left_img = cv2.remap(left_img, undistort_map_left_x,undistort_map_left_y, cv2.INTER_LINEAR)
    remap_right_img = cv2.remap(right_img, undistort_map_right_x,undistort_map_right_y, cv2.INTER_LINEAR)
    cv2.imshow('left_remap'+str(i),remap_left_img)
    cv2.imshow('right_remap' + str(i), remap_right_img)
    cv2.waitKey()
    cv2.destroyAllWindows()


np.savetxt('/home/sarthake/project_2a/parameters/R.csv', R, delimiter = ',')
np.savetxt('/home/sarthake/project_2a/parameters/T.csv', T, delimiter = ',')
np.savetxt('/home/sarthake/project_2a/parameters/F.csv', F, delimiter = ',')
np.savetxt('/home/sarthake/project_2a/parameters/E.csv', E, delimiter = ',')
np.savetxt('/home/sarthake/project_2a/parameters/Projection1.csv', projection1, delimiter = ',')
np.savetxt('/home/sarthake/project_2a/parameters/Projection2.csv', projection2, delimiter = ',')