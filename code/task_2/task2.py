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
    cv2.waitKey()

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
print(T)
print(R)
