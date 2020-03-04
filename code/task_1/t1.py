import cv2
import numpy as np

# # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# objp = np.zeros((6*9,3), np.float32)
# objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
# # print(objp)


d3 = []
ip_left = []
ip_right = []
d3_points = []
for i in range(6):
    for j in range(9):
        d3_points.append([float(j), float(i), 0.0])

d3_points = np.asarray(d3_points, np.float32)

print(d3_points)
str_left = '../../images/task_1/left_{}.png'
str_right = '../../images/task_1/right_{}.png'
df = (str_left.format(1))

for i in range(11):
    left = cv2.imread(str_left.format(i))
    right = cv2.imread(str_right.format(i))

    ret_l, point_cor_l = cv2.findChessboardCorners(left, (9, 6))
    ret_r, point_cor_r = cv2.findChessboardCorners(right, (9, 6))

    # print(point_cor[:,:,0])
    cv2.drawChessboardCorners(left, (9, 6), point_cor_l, ret_l)
    cv2.drawChessboardCorners(right, (9, 6), point_cor_r, ret_r)

    d3.append(d3_points)
    ip_left.append(point_cor_l)
    ip_right.append(point_cor_r)

    cv2.imshow('window_left', left)
    cv2.imshow('window_right', right)
    cv2.waitKey(100)
    cv2.destroyAllWindows()

print(point_cor_l.shape, d3_points.shape, left.shape[:-1])
ret_left, mtx_left, dist_left, rvecs_l, tvecs_l = cv2.calibrateCamera(d3, ip_left, left.shape[:-1], None, None)


print(mtx_left)
print(dist_left)
# print(rvecs,tvecs)

ret_right, mtx_right, dist_right, rvecs_r, tvecs_r = cv2.calibrateCamera(d3, ip_right, right.shape[:-1], None, None)

print(mtx_right)
print(dist_right)
# print(rvecs,tvecs)

np.savetxt('../../parameters/intrinsic_l.csv', mtx_left, delimiter=',')
np.savetxt('../../parameters/intrinsic_r.csv', mtx_right, delimiter=',')

np.savetxt('../../parameters/distortion_l.csv', dist_left, delimiter=',')
np.savetxt('../../parameters/distortion_r.csv', dist_right, delimiter=',')

# with open('/home/sarthake/project_2a/parameters/distortion.csv', 'w') as outfile:
#     # outfile.write('# Array shape: {0}\n'.format(dst.shape))
#
#     for slice_2d in dst:
#         # print(slice_2d)
#         np.savetxt(outfile, slice_2d,fmt='%-7.2f')
#         outfile.write('# New slice\n')
# print(dst)
h, w = left.shape[:2]

newcameramtx_l, roi_l =cv2.getOptimalNewCameraMatrix(mtx_left,dist_left,(w,h),1, (w,h))
newcameramtx_r, roi_r =cv2.getOptimalNewCameraMatrix(mtx_right,dist_right,(w,h),1, (w,h))

print(roi_l)

left = cv2.imread(str_left.format(1))
mapx_l, mapy_l = cv2.initUndistortRectifyMap(mtx_left, dist_left, None, None, (w,h), 5)
dst = cv2.remap(left, mapx_l, mapy_l, cv2.INTER_LINEAR)
# x,y,w,h = roi_l
# dst = dst[y:y+h, x:x+w]
print(dst.shape)
cv2.imshow('calibresult', dst)
cv2.imshow('original',left)
cv2.waitKey(0)
cv2.destroyAllWindows()
