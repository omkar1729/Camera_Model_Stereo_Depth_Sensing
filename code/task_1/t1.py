import cv2
import numpy as np



d3=[]
ip=[]
d3_points = []
for i in range(6):
    for j in range(9):
        d3_points.append([float(j),float(i),0.0])

d3_points = np.asarray(d3_points,np.float32)

#print(d3_points)
str_left = '/home/omkar/PycharmProjects/Perception-Project-2/images/task_1/left_{}.png'
str_right = '/home/omkar/PycharmProjects/Perception-Project-2/images/task_1/right_{}.png'
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
    ip.append(point_cor_l)


    cv2.imshow('window_left', left)
    cv2.imshow('window_right', right)
    cv2.waitKey(100)
    cv2.destroyAllWindows()

print(point_cor_l.shape, d3_points.shape, left.shape[:-1])
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(d3, ip, left.shape[:-1], None, None)


h,  w = left.shape[:-1]

print(mtx)
# print(rvecs,tvecs)
np.savetxt('/home/omkar/PycharmProjects/Perception-Project-2/parameters/intrinsic.csv', mtx, delimiter = ',')

left = cv2.imread(str_left.format(0))
mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, None, (w,h), 5)
dst = cv2.remap(left, mapx, mapy, cv2.INTER_LINEAR)
# print(dst.shape)
np.savetxt('/home/omkar/PycharmProjects/Perception-Project-2/parameters/distortion.csv', dist, delimiter = ',')
# with open('/home/sarthake/project_2a/parameters/distortion.csv', 'w') as outfile:
#     # outfile.write('# Array shape: {0}\n'.format(dst.shape))
#
#     for slice_2d in dst:
#         # print(slice_2d)
#         np.savetxt(outfile, slice_2d,fmt='%-7.2f')
#         outfile.write('# New slice\n')
# print(dst)
cv2.imshow('calibresult', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

