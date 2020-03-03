import cv2
import numpy as np


d3=[]
ip_l=[]
ip_r=[]
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
    ip_l.append(point_cor_l)
    ip_r.append(point_cor_r)



    cv2.imshow('window_left', left)
    cv2.imshow('window_right', right)
    cv2.waitKey(100)
    cv2.destroyAllWindows()

print(point_cor_l.shape, d3_points.shape, left.shape[:-1])
retl, mtxl, distl, rvecsl, tvecsl = cv2.calibrateCamera(d3, ip_l, left.shape[:-1], None, None,flags=cv2.CALIB_RATIONAL_MODEL)
retr, mtxr, distr, rvecsr, tvecsr = cv2.calibrateCamera(d3, ip_r, right.shape[:-1], None, None,flags=cv2.CALIB_RATIONAL_MODEL)



h,  w = left.shape[:-1]

newcameramtx_l, roi_l=cv2.getOptimalNewCameraMatrix(mtxl,distl,(w,h),1, (w,h))
newcameramtx_r, roi_r=cv2.getOptimalNewCameraMatrix(mtxr,distr,(w,h),1, (w,h))


print(mtxl,mtxr)
print(roi_l,roi_r)
np.savetxt('/home/omkar/PycharmProjects/Perception-Project-2/parameters/intrinsic_l.csv', mtxl, delimiter = ',')
np.savetxt('/home/omkar/PycharmProjects/Perception-Project-2/parameters/intrinsic_r.csv', mtxr, delimiter = ',')


left = cv2.imread(str_left.format(0))
mapx, mapy = cv2.initUndistortRectifyMap(mtxl, distl, None, newcameramtx_l, (w,h), 5)
dst = cv2.remap(left, mapx, mapy, cv2.INTER_LINEAR)
x,y,w,h = roi_l
dst = dst[y:y+h, x:x+w]
# print(dst.shape)
np.savetxt('/home/omkar/PycharmProjects/Perception-Project-2/parameters/distortion_l.csv', distl, delimiter = ',')
np.savetxt('/home/omkar/PycharmProjects/Perception-Project-2/parameters/distortion_r.csv', distr, delimiter = ',')

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

