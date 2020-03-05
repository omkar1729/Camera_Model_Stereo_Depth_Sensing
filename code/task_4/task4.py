import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
str_left = '../../images/task_3_and_4/left_{}.png'
str_right = '../../images/task_3_and_4/right_{}.png'
df = (str_left.format(1))

i=8
left = cv.imread(str_left.format(i), cv.IMREAD_GRAYSCALE)
right = cv.imread(str_right.format(i), cv.IMREAD_GRAYSCALE)



cv.imshow('window_left', left)
cv.imshow('window_right', right)
cv.waitKey(1000)
cv.destroyAllWindows()

#left only

mtx_l = np.loadtxt('../../parameters/left_camera_intrinsics/intrinsic_l.csv', delimiter=',')
dist_l = np.loadtxt('../../parameters/left_camera_intrinsics/distortion_l.csv', delimiter=',')
mtx_r = np.loadtxt('../../parameters/right_camera_intrinsics/intrinsic_r.csv', delimiter=',')
dist_r = np.loadtxt('../../parameters/right_camera_intrinsics/distortion_r.csv', delimiter=',')
Q = np.loadtxt('../../parameters/stereo_rectification/Q.csv', delimiter=',')

h,  w = left.shape
newcameramtx_l, roi_l=cv.getOptimalNewCameraMatrix(mtx_l,dist_l,(w,h),1, (w,h))

print(mtx_l)
print(roi_l)

mapx_l, mapy_l = cv.initUndistortRectifyMap(mtx_l, dist_l, None, None, (w,h), 5)
dst1 = cv.remap(left, mapx_l, mapy_l, cv.INTER_LINEAR)
# x,y,w,h = roi_l
# dst1 = dst1[y:y+h, x:x+w]
# print(dst.shape)


#right only

h,  w = right.shape
newcameramtx_r, roi_r=cv.getOptimalNewCameraMatrix(mtx_r,dist_r,(w,h),1, (w,h))

#right = cv.imread(str_right.format(0))
mapx_r, mapy_r = cv.initUndistortRectifyMap(mtx_r, dist_r, None, None, (w,h), 5)
dst2 = cv.remap(right, mapx_r, mapy_r, cv.INTER_LINEAR)
# x,y,w,h = roi_r
# dst2 = dst2[y:y+h, x:x+w]
# print(dst.shape)

stereo = cv.StereoBM_create(numDisparities=16, blockSize=21)
disparity = stereo.compute(dst1,dst2)
depth = cv.reprojectImageTo3D(disparity, Q)


print(depth[0,0])
print(disparity.shape)
plt.imshow(disparity)

vect = np.zeros((480,640))
for i in range(480):
    for j in range(640):

        vect[i,j] = np.linalg.norm(depth[i,j])


vect[np.isinf(vect)]=255
print(vect.max())
print(vect)
vect = ((255.0- vect)/(vect.max()))*255
#vect = vect/(float(vect.max())/255.0)
# for i in range(480):
#     for j in range(640):
#         if vect[i,j]=='inf':
#             vect[i,j] = 500000
#         vect[i,j] = ((255.0- vect[i,j])/(vect.max()))

cv.imshow('depth', vect)
#cv.imshow('disparity', disparity)
cv.waitKey(0)
cv.destroyAllWindows()
plt.imshow(disparity,'gray')
plt.show()



# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax=plt.axes(projection='3d')
# ax.scatter(depth[:][:][0],depth[:][:][1],depth[:][:][2])
# plt.show()

