import cv2 as cv
import numpy as np
# from math import multiply
#from matplotlib import pyplot as plt

d3=[]
ip=[]

d3_points = []
for i in range(6):
    for j in range(9):
        d3_points.append([float(j),float(i),0.0])

d3_points = np.asarray(d3_points,np.float32)

#print(d3_points)
str_left = '../../images/task_3_and_4/left_{}.png'
str_right = '../../images/task_3_and_4/right_{}.png'
df = (str_left.format(1))

for i in range(10,11):
    left = cv.imread(str_left.format(i), cv.IMREAD_GRAYSCALE)
    right = cv.imread(str_right.format(i), cv.IMREAD_GRAYSCALE)

    # ret_l, point_cor_l = cv.findChessboardCorners(left, (9, 6))
    # ret_r, point_cor_r = cv.findChessboardCorners(right, (9, 6))
    #
    # # print(point_cor[:,:,0])
    # cv.drawChessboardCorners(left, (9, 6), point_cor_l, ret_l)
    # cv.drawChessboardCorners(right, (9, 6), point_cor_r, ret_r)

    # d3.append(d3_points)
    # ip.append(point_cor_l)

    cv.imshow('window_left', left)
    cv.imshow('window_right', right)
    cv.waitKey(1000)
    cv.destroyAllWindows()

#left only
#print(point_cor_l.shape, d3_points.shape, left.shape)
#ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(d3, ip, left.shape, None, None,flags=cv.CALIB_RATIONAL_MODEL)

mtx = np.loadtxt('../../parameters/intrinsic.csv', delimiter=',')
dist = np.loadtxt('../../parameters/distortion.csv', delimiter=',')
h,  w = left.shape
newcameramtx, roi=cv.getOptimalNewCameraMatrix(mtx,dist,(w,h),1, (w,h))

print(mtx)
print(roi)
#np.savetxt('../../parameters/intrinsic.csv', mtx, delimiter = ',')

#left = cv.imread(str_left.format(0))
mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
dst1 = cv.remap(left, mapx, mapy, cv.INTER_LINEAR)
x,y,w,h = roi
dst1 = dst1[y:y+h, x:x+w]
# print(dst.shape)
#np.savetxt('../../parameters/intrinsic.csv', dist, delimiter = ',')


#right only
# print(point_cor_l.shape, d3_points.shape, right.shape)
# ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(d3, ip, right.shape, None, None,flags=cv.CALIB_RATIONAL_MODEL)
h,  w = right.shape
newcameramtx, roi=cv.getOptimalNewCameraMatrix(mtx,dist,(w,h),1, (w,h))

# print(mtx)
# print(roi)
# np.savetxt('../../parameters/intrinsic.csv', mtx, delimiter = ',')

#right = cv.imread(str_right.format(0))
mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
dst2 = cv.remap(right, mapx, mapy, cv.INTER_LINEAR)
x,y,w,h = roi
dst2 = dst2[y:y+h, x:x+w]
# print(dst.shape)
# np.savetxt('../../parameters/intrinsic.csv', dist, delimiter = ',')

#create orb class object for Left
orb1 = cv.ORB_create()
kp1 = orb1.detect(dst1,None)
kp1, des1 = orb1.compute(dst1, kp1)
img1 = cv.drawKeypoints(dst1, kp1, None, color=(0,255,0), flags=0)
cv.imshow('Feature Left', img1)#, plt.show()
cv.waitKey(1000)

#create orb class object for Right
orb2 = cv.ORB_create()
kp2 = orb2.detect(dst2,None)
kp2, des2 = orb2.compute(dst2, kp2)
img2 = cv.drawKeypoints(dst2, kp2, None, color=(0,255,0), flags=0)
cv.imshow('Feature Right', img2)#, plt.show()
cv.waitKey(1000)

des1 = np.array(des1)
des2 = np.array(des2)
print('DES1 ',len(des1),'*',len(des1[0]))
print('DES1 ',len(des2),'*',len(des2[0]))
print('MATX ',len(mtx),'*',len(mtx[0]))
# ans = np.dot(des1.transpose, mtx, des2)
# print(ans)
#print(kp1,kp2)

# creating BFMatcher object
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

# Matching the descriptors
matches = bf.match(des1,des2)

# Sorting them in the order of their distance
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 15 matches
img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:15],None,flags=2)

cv.imshow('Feature Matching',img3)#,plt.show()
cv.waitKey(1000)
#print(len(matches),len[0](matches))
matches = np.asarray(matches)
print(matches.shape)

intrinsic_matrix_left = np.loadtxt('../../parameters/intrinsic.csv', delimiter=',')
intrinsic_matrix_right = np.loadtxt('../../parameters/intrinsic.csv', delimiter=',')

# print(distortion)

R = np.loadtxt('../../parameters/R.csv', delimiter = ',')
T = np.loadtxt('../../parameters/T.csv', delimiter = ',')
F = np.loadtxt('../../parameters/F.csv', delimiter = ',')
E = np.loadtxt('../../parameters/E.csv', delimiter = ',')

translate = np.zeros([3,1])
rotate = np.identity(3)
proj1 = np.dot(intrinsic_matrix_left,np.concatenate((rotate,translate),axis=1))
R = np.asarray(R)
T = np.asarray(T)
T=np.reshape(T,(3,1))
print(R.shape,T.shape)
proj2 = np.dot(intrinsic_matrix_right, np.concatenate((R,T),axis=1))

list_kp1 = []
list_kp2 = []

# For each match...
for mat in matches:

    # Get the matching keypoints for each of the images
    img1_idx = mat.queryIdx
    img2_idx = mat.trainIdx

    # x - columns
    # y - rows
    # Get the coordinates
    (x1, y1) = kp1[img1_idx].pt
    (x2, y2) = kp2[img2_idx].pt

    # Append to each list
    list_kp1.append([(x1, y1)])
    list_kp2.append([(x2, y2)])

#undistorted_left = cv.undistortPoints(np.reshape(image_points_left,(108,1,2)), intrinsic_matrix_left, distortion)
#undistorted_right = cv.undistortPoints(np.reshape(image_points_right,(108,1,2)), intrinsic_matrix_right, distortion)
#print(np.shape(undistorted_left))

#print(list_kp1)
list_kp1 = np.asarray(list_kp1)
list_kp2 = np.asarray(list_kp2)
print(list_kp1)

triangulate = cv.triangulatePoints(proj1,proj2,list_kp1,list_kp2)
#print(triangulate.shape)

triangulate = np.array(triangulate)
print(triangulate)

cv.destroyAllWindows()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax=plt.axes(projection='3d')
ax.scatter(triangulate[0]/triangulate[3],triangulate[1]/triangulate[3],triangulate[2]/triangulate[3])
plt.show()
