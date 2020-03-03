import cv2 as cv
import numpy as np
#from matplotlib import pyplot as plt

d3=[]
ip=[]
d3_points = []
for i in range(6):
    for j in range(9):
        d3_points.append([float(j),float(i),0.0])

d3_points = np.asarray(d3_points,np.float32)

#print(d3_points)
str_left = '../../images/task_1/left_{}.png'                #change for images in task_3_and_4
str_right = '../../images/task_1/right_{}.png'
df = (str_left.format(1))

for i in range(2):
    left = cv.imread(str_left.format(i))
    right = cv.imread(str_right.format(i))

    ret_l, point_cor_l = cv.findChessboardCorners(left, (9, 6))
    ret_r, point_cor_r = cv.findChessboardCorners(right, (9, 6))

    # print(point_cor[:,:,0])
    cv.drawChessboardCorners(left, (9, 6), point_cor_l, ret_l)
    cv.drawChessboardCorners(right, (9, 6), point_cor_r, ret_r)

    d3.append(d3_points)
    ip.append(point_cor_l)


    cv.imshow('window_left', left)
    cv.imshow('window_right', right)
    cv.waitKey(1000)
    cv.destroyAllWindows()


#left only
print(point_cor_l.shape, d3_points.shape, left.shape[:-1])
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(d3, ip, left.shape[:-1], None, None,flags=cv.CALIB_RATIONAL_MODEL)
h,  w = left.shape[:-1]
newcameramtx, roi=cv.getOptimalNewCameraMatrix(mtx,dist,(w,h),1, (w,h))

print(mtx)
print(roi)
np.savetxt('../../parameters/intrinsic.csv', mtx, delimiter = ',')

left = cv.imread(str_left.format(0))
mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
dst1 = cv.remap(left, mapx, mapy, cv.INTER_LINEAR)
x,y,w,h = roi
dst1 = dst1[y:y+h, x:x+w]
# print(dst.shape)
np.savetxt('../../parameters/intrinsic.csv', dist, delimiter = ',')


#right only
print(point_cor_l.shape, d3_points.shape, right.shape[:-1])
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(d3, ip, right.shape[:-1], None, None,flags=cv.CALIB_RATIONAL_MODEL)
h,  w = right.shape[:-1]
newcameramtx, roi=cv.getOptimalNewCameraMatrix(mtx,dist,(w,h),1, (w,h))

print(mtx)
print(roi)
np.savetxt('../../parameters/intrinsic.csv', mtx, delimiter = ',')

right = cv.imread(str_right.format(0))
mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
dst2 = cv.remap(right, mapx, mapy, cv.INTER_LINEAR)
x,y,w,h = roi
dst2 = dst2[y:y+h, x:x+w]
# print(dst.shape)
np.savetxt('../../parameters/intrinsic.csv', dist, delimiter = ',')

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

# creating BFMatcher object
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

# Matching the descriptors
matches = bf.match(des1,des2)

# Sorting them in the order of their distance
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 15 matches
img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:15],None,flags=2)

cv.imshow('Feature Matching',img3)#,plt.show()
cv.waitKey(0)

cv.destroyAllWindows()
