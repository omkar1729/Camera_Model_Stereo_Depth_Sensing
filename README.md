
# Camera Model and Stereo Depth Sensing

**Task 1 – Pinhole camera model and calibration**
<p style='text-align: justify;'>
  Task 1 is the basic task of camera calibration. In this task we find out the camera intrinsic parameters from 3d – 2d point correspondences. We use the chess board as our reference for calibrating the camera. We calibrate left and right camera individually. We use cv2.findChessboardCorners to detect corners in the given image and draw those points on the image using drawChessboardCorners. We define the 3d points as demonstrated in task1 guidelines. Now we use these 3d and corresponding 2d points to calibrate camera using calibrateCamera function. The outputs of this function is the intrinsic matrix and distortion matrix. We save these in the parameters folder for future use. Below are the results obtained. Top 2 images are of points detected and original image respectively. Bottom image is the calibrated image. The intrinsic parameters are saved as csv files under parameters folder.

 </p>


Results for left Image 2 -

| Original Image  | Corners detected | Result |
| ------------- | ------------- | ----- |
| <img src="https://github.com/omkar1729/Perception_Project_2a/blob/master/output/task_1/original%20image%202.png" width="400" >  | <img src="https://github.com/omkar1729/Perception_Project_2a/blob/master/output/task_1/window%20left%202.png" width="400" >  | <img src="https://github.com/omkar1729/Perception_Project_2a/blob/master/output/task_1/calibrate%20result%202.png" width="400" > |





**Task 2 Stereo Calibration and Rectification**

Task 2 first focuses on acquiring extrinsic paramters of the left and right camera. Images from both camera of a chessboard is given. Both those images are loaded. Along with it, the intrinsic paramters are loaded from csv files, saved in &#39;parameters&#39; folder. First openCV&#39;s &#39;findchessboardcorners&#39; function is called to get the image coordinates of the corners in the images of left and right camera. These image points are stored and used along with the 3d world coordinates of the corners of the chessboard to calibrate both the left and right camera. To do so, we call the function &#39;stereocalibrate&#39; and obtain the rotation and transformation between the two cameras. On doing so, we then proceed to define the origin of first camera as (0,0,0). We set its transformation matrix to a 3x1 zero matrix and its rotation matrix to an identity matrix. We create the projection matrix of the first camera using the intrinsic matrix calculated in the first task and the the translation and rotation defined by us. We then create the projection matrix of the second camera using its intrinsic matrix and the rotation and translation obtained by the function &#39;stereocalibrate&#39;. Once we do this, we call the function,&#39;undistortpoints&#39; to undistort the imagepoints we saved for both cameras. Once we obtaint the undistorted image points, we use them in the function, &#39;traingulatepoints&#39; to triangulate the 3d position of the corners of the chessboard. On completing triangulation, we proceed to with the task of stereo rectification, i.e transformation of the source images onto a common plane such that the two images appear as they have been taken with only a horizontal displacement. We call the fucntion stereorectify followed by &#39;initUndistortRectifyMap&#39; to create two maps, to obtain the values of pixel coordinates of the new remapped image. We then call the fuction &#39;remap&#39; to remap the images from both cameras to the specified maps. It follows the formulas as specified below:

dst(x,y) = src(map\_x(x,y),map\_y(x,y))

Using these functions we obtain our new remapped images.

Below are the figures for original image on left and rectified image on right -

For Left camera -

| original image | Rectified Image |
| -------------  | ---------------- |
| <img src="https://github.com/omkar1729/Perception_Project_2a/blob/master/output/task_2/result_left%200.png" width="400" > | <img src="https://github.com/omkar1729/Perception_Project_2a/blob/master/output/task_2/Left%20remap%200.png" width="400" > |

For Right camera -

| original image | Rectified Image |
| -------------  | ---------------- |
| <img src="https://github.com/omkar1729/Perception_Project_2a/blob/master/output/task_2/result_right%200.png" width="400" > | <img src="https://github.com/omkar1729/Perception_Project_2a/blob/master/output/task_2/Right%20remap%200.png" width="400" > |

Below is plot for triangulation -
<p align="center">
  <img src="https://github.com/omkar1729/Perception_Project_2a/blob/master/output/task_2/Figure_3d%200.png" width="600" >
</p>



**Task 3: Sparse depth triangulation report**

The images are loaded in pairs and the image pair is selected using the value of i. The intrinsic and distortion values are loaded from the result of the previous task. The images are undistorted using &#39;initUndistortRectifyMap&#39; and &#39;remap&#39;. Then orb class objects are created for both the left and right images and ketpoints are extracted. The descriptors are then matched using BFMatcher and the image is displayed and saved after the matches are drawn using &#39;drawMatches&#39;. The essential matrix E is then calculated and only the good feature points are kept. The matches are drawn again using the good feature points and the 3d coordinates are plotted.

Results for scene 0 - 

| All feature points  | Good feature points | Matching |
| ------------- | ------------- | ----- |
| <img src="https://github.com/omkar1729/Perception_Project_2a/blob/master/output/task_3/Feature%20Right%200.png" width="400" >  | <img src="https://github.com/omkar1729/Perception_Project_2a/blob/master/output/task_3/Feature%20Match%20with%20good%20points%202%20-%200.png" width="400" >  | <img src="https://github.com/omkar1729/Perception_Project_2a/blob/master/output/task_3/Feature%20Matching%200.png" width="800" > |


Feature points are found and matched in the scene.

The 3D points received from the scene after using triangulatePoints()
<p align="center">
  <img src="https://github.com/omkar1729/Perception_Project_2a/blob/master/output/task_3/Plot%200.png" width="600" >
</p>



**Task 4: Calculating Depth from given RGB image**

In this task, we created a disparity map for the given image. I.e we calculated depth of every point on the image. We used stereoBM function for block matching and eventually getting the disparity map. We compute the depth using reprojectimageto 3d function which gives us the 3d cordinates for every pixel in image. We calculate the distance using norm of the position cordinates and clip the output between 0 to 255 range to produce the depth image. Below are the depth images for given image

| original image | Depth Image |
| -------------  | ---------------- |
| <img src="https://github.com/omkar1729/Perception_Project_2a/blob/master/images/task_3_and_4/left_8.png" width="400" > | <img src="https://github.com/omkar1729/Perception_Project_2a/blob/master/output/task_4/Disparity.png" width="400" > |

**How to Run the code**

Run the programs in order using

python3 t1.task

In the terminal change the directory to the correct folder (code) and use the above command. Repeat for other tasks, the images are stored after running the code.

