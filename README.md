
# Camera Model and Stereo Depth Sensing

**Task 1 – Pinhole camera model and calibration**

Task 1 is the basic task of camera calibration. In this task we find out the camera intrinsic parameters from 3d – 2d point correspondences. We use the chess board as our reference for calibrating the camera. We calibrate left and right camera individually. We use cv2.findChessboardCorners to detect corners in the given image and draw those points on the image using drawChessboardCorners. We define the 3d points as demonstrated in task1 guidelines. Now we use these 3d and corresponding 2d points to calibrate camera using calibrateCamera function. The outputs of this function is the intrinsic matrix and distortion matrix. We save these in the parameters folder for future use. Below are the results obtained. Top 2 images are of points detected and original image respectively. Bottom image is the calibrated image. The intrinsic parameters are saved as csv files under parameters folder.

Results for left Image 2 -

![](https://github.com/omkar1729/Perception_Project_2a/blob/master/output/task_1/calibrate%20result%202.png) ![](RackMultipart20210227-4-1oq288m_html_e68ea166647846f0.png)

![](RackMultipart20210227-4-1oq288m_html_b2d4293f21973fe5.png)

Results for Right Image 2 -

![](RackMultipart20210227-4-1oq288m_html_1de33517fdc9dddf.png) ![](RackMultipart20210227-4-1oq288m_html_e68ea166647846f0.png)

![](RackMultipart20210227-4-1oq288m_html_b2d4293f21973fe5.png)

**Task 2 Stereo Calibration and Rectification**

Task 2 first focuses on acquiring extrinsic paramters of the left and right camera. Images from both

camera of a chessboard is given. Both those images are loaded. Along with it, the intrinsic paramters are

loaded from csv files, saved in &#39;parameters&#39; folder. First openCV&#39;s &#39;findchessboardcorners&#39; function is

called to get the image coordinates of the corners in the images of left and right camera. These image

points are stored and used along with the 3d world coordinates of the corners of the chessboard to

calibrate both the left and right camera. To do so, we call the function &#39;stereocalibrate&#39; and obtain the

rotation and transformation between the two cameras. On doing so, we then proceed to define the origin

of first camera as (0,0,0). We set its transformation matrix to a 3x1 zero matrix and its rotation matrix to

an identity matrix. We create the projection matrix of the first camera using the intrinsic matrix calculated

in the first task and the the translation and rotation defined by us. We then create the projection matrix of

the second camera using its intrinsic matrix and the rotation and translation obtained by the function

&#39;stereocalibrate&#39;. Once we do this, we call the function,&#39;undistortpoints&#39; to undistort the imagepoints we

saved for both cameras. Once we obtaint the undistorted image points, we use them in the function,

&#39;traingulatepoints&#39; to triangulate the 3d position of the corners of the chessboard.

On completing triangulation, we proceed to with the task of stereo rectification, i.e transformation of the

source images onto a common plane such that the two images appear as they have been taken with only a

horizontal displacement. We call the fucntion stereorectify followed by &#39;initUndistortRectifyMap&#39; to

create two maps, to obtain the values of pixel coordinates of the new remapped image. We then call the

fuction &#39;remap&#39; to remap the images from both cameras to the specified maps. It follows the formulas as

specified below:

dst(x,y) = src(map\_x(x,y),map\_y(x,y))

Using these functions we obtain our new remapped images.

Below are the figures for original image on left and rectified image on right -

for right camera -

![](RackMultipart20210227-4-1oq288m_html_a663f6fefcdbb658.png) ![](RackMultipart20210227-4-1oq288m_html_afd706c5aabe3e10.png)

F ![](RackMultipart20210227-4-1oq288m_html_b2c69533fa5d6f48.png) or Left camera -

![](RackMultipart20210227-4-1oq288m_html_78b7121391f35464.png)

Below is plot for triangulation -

![](RackMultipart20210227-4-1oq288m_html_8761a5bfba9e4afb.png)

**Task 3: Sparse depth triangulation report**

The images are loaded in pairs and the image pair is selected using the value of i. The intrinsic and distortion values are loaded from the result of the previous task. The images are undistorted using &#39;initUndistortRectifyMap&#39; and &#39;remap&#39;. Then orb class objects are created for both the left and right images and ketpoints are extracted. The descriptors are then matched using BFMatcher and the image is displayed and saved after the matches are drawn using &#39;drawMatches&#39;. The essential matrix E is then calculated and only the good feature points are kept. The matches are drawn again using the good feature points and the 3d coordinates are plotted.

**Results for scene 4**

![](RackMultipart20210227-4-1oq288m_html_a552429d09454451.png) ![](RackMultipart20210227-4-1oq288m_html_177196db69dc3d59.png)

All feature points (left) versus only the good feature points (right)

![](RackMultipart20210227-4-1oq288m_html_5081f6e404935e90.png)

Feature points are found and matched in the scene.

![](RackMultipart20210227-4-1oq288m_html_ca96984e1a66a85e.png)

3D coordinates of the points that have a good match.

**Results for scene 0**

![](RackMultipart20210227-4-1oq288m_html_ef08a58003e2d62d.png) ![](RackMultipart20210227-4-1oq288m_html_12a4a2827eac28d9.png)

Good feature points (right) extracted from another scene

![](RackMultipart20210227-4-1oq288m_html_6ae0f3f3183d70c3.png)

Matching 15 points from the scene from the left and right image

![](RackMultipart20210227-4-1oq288m_html_ce6e8f56cb49dc18.png)

The 3D points received from the scene after using triangulatePoints()

**Task 4**

In this task, we created a disparity map for the given image. I.e we calculated depth of every point on the image. We used stereoBM function for block matching and eventually getting the disparity map. We compute the depth using reprojectimageto 3d function which gives us the 3d cordinates for every pixel in image. We calculate the distance using norm of the position cordinates and clip the output between 0 to 255 range to produce the depth image. Below are the depth images for given image

S ![](RackMultipart20210227-4-1oq288m_html_925319ca0a5faee8.png) ![](RackMultipart20210227-4-1oq288m_html_cc253f6a1477314d.png) ![](RackMultipart20210227-4-1oq288m_html_1cb9bca011d20150.png) cene 8

Scene 4 below -

![](RackMultipart20210227-4-1oq288m_html_ae0206e2ea0f0a53.png)

![](RackMultipart20210227-4-1oq288m_html_f66f498dcead956f.png)

![](RackMultipart20210227-4-1oq288m_html_fe0f37daf65d7577.png)

**How to Run the code**

Run the programs in order using

python3 t1.task

In the terminal change the directory to the correct folder (code) and use the above command. Repeat for

other tasks, the images are stored after running the code.

**Contributions:**

Mitul Magu (Tech Lead) – Completed task 3 and helped in the remaining tasks. Wrote section for task 3

in the reprort

Omkar Muglikar (Reporter) - Completed task 1 and task 4, helped in other tasks and wrote the report.

Sarthake Choudhary (Supporter) – Completed task 2 and helped in various other parts of the code.
