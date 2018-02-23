# **Project 4: Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./Images/Test1_Result.png "Test Image Result"
[image2]: ./Images/Test2_Result.png "Test Image Result"
[image3]: ./Images/Test3_Result.png "Test Image Result"
[image4]: ./Images/Test4_Result.png "Test Image Result"
[image5]: ./Images/Test5_Result.png "Test Image Result"
[image6]: ./Images/Test6_Result.png "Test Image Result"
[image7]: ./Images/Test1_Processed.png "Processed Image for Lane Finding"
[image8]: ./Images/Test2_Processed.png "Processed Image for Lane Finding"
[image9]: ./Images/Test3_Processed.png "Processed Image for Lane Finding"
[image10]: ./Images/Test4_Processed.png "Processed Image for Lane Finding"
[image11]: ./Images/Test5_Processed.png "Processed Image for Lane Finding"
[image12]: ./Images/Test6_Processed.png "Processed Image for Lane Finding"
[image13]: ./Images/Test1_Perspective.png "Perspective Transform"
[image14]: ./Images/Test2_Perspective.png "Perspective Transform"
[image15]: ./Images/Test3_Perspective.png "Perspective Transform"
[image16]: ./Images/Test4_Perspective.png "Perspective Transform"
[image17]: ./Images/Test5_Perspective.png "Perspective Transform"
[image18]: ./Images/Test6_Perspective.png "Perspective Transform"
[image19]: ./Images/Test5_Windows.png "Sliding Window Search"
[image20]: ./Images/Test5_Perspective_Binary.png "Binary Perspective Image"
[image21]: ./Images/Straight_Lines1_Perspective.png "Straight Lane Perspective Image"
[image22]: ./Images/Straight_Lines1_Processed.png "Straight Lane Perspective Image"
[image23]: ./Images/Straight_Lines1_Result.png "Straight Lane Perspective Image"
[image24]: ./Images/Straight_Lines2_Perspective.png "Straight Lane Perspective Image"
[image25]: ./Images/Straight_Lines2_Processed.png "Straight Lane Perspective Image"
[image26]: ./Images/Straight_Lines2_Result.png "Straight Lane Perspective Image"
[image27]: ./Images/Original_Camera_Calibration.png "Camera Calibration"
[image28]: ./Images/ChessBoardCorners.png "Chess Board Corners"
[image29]: ./Images/Test_Image_Distortion.png "Test Image Distortion"
[image30]: ./Images/ImageProcessingExample.png "Image Processing Example"
[image31]: ./Images/GrayScaleExample.png "Gray Scale Image Example"
[image32]: ./Images/SobelxExample.png "Sobel X Image Filtering Example"
[image33]: ./Images/HLSChannelExample.png "HLS Channel Filtering Example"
[image34]: ./Images/HSVChannelExample.png "HSV Channel Filtering Example"
[image35]: ./Images/SaturationThresholding.png "S-Channel Thresholding Example"
[image36]: ./Images/ValueThresholding.png "V-Channel Thresholding Example"
[image37]: ./Images/VariousFilteringExample.png "Combined Filtering Example"
[image38]: ./Images/PerspectiveCalibrationExample.png "Perspective Correction Example"
[image39]: ./Images/Perspective.png "Perspective Correction Example"
[image40]: ./Images/Test5_Histogram.png "Sliding Window Search"

[video1]: ./Project_Videos/project_video.mp4 "Project Video"
[video2]: ./LaneDetectedVideo.mp4 "Lane Detected Video"

The Rubric Points are listed in this following [link](https://review.udacity.com/#!/rubrics/571/view)   

---

### Camera Calibration

#### 1. Camera matrix and distortion coefficients Computation

The first step for this project is to correct the camera distortions present in the project video.  In order to do this, I used the various chessboard calibration images provided, which can be found in the "camera_cal" folder.  The code for this step is contained in the first code cell of the IPython notebook "AdvLaneFinding_Submit.ipynb" (or in lines 42 through 61 of the file called `AdvLaneFinding_Submit.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the physical space. Here I am assuming the chessboard has no out-of-plane component, therefore z=0 for all its coordinates and the object points are the same for each of the 20 calibration images.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  The following image shows an example of the detected chessboard corners:

![alt text][image28]

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function, which can be seen in line 61 of the code.  This step only need to be computed once since these parameters are the intrinsic properties of the camera.  As an demonstration,  I applied this distortion correction to the test image 'calibration2.jpg' using the `cv2.undistort()` function in the second code cell. Here are the images of the obtained result: 

![alt text][image27]

### Pipeline (single images)

The next step is to develop a pipeline to processed a number of test images in order to ensure the code is functional and robust enough for the project video.  The original development code can be seen in the Jupyter Notebook 'AdvLaneFinding_2.ipynb'.  Once the pipeline was proven to be adequate, many of the test codes were removed for clarity in the final version of the code, 'AdvLaneFinding_Submit.ipynb'.  In this section, I will describe how this pipeline works for individual test images.

#### 1. Distortion Correction on Camera Images

Using the distortion correction obtained from the previous step, I applied them on a test image to demostrate the effect of undistorting a road image:

![alt text][image29]

#### 2. Image Processing for Lane Finding - use color transforms and image gradients to create thresholded binary image  

In order to identify the lane lines on the road, a binary image of the road should be created, which will clearly identify the lane as "1" or True or other pixels as "0" or false.  Since the lanes can be continuous, discrete, displayed in different color and lighting condition, a robust image processing scheme need to be developed in order to identify the lanes at all times.  The image processing of the code is located in a function called "ProcessImg", which is located in code line# 108.  The grayscale conversion and thresholding is handled by code line# 109 to 112.  

At first, I converted the image to grayscale and generate a binary image with the goal of identifying the lane lines clearly.  To generate a binary image, I applied a lower and upper bound to the grayscale image, thus creating a threshold criteria.  Although this method works in ideal lighting and lane coloring conditions, they method is not versatile and fail to identify lanes some test images.  An example of grayscale conversion and thresholding can be seen below:

![alt text][image31]

In order to remove other artifacts, which are not part of the lane lines, a directional gradient filter was used.  The Sobel operator was used to extract the lane line from the road background image.  Since lane line is directionally biased, therefore this scheme should be able to distinguish the lane lines from the noisy background.  I experimented with the x and y direction of the Sobel operator, and it was found that the x direction works slightly better in picking out the lane lines. The implenmentation of this operation can be seen in code line# 114 to 119 inside the ProcessImg function.  Applying thresholds and creating a binary image of the input, the sample processed image looks like this:

![alt text][image32]

The Sobel X operator is a more robust approach to process an image for finding the lane lines.  In the example above, the grayscale binary image shows a pretty good result but if the lighting condition changed, the grayscale method may not work by itself.  Using a gradient operator, like Sobel, will allow a more versatile approach.

The challenges of identifying lanes in different lighting condition and color cannot be addressed fully by using just grayscale images.  Various color spaces that are less dependent on lighting and color can be used to improve the pipeline.  At first, the RGB color space was examined and the image was separated into 3 images according to their color components.  Although the Red channel shows the lane lines clearly in the image but if different color lane lines were used, the result will not be ideal.  For this reason, other color spaces should be used for additional image processing.

I explored the HLS and HSV color space in order to find a color components that are decoupled from color (Hue) and lighting conditions.  Examining the S Channel or the Saturation of the test image in the HLS color space, it was found that the S channel was the most distintive one.  The image in HLS color space can be seen below:

![alt text][image33]

However, using the S Channel from the HLS color space along still failed in other challenging light conditions.  Therefore, I explore using the V Channel or the Value of the HSV color space.  The image in HSV color space can be seen below:

![alt text][image34]

In theory, the Value Channel should provide a good indicator of the single color lane line even in low light shadow spots.  However, in practice the result is still not ideal since bright color sunlight will drown out the lane lines.  The Saturation Channel performs well in shadow and low light condition. The threshold binary image of S-Channel filtering can be seen here:

![alt text][image35]

And the V-Channel filtering can be seen here:

![alt text][image36]

Combining both the binaries of the S and V channels using a union or 'And' operator should reinforce the correct lane line detection.  Further improvement of the lane line image can be achieve by removing some of the noise from the image.  This can be done by combining the Sobel x grayscale binary image and the combined S-V color channel thresholded image.  The result can be seen in the image below:

![alt text][image37]

This image processing scheme was tested all the provided testing images and the results are good - lane lines are clearly identified in various lighting conditions:

Test 2 (Please disregard image titles - "Top-Down Perspective" should have been named "Processed Image")
![alt text][image7]

Test 3 (Please disregard image titles - "Top-Down Perspective" should have been named "Processed Image")
![alt text][image8] 

Test 4 (Please disregard image titles - "Top-Down Perspective" should have been named "Processed Image")
![alt text][image9] 

Test 5 (Please disregard image titles - "Top-Down Perspective" should have been named "Processed Image")
![alt text][image10] 

Test 6 (Please disregard image titles - "Top-Down Perspective" should have been named "Processed Image")
![alt text][image11] 


#### 3. Perspective Transform

The code for my perspective transform includes a function called `warp()`, which appears in lines 161 through 179 in the file `AdvLaneFinding_Submit.py`.  The `warp()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I picked the source points by using an image with a straight road, which can extend the distance between points and minimize the error associated with hand-picking the points.  Based on the location of the lower source points, I selected the coordinate of the destination points so that the area I defined as a straight road is a rectangle.  I iterate the selection until the lanes appear to be parallel and straight.  I chose to hardcode the source and destination points in the following manner:

```python
def warp(img):
    img_size = (img.shape[1], img.shape[0])
    src = np.float32(
        [[700,460],
         [1061,695],
         [241,695],
         [580,460]])         
    dstW = np.float32(
        [[1050,0],
         [1050,695],
         [225,695],
         [225,0]])
    
    M = cv2.getPerspectiveTransform(src, dstW)                           # Get transform matrix
    Minv = cv2.getPerspectiveTransform(dstW,src)                         # Get the inverse transform matrix
    
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    
    return [warped, Minv]
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 700, 460      | 1050, 0        | 
| 1061, 695      | 1050, 695      |
| 241, 695     | 225, 695      |
| 580, 460      | 225, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image:

![alt text][image38]
![alt text][image39] 

#### 4. Identified lane-line pixels

After the perspective view was found, the next step is to identify the lane lines so that they can be highlighted and the calculate the curvature of the lanes.  I defined a function called sliding_window, which implement a sliding window technique to segment a search area to identify the general location of the lane lines.  This function is located in line 198 throught 268.  In order to search for the lane lines, I first compute the histogram of the lower half of the image in order to find the beginning of the lane lines on the bottom of the screen.  This can be seen in the following image:

Test 5 Image: Perspective View and Histogram

![alt text][image40]

The two most prominent peaks in the histogram approximate the two lane line locations, this is in line 205 and 206 of the code. I segmented the search box to 10 segments along the height of the image for better resolution for curved and discreet lane lines.  The search window positions were updated every frame in order to determine where the lanes are.  This information is stored in the variables leftx_current and rightx_current.  The boundaries of the search windows are defined in lines 227 through 232, and the "occupied" pixel are identified in lines 239 to 242.  The mean position of the occupied pixels will be update for the next window search.  By fitting a second order polynomial to the mean positions of the windows, the lane line positions can be estimated.  The coefficients of the polynomial and the pixel positions of the lanes were returned from the function. The result is shown in the image below:

![alt text][image17]


#### 5. Calculate the radius of curvature of the lane and the position of the vehicle with respect to center

In order to determine the curvature of the lane, I developed a function called "get_Curvature", it is located in line 272 to 290.  The function also transform the measurement of the curvature in pixel space to physical space using the scaling factors provided, which are 30/720 meter per pixel in the y direction and 3.7/700 meter per pixel in the x direction. A pair of new polynormials were created (lines 283 to 284) based on physical space and the curvature of the lanes were determined (lines 287 to 288).

Assuming that the camera is mounted on the centerline of the vehicle, the departure distance of the vehicle and the center of the lane can be estimated.  By assumption, the middle of the image indicate the center of the vehicle, which is half the pixel count of the image width.  This equals to 640, this value is indicated in line 349.  Taking the base positions of the lane lines and calculate the median, the center of the lane can be estimated.  The pixel location of the left and right lanes can be taken by the 720th elements of the respective polynormials, which is indicated in the following code:

```python
# Departure from lane center information
    xm_per_pix = 3.7/700     
    laneDep = 640.0 - ((leftPolyx[719]+rightPolyx[719])/2)                   # Use y-intercept to determine lane positions in a 1280 pixel image
    laneDep = laneDep * xm_per_pix
```  

The difference between the the center of the lane and pixel 640 is the departure distance in pixel space.  Converting it to physical space, the physical departure distance can be estimated, this is shown in line 350.


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

After the curvatures and the departure distance were calculated, the next step is the plot the highlight lane back to the road and indicate the two values calculated.  This is done in the function "outputVideoImage", which takes both the warped and unwarped images, the polynormial information, curvature and departure values, as well as the inverse warp transform matrix.  The function bounded the area between the left and right lanes to create a polygon and then fill the polygon with green color in the top-down perspective view image.  Then the function use the cv2.warpPerspective method to transform the top-down view back to the warped view using the inverse transformation matrix. At the end, the warped image was added back to the original image along with the curvature and departure text to complete one cycle of the pipeline (lines 311 to 314).  If the curvature value is larger then 1.5 km, it is highly likely that the vehicle is travelling on a straight road, therefore the code will show the user "Straight Road" instead of providing a very large number.  This is shown in the following image:

![alt text][image5]

---

### Pipeline (video)

#### 1. Final video output

Here's a [link to my video result](./LaneDetectedVideo.mp4)

The code will call the main video processing function called "video_process".  Inside this function, it will called various image processing functions (lines 326 and 327), call the functions to calculate curvature (line 337) and compute the departure distance.  It will pass all these information to the outputVideoImage function for combining.  The result final image is passed back to the main function before writing it to file to create a video.

---

### Discussion

I had discussed the approach I took and the techniques I used through out this report.  There are number of areas that the pipeline can be improved since the pipeline may breakdown in a number of scenarios.  For example, if there are painted markings on the road, the histogram count make pick up the markings instead of the lane lines.  As a result, the sliding window search will fail.  Using history variables, performance tracking and uncertainty evaluation, this problem can be resolved.  The current method can also be improve by not calling the window search algorithm every frame.  This will improve the speed of the program.

