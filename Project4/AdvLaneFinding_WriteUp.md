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
[image19]: ./Images/Test5_Histogram.png "Sliding Window Search"
[image20]: ./Images/Test5_Perspective_Binary.png "Binary Perspective Image"
[image21]: ./Images/Straight_Lines1_Perspective.png "Straight Lane Perspective Image"
[image22]: ./Images/Straight_Lines1_Processed.png "Straight Lane Perspective Image"
[image23]: ./Images/Straight_Lines1_Result.png "Straight Lane Perspective Image"
[image24]: ./Images/Straight_Lines2_Perspective.png "Straight Lane Perspective Image"
[image25]: ./Images/Straight_Lines2_Processed.png "Straight Lane Perspective Image"
[image26]: ./Images/Straight_Lines2_Result.png "Straight Lane Perspective Image"
[image27]: ./Images/Original_Camera_Calibration.png "Camera Calibration"
[image28]: ./Images/ChessBoardCorners.png "Chess Board Corners"
[image29]: ./Images/Test_image_Distrortion.png "Test Image Distortion"
[image30]: ./Images/ImageProcessingExample.png "Image Processing Example"
[image31]: ./Images/GrayScaleExample.png "Gray Scale Image Example"
[image32]: ./Images/SobelxExample.png "Sobel X Image Filtering Example"
[image33]: ./Images/HLSChannelExample.png "HLS Channel Filtering Example"
[image34]: ./Images/HSVChannelExample.png "HSV Channel Filtering Example"
[image35]: ./Images/SaturationThresholding.png "S-Channel Thresholding Example"
[image36]: ./Images/ValueThresholding.png "V-Channel Thresholding Example"
[image37]: ./Images/VariousFilteringExample.png "Combined Filtering Example"
[image38]: ./Images/PerspectiveCalibrationExample.png "Perspective Correction Example"

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

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
