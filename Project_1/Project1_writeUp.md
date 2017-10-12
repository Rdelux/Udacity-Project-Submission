# **Finding Lane Lines on the Road - by Richard Lee** 

---

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report

Here are the output images from my pipeline using the provided test images:

[//]: # (Image References)

[image1]: ./Result_Images/solidWhiteCurve_after.jpg "solidWhiteCurve_after.jpg"
[image2]: ./Result_Images/whiteCarLaneSwitch_after.jpg "whiteCarLaneSwitch_after.jpg"
[image3]: ./Result_Images/whiteCarLaneSwitch_edge.jpg "whiteCarLaneSwitch_edge.jpg"
[image4]: ./Result_Images/whiteCarLaneSwitch_gray.jpg "whiteCarLaneSwitch_gray.jpg"
[image5]: ./Result_Images/whiteCarLaneSwitch_masked.jpg "whiteCarLaneSwitch_masked.jpg"
[image6]: ./Original_Images/solidWhiteCurve.jpg "solidWhiteCurve.jpg"

---

### Reflection

### 1. Pipeline Description

My pipeline consisted of 5 steps. As an illustration, the original image looks like this:

![alt text][image6]

Then I convert the images to grayscale with pixel values ranging from 0 to 255.  The converted images will like like this:

![alt text][image4]

Since there are yellow color lanes and white color lanes, color detection will not work for all cases.  Therefore, I would like to use the color gradient instead of just using color to determine the traffic lanes.  In order to do that, I apply Gaussian smoothing function with a kernel size of 3x3 and then apply the Canny function to extract the edges from the images. The result looks like this:

![alt text][image3]

Since there are many lines in the view and using the assumption that the camera is rigidly mounted on a vehicle, I created a region on the image where lines inside the region will be selected for further processing.  The result looks like this:

![alt text][image5]

Using Hough Transform to refine the positioning of the lines even further, I obtained an array of endpoint pairs to describe various line segments.  

The next step is to create a single left line to represent the left lane and a single right line to represent the right lane.  In order to do that I have tried various methods, which include simple averaging of slope, averaging and endpoint extrapolation and linear regression using line segment mid-points.  However, I found that a line-segment weighted average of line slope and intercept works the best in terms of consistency and accuracy.

For the cases where the algorithm returns a zero value for the line segment length, I used a temporal approximation method. I store the previous frame slope and intercept value as global variables, then I substitute them into the present case if zero value is detected.

The result is a smooth video file output with lane image overlay on top of the camera images.

![alt text][image1]

### 2. Potential shortcomings with the current pipeline

There are a number of shortcomings associated with the current method.  The pipeline only works when there are sufficient length of straight lane ahead of the vehicle.  If the vehicle is going into a curve, the current linear approximation of the traffic lane will fail.  In addition if there is a long line that is out of position, this will contribute to a lot of error in the annotated version of the input video.  Also, there's no adaptation to the lighting condition since all the parameters are hard coded to the pipeline for edge detection.  

### 3. Possible improvements to the pipeline

One potential improvement to the pipeline is to use high order polynomial to estimate the lanes.  The Hough Transform can be used to identify the end points of line segments, and then by using a second or third order polynomial, most of the traffic lanes can be approximated by a continuous line.

In order to further reduce the error of the lines drawn on the image, one can use the relational information from both the left and right lane.  One possibility is to assume that the road has a constant width so that the distance between the left and right lane should be a linear function of y in the images.

Regarding the changes in lighting condition, I would suggest to set the threshold for the Canny function to be a function of  the average brightness value of the entire picture, so that the pipeline will be more adaptive.
