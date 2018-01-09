# **Project 5: Vehicle Detection and Tracking Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier
* Apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector 
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images
* Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reduce outliers and follow detected vehicles
* Estimate a bounding box for vehicles detected

[//]: # (Image References)
[image1]: ./Images/Test1Output.png
[image2]: ./Images/Test2Output.png
[image3]: ./Images/Test3Output.png
[image4]: ./Images/Test4Output.png
[image5]: ./Images/Test5Output.png
[image6]: ./Images/Test6Output.png
[image7]: ./Images/Cars.png
[image8]: ./Images/Notcars.png
[image9]: ./Images/HOG.png
[video1]: ./VehDetectTrack_Video.mp4

The Rubric Points are listed in this following link: https://review.udacity.com/#!/rubrics/513/view

---


### Histogram of Oriented Gradients (HOG)

In order to create a feature set for training a classifier, the feature set needs to be robust and effective.  Even with a rich data set, generalization and over-fitting could be an issue, therefore Histogram of Oriented Gradients (HOG) technique was used in the project to identify vehicle objects.  The function implementation for this step is contained in the second code cell of the submitted IPython notebook (or in lines 150 through 164 of the file called `VehDetectTrack_Submit.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images, this is done in third code cell of the submitted IPython notebook.  Here are some examples of the `vehicle` and `non-vehicle` classes:

![Vehicle Class][image7]

![Non-vehicle Class][image8]

Visualization of these images and the HOG features are located in code cell number 4, 5 and 6.  Here are the HOG features associated with "Cars" images displayed above:

![Vehicle HOG Features][image9]

In addition to HOG features, I also explored the use of different color spaces for various feature extractions, including HSV, HLS, YCrCb, YUV and LUV.  The selected color space was also used for bin spatial and histogram of feature data extraction from the cars and non-cars images, the codes were implemented in line 103 to 105, and 107 to 109 respectively.  I used the cv2 input method from openCV to load the images hence the .png images have values from 0 to 255, and the values are actually in BGR format.  However, as long as the classifier and the detection feature vectors are consistent, the returned results are correct.  The only concern is when displaying the images, therefore a conversion step was added for image display.

I experimented with different color spaces by comparing the classifier accuracy and testing the classifier against the 6 test images.  The best solution was obtained by using the LUV color space, which correctly identified all the test images. The second best color space is HSV, which only correctly identify 5 out of the 6 images.


### Final choice of HOG parameters

Other parameters for HOG feature extraction includes number of orientation bins, number of pixel per cell, translation pixel number per block, number of HOG channel and spatial size. While holding other values constant and varying one parameter at a time, I varied the number of orientation bins from 8 to 12.  However, I did not find significant improvement in performance for the classifier beyond 8 orientation bins.  I did not vary the number of pixel per cell since varying other parameters provided adequate level of optimization to tune the classifier.  In order to achieve generalization and maximize the speed of the classifier, I used a 16 x 16 pixel spatial size instead of a 32 x 32.  The result was adequate and the classifier was able to identify all the images, therefore the final choice of the HOG parameters are:

|  HOG Parameters  |   Value    |
|:----------------:|:----------:|
|color_space       | LUV        |
| orient           | 8 |
| pix_per_cell | 8 |                                                   
| cell_per_block | 2 |                                               
| hog_channel | 0 |                                                 
| spatial_size | (16, 16) |

These values are declared in code cell number 7 between lines 276 to 281.

### Training a classifier using your selected HOG features and color features

I trained a linear SVM using...


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

