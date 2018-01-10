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

I started by reading in all the `Cars` and `non-Cars` images, this is done in third code cell of the submitted IPython notebook.  Here are some examples of the `Cars` and `non-Cars` classes:

![Vehicle Class][image7]

![Non-vehicle Class][image8]

Visualization of these images and the HOG features are located in code cell number 4, 5 and 6.  Here are the HOG features associated with "Cars" images displayed above:

![Vehicle HOG Features][image9]

In addition to HOG features, I also explored the use of different color spaces for various feature extractions, including HSV, HLS, YCrCb, YUV and LUV.  The selected color space was also used for bin spatial and histogram of feature data extraction from the cars and non-cars images, the codes were implemented in line 103 to 105, and 107 to 109 respectively.  I used the cv2 input method from openCV to load the images hence the .png images have values from 0 to 255, and the values are actually in BGR format.  However, as long as the classifier and the detection feature vectors are consistent, the returned results are correct.  The only concern is when displaying the images, therefore a conversion step was added for image display.

I experimented with different color spaces by comparing the classifier accuracy and testing the classifier against the 6 test images.  The best solution was obtained by using the LUV color space, which correctly identified all the test images. The second best color space is HSV, which only correctly identify 5 out of the 6 images.



### Final choice of HOG parameters

Other parameters for HOG feature extraction includes number of orientation bins, number of pixel per cell, translation pixel number per block, number of HOG channel and spatial size. While holding other values constant and varying one parameter at a time, I varied the number of orientation bins from 8 to 12.  However, I did not find significant improvement in performance for the classifier beyond 8 orientation bins.  I did not vary the number of pixel per cell since varying other parameters provided adequate level of optimization to tune the classifier.  In order to achieve generalization and maximize the speed of the classifier, I used a 16 x 16 pixel spatial size instead of a 32 x 32.  The result was adequate and the classifier was able to identify all the images, therefore the final choice of the HOG parameters are:

|  HOG Parameters  |   Values    |
|:----------------:|:----------:|
|color_space       | LUV        |
| orient           | 8 |
| pix_per_cell | 8 |                                                   
| cell_per_block | 2 |                                               
| hog_channel | 0 |                                                 
| spatial_size | (16, 16) |

These values are declared in code cell number 7 between lines 293 to 302.



### Training a classifier using your selected HOG features and color features

Prior to training a classifier, I check the number of images used for "Car" vs "non-Car" images to ensure that the data sets are balance.  The "non-Car" data set only has 2% more data than the "Car" data set, therefore it is assumed that the data set are sufficiently balanced.

For robustness, simplicity and performance, I used a linear Support Vector Machine (SVM) approach to build a classifier.  The first step for training the classifier is to prepare the data in the correct format.  "Cars" and "non-Cars" data were formatted to a 1-D vector in code line 318.  The feature vector created was then normalized by scaling their values (code line 303).   Since the data sets were loaded as "cars" and "non-cars", there are only 2 classes.  Car images are label with "1" and non-car images are labelled with "0" (code line number 329).  In order to test the accuracy of the trained linear SVM classifier, I split the data set into a training data set and a testing data set.  The testing data set represent only 20% of the entire data set. A linear SVM classifier was defined and used to train the data set in code line number 338 and 340.  The effectiveness and accuracy of the classifier was testing in code line number 342.  The final accuracy was found to be approximately 98.7% using the parameters listed above.  The accuracy changes slightly everytime when I retrain the classifier, which indicate the random sampling nature of the training and testing data set.  The difference in accuracy value is neglible and hence classifier is determined to be stable.


### Implementation of sliding window search

Once the classifier was trained, the next step in this project is to devise a scheme to detect and track the vehicles on the road.  In order to search for features in the image or video stream that resemble a car, a region or a window was defined and it was systematically move around to determine if a group of pixel can be identified as "car".  This is called a sliding window search.  The basic window search function was given by the Udacity course notes, however I implemented the sliding window search function by modifying the given function in order to achieve robustness and efficiency in detecting and track vehicles in the video stream.  

The main sliding window search function was implemented inside the 'find_cars' function.  A region in the given image will be specified by a "box's" corner positions, which are ystart, ystop, xstart, and xstop in code line 355.  Vehicles are assumed to enter into the view from the top and on the right-hand side of the image, since the vehicle is driving on the left lane of the road and did not change lane.  The size of the window should reflect the size of the vehicle appear on the image, therefore multiple scale is required to capture vehicle of different size and at different distance from the observer.  Hence the window "scale" is also an input to the 'find_cars' function.  The window was slide across the region of an image in the y and x directions by the "step" size provided to the function.  In this case, the step size is 2 as suggested in the Udacity course material, and it had shown satisfying results.  The loop for the sliding window search is mainly implemented in code lines 400 to 427.  A block size of 64 was used as it was suggested in Udacity course material and it was proven to be effective.

In order to test the classifier, I developed a testing code to test the classifier on a number of test images in code cell number 11, which correspond to code line number 599 to 624.  Given the specific location of the observer in the images and video, I narrowed the search region between 450 and 1280 x-pixel location for the width, and between 350 to 650 y-pixel location for the height of the image.  Since vehicle at the top part of the image will appear to be small, therefore a small window size should be used to search for vehicles.  A scale factor of 0.8 was used for this top portion of the image.  Additional multi-scale size windows were used to identify vehicles that are closer to the observer.  They include x1.5, x2 and x2.5 scale windows.  Although the largest vehicle appear in the video required a x3.5 scale window to encompress all the features, the heatmap technique explained in the next step will conglomerate all the smaller windows.  These multi-scale windows search are called in code line number 602 to 605.


### Test Image Sample for Vehicle Detection and Classifier Optimization

I used 4 different window scale to look for all car feature in the 6 test images, identified car feature is represented by a blue box and the associated heatmap are shown here:

![Test 1 Image][image1]
![Test 2 Image][image2]
![Test 3 Image][image3]
![Test 4 Image][image4]
![Test 5 Image][image5]
![Test 6 Image][image6]

All car assets were correctly identified by the classifier and the time requied for the search is recorded.  Given the number of frames of the video, which is 1261 frames, the estimated time for completing this exhaustive window search is approximately 50 minutes.  This is way too long for video processing, therefore an improved scheme need to be developed to streamline the bottle neck process, which is the sliding window search process.

First of all, since the relative velocity between the observer and the surrounding vehicle is relatively low in the video, therefore not all frames are required to detect and track the vehicles.  For this reason, I implenmented a toggle switch defined by the global variable "includFrame" to sample every other frame of the video stream instead effective reducing half of the processing time.  This is mainly implemented in code line 564 to 567 inside the 'processFrame' function, which is the main function for processing the video frames.

In addition, I know that vehicles that come into the view of the observer will either enter from the top and right hand side of the image, therefore I am going to focus on the initial vehicle search at those locations.  The plan is to first detect a vehicle entering into the view and then we can search around that location to identify where the vehicle has moved to.  This tracking mechanism will drastically reduce the number of window search operation required.  Even if we need to generalize this process to include the left of the observer, there's no way to test the pipeline for its robustness, therefore the left-hand side of the observer is omitted from the search region.  Once the initial search windows identified a vehicle (code line 516 to 518), search perimeters were defined as "searchWidth" and "searchHeight".  Literature search and experimentation identified the values indicated in code line 523 and the search perimeter parameters were defined.  Multi-scale search windows were used in subsequent steps to refine the search in the surrounding areas (code line 544 to 547).  The issues of multiple windows identifying the same vehicle object was resolved by using heatmap, which will be discussed further in the next section.  However, once the vehicle is identified, the 'label' function is used to identify the vehicle. Using the label information, another function, 'draw_labeled_bboxes', is called to draw a window around the identified object and to record that object in the "track_list" global variable for subsequent window tracking.  Using this targeted area detection and peripheral tracking, I was able to significantly reduce the processing time of the video stream to approximately 5 minutes.


### Video Implementation

Here's a [link to my video result](./project_video.mp4)


### False positives and combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I used a threshold number of 5 and most of the false positive detection was eliminated.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. In addition, I also use error-proving regional exclusion to ensure false positive is not identified in impossible areas.  This is implemented in code line 531 to 540.  The corresponding heatmaps of the test images are shown above.

---

### Discussion

The approach and techniques I used to implement the pipeline to detect and track the vehicles were discussed above.  Several false positive were still identified in the video stream.  A more robost tracking approach should be used to eliminate these false position.  For example, positive identification should not occur in areas that vehicle can't enter into the frame, such as right in front of the observer.  The bounding boxes for the vehicle was designed to have a smooth transition in changing size but the result is still not ideal.  I think the tracking algorithm and the box size control codes should be improve in future revisions.  Although it only takes 5 minutes to process the video, the ideal situation is 50 seconds since this is the length of the video.  If I can achieve 50 seconds of video processing time, then I will be able to track vehicle in near real-time.  The feature vectors may be reduced in length to improve classification time.  Seeding windows in the initial detection can be more intelligent if lane detection information was used to define a better search area. 
