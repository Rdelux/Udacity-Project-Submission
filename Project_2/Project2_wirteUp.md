## **Traffic Sign Classifier** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./images/trainingDataHist.png "Training Data Histogram"
[image2]: ./images/testDataHist.png "Test Data Histogram"
[image3]: ./images/validationDataHist.png "Validation Data Histogram"
[image4]: ./images/Speed-Limit-20.png "Speed Limit 20km/h Sign"
[image5]: ./images/Turn-Right-Ahead.png "Right Turn Ahead Sign"
[image6]: ./images/Wild-animal.png "Wild Animal Sign"
[image7]: ./images/2.png "Test image - 50 km/h Sign"
[image8]: ./images/11.png "Test image - Right of Way Sign"
[image9]: ./images/12.png "Test image - Priority Road Sign"
[image10]: ./images/17.png "Test image - Do not Enter Sign"
[image11]: ./images/28.png "Test image - Children Crossing Sign"


## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

Here is a link to my [project code and supporting materials](https://github.com/Rdelux/CarND-Traffic-Sign-Classifier-P2)

### Data Set Summary & Exploration

#### 1. Data set Summary

The code and analysis for this project was done using python, numpy, pandas and other libraries associated with python. 

I used the pandas library to calculate summary statistics of the German traffic
signs data set:

* The size of training set is 34799 samples
* The size of the validation set is 4410 samples
* The size of test set is 12630 samples
* The shape of a traffic sign image is 32 x 32 and since it's a color image so it has 3 channels - 32 x 32 x 3
* The number of unique classes/labels in the data set is 43 classes

#### 2. Exploratory visualization of the dataset

Here is an exploratory visualization of the data set. It is a histogram showing the distribution of the 43 classes of traffic signs provided by the project:

![alt text][image1]

![alt text][image2]

![alt text][image3]

### Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

## 1. Input Image Data and Image Processing
After all the images are downloaded and imported to my codes, the next step is to preprocess the image data so that the deep neural network can maximize its effectiveness during training.  The goal is to minimize the training time but maximize the validation accuracy.  By preprocessing the training data set, the gradient descent technique used to gradually reduce the loss of the system will be more stable and capable to converge the accuracy to the desired amount, which is 93% accuracy for the validation data. 

Since I have a CUDA enable laptop, which enable GPU computing, I decided to explore how much computational performance gain can be achieve by GPU acceleration.  For this reason, I decided to determine the weights and bias using color images instead of grayscale image data to increase the load of hardware computational resources.

The only image data preprocessing needed for this project is image normalization.  The goal is to use image normalization to achieve a zero statistical mean of the image data and equal variance by scaling the data.  There are many techniques that can be used for image normalization, however it was suggested a simple normalization can be used, which is (x - 128) / 128, where x is the pixel value of a color image.  x can be range from 0 - 255.  During the development of the system, it was found that an alternate normalization formula yield a more stable solution and allow the system to converge to a higher validation accuracy.  The formula is (x - 127.5) / 127.5

Here is an example of the image data:

![alt text][image4]

![alt text][image5]

![alt text][image6]

## 2. System Architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					|    Input             |   Output                | 
|:---------------------:|:-------------------------------------:|:---------------------------:| :----------------------:|
| Image Input         		| 32x32x3 RGB image   							|   
| Layer 1: Convolution      	| 1x1 stride, valid padding 	| 32 x 32 x 3             | 28 x 28 x 6             |
| RELU					|	 Activation											|    28 x 28 x 6                    |    28 x 28 x 6              |
| Max pooling	      	| 2x2 stride   			|   28 x 28 x 6  |  14 x 14 x 6             |
| Layer 2: Convolution 3x3	    | 1x1 stride, valid padding      									|   14 x 14 x 6   |   10 x 10 x 16     |
| RELU					|	 Activation											|    10 x 10 x 16                    |    10 x 10 x 16              |
| Max pooling	      	| 2x2 stride   			|   10 x 10 x 16  |  5 x 5 x 16             |
| Flatten  | Flatten matrix for fully connected layer |  5 x 5 x 16  | 400  |
| Layer 3: Fully Connected		| 400 -> 120 features        									|  400  |  120   |
| RELU					|	 Activation											| 120  | 120  |
| Dropout  |  Keep Probability = 50%  |  120  |  120 |
| Layer 4: Fully Connected   |  120 -> 84 features |  120  |  84 |
| RELU					|	 Activation											| 84  | 84  |
| Dropout  |  Keep Probability = 50%  |  84  |  84 |
| Layer 5: Fully Connected   |  84 -> 43 features  |  84  |  43 |

 

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

## 3. Model Training

I first used the LeNet architecture as a base architecture for my model.  Since I can't reach the targeted validation accuracy of 93%, I started to examine the data in order to identify the opportunities of how to improve the LeNet model.  I considered additional convolution layers, patch size, regularization to address overfitting issues, batch size, learning rate, layer depth and data conditioning.  

I used the default optimizer, which is the Adam algorithm to streamline stochastic gradient descent and backpropagation to minimize loss.  Batch size was increased from 128 to 160 in order to improve validation accuracy.  Although computationally more expensive, tensorflow-gpu works well with my GeForce 970M GPU graphics board, therefore hardware constraint was not an issue.  Learning rate was also lowered from 0.001 to 0.0009 in order to make the accuracy progression more stable and achieve higher accuracy.  I used 28 epochs to reach the desired accuracy.  I believe that the epoch and batch number can be lowered if the images are converted to grayscale and better normalization technique were used.  However, the targeted accuracy was achieved, therefore I stopped optimizing my model for the project but I will continue to find better ways to improve the model.


####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

## 4. Architecture Development

My model is based on the LeNet architecture.  When the LeNet architecture was first used, the training and validation accuracy plateau and did not improve.  In order to improve the validation accuracy, I iniatiatly added another Covolution layer before the flatten operation.  It provides a slight improvement but I was able to reach the objective accuracy.  After several attempts to optimize layer depth, batch sizes and training rate, the additional Convolution layer did not provide promising improvement to the model and thus it is abandon.  The training and validation accuracy rates were below 80% and the accuracy between epochs seemed to be unstable.

I experimented on alternative normalization techniques and found that by simply using (x - 127.5) / 127.5 to normalize the data improved the training accuracy, therefore this is the approach I adopted.  Training data converge to high accuracy early and rapidly, however validation accuracy still lacks behind and didn't improve, therefore additional techniques need to be used.  This observation seemed to imply overfitting occurred.  In order to resolve this issue, regularization technique was used, namely dropouts.  I used two dropouts at Layer 4 and 5 with the probability of keeping the data set to 50%.  Different value of dropout probability was used but 50% provided the most optimal value in the iteration.  Validation accuracy was increased however the accuracy took longer to converge and reach the maximum accuracy that my model can provide. Since I have a CUDA enable laptop, so that tensorflow-gpu can be used to train the model quickly for iterative runs, therefore number of epochs need to train the model is not an issue for this architecture.

My final model results were:
* training set accuracy of 99.6%
* test set accuracy of 94.5%
* validation set accuracy of 94.5% (PASS)

###Test a Model on New Images

## Testing of the New Model
####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image7] ![alt text][image8] ![alt text][image9] 
![alt text][image10] ![alt text][image11]


####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| ![alt text][image7]      		| "Speed limit (50km/h)"   									| 
| ![alt text][image8]     			| "Right-of-way at the next intersection" 										|
| ![alt text][image9]					| "Priority road"									|
| ![alt text][image10]      		| "No entry"					 				|
| ![alt text][image11]			| "Children crossing"      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 80%. When there are more test cases, it is expected that the result will be more than 80% since the validation accuracy is 94%.  Nonetheless the test results compare favorably to the accuracy on the validation accuracy.



