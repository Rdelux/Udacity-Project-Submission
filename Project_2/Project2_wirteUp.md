## **Traffic Sign Classifier - by Richard Lee** 

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


## 3. Model Training

I first used the LeNet architecture as a base architecture for my model.  Since I can't reach the targeted validation accuracy of 93%, I started to examine the data in order to identify the opportunities of how to improve the LeNet model.  I considered additional convolution layers, patch size, regularization to address overfitting issues, batch size, learning rate, layer depth and data conditioning.  

I used the default optimizer, which is the Adam algorithm to streamline stochastic gradient descent and backpropagation to minimize loss.  Batch size was increased from 128 to 160 in order to improve validation accuracy.  Although computationally more expensive, tensorflow-gpu works well with my GeForce 970M GPU graphics board, therefore hardware constraint was not an issue.  Learning rate was also lowered from 0.001 to 0.0009 in order to make the accuracy progression more stable and achieve higher accuracy.  I used 20 epochs to reach the desired accuracy.  I believe that the epoch and batch number can be lowered if the images are converted to grayscale and better normalization technique were used.  However, the targeted accuracy was achieved, therefore I stopped optimizing my model for the project but I will continue to find better ways to improve the model in the future.


## 4. Architecture Development

My model is based on the LeNet architecture.  When the LeNet architecture was first used, the training and validation accuracy plateau and did not improve.  In order to improve the validation accuracy, I iniatiatly added another Covolution layer before the flatten operation.  It provides a slight improvement but I was able to reach the objective accuracy.  After several attempts to optimize layer depth, batch sizes and training rate, the additional Convolution layer did not provide promising improvement to the model and thus it is abandon.  The training and validation accuracy rates were below 80% and the accuracy between epochs seemed to be unstable.

I experimented on alternative normalization techniques and found that by simply using (x - 127.5) / 127.5 to normalize the data improved the training accuracy, therefore this is the approach I adopted.  Training data converge to high accuracy early and rapidly, however validation accuracy still lacks behind and didn't improve, therefore additional techniques need to be used.  This observation seemed to imply overfitting occurred.  In order to resolve this issue, regularization technique was used, namely dropouts.  I used two dropouts at Layer 4 and 5 with the probability of keeping the data set to 50%.  Different value of dropout probability was used but 50% provided the most optimal value in the iteration.  Validation accuracy was increased however the accuracy took longer to converge and reach the maximum accuracy that my model can provide. Since I have a CUDA enable laptop, so that tensorflow-gpu can be used to train the model quickly for iterative runs, therefore number of epochs need to train the model is not an issue for this architecture.

My final model results were:
* training set accuracy of 99.3%
* test set accuracy of 93.3%
* validation set accuracy of 93.7% (PASS)

###Test a Model on New Images

## Testing of the New Model
####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

## 1. Random German Traffic Signs

Five German traffic signs were found on the web in order to feed into the model for further testing of the model:

![alt text][image7] ![alt text][image8] ![alt text][image9] 
![alt text][image10] ![alt text][image11]

These images have different dimensions and resolutions when they were first downloaded.  I cropped the area on the image where the signs is located in each image.  Using an online tool, I reduced the image resolution to 32 x 32, which is demanded by the current model architecture.  Although the distortion of the images are minimized from the cropping, the resolution of the images were significantly reduced.  This will help to prove the robustness of the current model I developed.

## 2. Prediction Results

The code for making predictions using my final model is located in the 10th cell of the Jupyter notebook. And the code for analyzing the overall accuracy of the prediction is located in the 11th cell of the Jupyter notebook. Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| ![alt text][image7]      		| "Speed limit (50km/h)"   									| 
| ![alt text][image8]     			| "Right-of-way at the next intersection" 										|
| ![alt text][image9]					| "Priority road"									|
| ![alt text][image10]      		| "No entry"					 				|
| ![alt text][image11]			| "Children crossing"      							|

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%, it's perfect! When there are more test cases, it is expected that the result will be lowered since the validation accuracy is 94%.  Nonetheless the test results compare favorably to the accuracy on the validation accuracy.

## 3. Top 5 Softmax Probabilities

The code for determining the top 5 softmax probabilities for each of the 5 images I downloaded is located in the 12th cell of the Jupyter notebook.  The model predicts the traffic sign very well with high certainty for all cases.  Here are the result summary:

| Image			        |     Probability	        					|   Description of Traffic Sign    |
|:-----------------:|:---------------------------------:|:---------------------------------|
|  ![alt text][image7]  |   84.0%   |   "Speed limit (50km/h)"   |
|  ![alt text][image7]  |   15.8%   |   "Speed limit (30km/h)"   |
|  ![alt text][image7]  |   0.1%    |   "Speed limit (80km/h)"   |
|  ![alt text][image7]  |   0.1%    |   "Speed limit (70km/h)"   |
|  ![alt text][image7]  |   0.0%    |   "Speed limit (100km/h)"   |
|   |   |   |
|  ![alt text][image8]  |   96.9%   |   "Right-of-way at the next intersection"   |
|  ![alt text][image8]  |   3.0%   |   "Beware of ice/snow"   |
|  ![alt text][image8]  |   0.0%    |   "Double curve"   |
|  ![alt text][image8]  |   0.0%    |   "Pedestrians"   |
|  ![alt text][image8]  |   0.0%    |   "Children crossing"   |
|   |   |   |
|  ![alt text][image9]  |   99.9%   |   "Priority road"   |
|  ![alt text][image9]  |   0.0%   |   "No entry"   |
|  ![alt text][image9]  |   0.0%    |   "No passing for vehicles over 3.5 metric tons"   |
|  ![alt text][image9]  |   0.0%    |   "End of all speed and passing limits"   |
|  ![alt text][image9]  |   0.0%    |   "Stop"   |
|   |   |   |
|  ![alt text][image10]  |   99.1%   |   "No entry"   |
|  ![alt text][image10]  |   0.9%   |   "Stop"   |
|  ![alt text][image10]  |   0.0%    |   "Priority road"   |
|  ![alt text][image10]  |   0.0%    |   "Speed limit (20km/h)"   |
|  ![alt text][image10]  |   0.0%    |   "Speed limit (70km/h)"   |
|   |   |   |
|  ![alt text][image11]  |   63.1%   |   "Children crossing"   |
|  ![alt text][image11]  |   23.4%   |   "Bicycles crossing"   |
|  ![alt text][image11]  |   3.8%    |   "Road narrows on the right"   |
|  ![alt text][image11]  |   2.4%    |   "Wild animals crossing"   |
|  ![alt text][image11]  |   2.1%    |   "Beware of ice/snow"   |

The first image is described as "Speed limit (50km/h)".  The model predicted it well, however the "3" and the "5" can be quite similar in different lighting condition and envirnomental factors, therefore there is a 15.8% that the sign can be a 30km/h sign.

The second image is described as "Right-of-way at the next intersection".  The model has a high confidence of 96.9% of the correct sign description.

The third image is described as "Priotity road".  The distintiveness and simplicity of the sign makes the model to be able to identify the sign easily with a confidence level of 99.9%.  This is also true for the fourth image, which is described as "No entry".

The fifth image is described as "Children crossing".  This image is more complicated and it involves picture-like qualities and not just symbolic signs, therefore the prediction of the this sign is more difficult.  However, my model predicted the sign with a confidence level of 63.1%, which is enough to ensure the accuracy of the prediction.
