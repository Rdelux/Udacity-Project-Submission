# **Project 3: Behavioral Cloning** 

---

**Behavioral Cloning Project**

The objectives of this project are as follow:
* Use the simulator provided by Udacity to collect data of human driving behavior
* Build a convolution neural network in Keras that predicts steering angles from image data collected
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/center_2017_11_25_18_25_27_636.jpg "Center Camera View"
[image2]: ./images/left_2017_11_25_18_25_27_636.jpg "Left Camera View"
[image3]: ./images/right_2017_11_25_18_25_27_636.jpg "Right Camera View"
[image4]: ./images/center_2017_11_25_18_49_34_669.jpg "Reverse Course Center Camera View"
[image5]: ./images/left_2017_11_25_18_49_34_669.jpg "Reverse Course Left Camera View"
[image6]: ./images/right_2017_11_25_18_49_34_669.jpg "Reverse Course Right Camera View"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model by reading batched image data using a python generator
* model_readAll.py containing the script to create and train the model by reading all images to the memory at once
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* Project3_writeup.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator, the drive.py file, and the model.h5 train network, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.
The model_readAll.py file contains similar code, however it read and store the preprocessed image data all at once withou using a generator.  It was found that my Alienware Laptop with a GTX 970M video card can train the model much faster than batching the image data using GPU acceleration.  My computer is able to store all the image data in the memory at once, thus minimizing communication resources on the bus, therefore the performance of the code in model_readAll.py is much faster than model.py using the dataset I collected.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I used the NVIDIA convolution neural network (CNN) outlined in the paper "End to End Learning for Self Driving Cars" as the model I used in this project.  The NVIDIA CNN contains five convolution layers after the data is normalized, then flattened and followed by three layers of fully-connected layers.
Using adequate amount of data while focusing on vehicle recovery and data generalization, the original NVIDIA CNN model performed well with my computing device, therefore no modification was made to the NVIDIA CNN.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen for the ultimate goal of keeping the vehicle driving on the road. I used a combination of center lane driving, reverse course driving, recovering from the left and right sides of the road and  repeated cornering around the curves in order to ensure that the vehicle will stay on the road and able to recover when the vehicle start to drift off from the center of the road.  

For details about how I created the training data, please see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use an existing proven CNN architecture designed for image classification or regression and improve the model as needed from the initial results.  A number of factors were considered when designing the overall solution approach, such as memory constraints, GPU acceleration, quality and quantity of image dataset, accuracy of trained model and project requirements.

I first used the LeNet CNN architecture with a moderate dataset in order to determine the areas for improvement.  The result from this initial model is poor, the vehicle went off the road in the first corner and was not able to recover.  While it was expected because there was no training data available for the vehicle to recover from mistakes in the initial dataset, which was developed by me using the Udacity vehicle simulator.  Futher collection of image dataset using the vehicle simulator yields some success but vehicle still drift off to the side and lose control.  Therefore, a more powerful CNN is needed in order to satisfy the project requirement.  Instead of fine tuning the LeNet model, I used the NVIDIA CNN model as a starting point for training an autonomous car steering control model, and I immediately noticed an improvement from the autonomous vehicle driving behavior.  

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I used 20% of the original image and steering angle dataset as validation.  I found that my first model had a monotonically decreasing low mean squared error on both the training set and the validation set, however the MSE on the validation set is slightly higher than the training set, about 20% after 5 epoches. This implied there is a slight amount of overfitting. In order to reduce overfitting, I decided to use early termination to minimize its effect.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
