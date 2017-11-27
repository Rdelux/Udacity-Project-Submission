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
[image7]: ./images/NVIDIA.png "NVIDIA Autonomous Car CNN Model"
[image8]: ./images/center_2017_11_25_18_51_09_020.jpg "Recovery1"
[image9]: ./images/center_2017_11_25_18_51_09_165.jpg "Recovery2"
[image10]: ./images/center_2017_11_25_18_51_09_324.jpg "Recovery3"
[image11]: ./images/center_2017_11_25_18_51_09_460.jpg "Recovery4"
[image12]: ./images/center_2017_11_25_18_51_09_618.jpg "Recovery5"


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

The model uses early termination to reduce the overfitting issue.  Since the results were acceptable, no further action was taken to address overfitting, such as adding a dropout layer.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 20 in the model.py version and line 67 in the model_readAll.py version). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 85;model_readAll.py line 66).

#### 4. Appropriate training data

Training data was chosen for the ultimate goal of keeping the vehicle driving on the road. I used a combination of center lane driving, reverse course driving, recovering from the left and right sides of the road and  repeated cornering around the curves in order to ensure that the vehicle will stay on the road and able to recover when the vehicle start to drift off from the center of the road.  

For details about how I created the training data, please see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use an existing proven CNN architecture designed for image classification or regression and improve the model as needed from the initial results.  A number of factors were considered when designing the overall solution approach, such as memory constraints, GPU acceleration, quality and quantity of image dataset, accuracy of trained model and project requirements.

I first used the LeNet CNN architecture with a moderate dataset in order to determine the areas for improvement.  The result from this initial model is poor, the vehicle went off the road in the first corner and was not able to recover.  While it was expected because there was no training data available for the vehicle to recover from mistakes in the initial dataset, which was developed by me using the Udacity vehicle simulator.  Futher collection of image dataset using the vehicle simulator yields some success but vehicle still drift off to the side and lose control.  Therefore, a more powerful CNN is needed in order to satisfy the project requirement.  Instead of fine tuning the LeNet model, I used the NVIDIA CNN model as a starting point for training an autonomous car steering control model, and I immediately noticed an improvement from the autonomous vehicle driving behavior.  

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I used 20% of the original image and steering angle dataset as validation.  I found that my first model had a monotonically decreasing low mean squared error on both the training set and the validation set, however the MSE on the validation set is slightly higher than the training set, a 1% difference after 5 epoches. This implied there is a slight amount of overfitting. In order to reduce overfitting, I decided to use early termination to minimize its effect.

Initial run using the trained model in autonomous mode showed that vehicle was able to execute the first few turns.  However, the vehicle did went off the track and got stuck on the side of the road.  Since the project requires that the car stay on the road 100% of the time, it is important to train the vehicle to have better recovery characteristics.  The MSE remains small for the training and validation set with a training accuracy of 97.96% and a validation accuracy of 97.43%.  Therefore it was concluded that I should improve the training image and steering angle data by providing the model with more appropriate information to combat the cornering issue.

After collection more data, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 69-80) consisted of a convolution neural network with the following layers and layer sizes:

| Layer         		|     Description	        					|    
|:---------------------:|:-------------------------------------:|
| Image Input         	| 160 x 320 x 3 RGB image   |
| Image Cropping        | 65 x 320 x 3 RGB image    |
| Layer 1: Convolution  | 31 x 158 x 24             |
| RELU					        |	 Activation								|
| Layer 2: Convolution  | 14 x 77 x 36              |
| RELU					        |	 Activation								|
| Layer 3: Convolution  | 5 x 37 x 48               |
| RELU					        |	 Activation								|
| Layer 4: Convolution  | 1 x 35 x 64               |
| RELU					        |	 Activation								|
| Layer 5: Convolution  | 1 x 33 x 64               |
| Flatten               |     2112 x 1              |
| Fully Connected 1     |     100                   |
| Fully Connected 2     |     50                    |
| Fully Connected 3     |     10                    |
| Fully Connected 4     |     1                     |

Here is a reference to the NVIDIA CNN architecture [[Ref 1]](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf):

![alt text][image7]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one focusing on driving in the middle of the lane. Here is an example image of center lane driving:

![alt text][image1]

In addition, I would like the model to be able to steer the vehicle back to the center when it is drifted off to the left or right side, and instead of data augmentation, I used the images from the left and right camera as well:

![alt text][image2]
![alt text][image3]

I order to generalize the dataset, I then drove the vehicle in a reversed course.  This could have been done by using mirror image of the collected data but I would like to collect more diverse data, therefore I chose to collect more data in the simulator.  The images of the center, left and right camera view in a reverse course can be seen here:

![alt text][image4]
![alt text][image5]
![alt text][image6]

In addition, I recorded the vehicle recovering from the left side and right sides of the road back to center numerous times in order to improve the recovery characteristics.  I steer the vehicle so that it approaches the side of the road, and then I started recording the images while steering the vehicle back to the center.  Here is a series of images where I drifted off to the left and then I recovered back to the center lane:

![alt text][image8]
![alt text][image9]
![alt text][image10]
![alt text][image11]
![alt text][image12]

Futhermore, I would like to improve the cornering characteristics of the vehicle, so I focused on gathering more data in just the curved part of the track.

I experimented on flipped, angled, mirrored images, however I found the biggest improvement of autonomous driving behavior came from providing the right data that are generalized enough to train the model.

After the collection process, I had 19099 number of data points from three camera images.  Validation is being done on 4775 samples, which represent 20% of the entire dataset.  I used 3 epoches to train the model since any more epoch will cause the validation loss to increase.

I used two approach to run the training and validation set to train the model: one approach is to import and process all the image and steering data, while the other approach is to use python generator to read the training and validation data as needed in batches.  Using my own hardware, I found that the first approach is much faster.  The training time for reading in all the data at once is 298s , while it took 11,658s by using a generator approach.
