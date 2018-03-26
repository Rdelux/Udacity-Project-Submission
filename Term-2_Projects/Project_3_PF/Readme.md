# **Term 2 Project 3: Kidnapped Vehicle - Particle Filter**
Self-Driving Car Engineer Nanodegree Program

The goals / steps of this project are the following:

* Implement a 2-dimensional particle filter in C++ to take observation and control data at each time step
* A map with landmarks and some noisy initial localization information are provided
* The particle filter should localize vehicle position and yaw to within the values specified in the grading code in the simulator
* The particle filter should complete execution within the time of 100 seconds


[//]: # (Image References)

[image1]: ./images/Pass_Overview.png "Pass_Overview"
[image2]: ./images/Pass_closeUp.png "Pass_closeUp"
[image3]: ./images/InAction.png "InAction"

The Rubric Points are listed in this following [link](https://review.udacity.com/#!/rubrics/747/view)   

---

### Particle Filter Code Implementation

The main steps of implementing a particle filter include initializaton, prediction, weight update and resampling.  


In a prediction step, the sigma points are generated based on the state vector, its dimension, process covariance and a design parameter.  The predicted mean and covariance of the state vector are obtained by predicting the sigma point based on the previous time-step.  In the update step, the measurement is predicted and mapped from the predicted sigma points, followed by a state update.  In addition, the root mean squared error (RMSE) is computed by comparing the UKF results and the ground truth.  

In order to implement the UKF algorithm, C++ codes were developed and organized in several .cpp and header files. Some of the important codes for implementation are briefly discussed in the following sections: 

#### Initialization


#### Prediction


#### Weight Update

The weight update step of a particle filter can be further divided into 4 steps.  They are observation transformation, data association, weight update and combining probabilities.


#### Resampling


### Results

![alt text][image2]

