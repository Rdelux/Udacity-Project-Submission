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

The main steps of implementing a particle filter include initializaton, prediction, weight update and resampling.  A simulator was used in this project, which provide a graphical visualization of how the car was driven, the landmark measurements and the particle transition calculation results.  The following figure shows the simulator in action:

![alt text][image3]

In order to implement the particle fileter, C++ codes were developed, which is included in this repo - particle_filter.cpp.  No other files were modified. The major components of the code are briefly discussed in the following sections. 

#### Initialization

In order to initialize the particle filter, GPS data was used.  Since GPS data is inherently noisy, the given position data for the vehicle as well as the standard deviations were provided.  The Gaussian noise was modelled as a normal distribution around the position data it is implement in line 42 to 44 of the code.  The standard deviation for x position, y position and yaw angle is [0.3, 0.3, 0.01].  Considering a 30 cm on both side of the normal distribution curve, 60 particles were assumed to be adequate for correct localization of the vehicle.  Higher number of particles were used but no significant improvement were observed. Particles were created in this step, which can be seen in line 52 to 58 of the code.

#### Prediction

Based on the previous velocity and yaw rate, as well as the standard deviation associated with the Gaussian noise, a prediction step was performed.  In order to avoid division-by-very-small number or by zero, if the yaw rate is less then 0.0001, the motion model that correspond to zero yaw rate was used.  Otherwise a non-zero yaw rate motion model was used.

#### Weight Update

The weight update step of a particle filter can be further divided into 4 steps.  They are observation transformation, data association, weight update and combining probabilities.  
Since the observations were given in the vehicle coordinate system, therefore they need to be transformed into the map coordinate system using homogeneous transformation.  Prior to the transform, in order to reduce the computation burden of considering all the landmarks and to simulate real sensor limitation based on the given sensor range (=50 m), only the landmarks within range were taken into consideration.
After the observation measurements were transformed, a data association was performed.  The nearest neighbour technique was used to find the predicted measurement that is closest to each observed measurement.  The observed measurement was assigned to this specific landmark.
Using the multi-variate Gaussian probability density function, the weight associate with the particles were determined based on the distance between the landmark and the observation, as well as the measurement noise.  By combining all the measurement probabilities, the particle final weight can be determined.

#### Resampling

The final step of localization using a particle filter is resampling.  By resampling, particles with higher weight will be more likely to remain as a candidate for localization.  The resampling wheel algorithm given in Lesson 14, Topic 20 was used to carry out this step.  This is implemented in line 256 to 270 of the code

### Results

Using the modified code and the simulator provided, visualization of how the car is being track by particle can be seen.  The particle filter successfully localize the vehicle within the desired accuracy.  In addition, the particle run within the specified time of 100 seconds.  This is shown in the images below:

![alt text][image2]

This project is completed successfully!


