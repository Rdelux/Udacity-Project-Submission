# **Term 2 Project 4: PID Controller Project**
Self-Driving Car Engineer Nanodegree Program

The goals / steps of this project are the following:

* Implement a PID controller in C++ to maneuver the vehicle around the lake race track from the Behavioral Cloning Project
* The PID procedure follows what was taught in the lessons
* Calculate the steering angle based on the cross track error (CTR) and the chosen vehicle velocity value
* The car should not pop up onto ledges or roll over any surfaces that would otherwise be considered unsafe


[//]: # (Image References)

[image1]: ./images/Pass_Overview.png "Pass_Overview"
[image2]: ./images/Pass_closeUp.png "Pass_closeUp"
[image3]: ./images/InAction.png "InAction"

The Rubric Points are listed in this following [link](https://review.udacity.com/#!/rubrics/824/view)   

---

### Particle Filter Code Implementation

The main steps of implementing a particle filter include initializaton, prediction, weight update and resampling.  A simulator was used in this project, which provide a graphical visualization of how the car was driven, the landmark measurements and the particle transition calculation results.  The following figure shows the simulator in action:

![alt text][image3]

In order to implement the particle fileter, C++ codes were developed, which is included in this repo - particle_filter.cpp.  No other files were modified. The major components of the code are briefly discussed in the following sections. 

#### Initialization

In order to initialize the particle filter, GPS data was used.  Since GPS data is inherently noisy, the given position data for the vehicle as well as the standard deviations were provided.  The Gaussian noise was modelled as a normal distribution around the position data it is implement in line 42 to 44 of the code.  The standard deviation for x position, y position and yaw angle is [0.3, 0.3, 0.01].  Considering a 30 cm on both side of the normal distribution curve, 60 particles were assumed to be adequate for correct localization of the vehicle.  Higher number of particles were used but no significant improvement were observed. Particles were created in this step, which can be seen in line 52 to 58 of the code.

