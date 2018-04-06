# **Term 2 Project 4: PID Controller Project**
Self-Driving Car Engineer Nanodegree Program

The goals / steps of this project are the following:

* Implement a PID controller in C++ to maneuver the vehicle around the lake race track from the Behavioral Cloning Project
* The PID procedure follows what was taught in the lessons
* Calculate the steering angle based on the cross track error (CTR) and the chosen vehicle velocity value
* The car should not pop up onto ledges or roll over any surfaces that would otherwise be considered unsafe


[//]: # (Image References)

[video1]: ./images/Normal_Driving.mov "Normal"


The Rubric Points are listed in this following [link](https://review.udacity.com/#!/rubrics/824/view)   

---

### PID Controller Implementation

Using the simulator provided in the Behavioral Cloning Project, a vehicle is to be maneuvered by a PID controller algorithm, implemented in C++ code.  The PID class is implemented in the PID.cpp and PID.h files.  The simulator provides the cross track error (CTE) to the code and the PID algorithm will compute the steering angle in order to minimize the deviation or error of the vehicle relative to the track.  

#### Initialization

The initialization of the algorithm was implemented in the PID::Init method, where the proportional (Kp), integral (Ki), and derivative (Kd) controller constants were statically initialized.  The errors used to calculate the required steering angle associated with each of the three controller parameters were also initialized in PID.cpp file, code line 14 to 26.  The selection of the controller constants will be discussed in the result section.  A preset vehicle was also chosen based on the criteria specified in the rubric points, namely safety operation and speed limit.

#### Error Calculation

In order to steer the vehicle back to the center of the track, the CTE was calculated and provided by the simulator.  Using the CTE, the steering response can be efficiently described by Kp, Ki and Kd.

For the proportional response, Kp is used to scale the CTE to provide a steering response.  As a result, the higher the CTE, the higher the steering response is.  This characteristic of control allows the vehicle to steer quickly back to the center of the track thus avoiding leaving the track area.  This is particlarly important during high speed cornering.

