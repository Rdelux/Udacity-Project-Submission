# **Term 2 Project 2: Unscented Kalman Filter**
Self-Driving Car Engineer Nanodegree Program

The goals / steps of this project are the following:

* Implement an Unscented Kalman filter in C++ to take LiDAR and RaDAR measurement to track a car's position and velocity
* Achieve RMSE values less than [.09, .10, .40, .30] for position in x and y, and velocity in x and y respectively
* The Unscented Kalman filter algorithm initializes the first measurements
* Upon receiving a measurement after the first, the algorithm should predict object position to the current timestep and then update the prediction using the new measurement.
* The algorithm should set up the appropriate matrices given the type of measurement and calls the correct measurement function for a given sensor type
* It must be possible to run the project in three different modes: considering laser only, with considering radar only, or with using both sensors.
* For every mode, the overall RMSE (2d position only) may not be more than 10% increased to what the original solution is able to reach (this number depends on the individual measurement sequence)
* The RMSE of laser AND radar must be lower than radar only or laser only
* The NIS of radar measurements must be between 0.35 and 7.81 in at least 80% of all radar update steps.

[//]: # (Image References)

[image1]: ./images/L_Only_Overview.png "L_Only_Overview"
[image2]: ./images/L_Only_closeup.png "L_Only_closeup"
[image3]: ./images/R_Only_Overview.png "R_Only_Overview"
[image4]: ./images/R_Only_closeup.png "R_Only_closeup"
[image5]: ./images/R_and_L_Overview.png "R_and_L_Overview"
[image6]: ./images/R_and_L_closeup.png "R_and_L_closeup"





The Rubric Points are listed in this following [link](https://review.udacity.com/#!/rubrics/783/view)   

---

### uWebSocketIO Implementation

uWebSocketIO is a WebSocket and HTTP implementation for web clients and servers.  It is used to facilitates the connection between the simulator and Unscented Kalman filter C++ code, which act as the web server.

Since macOS was used in this project, [Homebrew](http://brew.sh) was installed in order to install all the required dependencies.  Using the provided setup script, all the necessary libraries required for uWebSocketIO implementation was installed. 

### Unscented Kalman Filter Code Implementation

The main steps of an Unscented Kalman filter (EKF) code include a prediction step and an update step.  In a prediction step, the sigma points are generated based on the state vector and its dimension, process covariance and a design parameter.  The predicted mean and covariance of the state vector are obtained by predicting the sigma point based on the previous time-step.  In the update step, the measurement is predicted and mapped from the predicted sigma points, followed by a state update.  In addition, the root mean squared error (RMSE) is computed by comparing the UKF results and the ground truth.  

In order to implement the UKF algorithm, C++ codes were developed and organized in several .cpp and header files. Some of the important codes for implementation are briefly discussed in the following sections: 

#### main.cpp
The main.cpp file reads in data, calls a function to run the Unscented Kalman filter and calls a function to calculate RMSE

#### ukf.cpp
This file initializes the Unscented Kalman filter, calls the predict and update function, defines the predict and update functions. The Normalized Innovation Square (NIS) is calculated in order perform the Chi square test to assess the prediction accuracy.

#### tools.cpp
The performance of the UKF is defined by the RMSE. The function CalculateRMSE() defines and compute the RMSE values.

Three modes of the algorithm were executed in order to compare the RMSE and NIS values.  These modes are:
* Lidar measurement only
* Radar measurement only
* Lidar and Radar measurement together

### Results
The final RMSE values for Dataset #1 are [0.0973, 0.0855, 0.4513, 0.4399] for [px,py,vx,vy], which is smaller than the target accuracy of [.11, .11, 0.52, 0.52], therefore the result of the EKF implementation is satisfactory.  A sample of the state estimate vector and the uncertainty covariance matrix output is shown below:

![alt text][image2]

The simulation result for Dataset #1 is shown below:

![alt text][image3]

Note that the LiDAR measure is shown in red, the RaDAR measurement is shown in blue and the EKF output is shown in green:

![alt text][image4]




# Unscented Kalman Filter Project Starter Code
Self-Driving Car Engineer Nanodegree Program

In this project utilize an Unscented Kalman Filter to estimate the state of a moving object of interest with noisy lidar and radar measurements. Passing the project requires obtaining RMSE values that are lower that the tolerance outlined in the project rubric. 

This project involves the Term 2 Simulator which can be downloaded [here](https://github.com/udacity/self-driving-car-sim/releases)

This repository includes two files that can be used to set up and intall [uWebSocketIO](https://github.com/uWebSockets/uWebSockets) for either Linux or Mac systems. For windows you can use either Docker, VMware, or even [Windows 10 Bash on Ubuntu](https://www.howtogeek.com/249966/how-to-install-and-use-the-linux-bash-shell-on-windows-10/) to install uWebSocketIO. Please see [this concept in the classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/16cf4a78-4fc7-49e1-8621-3450ca938b77) for the required version and installation scripts.

Once the install for uWebSocketIO is complete, the main program can be built and ran by doing the following from the project top directory.

1. mkdir build
2. cd build
3. cmake ..
4. make
5. ./UnscentedKF

Tips for setting up your environment can be found [here](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/23d376c7-0195-4276-bdf0-e02f1f3c665d)

Note that the programs that need to be written to accomplish the project are src/ukf.cpp, src/ukf.h, tools.cpp, and tools.h

The program main.cpp has already been filled out, but feel free to modify it.

Here is the main protcol that main.cpp uses for uWebSocketIO in communicating with the simulator.


INPUT: values provided by the simulator to the c++ program

["sensor_measurement"] => the measurment that the simulator observed (either lidar or radar)


OUTPUT: values provided by the c++ program to the simulator

["estimate_x"] <= kalman filter estimated position x
["estimate_y"] <= kalman filter estimated position y
["rmse_x"]
["rmse_y"]
["rmse_vx"]
["rmse_vy"]

---

## Other Important Dependencies
* cmake >= 3.5
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./UnscentedKF` Previous versions use i/o from text files.  The current state uses i/o
from the simulator.

## Editor Settings

We've purposefully kept editor configuration files out of this repo in order to
keep it as simple and environment agnostic as possible. However, we recommend
using the following settings:

* indent using spaces
* set tab width to 2 spaces (keeps the matrices in source code aligned)

## Code Style

Please stick to [Google's C++ style guide](https://google.github.io/styleguide/cppguide.html) as much as possible.

## Generating Additional Data

This is optional!

If you'd like to generate your own radar and lidar data, see the
[utilities repo](https://github.com/udacity/CarND-Mercedes-SF-Utilities) for
Matlab scripts that can generate additional data.

## Project Instructions and Rubric

This information is only accessible by people who are already enrolled in Term 2
of CarND. If you are enrolled, see [the project page](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/c3eb3583-17b2-4d83-abf7-d852ae1b9fff/concepts/f437b8b0-f2d8-43b0-9662-72ac4e4029c1)
for instructions and the project rubric.

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).

