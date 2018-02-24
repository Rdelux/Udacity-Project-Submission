# **Term 2 Project 1: Extended Kalman Filter**
Self-Driving Car Engineer Nanodegree Program

The goals / steps of this project are the following:

* Download the Term 2 Simulator
* Set up and install uWebSocketIO for Mac system 
* Implement an Extended Kalman filter in C++ to take LiDAR and RaDAR measurement to track a car's position and velocity
* Achieve RMSE values less than [.11, .11, 0.52, 0.52] for position in x and y, and velocity in x and y respectively
* The Extended Kalman filter algorithm initializes the first measurements
* The Extended Kalman filter algorithm first predicts then updates the vehicle's position and velocity
* The Extended Kalman filter algorithm processes the subsequent LiDAR and RaDAR measurements

[//]: # (Image References)

[image1]: ./images/EKF_Process_Flow.png "EKF Process Flow"
[image2]: ./Images/Test2_Result.png "Test Image Result"

The Rubric Points are listed in this following [link](https://review.udacity.com/#!/rubrics/748/view)   

---

### uWebSocketIO Implementation

uWebSocketIO is a WebSocket and HTTP implementation for web clients and servers.  It is used to facilitates the connection between the simulator and Extended Kalman filter C++ code, which act as the web server.

Since macOS was used in this project, [Homebrew](http://brew.sh) was installed in order to install all the required dependencies.  Using the provided setup script, all the necessary libraries required for uWebSocketIO implementation was installed. 

### Extended Kalman Filter Code Implementation

The main steps an Extended Kalman filter (EKF) code should include the following:
* initializing Kalman filter variables
* predicting the position (x,y) and velocity (vx,vy) of the object after a time step Δt
* updating where the object is based on sensor measurements - LiDAR and RaDAR measurements

In addition, the root mean squared error (RMSE) is computed by comparing the EKF results and the ground truth.  The general process flow of the EKF algorithm is shown below:

![alt text][image1]

In order to implement the EKF algorithm, C++ codes were developed and organized in several .cpp and header files. Some of the important codes for implementation are discussed in the following sections. 

#### main.cpp
The main.cpp code communicates with the simulator through the uWebSocketIO.  The code reads in the LiDAR and RaDar sensor measurement data from the client and pass the information to FusionEKF.cpp and kalman_filter.cpp for Kalman filter processing.  The main.cpp code also calls a function to compare the ground truth and the EKF result to provide the RMSE information.

#### FusionEKF.cpp
Initializations of the kalman filter and the associated variables are performed in this file.  Then the code calls the predict function and the update function.  The code also output the state estimate and the uncertainty covariance at every timestep.  The following EKF elements were initialized:
* Measurement covariance matrices for LiDAR and RaDar (R_laser and R_radar)
* Measurement function matrix for LiDAR (H_laser)
* Jacobian matrix for RaDAR (Hj_ and call tools.cpp to calculate the matrix at every timestep)
* Uncertainty covariance matrix (P_)
* State estimate vector (x_)
* State transition matrix (F_)
* Process covariance matrix (Q_)

RaDAR measurement is converted from Polar coordinates to Cartesian coordinate to generate the state estimate vector.  Timestep is calculated in this file based on data input. Error prevention codes were implemented in order to avoid invalid elements in Jacobian matrix.  This is achieved by assigning a threshold if the position values (px and py) are too small.

#### kalman_filter.cpp
The EKF equations are implemented in this file.  The predict and update functions are define here.  
The state estimate vector and the uncertainty covariance matrix in the Predict() method.  The sensor measurement noise(y) is calculated in the Update() method and it is used to update the state estimate vector and the uncertainty covariance by calling the MSRUpdate method.  The UpdateEKF() method is similar to the Update() method but it handles RaDAR measurement since RaDAR measurement processing required non-linear equation in the Cartesian coordinate space.  For RaDAR measurement, the resulting angle phi in the sensor measurement noise vector(y) is adjusted so that it is between -pi and pi.  It is necessary because atan2() returns values between -pi and pi in C++.  This normalization step is achieved by adding or subtracting 2π from the angle until the desired range is realized.







kalman_filter.cpp- defines the predict function, the update function for lidar, and the update function for radar
tools.cpp- function to calculate RMSE and the Jacobian matrix
The only files you need to modify are FusionEKF.cpp, kalman_filter.cpp, and tools.cpp.
How the Files Relate to Each Other
Here is a brief overview of what happens when you run the code files:

Main.cpp reads in the data and sends a sensor measurement to FusionEKF.cpp
FusionEKF.cpp takes the sensor data and initializes variables and updates variables. The Kalman filter equations are not in this file. FusionEKF.cpp has a variable called ekf_, which is an instance of a KalmanFilter class. The ekf_ will hold the matrix and vector values. You will also use the ekf_ instance to call the predict and update equations.
The KalmanFilter class is defined in kalman_filter.cpp and kalman_filter.h. You will only need to modify 'kalman_filter.cpp', which contains functions for the prediction and update steps.







Note that the programs that need to be written to accomplish the project are src/FusionEKF.cpp, src/FusionEKF.h, kalman_filter.cpp, kalman_filter.h, tools.cpp, and tools.h

The program main.cpp has already been filled out, but feel free to modify it.

Here is the main protcol that main.cpp uses for uWebSocketIO in communicating with the simulator.


INPUT: values provided by the simulator to the c++ program

["sensor_measurement"] => the measurement that the simulator observed (either lidar or radar)


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
   * On windows, you may need to run: `cmake .. -G "Unix Makefiles" && make`
4. Run it: `./ExtendedKF `

## Editor Settings

We've purposefully kept editor configuration files out of this repo in order to
keep it as simple and environment agnostic as possible. However, we recommend
using the following settings:

* indent using spaces
* set tab width to 2 spaces (keeps the matrices in source code aligned)

## Code Style

Please (do your best to) stick to [Google's C++ style guide](https://google.github.io/styleguide/cppguide.html).

## Generating Additional Data

This is optional!

If you'd like to generate your own radar and lidar data, see the
[utilities repo](https://github.com/udacity/CarND-Mercedes-SF-Utilities) for
Matlab scripts that can generate additional data.

## Project Instructions and Rubric

Note: regardless of the changes you make, your project must be buildable using
cmake and make!

More information is only accessible by people who are already enrolled in Term 2
of CarND. If you are enrolled, see [the project resources page](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/382ebfd6-1d55-4487-84a5-b6a5a4ba1e47)
for instructions and the project rubric.

## Hints and Tips!

* You don't have to follow this directory structure, but if you do, your work
  will span all of the .cpp files here. Keep an eye out for TODOs.
* Students have reported rapid expansion of log files when using the term 2 simulator.  This appears to be associated with not being connected to uWebSockets.  If this does occur,  please make sure you are conneted to uWebSockets. The following workaround may also be effective at preventing large log files.

    + create an empty log file
    + remove write permissions so that the simulator can't write to log
 * Please note that the ```Eigen``` library does not initialize ```VectorXd``` or ```MatrixXd``` objects with zeros upon creation.

## Call for IDE Profiles Pull Requests

Help your fellow students!

We decided to create Makefiles with cmake to keep this project as platform
agnostic as possible. Similarly, we omitted IDE profiles in order to we ensure
that students don't feel pressured to use one IDE or another.

However! We'd love to help people get up and running with their IDEs of choice.
If you've created a profile for an IDE that you think other students would
appreciate, we'd love to have you add the requisite profile files and
instructions to ide_profiles/. For example if you wanted to add a VS Code
profile, you'd add:

* /ide_profiles/vscode/.vscode
* /ide_profiles/vscode/README.md

The README should explain what the profile does, how to take advantage of it,
and how to install it.

Regardless of the IDE used, every submitted project must
still be compilable with cmake and make.

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).

