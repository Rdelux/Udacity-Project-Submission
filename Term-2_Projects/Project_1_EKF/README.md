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
[image2]: ./images/x_and_P_Output.png "x and P Output"
[image3]: ./images/Sim_output1.png "Simulation Overview DS1"
[image4]: ./images/Sim_output3.png "Simulation Closeup DS1"



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

#### tools.cpp
In this file, two functions are define to support the EKF implementation.  The Jacobian matrix required for RaDAR measurement processing is defined in the CalculateJacobian() method.  The performance of the EKF is defined by the RMSE, and the function CalculateRMSE() define and compute the RMSE values.


### Results
The final RMSE values for Dataset #1 are [0.0973, 0.0855, 0.4513, 0.4399] for [px,py,vx,vy], which is smaller than the target accuracy of [.11, .11, 0.52, 0.52], therefore the result of the EKF implementation is satisfactory.  A sample of the state estimate vector and the uncertainty covariance matrix output is shown below:

![alt text][image2]

The simulation result for Dataset #1 is shown below:

![alt text][image3]

Note that the LiDAR measure is shown in red, the RaDAR measurement is shown in blue and the EKF output is shown in green:

![alt text][image4]





