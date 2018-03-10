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
[image7]: ./images/NIS_Values.png "NIS_Values"


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
This file initializes the Unscented Kalman filter, calls the predict and update function, defines the predict and update functions. The Normalized Innovation Square (NIS) is calculated in order perform the Chi square test to assess the prediction accuracy.  Although measurement noise values were given, the process noise values need to be tuned.

#### tools.cpp
The performance of the UKF is defined by the RMSE. The function CalculateRMSE() defines and compute the RMSE values.

Three modes of the algorithm were executed in order to compare the RMSE and NIS values.  These modes are:
* Lidar measurement only
* Radar measurement only
* Lidar and Radar measurement together

#### Process Noise Standard Deviation Parameter Tuning
In order to determine the process noise, the standard deviation for the longitudinal acceleration and yaw acceleration need to be estimated.  The maximum acceleration for a vehicle operating in an urban environment rarely exceed 6 m/s^2 as mentioned in the Lesson 8, Topic 31 of the course.  Since the measurement is from a bicycle, the maximum acceleration is even less.  Therefore, it was decided that half the acceleration of a vehicle will be used for bicycle.  Since the maximum longitudinal acceleration is estimated to be 3 m/s^2, the longitudinal acceleration standard deviation was calculated to be 1.5 and the variance is 2.25.  A rounded number of 2.0 was used initially to start the parameter tuning process.  


### Results



The final RMSE values for Dataset #1 are [0.0973, 0.0855, 0.4513, 0.4399] for [px,py,vx,vy], which is smaller than the target accuracy of [.11, .11, 0.52, 0.52], therefore the result of the EKF implementation is satisfactory.  A sample of the state estimate vector and the uncertainty covariance matrix output is shown below:

![alt text][image2]

The simulation result for Dataset #1 is shown below:

![alt text][image3]

Note that the LiDAR measure is shown in red, the RaDAR measurement is shown in blue and the EKF output is shown in green:

![alt text][image4]






