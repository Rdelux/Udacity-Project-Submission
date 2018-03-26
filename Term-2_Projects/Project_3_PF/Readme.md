# **Term 2 Project 3: Kidnapped Vehicle - Particle Filter**
Self-Driving Car Engineer Nanodegree Program

The goals / steps of this project are the following:

* Implement a 2-dimensional particle filter in C++ to take observation and control data at each time step
* A map with landmarks and some noisy initial localization information are provided
* The particle filter should localize vehicle position and yaw to within the values specified in the grading code in the simulator
* The particle filter should complete execution within the time of 100 seconds


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

### Unscented Kalman Filter Code Implementation

The main steps of an Unscented Kalman filter (UKF) code include a prediction step and an update step.  In a prediction step, the sigma points are generated based on the state vector, its dimension, process covariance and a design parameter.  The predicted mean and covariance of the state vector are obtained by predicting the sigma point based on the previous time-step.  In the update step, the measurement is predicted and mapped from the predicted sigma points, followed by a state update.  In addition, the root mean squared error (RMSE) is computed by comparing the UKF results and the ground truth.  

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
In order to determine the process noise, the standard deviation for the longitudinal acceleration and yaw acceleration need to be estimated.  

The maximum acceleration for a vehicle operating in an urban environment rarely exceed 6 m/s^2 as mentioned in the Lesson 8, Topic 31 of the course.  Since the measurement is from a bicycle, the maximum acceleration is even less.  Therefore, it was decided that half the acceleration of a vehicle will be used for bicycle.  Since the maximum longitudinal acceleration is estimated to be 3 m/s^2, the longitudinal acceleration standard deviation was calculated to be 1.5 and the variance is 2.25.  A rounded number of 2.0 was used initially to start the parameter tuning process.  

In order to estimate the yaw acceleration noise, I started by estimating a centrifugal acceleration of 1g, which is equivalent to 9.81 m/s^2.  If we assumed that the radius of a turn is 9.81 m, then the angular velocity to achieve 1g of centrifugal acceleration is 1 rad/s.  Using this maximum angular velocity and assuming an linear acceleration of 3 m/s^s over a 90 degrees turn with a 9.81 m radius, the maximum yaw acceleration is 0.31 rad/s^2, therefore the standard deviation for yaw acceleration noise can be estimated to be 0.02.  When this value was first tested, poor RMSE was observed and the NIS numbers were too high.  This indicates that we are under-estimating the process noise.  The yaw acceleration noise was increased to 1.0 and the results were satisfactory.

### Results

In order to assess if the process noise parameters were chosen properly, the Normalized Innovation Squared (NIS) was calculated for the Lidar and Radar measurement prediction.  The Chi Squared test was used to ensure that our measurement hypothesis is true with a confidence level of 90%.  For Radar measurement, which has a dimension of 3, the upper and lower bound of the Chi Squared test are 7.815 and 0.352 respectively.  For Lidar measurement, which as a dimension of 2, the upper and lower bound of the Chi Squared test are 5.991 and 0.103 respectively.  Three cases were considered, they were using Lidar measurement only, using Radar measurement only, and using Lidar and Radar measurements.  The result of the NIS values vs time-steps are illustrated in the chart below.  The top chart shows Lidar and Radar measurements used separately, while the bottow chart illustrate the results of using both Lidar and Radar measurements but the results were plotted separately:

![alt text][image7]

It can be seen that both charts indicate the NIS values for both Lidar and Radar measurements are mostly within the required confidence level, therefore they indicated that the noise parameters were chosen correctly.  The following table shows the NIS values that are within the allowable range for 90% confidence level:

| Scenario         		|     Percent of NIS within Range	        					|    
|:---------------------:|:-------------------------------------:|
| Lidar Only         		| 88%  							|   
| Radar Only     	| 90.4% 	|
| Lidar and Radar (Radar Measurement)     	| 89.2% 	|
| Lidar and Radar (Lidar Measurement)     	| 91.6% 	|

The final RMSE values for Dataset #1 using both Lidar and Radar measurements are [0.0702, 0.0839, 0.3407, 0.2457] for [px,py,vx,vy], which is smaller than the target accuracy of [.09, .10, .40, .30] , therefore the result of the UKF implementation is satisfactory.  As a comparison, the RMSE for EKF of the same dataset is [0.0973, 0.0855, 0.4513, 0.4399].  The simulation result for dataset #1 are shown below: (note that the LiDAR measure is shown in red, the RaDAR measurement is shown in blue and the UKF output is shown in green)

![alt text][image5]
![alt text][image6]

Using only Lidar measurement, the RMSE for the same UKF implementation using the same dataset is [0.1086, 0.1009, 0.6201, 0.2836], which is significantly higher than using both Lidar and Radar measurement.  The simulation result is shown below:

![alt text][image1]
![alt text][image2]

Using only Radar measurement, the RMSE for the same UKF implementation using the same dataset is [0.1627, 0.2241, 0.3978, 0.3041], which is significantly higher than using both Lidar and Radar measurement.  In addition, the position estimate is worse than using Lidar measure, which is expected since Lidar is more superior in measuring position than Radar.  Radar, however, provide a better accuracy in x-velocity estimate than Lidar.  The simulation result is shown below:

![alt text][image3]
![alt text][image4]








