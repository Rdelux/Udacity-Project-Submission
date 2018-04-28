# **Term 2 Project 5: Model Predictive Control Project**
Self-Driving Car Engineer Nanodegree Program

The goals / steps of this project are the following:

* Implement a MPC controller in C++ to maneuver the vehicle around the lake race track 
* The car should not pop up onto ledges or roll over any surfaces that would otherwise be considered unsafe
* Create a polynomial to fit the waypoints provided
* Find the optimized timestep length, elasped duration and different weighting factors for state control
* The MPC should be able to handle a 100 millisecond of latency

[//]: # (Image References)

[image1]: ./images/MPC_1.png "MPC"
[image2]: ./images/state.png "Update equations"
[image3]: ./images/cte.png "cross track error"
[image4]: ./images/orientationError.png "orientation error"

The Rubric Points are listed in this following [link](https://review.udacity.com/#!/rubrics/896/view)   

---

![alt text][image1]

### The Model

In order to control the vehicle in the simulator to drive itself autonomously around the track, a model predictive control (MPC) approach was used.  This approach essentially turn a control problem into an optimization problem.  The optimization parameters include the state of the vehicle and the actuator constraints.  A kinematic model was used to model the dynamic behavior of the vehicle since it is more tractable than the dynamic model approach and it is simpler to implement.  The update equations from the kinematic model along with the state and actuator constraints made up the "model" that is described in this section.

The state of the vehicle includes the following parameters:

| Variable     |    State    |
|:--------------|-------------:|
| px            | x-position of vehicle in global coordinate       |
| py     |    y-position of vehicle in global coordinate    |
| psi | orientation of vehicle |
| v | current vehicle speed  |

The actuator constraints include:

| Variable     |    Actuator    |
|:--------------|-------------:|
| delta            | steering angle       |
| acc     |    acceleration or throttle    |

The update equations from the kinematic model that describe the next state are listed below:

![alt text][image2]

These are implemented in line 130 - 133 of main.cpp

In addition to the model, one must define the error between the desire state and the current state so that we can control the vehicle to follow a specific path.  Once the error is defined, the optimization algorithm can minimize it to zero, making the vehicle to follow the path as much as possible.  The errors for the model are:

| Variable     |    Error    |
|:--------------|-------------:|
| cte            | cross track error       |
| epsi     |    orientation error    |

The equation for the aforementioned errors are:

![alt text][image3]
![alt text][image4]

These error will be used to define the cost function for optimization and they are implemented in line 134 - 135 of main.cpp

Once the model is defined, the next step is fit a polynormial to define the desire path and preprocess the variables for the MPC algorithm.

### Polynomial Fitting and MPC Preprocessing

The waypoints of the track are given by the lake_track_waypoints.csv file and are incorporated into the code via the data JSON object in line 88 - 89 of main.cpp as ptsx and ptsy.  Since both the position of the waypoints and the vehicle are given in global map coordinate, the first step for MPC preprocessing is to transform the waypoints to the vehicle coordinate system, which include a translational and rotational transform.  These transforms are performed in line 111 - 112 for the translational transform and line 113 - 114 for the rotational transform.

Using the polyfit() function, a third order polynomial curve can be created using the transformed waypoints.  This curve identify the desire path for the vehicle.  

Now that the waypoints are transformed into the vehicle coordinate system, the initial state for the vehicle position and orientation (px, py, psi) can be considered to be zeros (see line 122 to 124). Using the zero-order term of the polynormial function created, the cte can be calculated since it represent the cte at the current vehicle position.  The orientation of the vehicle, or the heading, is the tangential slope value at the current vehicle position, therefore the negative arc tangent of the second term in the polynormial will provide the appropriate value.  These are implemented in line 125 - 126 of main.cpp.

### Model Predictive Control with Latency




### Timestep Length, Elapsed Duration and Optimization Factors Tuning 







Based on the update equations, vehicle state and the actuator inputs, the cost functions can be calculated

cost function include both state and control input so that we can also control the magnitude and change rate of input



### Results

Manual tuning of the hyperparameters were used in the code implementation.  The proportional coefficient, Kp, was tuned first in order to provide the fundamental restoring steering response to keep the vehicle on the track.  Oscillation and overshoot was observed, therefore the derivative coefficient, Kd, was tuned next.  It is suspected that system bias exist, therefore the integral coefficient was tuned as well.  Once the controller parameters were optimized and successfully control the vehicle to drive around the track, the speed of the vehicle was increased in order to channel my inner Vin Diesel to this project.  By setting the vehicle velocity to 50 mph, the vehicle went off the track during cornering, thus another round of coefficient optimization was done.  The result coefficients required to keep the vehicle on the track at all times and provide a stable trajectory are summarized below: 

| Parameter     |    Value    |
|:--------------|-------------:|
| Vehicle speed            |50 mph        |
| Kp     |    0.1     |
| Kd | 2.0 |
| Ki | 0.0001 |


