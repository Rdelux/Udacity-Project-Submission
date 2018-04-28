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
[image2]: ./images/state.png "state"

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

cost function include both state and control input so that we can also control the magnitude and change rate of input


### Polynomial Fitting and MPC Preprocessing


### Model Predictive Control with Latency



### Timestep Length, Elapsed Duration and Optimization Factors Tuning 











### Results

Manual tuning of the hyperparameters were used in the code implementation.  The proportional coefficient, Kp, was tuned first in order to provide the fundamental restoring steering response to keep the vehicle on the track.  Oscillation and overshoot was observed, therefore the derivative coefficient, Kd, was tuned next.  It is suspected that system bias exist, therefore the integral coefficient was tuned as well.  Once the controller parameters were optimized and successfully control the vehicle to drive around the track, the speed of the vehicle was increased in order to channel my inner Vin Diesel to this project.  By setting the vehicle velocity to 50 mph, the vehicle went off the track during cornering, thus another round of coefficient optimization was done.  The result coefficients required to keep the vehicle on the track at all times and provide a stable trajectory are summarized below: 

| Parameter     |    Value    |
|:--------------|-------------:|
| Vehicle speed            |50 mph        |
| Kp     |    0.1     |
| Kd | 2.0 |
| Ki | 0.0001 |


