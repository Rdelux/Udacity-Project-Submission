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

In order to control the vehicle in the simulator to drive itself autonomously around the track, a model predictive control (MPC) approach was used.  This approach essentially turn a control problem into an optimization problem.  The optimization parameters include the state of the vehicle and the actuator constraints.  A kinematic model was used to model the dynamic behavior of the vehicle since it is more tractable than the dynamic model approach and it is simpler to implement.  Update equations from the kinematic model along with the state and actuator constraints made up the "model" that is described in this section.

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

These are implemented in line 146 - 149 of MPC.cpp

In addition to the model, one must define the error between the desire state and the current state so that we can control the vehicle to follow a specific path.  Once the error is defined, the optimization algorithm can minimize it to zero, making the vehicle to follow the path as much as possible.  The errors for the model are:

| Variable     |    Error    |
|:--------------|-------------:|
| cte            | cross track error       |
| epsi     |    orientation error    |

The equation for the aforementioned errors are:

![alt text][image3]
![alt text][image4]

These error will be used to define the cost function for optimization and they are implemented in line 150 - 151 of MPC.cpp

Once the model is defined, the next step is fit a polynormial to define the desire path and preprocess the variables for the MPC algorithm.

### Polynomial Fitting and MPC Preprocessing

The waypoints of the track are given by the lake_track_waypoints.csv file and are incorporated into the code via the data JSON object in line 88 - 89 of main.cpp as ptsx and ptsy.  Since both the position of the waypoints and the vehicle are given in global map coordinate, the first step for MPC preprocessing is to transform the waypoints to the vehicle coordinate system, which include a translational and rotational transform.  These transforms are performed in line 111 - 112 for the translational transform and line 113 - 114 for the rotational transform.

Using the polyfit() function, a third order polynomial curve can be created using the transformed waypoints.  This curve identify the desire path for the vehicle.  

Now that the waypoints are transformed into the vehicle coordinate system, the initial state for the vehicle position and orientation (px, py, psi) can be considered to be zeros (see line 122 to 124). Using the zero-order term of the polynormial function created, the cte can be calculated since it represent the cte at the current vehicle position.  The orientation of the vehicle, or the heading, is the tangential slope value at the current vehicle position, therefore the negative arc tangent of the second term in the polynormial will provide the appropriate value.  These are implemented in line 125 - 126 of main.cpp.

### Model Predictive Control with Latency

One of the main advantage of using MPC instead of PID controller is that latency caused by actuation and inertia of the vehicle can be taken into account when controlling a vehicle.  In this project, it was assumed that there's a 100 millisecond latency between the control command and the actuation response.  In order to implement latency into the model, the delta t value, dt, was set equal to the latency and the delay state was calculated.  This was implemented in line 130 - 135 of main.cpp, using the same update equations mentioned earlier.  This created a predicted state, 100 millisecond into the future, and this state is fed into the the mpc solver in line 139 and 141 of main.cpp.

### Timestep Length, Elapsed Duration and Optimization Factors Tuning 

Once the model and the algorithm is all defined and set up, the next step is to tune the various hyperparameters in order to achieve the desire behavior from the vehicle simulator.  

The prediction time horizon is an important characteristic that need to be tuned.  In order to do that, the timestep length, dt, and the elapsed duration, N, need to be selected (line 9 & 10 of MPC.cpp).The predicted horizon, T, is the duration over which future predictions are made, therefore N x dt = T.  Since the environment is changing rapidly when driving a vehicle, T should only be a few seconds.  In this case, I aimed to have the vehicle to attain a top speed of 100 km/h, therefore the time horizon should be small since the environment will be changing very fast. The initial T value was set to 0.5 second.  In order to accurately control the vehicle at high speed and to minimize the discretization error, a small dt value was chosen be to 0.05 s, which means N is equal to 10.  While the vehicle was able to smoothly complete several lap in the course, the top speed was only around 70 km/h.  Increasing N to 20 allow the top speed of the vehicle to increase drastically.  However, the vehicle became unstable and crashed.  Instead of increasing N, dt was increased to 0.1 s, and a higher speed was attained without sacraficing stability.

In order to control the vehicle performance around the track, optimization algorithm needs to be used.  An open-source software package, Interior Point OPTimizer or Ipopt, was used to handle a large scale non-linear optimization problem.  Ipopt needs to jacobians and hessians for computation, therefore CppAD was used to perform automatic differentiation.  Weighting factors need to be defined in order to tune the vehicle behaviour.  The criteria for tuning is to be safely drive the vehicle around the track once while attaining the desired top speed, which was set to 100 km/h (this is a self-inflicted goal).  The faster the vehicle is, the more responsive that the actuation needs to be.  However, if the actuations are too responsive, overshooting may happened and the vehicle performance become unstable.  Sets of parameters were tested in a trail-and-error passes, and their the top-speed attained is recorded in the table below:

Initial parameter set:

| Tuning Parameters     | Variable | Value    |
|:--------------|-------------|----------------:|
| Reference CTE           | ref_cte        | 0    |
| Reference Error psi     |    ref_epsi    | 0    |
| Reference Velocity           | ref_v          | 120  |
| CTE Weighting Factor     |    cte_weight    | 1000    |
| psi Error Weighting Factor            | epsi_weight        | 1000    |
| Velocity Weighting Factor      |    v_weight    | 1    |
| Steering Angle Weighting Factor           | delta_weight        | 50    |
| Throttle Weighting Factor      |    acc_weight    | 50    |
| Steering Angle Change Weighting Factor           | delta_change_weight        | 2500000    |
| Throttle Change Weighting Factor      |    acc_change_weight    | 5000    |
| Performance: Stable |   Top Speed: 88 Km/h |

Modified parameter set (still can't attain desired speed):

| Tuning Parameters     | Variable | Value    |
|:--------------|-------------|----------------:|
| Reference CTE           | ref_cte        | 0    |
| Reference Error psi     |    ref_epsi    | 0    |
| Reference Velocity           | ref_v          | 120  |
| CTE Weighting Factor     |    cte_weight    | 1000    |
| psi Error Weighting Factor            | epsi_weight        | 1000    |
| Velocity Weighting Factor      |    v_weight    | 1    |
| Steering Angle Weighting Factor           | delta_weight        | 50    |
| Throttle Weighting Factor      |    acc_weight    | 35    |
| Steering Angle Change Weighting Factor           | delta_change_weight        | 3000000    |
| Throttle Change Weighting Factor      |    acc_change_weight    | 5000    |
| Performance: Stable |   Top Speed: 98 Km/h |

Final parameter set:

| Tuning Parameters     | Variable | Value    |
|:--------------|-------------|----------------:|
| Reference CTE           | ref_cte        | 0    |
| Reference Error psi     |    ref_epsi    | 0    |
| Reference Velocity           | ref_v          | 120  |
| CTE Weighting Factor     |    cte_weight    | 800    |
| psi Error Weighting Factor            | epsi_weight        | 800    |
| Velocity Weighting Factor      |    v_weight    | 1    |
| Steering Angle Weighting Factor           | delta_weight        | 50    |
| Throttle Weighting Factor      |    acc_weight    | 50    |
| Steering Angle Change Weighting Factor           | delta_change_weight        | 3500000    |
| Throttle Change Weighting Factor      |    acc_change_weight    | 4000    |
| Performance: Stable |   Top Speed: 106 Km/h |

Note: I need to specify the library location for ipopt in order the the program to compile.  This was done in the CMakeLists.txt file, which is included in this repo
