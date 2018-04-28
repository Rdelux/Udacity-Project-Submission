# **Term 2 Project 4: PID Controller Project**
Self-Driving Car Engineer Nanodegree Program

The goals / steps of this project are the following:

* Implement a PID controller in C++ to maneuver the vehicle around the lake race track from the Behavioral Cloning Project
* The PID procedure follows what was taught in the lessons
* Calculate the steering angle based on the cross track error (CTR) and the chosen vehicle velocity value
* The car should not pop up onto ledges or roll over any surfaces that would otherwise be considered unsafe


[//]: # (Image References)

[image1]: ./images/High_Speed_Cornering.png "HSC"
[image2]: ./images/Recovering.png "Recover"



The Rubric Points are listed in this following [link](https://review.udacity.com/#!/rubrics/824/view)   

---

### PID Controller Implementation

Using the simulator provided in the Behavioral Cloning Project, a vehicle is to be maneuvered by a PID controller algorithm, implemented in C++ code.  The PID class is implemented in the PID.cpp and PID.h files.  The simulator provides the cross track error (CTE) to the code and the PID algorithm will compute the steering angle in order to minimize the deviation or error of the vehicle relative to the track.  

### Initialization

The initialization of the algorithm was implemented in the PID::Init method, where the proportional (Kp), integral (Ki), and derivative (Kd) controller constants were statically initialized.  The errors used to calculate the required steering angle associated with each of the three controller parameters were also initialized in PID.cpp file, code line 14 to 26.  The selection of the controller constants will be discussed in the result section.  A preset vehicle was also chosen based on the criteria specified in the rubric points, namely safety operation and speed limit.

### Error Calculation

In order to steer the vehicle back to the center of the track, the CTE was calculated and provided by the simulator.  Using the CTE, the steering response can be efficiently described by Kp, Ki and Kd.

#### Proportional Controller

For the proportional response, Kp is used to scale the CTE to provide a steering response.  As a result, the higher the CTE, the higher the steering response is.  This characteristic of control allows the vehicle to steer quickly back to the center of the track thus avoiding leaving the track area.  This is particlarly important during high speed cornering.  The following picture shows that the vehicle is approaching a corner at high speed (54 mph)

![alt text][image1]

As the vehicle deviate from the center of the track by a significant amount, the CTE will be high, which cause the proportional error to be high.  The parameter Kp will control the required steering response to steer the vehicle back to the desired location.  The picture below shows that the vehicle recover from under-steering and prevent the vehicle from leaving the track:

![alt text][image2]

The calculation of the proportional error is implemented in the PID.cpp file, line 32, which is simply equal to the CTE.

#### Derivative Controller

Using the proportional controller, the vehicle response from center track departure can be controlled.  The higher the vehicle speed, the higher the vehicle response needs to be in order for the vehicle to stay on the track.  However, the higher the proportional constant Kp, the more overshoot that the vehicle trajectory would be.  This results in high vehicle oscillation behaviour, which is undesirable for safety, efficiency and comfort concern. As the oscillation became too excessive or unbound, the vehicle would leave the road thus fail the criteria of this project.  The oscillation behaviour of the vehicle can be seen in a video file - Oscillation.mov in the images folder.  In order to control the oscillation, the derivative controller was implemented to address this concern.  The derivative constant, Kd, provides a scaling factor for the CTE rate of change, thus damping out the proportional response.  The derivative error can be calculated by dividing the difference in CTE in two different timestep by the change in time.  Assuming the change is time is equals to 1, the derivative error was calculated in line 35 of the PID.cpp file.  

#### Integral Controller

If there is no drift or bias in the vehicle system, the PD controller will be sufficient to efficiently control the vehicle.  However, every physically will have some bias, therefore it is important to incorporate a integral controller in a controller.  The integral error is a function of time and it is scaled by the integral constant Ki in the code.  The integral is the sum of of CTE over time, which is designed to capture the inherent system bias.  Since the integral error accumulate the CTE over time, it is expected that the scaling constant is small.  This implementation is done in line 39 of the PID.cpp file.

### Results

Manual tuning of the hyperparameters were used in the code implementation.  The proportional coefficient, Kp, was tuned first in order to provide the fundamental restoring steering response to keep the vehicle on the track.  Oscillation and overshoot was observed, therefore the derivative coefficient, Kd, was tuned next.  It is suspected that system bias exist, therefore the integral coefficient was tuned as well.  Once the controller parameters were optimized and successfully control the vehicle to drive around the track, the speed of the vehicle was increased in order to channel my inner Vin Diesel to this project.  By setting the vehicle velocity to 50 mph, the vehicle went off the track during cornering, thus another round of coefficient optimization was done.  The result coefficients required to keep the vehicle on the track at all times and provide a stable trajectory are summarized below: 

| Parameter     |    Value    |
|:--------------|-------------:|
| Vehicle speed            |50 mph        |
| Kp     |    0.1     |
| Kd | 2.0 |
| Ki | 0.0001 |

Going a step further, the speed was subsequently increased to 60 mph and the vehicle behaviour became marginally stable.  The vehicle was able to stay on the track for one lap, however it eventually went off the track after an extended period of time of testing.  At 70 mph, the vehicle exhibited high oscillation behaviour and the vehicle went off the track and flipped.  At higher speed, the current model under-steer thus the vehicle does not response fast enough at sharp corners.  In order to combat this effect, a higher Kp was chosen.  As Kp becomes higher, Kd needs to be higher too in order to reduce oscillation and overshooting.  Ki remains unchanged assuming system bias is independent of vehicle speed.  The following parameters allows the vehicle to stay on the track even after long period of autonomous driving even though elevated oscillation behaviour persist.

| Parameter     |    Value    |
|:--------------|-------------:|
| Vehicle speed            |70 mph        |
| Kp     |    0.13     |
| Kd | 4.0 |
| Ki | 0.0001 |

The resultant vehicle behaviour at 50 mph can be seen in the video Normal_Driving.mov in the images folder.

