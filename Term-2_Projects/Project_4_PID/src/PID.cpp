#include "PID.h"
#include <algorithm>                // darienmt

using namespace std;

/*
* TODO: Complete the PID class.
*/

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp, double Ki, double Kd) {
    PID::Kp = Kp;                                   // proportional constant
    PID::Ki = Ki;                                   // integrate constant
    PID::Kd = Kd;                                   // derivative constant
    
    p_error = 10.0;                                  // initialization
    i_error = 10.0;
    d_error = 10.0;
    
    p_cte = 0.0;                             // previous cross track error
    counter = 0;
    int_cte = 0.0;
}


void PID::UpdateError(double cte) {
    
    // Proportional Controller
    p_error = cte;
    
    // Differential Controller - change in error
    d_error = cte - p_cte;
    p_cte = cte;
    
    // Integral error - sum of errors over time
      i_error += cte;
    
    int_cte += cte;
    counter++;
}


double PID::TotalError() {
    return p_error * Kp + i_error * Ki + d_error * Kd;                          // How much it will steer
}



