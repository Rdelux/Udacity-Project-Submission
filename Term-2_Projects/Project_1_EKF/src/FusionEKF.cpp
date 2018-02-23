#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/* To do summary:
 1. initialize variables and matrices (x, F, H_laser, H_jacobian, P, etc.)
 2. initialize the Kalman filter position vector with the first sensor measurements
 3. modify the F and Q matrices prior to the prediction step based on the elapsed time between measurements
 4. call the update step for either the lidar or radar sensor measurement. Because the update step for lidar and radar are slightly different, there are different functions for updating lidar and radar.
 */

double threshold = 0.0001;                     // Threshold for preventing error
int timestep = 0;

/*
 * Constructor - FusionEKF
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;              // Set to "False" first to facilitate initialization

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);            // Measurement noise for laser
  R_radar_ = MatrixXd(3, 3);            // Measurement noise for radar
  H_laser_ = MatrixXd(2, 4);            // Measurement function for laser
  Hj_ = MatrixXd(3, 4);                 // Measure function for Radar - Jacobian to be called in tools.cpp

  //measurement covariance matrix - laser (given)
  R_laser_ << 0.0225, 0,
        0, 0.0225;

  //measurement covariance matrix - radar (given)
  R_radar_ << 0.09, 0, 0,
        0, 0.0009, 0,
        0, 0, 0.09;

  /**
  TODO:
    * Finish initializing the FusionEKF.
    * Set the process and measurement noises 
  */
  
    // Initial State covariance matrix - L6T12
  ekf_.P_ = MatrixXd(4, 4);
  ekf_.P_ << 1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1000, 0,
    0, 0, 0, 1000;
    
    H_laser_ << 1,0,0,0,
    0,1,0,0;                            // typical - Laser only have position information ; Radar Hj in tools.cpp

}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {

  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    /**
    TODO:
      * Initialize the state ekf_.x_ with the first measurement.
      * Create the covariance matrix.
      * Remember: you'll need to convert radar from polar to cartesian coordinates.
    */
      
    // first measurement
    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /*
      Convert radar from polar to cartesian coordinates and initialize state.
      */
        double rho = measurement_pack.raw_measurements_[0];                  // range
        double phi = measurement_pack.raw_measurements_[1];                  // bearing
        double rho_dot = measurement_pack.raw_measurements_[2];              // velocity
        
        // Polar to Cartesian
        double x = rho * cos(phi);
        double y = rho * sin(phi);
        double vx = rho_dot * cos(phi);
        double vy = rho_dot * sin(phi);
        
        // Error prevention
        if (x < threshold)
            x = threshold;
        if (y < threshold)
            y = threshold;
        
        ekf_.x_ << x, y, vx , vy;                                           // storge state vector x
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
        // Assume 0 velocity for LiDAR initialization
        ekf_.x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;
    }
      
    // Error prevention for initialization issues
    if (fabs(ekf_.x_(0)) < threshold and fabs(ekf_.x_(1)) < threshold){
        ekf_.x_(0) = threshold;
        ekf_.x_(1) = threshold;
    }
      
    previous_timestamp_ = measurement_pack.timestamp_ ;
    
    is_initialized_ = true;                             // done initializing, no need to predict or update
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  /**
   TODO:
     * Update the state transition matrix F according to the new elapsed time.
      - Time is measured in seconds.
     * Update the process noise covariance matrix.
     * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */
    
    double noise_ax = 9.0;               // Sigma ax is a process covariance variable to describe the variance in x acc
    double noise_ay = 9.0;               // Sigma ay is a process covariance variable to describe the variance in y acc
    
    // Timestep Calculation
    double dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;           // Calculate duration
    previous_timestamp_ = measurement_pack.timestamp_;                          // store time for next calculation
    
    // State transition matrix update
    ekf_.F_ = MatrixXd(4, 4);
    ekf_.F_ << 1, 0, dt, 0,
    0, 1, 0, dt,
    0, 0, 1, 0,
    0, 0, 0, 1;

    // Noise covariance matrix computation
    double dt_2 = dt * dt;                           //dt^2
    double dt_3 = dt_2 * dt;                         //dt^3
    double dt_4 = dt_3 * dt;                         //dt^4
    
    ekf_.Q_ = MatrixXd(4, 4);
    ekf_.Q_ << dt_4 * noise_ax / 4., 0, dt_3 * noise_ax / 2, 0,
    0, dt_4 * noise_ay / 4, 0, dt_3 * noise_ay / 2,
    dt_3 * noise_ax / 2, 0, dt_2 * noise_ax, 0,
    0, dt_3 * noise_ay / 2, 0, dt_2 * noise_ay;

    ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
   TODO:
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
   */

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
        // Radar updates
        ekf_.H_ = tools.CalculateJacobian(ekf_.x_);                             // call method in tools.cpp
        ekf_.R_ = R_radar_;
        ekf_.UpdateEKF(measurement_pack.raw_measurements_);
    } else {
        // Laser updates
        ekf_.H_ = H_laser_;
        ekf_.R_ = R_laser_;
        ekf_.Update(measurement_pack.raw_measurements_);
    }

  timestep += 1;
    
  // print the output
  cout << "Timestep = " << timestep << endl;
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
    cout << "-------------------------" << endl;
}
