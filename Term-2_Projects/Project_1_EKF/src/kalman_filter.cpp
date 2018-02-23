#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;                                                                    // State vector
  P_ = P_in;                                                                    // State Covariance matrix
  F_ = F_in;                                                                    // State Transition matrix
  H_ = H_in;                                                                    // Measurement function
  R_ = R_in;                                                                    // Measurement Covariance matrix
  Q_ = Q_in;                                                                    // Process Covariance matrix
}

void KalmanFilter::Predict() {
  /**
  TODO:
    * predict the state
  */
    
    x_ = F_ * x_ ;                                          // There is no external motion, no "+u"
    MatrixXd Ft = F_.transpose();
    P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Kalman Filter equations
  */
    VectorXd y = z - H_ * x_;                               // measurement error calculation ; z is measurement
    MSRUpdate(y);                                           // measurement update using y
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Extended Kalman Filter equations
  */
    // convert cartesian state back to radar space
    double rho = sqrt(x_(0)*x_(0) + x_(1)*x_(1));
    double theta = atan2(x_(1), x_(0));
    double rho_dot = (x_(0)*x_(2) + x_(1)*x_(3)) / rho;
    
    VectorXd h = VectorXd(3);                               // h'(x) for radar
    h << rho, theta, rho_dot;
    
    VectorXd y = z - h;                                     // Measurement error/noise
    
    // Normalizing angle - adjust angle so that it is between -pi and pi
    while ( y(1) > M_PI || y(1) < -M_PI ) {
        if ( y(1) > M_PI ) {
            y(1) -= M_PI;
        } else {
            y(1) += M_PI;
        }
    }
    MSRUpdate(y);                                             // Linearlized; call measurement update similar to Laser
}

// update state using Kalman Filter
void KalmanFilter::MSRUpdate(const VectorXd &y){
    MatrixXd Ht = H_.transpose();                               // Measurement function H transpose
    MatrixXd S = H_ * P_ * Ht + R_;                             // State covariance map to measurement space
    MatrixXd Si = S.inverse();                                  // Inverse of S
    MatrixXd K =  P_ * Ht * Si;                                 // Kalman gain
    
    // Next state
    x_ = x_ + (K * y);                                          // New state vector
    long x_size = x_.size();
    MatrixXd I = MatrixXd::Identity(x_size, x_size);
    P_ = (I - K * H_) * P_;                                     // New state covariance matrix
}
