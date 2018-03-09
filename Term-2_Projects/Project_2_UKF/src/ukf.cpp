#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2 (May need to change)
  // std_a_ = 30;
  // For a max longitudinal acceleration of 12 m/s^2
    std_a_ = 2.0;

  // Process noise standard deviation yaw acceleration in rad/s^2 (May need to change)
  // std_yawdd_ = 30;
    std_yawdd_ = 1.0;
  
  //DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.***********************************
  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  //DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.************************************
  
  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
    P_ << 1, 0, 0, 0, 0,
    0, 1, 0, 0, 0,
    0, 0, 1, 0, 0,
    0, 0, 0, 1, 0,
    0, 0, 0, 0, 1;
  
    // State Dimension
    n_x_ = 5;
    
    // Lambda
    lambda_ = 3 - n_x_;
    
    // Augmentation State Dimension
    n_aug_ = n_x_ + 2;
    
    // Number of Sigma Points
    n_sig_ = 2 * n_aug_ + 1;
    
    // Weight Initialization - L8T30
    weights_ = VectorXd(n_sig_);
    weights_(0) = lambda_ / (lambda_ + n_aug_);
    for (int i = 1; i < n_sig_; i++) {
        weights_(i) = 0.5 / (lambda_ + n_aug_);
    }
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
    
    if (!is_initialized_) {
        double px = 0;                                                          // x-position in Cartesian
        double py = 0;                                                          // y-position in Cartesian
        if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
            double rho = meas_package.raw_measurements_[0];                     // range in polar
            double phi = meas_package.raw_measurements_[1];                     // bearing in polar
            double rho_dot = meas_package.raw_measurements_[2];                 // velocity in polar
            
            px = rho * cos(phi);                                                // x-position in Cartesian
            py = rho * sin(phi);                                                // y-position in Cartesian
            double vx = rho_dot * cos(phi);                                     // x-velocity in Cartesian
            double vy = rho_dot * sin(phi);                                     // y-velocity in Cartesian
            double v = sqrt(vx * vx + vy * vy);                                 // velocity magnitude
            
            x_ << px, py, v, 0, 0;
        } else {
            px = meas_package.raw_measurements_[0];                             // Lidar x position
            py = meas_package.raw_measurements_[1];                             // Lidar y position
            x_ << px, py, 0, 0, 0;                                              // Lidar measurement
        }
        
        time_us_ = meas_package.timestamp_ ;                                    //  Timestamp in seconds
        
        is_initialized_ = true;                                                 // done initializing, no need to predict or update
        
        return;
    }
    
    double dt = (meas_package.timestamp_ - time_us_) / 1000000.0;                // Calculate duration dt
    time_us_ = meas_package.timestamp_;
    
    Prediction(dt);                                                              // Prediction step
    
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
        UpdateRadar(meas_package);
    }
    if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_) {
        UpdateLidar(meas_package);
    }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */
    // Generate sigma points *****************************************************************************
    
    // Augmented mean vector
    VectorXd x_aug = VectorXd(n_aug_);
    x_aug.head(5) = x_;                                                             // mean
    x_aug(5) = 0;                                                                   // acceleration noise
    x_aug(6) = 0;                                                                   // Yaw acceleration noise
    
    // Augmented state covariance
    MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
    P_aug.fill(0.0);
    P_aug.topLeftCorner(n_x_,n_x_) = P_;
    P_aug(5,5) = std_a_ * std_a_;
    P_aug(6,6) = std_yawdd_ * std_yawdd_;
    
    // Sigma points - L8T18
    MatrixXd Xsig_aug = GenerateSigmaPoints(x_aug, P_aug, lambda_, n_sig_);
    
    // Predict Sigma Points *****************************************************************************
    
    Xsig_pred_ = SigmaPointPrediction(Xsig_aug, delta_t, n_x_, n_sig_, std_a_, std_yawdd_);
    
    // Predict Mean and Covariance *****************************************************************************
    
    x_ = Xsig_pred_ * weights_;                                                         //predicted state mean
    
    //predicted state covariance matrix
    P_.fill(0.0);
    for (int i = 0; i < n_sig_; i++) {
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        NormalizeAngle(x_diff, 3);                                           //angle normalization
        P_ = P_ + weights_(i) * x_diff * x_diff.transpose() ;
    }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
    // <Predict>
    int n_z = 2;                                            // Lidar has a dimension of 2 px and py
    MatrixXd Zsig = Xsig_pred_.block(0, 0, n_z, n_sig_);
    
    //mean predicted measurement
    VectorXd z_pred = VectorXd(n_z);
    z_pred.fill(0.0);
    for (int i=0; i < n_sig_; i++) {
        z_pred = z_pred + weights_(i) * Zsig.col(i);
    }
    
    //measurement covariance matrix S
    MatrixXd S = MatrixXd(n_z,n_z);
    S.fill(0.0);
    for (int i = 0; i < n_sig_; i++) {  //2n+1 simga points
        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;
        
        S = S + weights_(i) * z_diff * z_diff.transpose();
    }
    
    // Lidar measurement noice covarieance matrix ./
    R_lidar_ = MatrixXd(2, 2);
    R_lidar_ << std_laspx_*std_laspx_,0,
    0,std_laspy_*std_laspy_;
    
    //add measurement noise covariance matrix
    S = S + R_lidar_;
    
    // <Update>
    // Incoming radar measurement
    VectorXd z = meas_package.raw_measurements_;
    
    //create matrix for cross correlation Tc
    MatrixXd Tc = MatrixXd(n_x_, n_z);
    
    Tc.fill(0.0);
    for (int i = 0; i < n_sig_; i++) {  //2n+1 simga points
        
        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;
        
        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        
        Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }
    
    //Kalman gain K;
    MatrixXd K = Tc * S.inverse();
    
    //residual
    VectorXd z_diff = z - z_pred;
    
    //update state mean and covariance matrix
    x_ = x_ + K * z_diff;
    P_ = P_ - K*S*K.transpose();
    
    //NIS Lidar Update
    NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */

    // <Predict>
    int n_z = 3;
    
    //create matrix for sigma points in measurement space
    MatrixXd Zsig = MatrixXd(n_z, n_sig_);
    
    //transform sigma points into measurement space - L8T27
    for (int i = 0; i < n_sig_; i++) {

        double p_x = Xsig_pred_(0,i);
        double p_y = Xsig_pred_(1,i);
        double v  = Xsig_pred_(2,i);
        double yaw = Xsig_pred_(3,i);
        
        double v1 = cos(yaw)*v;
        double v2 = sin(yaw)*v;
        
        // measurement model
        Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                                        //r
        Zsig(1,i) = atan2(p_y,p_x);                                                 //phi
        Zsig(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);                   //r_dot
    }
    
    //mean predicted measurement
    VectorXd z_pred = VectorXd(n_z);
    z_pred.fill(0.0);
    for (int i=0; i < n_sig_; i++) {
        z_pred = z_pred + weights_(i) * Zsig.col(i);
    }
    
    //measurement covariance matrix S
    MatrixXd S = MatrixXd(n_z,n_z);
    S.fill(0.0);
/*    for (int i = 0; i < n_sig_; i++) {
        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;
        
        //angle normalization
        NormalizeAngle(z_diff, 1);
        
        S = S + weights_(i) * z_diff * z_diff.transpose();
*/
    for (int i = 0; i < n_sig_; i++) {
        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;
        
        //angle normalization
        NormalizeAngle(z_diff, 1);
        
        S = S + weights_(i) * z_diff * z_diff.transpose();
    }
    
    // Radar measurement noice covarieance matrix initialization
    R_radar_ = MatrixXd(3, 3);
    R_radar_ << std_radr_*std_radr_, 0, 0,
    0, std_radphi_*std_radphi_, 0,
    0, 0,std_radrd_*std_radrd_;
    
    //add measurement noise covariance matrix
    S = S + R_radar_;
    
    // <Update> - T8L30
    // Incoming radar measurement
    VectorXd z = meas_package.raw_measurements_;
    
    //create matrix for cross correlation Tc
    MatrixXd Tc = MatrixXd(n_x_, n_z);
    
    Tc.fill(0.0);
    for (int i = 0; i < n_sig_; i++) {
        
        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;
        //angle normalization
        NormalizeAngle(z_diff, 1);
        
        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        //angle normalization
        NormalizeAngle(x_diff, 3);
        
        Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }
    
    //Kalman gain K;
    MatrixXd K = Tc * S.inverse();
    
    //residual
    VectorXd z_diff = z - z_pred;
    
    //angle normalization
    NormalizeAngle(z_diff, 1);
    
    //update state mean and covariance matrix
    x_ = x_ + K * z_diff;
    P_ = P_ - K*S*K.transpose();
    
    //NIS Update
    NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
}

// Normalize angle
void UKF::NormalizeAngle(VectorXd vector, int index) {
    while (vector(index)> M_PI) vector(index)-=2.*M_PI;
    while (vector(index)<-M_PI) vector(index)+=2.*M_PI;
}

// Predict sigma points based on sigma pts, dt, state dimension, sigma pt dimension, acceleration noise, yaw acceleration noise
MatrixXd UKF::SigmaPointPrediction(MatrixXd Xsig, double delta_t, int n_x, int n_sig, double nu_acc, double nu_yawdd) {
    
    MatrixXd Xsig_pred = MatrixXd(n_x, n_sig);

    for (int i = 0; i< n_sig; i++)
    {
        //extract values for better readability
        double p_x = Xsig(0,i);
        double p_y = Xsig(1,i);
        double v = Xsig(2,i);
        double yaw = Xsig(3,i);
        double yawd = Xsig(4,i);
        double nu_a = Xsig(5,i);
        double nu_yawdd = Xsig(6,i);
        
        //predicted state values
        double px_p, py_p;
        
        //avoid division by zero
        if (fabs(yawd) > 0.001) {
            px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
            py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
        }
        else {
            px_p = p_x + v*delta_t*cos(yaw);
            py_p = p_y + v*delta_t*sin(yaw);
        }
        
        double v_p = v;
        double yaw_p = yaw + yawd*delta_t;
        double yawd_p = yawd;
        
        //add noise
        px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
        py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
        v_p = v_p + nu_a*delta_t;
        
        yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
        yawd_p = yawd_p + nu_yawdd*delta_t;
        
        //write predicted sigma point into right column
        Xsig_pred(0,i) = px_p;
        Xsig_pred(1,i) = py_p;
        Xsig_pred(2,i) = v_p;
        Xsig_pred(3,i) = yaw_p;
        Xsig_pred(4,i) = yawd_p;
    }
    return Xsig_pred;
}

// Generate sigma points based on state vector, covariance matrix, lambda and dimension - L8T15
MatrixXd UKF::GenerateSigmaPoints(VectorXd x, MatrixXd P, double lambda, int n_sig) {
    int n = x.size();
    // Sigma point matrix
    MatrixXd Xsig = MatrixXd( n, n_sig );
    
    //calculate square root of P
    MatrixXd A = P.llt().matrixL();
    
    //set first column of sigma point matrix
    Xsig.col(0) = x;
    
    for (int i = 0; i < n; i++){
        Xsig.col(i+1)     = x + sqrt(lambda+n) * A.col(i);
        Xsig.col(i+1+n) = x - sqrt(lambda+n) * A.col(i);
    }
    return Xsig;
}

