#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

double threshold1 = 0.0001;                                      // small number for error prevention

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,const vector<VectorXd> &ground_truth)
{
    VectorXd rmse(4);
    rmse << 0,0,0,0;
    
    // Input Error Checking
    if(estimations.size() != ground_truth.size())
    {
        cout << "Error - Invalid data size" << endl;
        return rmse;
    }
    
    if(estimations.size() == 0)
    {
        cout << "Error - size is zero" << endl;
        return rmse;
    }
    
    // Cumulative sum of residual
    for(unsigned int i = 0 ; i < estimations.size() ; ++i)
    {
        VectorXd residual = estimations[i] - ground_truth[i];
        residual = residual.array() * residual.array();
        rmse += residual;
    }
    
    // Mean
    rmse = rmse / estimations.size();
    
    // Square root
    rmse = rmse.array().sqrt();
    
    return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state)
{
    double px = x_state(0);
    double py = x_state(1);
    double vx = x_state(2);
    double vy = x_state(3);
    MatrixXd Hj(3,4);
    
    // Deal with the special case problems
    if (fabs(px) < threshold1 and fabs(py) < threshold1)
    {
        px = threshold1;
        py = threshold1;
    }
    
    // Pre-compute a set of terms to avoid repeated calculation
    double c1 = px*px+py*py;
    
    // Error Prevention
    if(fabs(c1) < threshold1)
    {
        c1 = threshold1;
    }
    double c2 = sqrt(c1);
    double c3 = (c1*c2);
    
    // Compute the Jacobian matrix
    Hj << (px/c2), (py/c2), 0, 0,
    -(py/c1), (px/c1), 0, 0,
    py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;
    return Hj;
}
