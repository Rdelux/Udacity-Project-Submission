/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 *      Student: Richard Lee modified it on March 25, 2018
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    
    if (is_initialized)                                             // Check flag to see if it has initialized
        return;
    
    // Set the number of particles
    num_particles = 60;
    
    // Initialize all particles to first position
    // Read Standard Deviation - GPS measurement uncertainty [x [m], y [m], theta [rad]]
    double std_x = std[0];                                           // Reading sigma_pos
    double std_y = std[1];
    double std_T = std[2];
    
    // Create Normal Distributions - L15T5
    normal_distribution<double> dist_x(x,std_x);
    normal_distribution<double> dist_y(y,std_y);
    normal_distribution<double> dist_T(theta,std_T);
    
    // Add random number generator
    std::default_random_engine gen;                                 // L15T5Ln23
    
    // Generate particles
    for (int i = 0 ; i < num_particles ; i++)
    {
        Particle myParticle;                                            // Create particles
        myParticle.id = i;
        myParticle.x = dist_x(gen);
        myParticle.y = dist_y(gen);
        myParticle.theta = dist_T(gen);
        myParticle.weight = 1.0;
        particles.push_back(myParticle);                                // set initialized particles to set
    }
    is_initialized = true;                                              // Complete initialization
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    
    // Read standard deviation - [standard deviation of x [m], of y [m], of yaw [rad]]
    double std_x = std_pos[0];                                           // Reading sigma_pos
    double std_y = std_pos[1];
    double std_T = std_pos[2];
    
    // Create Normal Distributions - L15T5
    normal_distribution<double> dist_x(0,std_x);                        // 0 mean
    normal_distribution<double> dist_y(0,std_y);
    normal_distribution<double> dist_T(0,std_T);

    // Add random number generator
    std::default_random_engine gen;                                 // L15T5Ln23
    
    for (int i = 0 ; i < num_particles ; i++)                       // Predict motion
    {
        double yawAngle = particles[i].theta;
        if (fabs(yaw_rate) < 0.0001)                                // Motion model for yaw rate ~= 0
        {
            particles[i].x += velocity * delta_t * cos(yawAngle);
            particles[i].y += velocity * delta_t * sin(yawAngle);
        }
        else                                                        // Motion model for yaw rate != 0
        {
            particles[i].x += velocity / yaw_rate * ( sin( yawAngle + yaw_rate * delta_t ) - sin( yawAngle ) );
            particles[i].y += velocity / yaw_rate * ( cos( yawAngle ) - cos( yawAngle + yaw_rate * delta_t ) );
            particles[i].theta += yaw_rate * delta_t;
        }
        // Add noise
        particles[i].x += dist_x(gen);
        particles[i].y += dist_y(gen);
        particles[i].theta += dist_T(gen);                                      // particles positions predicted
    }   
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

    unsigned long nObservations = observations.size();                  // observation
    unsigned long nPredictions = predicted.size();                      // In range LM
    
    for (int i = 0 ; i < nObservations ; i++)                           // For every observation from sensor
    {
        double minD = 9999999;                                          // Initialize to large number
        int landMark = -1;                                              // Initialize index
        for (int j = 0 ; j < nPredictions ; j++)                        // Go through LM in range
        {
            double dx = observations[i].x - predicted[j].x;
            double dy = observations[i].y - predicted[j].y;
            double dist = dx * dx + dy * dy;
            if (dist < minD)                                            // Check for nearest neighbour
            {
                minD = dist;                                            // update the nearest neighbour
                landMark = predicted[j].id;
            }
        }
        observations[i].id = landMark;                                  // Match observation for nearest neighbour
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
    
    double stdLMX = std_landmark[0];                                    // Landmark measurement uncertainty
    double stdLMY = std_landmark[1];
    
    // Determine which landmarks are in range
    for (int i = 0 ; i < num_particles ; i++)
    {
        double x = particles[i].x;                                      // particle pose before transform
        double y = particles[i].y;
        double theta = particles[i].theta;
        
        vector<LandmarkObs> inRangeLM;                                  // Landmarks-in-range vector
        double sensor_range_sqrt = sensor_range * sensor_range;
        
        for(int j = 0 ; j < map_landmarks.landmark_list.size(); j++)                    // For each landmark in map
        {
            float LMx = map_landmarks.landmark_list[j].x_f;                             // x, y, id of landmark
            float LMy = map_landmarks.landmark_list[j].y_f;
            int id = map_landmarks.landmark_list[j].id_i;
            
            double dX = x - LMx;                                                    // delta of landmark and particle
            double dY = y - LMy;
            if ( dX * dX + dY * dY <= sensor_range_sqrt )                           // landmark is in sensor range
            {
                inRangeLM.push_back(LandmarkObs{id, LMx, LMy});
            }
        }
        
        // Transform Landmark Observation
        vector<LandmarkObs> MObservations;                                          // Mapped observation vector
        
        for(int j = 0; j < observations.size() ; j++)                               // Homogeneous transformation
        {
            double Xm = cos(theta) * observations[j].x - sin(theta) * observations[j].y + x;
            double Ym = sin(theta) * observations[j].x + cos(theta) * observations[j].y + y;
            MObservations.push_back(LandmarkObs{observations[j].id, Xm, Ym });
        }
        
        // Determine weight
        dataAssociation(inRangeLM, MObservations);                     // nearest neigbour found for every sensor data
        
        particles[i].weight = 1.0;                                      // Reset weight for particle
        int m = 0;                                                      // Landmark counter
        double landmarkX, landmarkY;
        bool found = false;
        unsigned long nLandmarks = inRangeLM.size();
        int landmarkId = 0;
        
        for(int j = 0 ; j < MObservations.size() ; j++)                 // Go through every LM observation
        {
            landmarkId = MObservations[j].id;
            m = 0;
            nLandmarks = inRangeLM.size();
            found = false;
            
            while( !found && m < nLandmarks )                           // find the landmark id
            {
                if (inRangeLM[m].id == landmarkId)
                {
                    found = true;
                    landmarkX = inRangeLM[m].x;
                    landmarkY = inRangeLM[m].y;
                }
                m++;
            }
            
            // Calculating weight.
            double dX = MObservations[j].x - landmarkX;
            double dY = MObservations[j].y - landmarkY;
            
            // Multivariate Gaussian Probability Density
            double weight = ( 1/(2 * M_PI * stdLMX * stdLMY)) * exp( -( dX * dX /(2 * stdLMX * stdLMX) + (dY * dY / (2 * stdLMY * stdLMY)) ) );
            
            if (weight == 0)                                                // Error prevention
            {
                particles[i].weight *= 0.0000001;
            }
            else
            {
                particles[i].weight *= weight;
            }
        }
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    
    vector<double> weights;
    double maxWeight = numeric_limits<double>::min();
    
    for(int i = 0; i < num_particles; i++)
    {
        weights.push_back(particles[i].weight);
        
        if ( particles[i].weight > maxWeight )                              // Get max weight of all particles
        {
            maxWeight = particles[i].weight;
        }
    }
    
    // Create distributions
    uniform_real_distribution<double> distDouble(0.0, maxWeight);
    uniform_int_distribution<int> distInt(0, num_particles - 1);
    
    std::default_random_engine gen;                                 // L15T5Ln23
    int index = distInt(gen);
    
    double beta = 0.0;
    
    // Resampling wheel - L14T20
    vector<Particle> resampledParticles;
    for(int i = 0; i < num_particles; i++)
    {
        beta += distDouble(gen) * 2.0;
        while( beta > weights[index])
        {
            beta -= weights[index];
            index = (index + 1) % num_particles;
        }
        resampledParticles.push_back(particles[index]);
    }
    
    particles = resampledParticles;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
    
    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
