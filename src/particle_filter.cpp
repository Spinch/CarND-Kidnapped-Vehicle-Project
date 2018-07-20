/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

// using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	
    num_particles = 100;
    
    // Create normal (Gaussian) distributions.
    std::normal_distribution<double> dist_x(x, std[0]);
    std::normal_distribution<double> dist_y(y, std[1]);
    std::normal_distribution<double> dist_theta(theta, std[2]);

    double w = 1./num_particles;
    
    for (int i=0; i<num_particles; ++i) {
	Particle p({i, dist_x(_gen), dist_y(_gen), dist_theta(_gen), w});
// 	p.norm();
	
	particles.push_back(p);
    }
    
    is_initialized = true;
    
    return;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    
    for (auto &p : particles) {
	// Movement model
	if (fabs(yaw_rate) > 0.0001) {
	    double t1 = velocity/yaw_rate;
	    p.x += t1*(sin(p.theta+yaw_rate*delta_t)-sin(p.theta));
	    p.y += t1*(cos(p.theta)-cos(p.theta+yaw_rate*delta_t));
	    p.theta += yaw_rate*delta_t;
	}
	else
	{
	    double t1 = velocity*delta_t;
	    p.x += t1*cos(p.theta);
	    p.y += t1*sin(p.theta);
	}
	
	// Add noise
	std::normal_distribution<double> dist_x(p.x, std_pos[0]);
	std::normal_distribution<double> dist_y(p.y, std_pos[1]);
	std::normal_distribution<double> dist_theta(p.theta, std_pos[2]);
	p.x = dist_x(_gen);
	p.y = dist_y(_gen);
	p.theta = dist_theta(_gen);
    }
    
    return;
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    
    // Not usefull
    
    return;
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
    
    double totalW = 0;
    
    for (auto &p : particles) {
	long double w = 1;
	
	std::vector<int> associations;
	std::vector<double> sense_x;
	std::vector<double> sense_y;
	
	for (auto &obs : observations) {
	    LandmarkObs lobs;
	    lobs.x = p.x + cos(p.theta)*obs.x - sin(p.theta)*obs.y;
	    lobs.y = p.y + sin(p.theta)*obs.x + cos(p.theta)*obs.y;
	    lobs.id = obs.id;
	    
	    // Find closest landmark
	    double dist = std::numeric_limits<double>::max();
	    Map::single_landmark_s betsLandmark;
	    for (auto &landmark : map_landmarks.landmark_list) {
		double dx = landmark.x_f - lobs.x;
		double dy = landmark.y_f - lobs.y;
		double d = sqrt(dx*dx + dy*dy);
		if (d < dist) {
		    dist = d;
		    betsLandmark = landmark;
		}
	    }
	    
	    associations.push_back(betsLandmark.id_i);
	    sense_x.push_back(betsLandmark.x_f);
	    sense_y.push_back(betsLandmark.y_f);
	    
	    double k = 1. / (2*M_PI*std_landmark[0]*std_landmark[1]);
	    double dx = betsLandmark.x_f - lobs.x;
	    double dy = betsLandmark.y_f - lobs.y;
	    long double w1 = k*exp( -(dx*dx/(2*std_landmark[0]*std_landmark[0]) + dy*dy/(2*std_landmark[1]*std_landmark[1]) ) );
	    w *= w1;
	}
	
	this->SetAssociations(p, associations, sense_x, sense_y);
	
	p.weight = w;
	totalW += p.weight;
    }
    
    // Normalize
    for (auto &p : particles)
	p.weight /= totalW;
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    
    std::vector<double> weights;
    for (auto &p : particles)
	weights.push_back(p.weight);
    std::discrete_distribution<> d(weights.begin(), weights.end());
    
    std::vector<Particle> newParticles;
    for (unsigned int i=0; i<particles.size(); ++i) {
	newParticles.push_back(particles[d(_gen)]);
    }
    
    particles = newParticles;
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

std::string ParticleFilter::getAssociations(Particle best)
{
    std::vector<int> v = best.associations;
    std::stringstream ss;
    copy( v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
    std::string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
std::string ParticleFilter::getSenseX(Particle best)
{
    std::vector<double> v = best.sense_x;
    std::stringstream ss;
    copy( v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
    std::string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
std::string ParticleFilter::getSenseY(Particle best)
{
    std::vector<double> v = best.sense_y;
    std::stringstream ss;
    copy( v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
    std::string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
