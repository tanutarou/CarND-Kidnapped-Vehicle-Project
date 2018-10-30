/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
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
    default_random_engine gen;
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);
    for(int i=0; i<num_particles; i++){
        Particle p;
        p.id = i;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = 1.0;
        particles.push_back(p);
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    default_random_engine gen;
    normal_distribution<double> noise_x(0, std_pos[0]);
    normal_distribution<double> noise_y(0, std_pos[1]);
    normal_distribution<double> noise_theta(0, std_pos[2]);

    for(int i=0; i<num_particles; i++){
        if(fabs(yaw_rate) > 1e-8){
            particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
            particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) -  cos(particles[i].theta + yaw_rate * delta_t));
            particles[i].theta += yaw_rate * delta_t;
        }else{
            particles[i].x += velocity * delta_t * cos(particles[i].theta);
            particles[i].y += velocity * delta_t * sin(particles[i].theta);
        }

        // add noise
        particles[i].x += noise_x(gen);
        particles[i].y += noise_y(gen);
        particles[i].theta += noise_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

    for(int i=0; i < observations.size(); i++){
        double min_dist = 1e10;
        for(int j=0; j < predicted.size(); j++){
            double d = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
            if(min_dist > d){
                min_dist = d;
                observations[i].id = j;
            }
        }
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

    for(int i=0; i < num_particles; i++){
       vector<LandmarkObs>  predicted;
       for(int j=0; j < map_landmarks.landmark_list.size(); j++){
           double d = dist(particles[i].x, particles[j].y, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f);
           if (d < sensor_range){
               LandmarkObs p_obs;
               p_obs.id = map_landmarks.landmark_list[j].id_i;
               p_obs.x = map_landmarks.landmark_list[j].x_f;
               p_obs.y = map_landmarks.landmark_list[j].y_f;

               predicted.push_back(p_obs);
           }
       }

       vector<LandmarkObs> observations_map;
       for(int j=0; j < observations.size(); j++){
           const double px = particles[i].x;
           const double py = particles[i].y;
           const double ptheta = particles[i].theta;

           LandmarkObs obs;
           obs.id = observations[j].id;
           obs.x = px + cos(ptheta) * observations[j].x - sin(ptheta) * observations[j].y;
           obs.y = py + sin(ptheta) * observations[j].x + cos(ptheta) * observations[j].y;
           observations_map.push_back(obs);
       }
       
       dataAssociation(predicted, observations_map);

       // weights update
       double w = 1.0;
       //printf("before %f\n", w);
       for(int j=0; j < observations_map.size(); j++){
           double x = observations_map[j].x;
           double y = observations_map[j].y;

           double mx = predicted[observations_map[j].id].x;
           double my = predicted[observations_map[j].id].y;

           //printf("x,y,mx,my %f %f %f %f %f\n", x, y, mx, my, exp(-(pow(x-mx, 2)/(2*pow(std_landmark[0], 2)) + pow(y-my, 2)/(2*pow(std_landmark[1], 2)))) / (2 * M_PI * std_landmark[0] * std_landmark[1]));
           w *= exp(-(pow(x-mx, 2)/(2*pow(std_landmark[0], 2)) + pow(y-my, 2)/(2*pow(std_landmark[1], 2)))) / (2 * M_PI * std_landmark[0] * std_landmark[1]);
       }
       //printf("after %f\n", w);
       particles[i].weight = w;
        
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    vector<double> weights_list;
    for(int i=0; i < num_particles; i++){
        weights_list.push_back(particles[i].weight);
    }

    random_device rd;
    mt19937 gen(rd());
    discrete_distribution<> d(weights_list.begin(), weights_list.end());
    for(int i=0; i < num_particles; i++){
        int index = d(gen);
        particles[i] = particles[index];
    }
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
