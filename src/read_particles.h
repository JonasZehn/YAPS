#ifndef READ_PARTICLES_H
#define READ_PARTICLES_H

#include <Eigen/Core>

bool read_particles_posgz(Eigen::MatrixXf &x, Eigen::VectorXf &d, const char *filename);
bool save_particles_posgz(Eigen::MatrixXf &x, const char *filename);

#endif
