#include "read_particles.h"

#include <zlib.h>

#include <iostream>
#include <vector>
//
bool read_particles_posgz(Eigen::MatrixXf &x, Eigen::VectorXf &d, const char *filename)
{
	gzFile gzf = gzopen(filename, "rb");
	if (!gzf)
	{
		std::cout << "can't open file " << filename << std::endl;
		return false;
	}

	std::vector<Eigen::Vector3f> particles;

	float xyz[3];
	int numBytes = 3 * sizeof(float);
	while (gzread(gzf, xyz, numBytes) == numBytes)
	{
		particles.push_back(Eigen::Vector3f(xyz[0], xyz[1], xyz[2]));
	}

	x.resize(3, particles.size());
	d.resize(particles.size());

#pragma omp parallel for
	for (int i = 0; i < particles.size(); i++)
	{
		x.col(i) = particles[i];
		d[i] = 1.0;
	}

	return true;
}
bool save_particles_posgz(Eigen::MatrixXf &x, const char *filename)
{
	gzFile gzf = gzopen(filename, "wb1");
	if (!gzf)
	{
		std::cout << "can't open file " << filename << std::endl;
		return false;
	}
	for (int i = 0; i<x.cols(); ++i)
	{
		float xyz[3];
		int numBytes = 3 * sizeof(float);
		xyz[0] = x(0, i);
		xyz[1] = x(1, i);
		xyz[2] = x(2, i);
		gzwrite(gzf, xyz, numBytes);
	}
	gzclose(gzf);

	return true;
}
