#ifndef NORMALS_H
#define NORMALS_H

#include <Eigen/Core>
#include <iostream>
#include <vector>
#include <array>
#include <cmath>

namespace WeightType
{
enum Type
{
	CONSTANT, AREA, ANGLE, ANGLE_AREA
};
}

void perVertexNormals(const Eigen::MatrixXd &vertices,
	const Eigen::MatrixXi &triangles,
	Eigen::MatrixXd &normals,
	WeightType::Type weightType = WeightType::ANGLE_AREA);

#endif