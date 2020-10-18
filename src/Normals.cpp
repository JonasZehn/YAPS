#include "Normals.h"

#include <Eigen/Geometry>

void perVertexNormals(const Eigen::MatrixXd &vertices,
	const Eigen::MatrixXi &triangles,
	Eigen::MatrixXd &normals,
	WeightType::Type weightType)
{
	normals.setZero(vertices.rows(), vertices.cols());
	std::vector<double> weightSum(vertices.cols(), 0.0);

	for (int tri = 0; tri < triangles.cols(); tri++)
	{
		std::array<int, 3> vIdcs = { triangles(0, tri), triangles(1, tri), triangles(2, tri) };
		std::array<Eigen::Vector3d, 3> v;
		for (int i = 0; i < 3; i++) {
			if (vIdcs[i] < 0 || vIdcs[i] >= vertices.cols()) {
				throw std::logic_error("err");
			}
			v[i] = vertices.col(vIdcs[i]);
		}
		Eigen::Vector3d normal = (v[1] - v[0]).cross(v[2] - v[0]);
		double twiceArea = normal.norm();
		normal /= twiceArea;
		for (int i = 0; i < 3; i++)
		{
			Eigen::Vector3d e1 = v[(i + 1) % 3] - v[i];
			Eigen::Vector3d e2 = v[(i + 2) % 3] - v[i];
			double angleBetween = std::acos(std::min(1.0, std::max(-1.0, e1.normalized().dot(e2.normalized()))));
			double weight;
			switch (weightType)
			{
			case WeightType::CONSTANT:
				weight = 1.0;
				break;
			case WeightType::AREA:
				weight = twiceArea;
				break;
			case WeightType::ANGLE:
				weight = angleBetween;
				break;
			case WeightType::ANGLE_AREA:
				weight = twiceArea * angleBetween;
				break;
			default: assert(false);
			}
			//double weight = 1.0;
			Eigen::Vector3d wn = weight*normal;
			normals.col(vIdcs[i]) += wn;
			weightSum[vIdcs[i]] += weight;
		}
	}

	for (int i = 0; i < vertices.cols(); i++)
	{
		if (weightSum[i] != 0.0)
		{
			normals.col(i) /= weightSum[i];
		}
	}
}
