#ifndef SIMPLE_TRI_MESH_H
#define SIMPLE_TRI_MESH_H

#include <Eigen/Core>

#include <vector>
#include <cmath>

class SimpleTriMesh
{
public:
	typedef std::vector<Eigen::Vector3d>& VertexRange;

	void addVertex(double x, double y, double z);
	void addTri(int i1, int i2, int i3);

	void setFromMatrices(const Eigen::MatrixXd &vertices, const Eigen::MatrixXi &tris);
	void toMatrices(Eigen::MatrixXd &vertices, Eigen::MatrixXi &tris) const;

	void upsample(int nAdditionalPointsPerEdge);

	VertexRange vertices()
	{
		return m_vertices;
	}

	static SimpleTriMesh icosphere();
	static SimpleTriMesh icosphere(int nSubdivisions);

private:
	std::vector<Eigen::Vector3d> m_vertices;
	std::vector<Eigen::Vector3i> m_triangles;
};

#endif