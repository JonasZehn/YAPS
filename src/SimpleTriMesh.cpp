#include "SimpleTriMesh.h"

#include <Eigen/StdVector>

#include <map>

void SimpleTriMesh::addVertex(double x, double y, double z)
{
	m_vertices.push_back(Eigen::Vector3d(x, y, z));
}
void SimpleTriMesh::addTri(int i1, int i2, int i3)
{
	m_triangles.push_back(Eigen::Vector3i(i1, i2, i3));
}

void SimpleTriMesh::setFromMatrices(const Eigen::MatrixXd &vertices, const Eigen::MatrixXi &tris)
{
	m_vertices.resize(vertices.cols());
	m_triangles.resize(tris.cols());

	for (int i = 0; i < (int)m_vertices.size(); i++)
	{
		m_vertices[i] = vertices.col(i);
	}
	for (int i = 0; i < (int)m_triangles.size(); i++)
	{
		m_triangles[i] = tris.col(i);
	}
}
void SimpleTriMesh::toMatrices(Eigen::MatrixXd &vertices, Eigen::MatrixXi &tris) const
{
	vertices.resize(3, m_vertices.size());
	tris.resize(3, m_triangles.size());

	for (int i = 0; i < (int)m_vertices.size(); i++)
	{
		vertices.col(i) = m_vertices[i];
	}
	for (int i = 0; i < (int)m_triangles.size(); i++)
	{
		tris.col(i) = m_triangles[i];
	}
}
class TriBaryCoords
{
public:
	TriBaryCoords(int triIdx, const Eigen::Vector2d &u)
		:m_triIdx(triIdx),
		m_u(u)
	{
	}

	int getTriIdx() const
	{
		return m_triIdx;
	}

	Eigen::Vector3d getN() const
	{
		Eigen::Vector3d N;
		N << 1.0 - m_u.sum(), m_u;
		return N;
	}

private:
	Eigen::Vector2d m_u;
	int m_triIdx;
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

void SimpleTriMesh::upsample(int nAdditionalPointsPerEdge)
{
	//the following implementation removes non used vertices

	int numPointsPerEdge = nAdditionalPointsPerEdge + 2;
	int numTrisPerOldFace = (numPointsPerEdge - 1)*(numPointsPerEdge - 1);
	int numTris = numTrisPerOldFace * this->m_triangles.size();

	std::vector<TriBaryCoords, Eigen::aligned_allocator<TriBaryCoords> > baryCoords;

	std::vector<int> oldVertexToBaryCoords(this->m_vertices.size(), -1);
	std::vector<std::map<int, int> > edgeToBaryCoords(this->m_vertices.size());

	auto getOldVertexIdx = [&oldVertexToBaryCoords, &baryCoords](int oldTriIdx, int v, const Eigen::Vector2d &u) {
		if (oldVertexToBaryCoords[v] == -1)
		{
			baryCoords.push_back(TriBaryCoords(oldTriIdx, u));
			oldVertexToBaryCoords[v] = baryCoords.size() - 1;
		}
		return oldVertexToBaryCoords[v];
	};
	auto getVertexIdxOnEdge = [numPointsPerEdge, &getOldVertexIdx, &baryCoords, &edgeToBaryCoords](
		int oldTriIdx, int v0, const Eigen::Vector2d &u0, int v1, const Eigen::Vector2d &u1, int col) {
		if (col == 0) return getOldVertexIdx(oldTriIdx, v0, u0);
		if (col == numPointsPerEdge - 1) return getOldVertexIdx(oldTriIdx, v1, u1);

		Eigen::Vector2d uCopy0 = u0;
		Eigen::Vector2d uCopy1 = u1;

		if (v0 > v1)
		{
			std::swap(v0, v1);
			Eigen::Vector2d ut = uCopy0;
			uCopy0 = uCopy1;
			uCopy1 = ut;
			col = numPointsPerEdge - col - 1;
		}

		if (edgeToBaryCoords[v0].find(v1) == edgeToBaryCoords[v0].end())
		{
			double lambda = col / double(numPointsPerEdge - 1);
			Eigen::Vector2d u = (1.0 - lambda) * uCopy0 + lambda * uCopy1;
			baryCoords.push_back(TriBaryCoords(oldTriIdx, u));
			edgeToBaryCoords[v0].emplace(v1, baryCoords.size() - 1);
		}

		return edgeToBaryCoords[v0].find(v1)->second;
	};

	auto getVertexIdx = [this, &getVertexIdxOnEdge, &baryCoords, numPointsPerEdge](int oldTriIdx, int row, int col) {
		const Eigen::Vector3i &tri = this->m_triangles[oldTriIdx];

		Eigen::Vector2d u0(0.0, 0.0);
		Eigen::Vector2d u1(1.0, 0.0);
		Eigen::Vector2d u2(0.0, 1.0);

		if (row == 0)
		{
			return getVertexIdxOnEdge(oldTriIdx, tri[0], u0, tri[1], u1, col);
		}
		if (col == 0)
		{
			return getVertexIdxOnEdge(oldTriIdx, tri[0], u0, tri[2], u2, row);
		}
		if (row - col == 0)
		{
			return getVertexIdxOnEdge(oldTriIdx, tri[1], u1, tri[2], u2, row);
		}

		int numPointsInRow = numPointsPerEdge - row;

		double lambda = row / double(numPointsPerEdge - 1);
		double lambda2 = col / double(numPointsInRow - 1);

		//Eigen::Vector2d p1 = (1.0 - lambda) * u0 + lambda * u2;
		//Eigen::Vector2d p2 = (1.0 - lambda) * u1 + lambda * u2;

		//Eigen::Vector2d u = (1.0 - lambda2) * p1 + lambda2 * p2;

		Eigen::Vector2d u(lambda2 * (1.0 - lambda), lambda);

		//if (numPointsInRow == 1) u = Eigen::Vector2d(0.0, lambda);

		baryCoords.push_back(TriBaryCoords(oldTriIdx, u));

		return (int)baryCoords.size() - 1;
	};

	Eigen::MatrixXi newFaces(3, numTris);

	for (int oldTriIdx = 0; oldTriIdx < this->m_triangles.size(); oldTriIdx++)
	{
		int ciTri = 0;
		for (int row = 0; row < numPointsPerEdge - 1; row++)
		{
			int numTrisInRow = numPointsPerEdge - 2 * row;
			for (int k = 0, baseVertex = 0; k < numTrisInRow; k++, baseVertex++)
			{
				Eigen::Vector3i newTri;
				newTri[0] = getVertexIdx(oldTriIdx, row, baseVertex);
				newTri[1] = getVertexIdx(oldTriIdx, row, baseVertex + 1);
				newTri[2] = getVertexIdx(oldTriIdx, row + 1, baseVertex);
				newFaces.col(oldTriIdx * numTrisPerOldFace + ciTri) = newTri;
				ciTri += 1;

				k++;
				if (k >= numTrisInRow) break;

				newTri[0] = getVertexIdx(oldTriIdx, row, baseVertex + 1);
				newTri[1] = getVertexIdx(oldTriIdx, row + 1, baseVertex + 1);
				newTri[2] = getVertexIdx(oldTriIdx, row + 1, baseVertex);
				newFaces.col(oldTriIdx * numTrisPerOldFace + ciTri) = newTri;
				ciTri += 1;
			}
		}
		assert(ciTri == numTrisPerOldFace);
	}

	Eigen::MatrixXd newVertices(3, baryCoords.size());
	for (int i = 0; i < baryCoords.size(); i++)
	{
		const Eigen::Vector3i &tri = this->m_triangles[baryCoords[i].getTriIdx()];
		Eigen::Vector3d N = baryCoords[i].getN();
		Eigen::Vector3d v = Eigen::Vector3d::Zero();
		for (int j = 0; j < 3; j++)
		{
			v += N[j] * this->m_vertices[tri[j]];
		}
		newVertices.col(i) = v;
	}

	setFromMatrices(newVertices, newFaces);
}

SimpleTriMesh SimpleTriMesh::icosphere()
{
	double t = (1.0 + std::sqrt(5.0)) / 2.0;
	double normalizationFactor = 1.0 / (1.0 + t*t);
	double u = t * normalizationFactor;
	double o = 1.0 * normalizationFactor;
	SimpleTriMesh mesh;

	mesh.addVertex(-o, u, 0);
	mesh.addVertex(o, u, 0);
	mesh.addVertex(-o, -u, 0);
	mesh.addVertex(o, -u, 0);

	mesh.addVertex(0, -o, u);
	mesh.addVertex(0, o, u);
	mesh.addVertex(0, -o, -u);
	mesh.addVertex(0, o, -u);

	mesh.addVertex(u, 0, -o);
	mesh.addVertex(u, 0, o);
	mesh.addVertex(-u, 0, -o);
	mesh.addVertex(-u, 0, o);

	mesh.addTri(0, 5, 1);
	mesh.addTri(0, 11, 5);
	mesh.addTri(0, 1, 7);
	mesh.addTri(0, 10, 11);
	mesh.addTri(0, 7, 10);

	mesh.addTri(3, 9, 4);
	mesh.addTri(3, 4, 2);
	mesh.addTri(3, 2, 6);
	mesh.addTri(3, 6, 8);
	mesh.addTri(3, 8, 9);

	mesh.addTri(1, 5, 9);
	mesh.addTri(7, 1, 8);
	mesh.addTri(10, 7, 6);
	mesh.addTri(5, 11, 4);
	mesh.addTri(11, 10, 2);

	mesh.addTri(4, 9, 5);
	mesh.addTri(9, 8, 1);
	mesh.addTri(6, 2, 10);
	mesh.addTri(2, 4, 11);
	mesh.addTri(8, 6, 7);

	return mesh;
}
SimpleTriMesh SimpleTriMesh::icosphere(int nSubdivisions)
{
	SimpleTriMesh mesh = SimpleTriMesh::icosphere();
	for (int i = 0; i < nSubdivisions; i++)
	{
		mesh.upsample(1);
		for (auto &vertex : mesh.vertices())
		{
			vertex.normalize();
		}
	}
	return mesh;
}