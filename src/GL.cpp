#include "GL.h"

#include <cassert>

#include "Normals.h"

ArrayBufferGL::ArrayBufferGL()
{
	glGenBuffers(1, &m_buffer);
	assert(glGetError() == GL_NO_ERROR);
}
ArrayBufferGL::~ArrayBufferGL()
{
	glDeleteBuffers(1, &m_buffer);
}
void ArrayBufferGL::bind()
{
	glBindBuffer(GL_ARRAY_BUFFER, m_buffer);
}

void ArrayBufferGL::unbind()
{
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void ArrayBufferGL::setData(
	GLsizeiptr sizeInBytes,
	const GLvoid * data,
	GLenum usage)
{
	assert(glGetError() == GL_NO_ERROR);

	bind();
	assert(glGetError() == GL_NO_ERROR);

	glBufferData(GL_ARRAY_BUFFER, sizeInBytes, data, usage);
	GLuint error = glGetError();
	assert(error == GL_NO_ERROR);
	unbind();
	assert(glGetError() == GL_NO_ERROR);
}

GLuint ArrayBufferGL::getId()
{
	return m_buffer;
}

ElementArrayBufferGL::ElementArrayBufferGL()
{
	glGenBuffers(1, &m_buffer);
	assert(glGetError() == GL_NO_ERROR);
}

ElementArrayBufferGL::~ElementArrayBufferGL()
{
	glDeleteBuffers(1, &m_buffer);
}

void ElementArrayBufferGL::bind()
{
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_buffer);
}

void ElementArrayBufferGL::unbind()
{
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void ElementArrayBufferGL::setData(
	GLsizeiptr sizeInBytes,
	const GLvoid * data,
	GLenum usage)
{
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_buffer);
	assert(glGetError() == GL_NO_ERROR);

	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeInBytes, data, usage);
	assert(glGetError() == GL_NO_ERROR);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	assert(glGetError() == GL_NO_ERROR);
}

GLuint ElementArrayBufferGL::getId()
{
	return m_buffer;
}

VAOGL::VAOGL()
{
	glGenVertexArrays(1, &m_vao);
}

VAOGL::~VAOGL()
{
	glDeleteVertexArrays(1, &m_vao);
}

void VAOGL::bind()
{
	glBindVertexArray(m_vao);
}

void VAOGL::unbind()
{
	glBindVertexArray(0);
}

GLuint VAOGL::getId()
{
	return m_vao;
}

void MeshCPU::setMesh(const Eigen::MatrixXd &vertices, const Eigen::MatrixXi &triangles)
{
	m_vertices = vertices;
	perVertexNormals(vertices, triangles, m_normals);
	m_triangles = triangles;
}

void MeshCPU::setMesh(const Eigen::MatrixXd &vertices, const Eigen::MatrixXd &normals, const Eigen::MatrixXi &triangles)
{
	m_vertices = vertices;
	m_normals = normals;
	m_triangles = triangles;
}

const Eigen::MatrixXd& MeshCPU::getVertices() const
{
	return m_vertices;
}

const Eigen::MatrixXd& MeshCPU::getVertexNormals() const
{
	return m_normals;
}

const Eigen::MatrixXi & MeshCPU::getIndices() const
{
	return m_triangles;
}

int MeshCPU::getTriangleCount() const
{
	return m_triangles.cols();
}
VAOGL& MeshGPU::getVAO()
{
	return m_vao;
}

MeshGPU::MeshGPU(const MeshCPU& cpuMesh)
{
	const Eigen::MatrixXd &vertices = cpuMesh.getVertices();
	m_vertexBuffer.setData(vertices.rows()*vertices.cols() * sizeof(Eigen::MatrixXd::Scalar), vertices.data());

	assert(glGetError() == GL_NO_ERROR);

	const Eigen::MatrixXd &normals = cpuMesh.getVertexNormals();
	m_normalBuffer.setData(normals.rows()*normals.cols() * sizeof(Eigen::MatrixXd::Scalar), normals.data());

	assert(glGetError() == GL_NO_ERROR);

	const Eigen::MatrixXi &indices = cpuMesh.getIndices();
	m_indexBuffer.setData(indices.rows()*indices.cols() * sizeof(Eigen::MatrixXi::Scalar), indices.data());

	assert(glGetError() == GL_NO_ERROR);
}
void MeshManaged::initGL()
{
	m_gpuMesh.reset(new MeshGPU(m_cpuMesh));

	assert(glGetError() == GL_NO_ERROR);

	VAOGL &vao = m_gpuMesh->getVAO();
	vao.bind();

	assert(glGetError() == GL_NO_ERROR);

	m_gpuMesh->getVertexBuffer().bind();

	assert(glGetError() == GL_NO_ERROR);

	glVertexAttribPointer(0, 3, GL_DOUBLE, GL_FALSE, 3 * sizeof(double), 0);
	glEnableVertexAttribArray(0);

	assert(glGetError() == GL_NO_ERROR);

	m_gpuMesh->getNormalBuffer().bind();

	assert(glGetError() == GL_NO_ERROR);

	glVertexAttribPointer(1, 3, GL_DOUBLE, GL_FALSE, 3 * sizeof(double), 0);
	glEnableVertexAttribArray(1);

	assert(glGetError() == GL_NO_ERROR);

	m_gpuMesh->getIndexBuffer().bind(); // need to bind element array buffer

	assert(glGetError() == GL_NO_ERROR);

	vao.unbind();

	assert(glGetError() == GL_NO_ERROR);
}

void MeshManaged::shutGL()
{
	m_gpuMesh.reset();
}
ArrayBufferGL& MeshGPU::getVertexBuffer()
{
	return m_vertexBuffer;
}
ArrayBufferGL& MeshGPU::getNormalBuffer()
{
	return m_normalBuffer;
}
ElementArrayBufferGL& MeshGPU::getIndexBuffer()
{
	return m_indexBuffer;
}
void MeshManaged::setMesh(const Eigen::MatrixXd &vertices, const Eigen::MatrixXi &triangles)
{
	m_cpuMesh.setMesh(vertices, triangles);
}
MeshGPU& MeshManaged::getGPUMesh()
{
	return *m_gpuMesh;
}


void MeshManaged::draw()
{
	VAOGL &vao = m_gpuMesh->getVAO();

	assert(glGetError() == GL_NO_ERROR);
	vao.bind();
	assert(glGetError() == GL_NO_ERROR);

	glDrawElements(GL_TRIANGLES, 3 * m_cpuMesh.getTriangleCount(), GL_UNSIGNED_INT, 0);

	assert(glGetError() == GL_NO_ERROR);
	vao.unbind();
	assert(glGetError() == GL_NO_ERROR);
}