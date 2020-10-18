#ifndef PARTICLE_RENDERER_GL_H
#define PARTICLE_RENDERER_GL_H

#include <glad/glad.h>

#include <Eigen/Core>
#include <memory>

class ArrayBufferGL
{
public:
	ArrayBufferGL();
	ArrayBufferGL(const ArrayBufferGL&) = delete;
	ArrayBufferGL& operator=(const ArrayBufferGL&) = delete;

	~ArrayBufferGL();

	void setData(
		GLsizeiptr sizeInBytes,
		const GLvoid * data,
		GLenum usage = GL_STATIC_DRAW);

	void bind();

	void unbind();

	GLuint getId();

protected:
	GLuint m_buffer;
};

class ElementArrayBufferGL
{
public:
	ElementArrayBufferGL();
	ElementArrayBufferGL(const ElementArrayBufferGL&) = delete;
	ElementArrayBufferGL& operator=(const ElementArrayBufferGL&) = delete;

	~ElementArrayBufferGL();

	void bind();

	void unbind();

	void setData(
		GLsizeiptr sizeInBytes,
		const GLvoid * data,
		GLenum usage = GL_STATIC_DRAW);

	GLuint getId();
protected:
	GLuint m_buffer;
};

class VAOGL
{
public:
	VAOGL();
	VAOGL(const VAOGL&) = delete;
	VAOGL& operator=(const VAOGL&) = delete;

	~VAOGL();

	void bind();

	void unbind();

	GLuint getId();
private:
	GLuint m_vao;
};

class MeshCPU
{
public:

	void setMesh(const Eigen::MatrixXd &vertices, const Eigen::MatrixXi &triangles);
	void setMesh(const Eigen::MatrixXd &vertices, const Eigen::MatrixXd &normals, const Eigen::MatrixXi &triangles);

	const Eigen::MatrixXd& getVertices() const;

	const Eigen::MatrixXd& getVertexNormals() const;

	const Eigen::MatrixXi & getIndices() const;

	int getTriangleCount() const;
private:
	Eigen::MatrixXd m_vertices;
	Eigen::MatrixXd m_normals;
	Eigen::MatrixXi m_triangles;
};

class MeshGPU
{
public:
	MeshGPU(const MeshCPU& cpuMesh);

	VAOGL& getVAO();

	ArrayBufferGL& getVertexBuffer();
	ArrayBufferGL& getNormalBuffer();
	ElementArrayBufferGL& getIndexBuffer();
public:
	ArrayBufferGL m_vertexBuffer;
	ArrayBufferGL m_normalBuffer;
	ElementArrayBufferGL m_indexBuffer;
	VAOGL m_vao;
};

class MeshManaged
{
public:

	void setMesh(const Eigen::MatrixXd &vertices, const Eigen::MatrixXi &triangles);

	void initGL();
	void shutGL();

	MeshGPU& getGPUMesh();

	void draw();

private:
	MeshCPU m_cpuMesh;
	std::unique_ptr<MeshGPU> m_gpuMesh;
};


#endif