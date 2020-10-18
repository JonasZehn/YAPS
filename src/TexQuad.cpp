#include "TexQuad.h"

GPUTexQuad::GPUTexQuad(const Eigen::MatrixXd &vertices, const Eigen::MatrixXd &textureData)
{
	vertexBuffer.setData(vertices.rows()*vertices.cols() * sizeof(Eigen::MatrixXd::Scalar), vertices.data());

	assert(glGetError() == GL_NO_ERROR);

	textureBuffer.setData(textureData.rows()*textureData.cols() * sizeof(Eigen::MatrixXd::Scalar), textureData.data());

	assert(glGetError() == GL_NO_ERROR);
}

void TexQuad::init()
{
	Eigen::MatrixXd verts(3, 4);
	verts <<
		-1.0f, -1.0f, 1.0f, 1.0f,
		-1.0f, 1.0f, -1.0f, 1.0f,
		0.0f, 0.0f, 0.0f, 0.0f;

	Eigen::MatrixXd tex(2, 4);
	tex <<
		0.0f, 0.0f, 1.0f, 1.0f,
		0.0f, 1.0f, 0.0f, 1.0f;

	m_gpuMesh.reset(new GPUTexQuad(verts, tex));

	assert(glGetError() == GL_NO_ERROR);

	VAOGL &vao = m_gpuMesh->getVAO();
	vao.bind();

	assert(glGetError() == GL_NO_ERROR);

	m_gpuMesh->getVertexBuffer().bind();

	assert(glGetError() == GL_NO_ERROR);

	glVertexAttribPointer(0, 3, GL_DOUBLE, GL_FALSE, 3 * sizeof(double), 0);
	glEnableVertexAttribArray(0);

	assert(glGetError() == GL_NO_ERROR);

	m_gpuMesh->getTextureBuffer().bind();

	assert(glGetError() == GL_NO_ERROR);

	glVertexAttribPointer(1, 2, GL_DOUBLE, GL_FALSE, 2 * sizeof(double), 0);
	glEnableVertexAttribArray(1);

	assert(glGetError() == GL_NO_ERROR);

	vao.unbind();

	assert(glGetError() == GL_NO_ERROR);
}
void TexQuad::draw()
{
	VAOGL &vao = m_gpuMesh->getVAO();

	assert(glGetError() == GL_NO_ERROR);
	vao.bind();
	assert(glGetError() == GL_NO_ERROR);

	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

	GLenum err = glGetError();
	assert(err == GL_NO_ERROR);
	assert(glGetError() == GL_NO_ERROR);
	vao.unbind();
	assert(glGetError() == GL_NO_ERROR);
}