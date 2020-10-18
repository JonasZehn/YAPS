#include "Particles.h"

#include <cuda_runtime_api.h>

void ParticlesCPU::setParticles(const Eigen::MatrixXf &vertices, const Eigen::MatrixXf &colors,
	const Eigen::VectorXi &types,
	const Eigen::MatrixXf &additionalData1,
	const Eigen::MatrixXf &additionalData2)
{
	m_vertices = vertices;
	m_colors = colors;
	m_illuminated = colors;
	m_types = types;
	m_additionalData1 = additionalData1;
	m_additionalData2 = additionalData2;
}
ParticlesGPU::ParticlesGPU(const ParticlesCPU& cpuParticles)
{
	initializeBuffers(cpuParticles);
}
void ParticlesGPU::unregisterCudaResources()
{
	cudaGraphicsUnregisterResource(idcs_resource);
	cudaGraphicsUnregisterResource(illuminated_resource);
	cudaGraphicsUnregisterResource(additionalData2_resource);
	cudaGraphicsUnregisterResource(additionalData1_resource);
	cudaGraphicsUnregisterResource(types_resource);
	cudaGraphicsUnregisterResource(colors_resource);
	cudaGraphicsUnregisterResource(resource);
}
void ParticlesGPU::initializeBuffers(const ParticlesCPU& cpuParticles)
{

	const Eigen::MatrixXf &vertices = cpuParticles.getVertices();
	m_vertexBuffer.setData(vertices.rows()*vertices.cols() * sizeof(Eigen::MatrixXf::Scalar), vertices.data(), GL_DYNAMIC_DRAW);

	assert(glGetError() == GL_NO_ERROR);

	cudaGraphicsGLRegisterBuffer(&resource, m_vertexBuffer.getId(), cudaGraphicsMapFlagsNone);

	assert(glGetError() == GL_NO_ERROR);

	const Eigen::MatrixXf &colors = cpuParticles.getColors();
	m_colorBuffer.setData(colors.rows()*colors.cols() * sizeof(Eigen::MatrixXf::Scalar), colors.data(), GL_DYNAMIC_DRAW);

	assert(glGetError() == GL_NO_ERROR);

	cudaGraphicsGLRegisterBuffer(&colors_resource, m_colorBuffer.getId(), cudaGraphicsMapFlagsNone);

	assert(glGetError() == GL_NO_ERROR);

	const Eigen::VectorXi &types = cpuParticles.getTypes();
	m_typeBuffer.setData(types.rows()*types.cols() * sizeof(Eigen::VectorXi::Scalar), types.data(), GL_DYNAMIC_DRAW);

	assert(glGetError() == GL_NO_ERROR);

	cudaGraphicsGLRegisterBuffer(&types_resource, m_typeBuffer.getId(), cudaGraphicsMapFlagsNone);

	assert(glGetError() == GL_NO_ERROR);

	const Eigen::MatrixXf &additionalData1 = cpuParticles.getAdditionalData1();
	m_additionalData1Buffer.setData(additionalData1.rows()*additionalData1.cols() * sizeof(Eigen::MatrixXf::Scalar), additionalData1.data(), GL_DYNAMIC_DRAW);

	assert(glGetError() == GL_NO_ERROR);

	cudaGraphicsGLRegisterBuffer(&additionalData1_resource, m_additionalData1Buffer.getId(), cudaGraphicsMapFlagsNone);

	assert(glGetError() == GL_NO_ERROR);

	const Eigen::MatrixXf &additionalData2 = cpuParticles.getAdditionalData2();
	m_additionalData2Buffer.setData(additionalData2.rows()*additionalData2.cols() * sizeof(Eigen::MatrixXf::Scalar), additionalData2.data(), GL_DYNAMIC_DRAW);

	assert(glGetError() == GL_NO_ERROR);

	cudaGraphicsGLRegisterBuffer(&additionalData2_resource, m_additionalData2Buffer.getId(), cudaGraphicsMapFlagsNone);

	assert(glGetError() == GL_NO_ERROR);

	const Eigen::MatrixXf &illuminated = cpuParticles.getIlluminated();
	m_illuminatedBuffer.setData(illuminated.rows()*illuminated.cols() * sizeof(Eigen::MatrixXf::Scalar), illuminated.data(), GL_DYNAMIC_DRAW);

	assert(glGetError() == GL_NO_ERROR);

	cudaGraphicsGLRegisterBuffer(&illuminated_resource, m_illuminatedBuffer.getId(), cudaGraphicsMapFlagsNone);

	assert(glGetError() == GL_NO_ERROR);

	Eigen::MatrixXi indices(1, vertices.cols());
	for (int i = 0; i < indices.cols(); i++) indices(0, i) = i;
	m_indexBuffer.setData(indices.rows()*indices.cols() * sizeof(Eigen::MatrixXi::Scalar), indices.data(), GL_DYNAMIC_DRAW);

	assert(glGetError() == GL_NO_ERROR);

	cudaGraphicsGLRegisterBuffer(&idcs_resource, m_indexBuffer.getId(), cudaGraphicsMapFlagsNone);

	assert(glGetError() == GL_NO_ERROR);
}
void ParticlesGPU::setParticles(const ParticlesCPU& cpuParticles)
{
	unregisterCudaResources();
	initializeBuffers(cpuParticles);
}
ParticlesGPU::~ParticlesGPU()
{
	unregisterCudaResources();
}
void ParticlesManaged::initGL()
{
	m_gpuParticles.reset(new ParticlesGPU(m_cpuParticles));

	assert(glGetError() == GL_NO_ERROR);

	VAOGL &vao = m_gpuParticles->getVAO();
	vao.bind();

	assert(glGetError() == GL_NO_ERROR);

	m_gpuParticles->getVertexBuffer().bind();

	assert(glGetError() == GL_NO_ERROR);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0);
	glEnableVertexAttribArray(0);

	assert(glGetError() == GL_NO_ERROR);

	m_gpuParticles->getColorBuffer().bind();

	assert(glGetError() == GL_NO_ERROR);

	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0);
	glEnableVertexAttribArray(1);

	assert(glGetError() == GL_NO_ERROR);

	m_gpuParticles->getTypeBuffer().bind();

	assert(glGetError() == GL_NO_ERROR);

	glVertexAttribPointer(2, 1, GL_INT, GL_FALSE, sizeof(int), 0);
	glEnableVertexAttribArray(2);

	assert(glGetError() == GL_NO_ERROR);

	m_gpuParticles->getAdditionalData1Buffer().bind();

	assert(glGetError() == GL_NO_ERROR);

	glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0);
	glEnableVertexAttribArray(3);

	assert(glGetError() == GL_NO_ERROR);

	m_gpuParticles->getAdditionalData2Buffer().bind();

	assert(glGetError() == GL_NO_ERROR);

	glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0);
	glEnableVertexAttribArray(4);

	assert(glGetError() == GL_NO_ERROR);

	m_gpuParticles->getIlluminatedBuffer().bind();

	assert(glGetError() == GL_NO_ERROR);

	glVertexAttribPointer(5, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0);
	glEnableVertexAttribArray(5);

	assert(glGetError() == GL_NO_ERROR);

	m_gpuParticles->getIndexBuffer().bind(); // need to bind element array buffer

	assert(glGetError() == GL_NO_ERROR);

	vao.unbind();

	assert(glGetError() == GL_NO_ERROR);
}

void ParticlesManaged::shutGL()
{
	m_gpuParticles.reset();
}
void ParticlesManaged::draw()
{
	VAOGL &vao = m_gpuParticles->getVAO();

	assert(glGetError() == GL_NO_ERROR);
	vao.bind();
	assert(glGetError() == GL_NO_ERROR);

	glDrawElements(GL_POINTS, m_cpuParticles.getVertices().cols(), GL_UNSIGNED_INT, 0);

	assert(glGetError() == GL_NO_ERROR);
	vao.unbind();
	assert(glGetError() == GL_NO_ERROR);
}

void ParticlesManaged::sort(float dx, float dy, float dz)
{
	//sort_points(getGPUParticles().getCUDARessource(), dx, dy, dz);
	sort_points(getGPUParticles().getCUDAResource(), getGPUParticles().getCUDAIdcsResource(), dotProductBuffer, dx, dy, dz);
}
void ParticlesManaged::transferGPUToCPU()
{
	copyParticles(getGPUParticles().getCUDAResource(), m_cpuParticles.getVertices().data());
}

