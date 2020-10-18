#ifndef PARTICLES_H
#define PARTICLES_H

#include "sort_pixels.h"

#include "ComputeColorsParameters.h"

#include "GL.h"

#include <Eigen/Core>
#include <memory>

class TextureBuffer;

void copyParticles(cudaGraphicsResource *vtcsResource, float *dest);

void computeColors(TextureBuffer *illuTexture, cudaGraphicsResource *illuminatedResource, const ComputeColorsParameters &parameters);

void computeColors2(cudaGraphicsResource *colorsResource, TextureBuffer *illuTexture, cudaGraphicsResource *illuminatedResource, const ComputeColorsParameters &parameters);

class ParticlesCPU
{
public:

	void setParticles(
		const Eigen::MatrixXf &vertices,
		const Eigen::MatrixXf &colors,
		const Eigen::VectorXi &types,
		const Eigen::MatrixXf &additionalData1,
		const Eigen::MatrixXf &additionalData2);

	Eigen::MatrixXf& getVertices()
	{
		return m_vertices;
	}
	const Eigen::MatrixXf& getVertices() const
	{
		return m_vertices;
	}
	const Eigen::MatrixXf& getColors() const
	{
		return m_colors;
	}
	const Eigen::VectorXi& getTypes() const
	{
		return m_types;
	}
	const Eigen::MatrixXf& getAdditionalData1() const
	{
		return m_additionalData1;
	}
	const Eigen::MatrixXf& getAdditionalData2() const
	{
		return m_additionalData2;
	}
	const Eigen::MatrixXf& getIlluminated() const
	{
		return m_illuminated;
	}

private:
	Eigen::MatrixXf m_vertices;
	Eigen::MatrixXf m_colors;
	Eigen::VectorXi m_types;
	Eigen::MatrixXf m_additionalData1;
	Eigen::MatrixXf m_additionalData2;
	Eigen::MatrixXf m_illuminated;
};

void fillFloat(cudaGraphicsResource *resource, float val);

class ParticlesGPU
{
public:
	ParticlesGPU(const ParticlesCPU& cpuParticles);
	void setParticles(const ParticlesCPU& cpuParticles);
	~ParticlesGPU();

	VAOGL& getVAO()
	{
		return m_vao;
	}

	ArrayBufferGL& getVertexBuffer()
	{
		return m_vertexBuffer;
	}

	ArrayBufferGL& getColorBuffer()
	{
		return m_colorBuffer;
	}

	ArrayBufferGL& getTypeBuffer()
	{
		return m_typeBuffer;
	}

	ArrayBufferGL& getAdditionalData1Buffer()
	{
		return m_additionalData1Buffer;
	}
	ArrayBufferGL& getAdditionalData2Buffer()
	{
		return m_additionalData2Buffer;
	}
	ArrayBufferGL& getIlluminatedBuffer()
	{
		return m_illuminatedBuffer;
	}

	ElementArrayBufferGL& getIndexBuffer()
	{
		return m_indexBuffer;
	}

	cudaGraphicsResource * getCUDAResource()
	{
		return resource;
	}

	cudaGraphicsResource * getCUDAColorsResource()
	{
		return colors_resource;
	}

	cudaGraphicsResource * getCUDATypesResource()
	{
		return types_resource;
	}

	cudaGraphicsResource * getCUDAIlluminatedResource()
	{
		return illuminated_resource;
	}

	cudaGraphicsResource * getCUDAIdcsResource()
	{
		return idcs_resource;
	}

private:
	void unregisterCudaResources();
	void initializeBuffers(const ParticlesCPU& cpuParticles);

	ArrayBufferGL m_vertexBuffer;
	ArrayBufferGL m_colorBuffer;
	ArrayBufferGL m_typeBuffer;
	ArrayBufferGL m_additionalData1Buffer;
	ArrayBufferGL m_additionalData2Buffer;
	ArrayBufferGL m_illuminatedBuffer;
	ElementArrayBufferGL m_indexBuffer;
	VAOGL m_vao;
	cudaGraphicsResource *resource;
	cudaGraphicsResource *colors_resource;
	cudaGraphicsResource *types_resource;
	cudaGraphicsResource *additionalData1_resource;
	cudaGraphicsResource *additionalData2_resource;
	cudaGraphicsResource *illuminated_resource;
	cudaGraphicsResource *idcs_resource;
};
class ParticlesManaged
{
public:

	int getParticleCount() const { return m_cpuParticles.getVertices().cols(); }

	void setParticles(const Eigen::MatrixXf &vertices, const Eigen::MatrixXf &colors, const Eigen::VectorXi &types, const Eigen::MatrixXf &additionalData1, const Eigen::MatrixXf &additionalData2)
	{
		m_cpuParticles.setParticles(vertices, colors, types, additionalData1, additionalData2);
		if (m_gpuParticles)
		{
			//std::cout << " before " << std::endl;
			//outputGPUMemoryUsage();
			//shutGL();
			//outputGPUMemoryUsage();
			//initGL();
			//outputGPUMemoryUsage();
			m_gpuParticles->setParticles(m_cpuParticles);
		}
	}

	void initGL();
	void shutGL();

	ParticlesGPU& getGPUParticles()
	{
		return *m_gpuParticles;
	}

	ParticlesCPU& getCPUParticles()
	{
		return m_cpuParticles;
	}

	void draw();

	void sort(float dx, float dy, float dz);

	void transferGPUToCPU();

private:
	ParticlesCPU m_cpuParticles;
	std::unique_ptr<ParticlesGPU> m_gpuParticles;
	thrust::device_vector<float> dotProductBuffer;
};


#endif