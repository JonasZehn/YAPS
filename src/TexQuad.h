#ifndef TEX_QUAD_H
#define TEX_QUAD_H

#include "GL.h"

#include <Eigen/Core>

#include <memory>
#include <cassert>

class GPUTexQuad
{
public:
	GPUTexQuad(const Eigen::MatrixXd &vertices, const Eigen::MatrixXd &textureData);

	VAOGL& getVAO()
	{
		return vao;
	}

	ArrayBufferGL& getVertexBuffer()
	{
		return vertexBuffer;
	}

	ArrayBufferGL& getTextureBuffer()
	{
		return textureBuffer;
	}

private:
	VAOGL vao;
	ArrayBufferGL vertexBuffer;
	ArrayBufferGL textureBuffer;
};

class TexQuad
{
public:

	void init();

	void draw();

private:

	std::unique_ptr<GPUTexQuad> m_gpuMesh;

};

#endif