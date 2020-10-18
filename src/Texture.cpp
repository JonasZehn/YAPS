#include "Texture.h"

#include "GLCUDA.h"
#include <cassert>

#include <Eigen/Core>

TextureBuffer::TextureBuffer(int width)
	:
	m_width(width)
{
	glChk(glGenBuffers(1, &m_buffer));
	glChk(glBindBuffer(GL_TEXTURE_BUFFER, m_buffer));
	glChk(glBufferData(GL_TEXTURE_BUFFER, width * sizeof(int), NULL, GL_DYNAMIC_COPY));
	cuChk(cudaGraphicsGLRegisterBuffer(&m_resource, m_buffer, cudaGraphicsMapFlagsNone));
	glChk(glBindBuffer(GL_TEXTURE_BUFFER, 0));

	glChk(glGenTextures(1, &m_textureId));

	glChk(glBindTexture(GL_TEXTURE_BUFFER, m_textureId));

	glChk(glTexBuffer(GL_TEXTURE_BUFFER, GL_R32I, m_buffer));

	glChk(glBindTexture(GL_TEXTURE_BUFFER, 0));
}
TextureBuffer::~TextureBuffer()
{
	glChk(glDeleteTextures(1, &m_textureId));
	cuChk(cudaGraphicsUnregisterResource(m_resource));
	glChk(glDeleteBuffers(1, &m_buffer));
}

void TextureBuffer::setConstant(int val)
{
	fillIntBuffer(m_resource, val);
}

Texture2D::Texture2D(int width, int height)
	:m_width(width),
	m_height(height)
{
	//glGenBuffers(1, &m_buffer);
	//glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, m_buffer);
	//glBufferData(GL_ATOMIC_COUNTER_BUFFER, width * height * sizeof(int), NULL, GL_DYNAMIC_COPY);
	//cudaGraphicsGLRegisterBuffer(&m_resource, m_buffer, cudaGraphicsMapFlagsNone);
	
	glChk(0);

	glChk(glGenTextures(1, &m_textureId));
	
	glChk(glBindTexture(GL_TEXTURE_2D, m_textureId));

	glChk(glTexImage2D(GL_TEXTURE_2D, 0, GL_R32I, width, height, 0, GL_RED_INTEGER, GL_INT, NULL));
	//glChk(glTexStorage2D(GL_TEXTURE_2D, 0, GL_R32I, width, height));

	glChk(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
	glChk(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));

	cuChk(cudaGraphicsGLRegisterImage(&m_textureResource, m_textureId, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone));

	glChk(glBindTexture(GL_TEXTURE_2D, 0));
}
Texture2D::~Texture2D()
{
	cuChk(cudaGraphicsUnregisterResource(m_textureResource));
	glChk(glDeleteTextures(1, &m_textureId));
	//cudaGraphicsUnregisterResource(m_resource);
	//glDeleteBuffers(1, &m_buffer);
}

void Texture2D::setZero()
{
	//fillIntTexture(m_textureResource, m_width, m_height, val);
	glChk(glClearTexImage(m_textureId, 0, GL_RED_INTEGER, GL_INT, NULL));
}
std::vector<int> Texture2D::read()
{
	glChk(glBindTexture(GL_TEXTURE_2D, m_textureId));
	std::vector<int> result(m_width *  m_height);
	glChk(glGetTexImage(GL_TEXTURE_2D, 0, GL_RED_INTEGER, GL_INT, result.data()));
	glChk(glBindTexture(GL_TEXTURE_2D, 0));
	return result;
}