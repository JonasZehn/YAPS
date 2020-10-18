#ifndef TEXTURE_H
#define TEXTURE_H

#include <glad/glad.h>
#include <cuda_gl_interop.h>

#include <vector>

void fillIntBuffer(cudaGraphicsResource *resource, int val);
void fillIntTexture(cudaGraphicsResource *resource, int width, int height, int val);

class TextureBuffer
{
public:
	TextureBuffer(int width);
	~TextureBuffer();

	void setConstant(int val);

	int getWidth() const
	{
		return m_width;
	}

	cudaGraphicsResource * getCUDAResource()
	{
		return m_resource;
	}

	//GLuint getBufferId()
	//{
	//	return m_buffer;
	//}

	GLuint getTextureId()
	{
		return m_textureId;
	}

private:
	GLuint m_buffer;
	cudaGraphicsResource *m_resource;
	GLuint m_textureId;
	int m_width;
};

class Texture2D
{
public:
	Texture2D(int width, int height);
	~Texture2D();

	void setZero();

	int getWidth() const
	{
		return m_width;
	}

	int getHeight() const
	{
		return m_height;
	}

	//GLuint getBufferId()
	//{
	//	return m_buffer;
	//}

	GLuint getTextureId()
	{
		return m_textureId;
	}

	//Note:this method changes bound texture
	std::vector<int> read();

private:
	//GLuint m_buffer;
	//cudaGraphicsResource *m_resource;
	GLuint m_textureId;
	cudaGraphicsResource *m_textureResource;
	int m_width, m_height;
};

#endif