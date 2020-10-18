#ifndef FRAMEBUFFER_H
#define FRAMEBUFFER_H

#include <glad/glad.h>

#include <string>

class Framebuffer
{
public:
	Framebuffer(int width, int height);

	//note this calls glViewport, so make sure you are using correct glViewport at the beginning of drawing to default framebuffer or after unbinding
	void bind();
	void unbind();

	int getWidth() const
	{
		return m_width;
	}

	int getHeight() const
	{
		return m_height;
	}

private:
	GLuint m_framebufferId;
	GLuint m_renderTexture;

	int m_width, m_height;
};

class FramebufferHDR
{
public:
	FramebufferHDR(int width, int height);

	//note this calls glViewport, so make sure you are using correct glViewport at the beginning of drawing to default framebuffer or after unbinding
	void bind();
	void unbind();

	int getWidth() const
	{
		return m_width;
	}

	int getHeight() const
	{
		return m_height;
	}

	GLuint colorTexture()
	{
		return m_renderTexture;
	}

	bool render_to_png(const std::string png_file);

private:
	GLuint m_framebufferId;
	GLuint m_renderTexture;

	int m_width, m_height;
};

#endif