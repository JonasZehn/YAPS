#include "Framebuffer.h"

#include <stdexcept>

#include "GLCUDA.h"

#include <vector>
#include <algorithm>

#include <png.h>

#include <cassert>
#include <iostream>

Framebuffer::Framebuffer(int width, int height)
	:m_width(width), m_height(height)
{
	glGenFramebuffers(1, &m_framebufferId);
	glBindFramebuffer(GL_FRAMEBUFFER, m_framebufferId);

	glGenTextures(1, &m_renderTexture);

	glBindTexture(GL_TEXTURE_2D, m_renderTexture);

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, m_renderTexture, 0);

	GLenum DrawBuffers[1] = { GL_COLOR_ATTACHMENT0 };
	glDrawBuffers(1, DrawBuffers);

	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
	{
		std::cout << " framebuffer not complete " << std::endl;
		throw std::logic_error("framebuffer not complete");
	}

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void Framebuffer::bind()
{
	glBindFramebuffer(GL_FRAMEBUFFER, m_framebufferId);
	glViewport(0, 0, m_width, m_height);
}

void Framebuffer::unbind()
{
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}


FramebufferHDR::FramebufferHDR(int width, int height)
	:m_width(width), m_height(height)
{
	glGenFramebuffers(1, &m_framebufferId);
	glBindFramebuffer(GL_FRAMEBUFFER, m_framebufferId);

	glGenTextures(1, &m_renderTexture);

	// "Bind" the newly created texture : all future texture functions will modify this texture
	glBindTexture(GL_TEXTURE_2D, m_renderTexture);

	// Give an empty image to OpenGL ( the last "0" )
	glChk(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, width, height, 0, GL_RGB, GL_FLOAT, 0));

	// Poor filtering. Needed !
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

	//use glTexImage2DMultisample for multisampling, but we don't want that here

	////create and bind depth buffer attachment
	GLuint depthrenderbuffer;
	glGenRenderbuffers(1, &depthrenderbuffer);
	glBindRenderbuffer(GL_RENDERBUFFER, depthrenderbuffer);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthrenderbuffer);

	// Set "renderedTexture" as our colour attachement #0
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, m_renderTexture, 0);

	// Set the list of draw buffers.
	GLenum DrawBuffers[1] = { GL_COLOR_ATTACHMENT0 };
	glDrawBuffers(1, DrawBuffers); // "1" is the size of DrawBuffers

	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
	{
		throw std::logic_error("framebuffer not complete");
	}

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void FramebufferHDR::bind()
{
	glBindFramebuffer(GL_FRAMEBUFFER, m_framebufferId);
	glViewport(0, 0, m_width, m_height);
	assert(glGetError() == GL_NO_ERROR);
}

void FramebufferHDR::unbind()
{
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	assert(glGetError() == GL_NO_ERROR);
}

bool writePNGFileFromBuffer(const char *filename, unsigned char *pixels, int w, int h)
{
	png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);//8
	if (!png)
		return false;

	png_infop info = png_create_info_struct(png);//7
	if (!info)
	{
		png_destroy_write_struct(&png, &info);//
		return false;
	}

	FILE *fp = fopen(filename, "wb");
	if (!fp)
	{
		png_destroy_write_struct(&png, &info);//
		return false;
	}
	png_init_io(png, fp);//9
	png_set_IHDR(png,
		info,
		w,
		h,
		8 /* depth */,
		PNG_COLOR_TYPE_RGBA,
		PNG_INTERLACE_NONE,
		PNG_COMPRESSION_TYPE_DEFAULT,
		PNG_FILTER_TYPE_DEFAULT);//10

	png_write_info(png, info);//1

	png_bytepp rows = (png_bytepp)png_malloc(png, h * sizeof(png_bytep));//
	for (int i = 0; i < h; ++i)
		rows[i] = (png_bytep)(pixels + i * w * 4);

	png_write_image(png, rows);//2
	png_write_end(png, info);//6
	png_destroy_write_struct(&png, &info);//3

	fclose(fp);
	delete[] rows;
	return true;
}

bool FramebufferHDR::render_to_png(
	const std::string png_file)
{
	glChk(0);

	glChk(glBindTexture(GL_TEXTURE_2D, m_renderTexture));
	int width = m_width;
	int height = m_height;
	std::vector<float> data(3 * width * height);
	std::vector<unsigned char> data2(4 * width * height);
	glChk(glGetTexImage(
		GL_TEXTURE_2D,
		0,
		GL_RGB,
		GL_FLOAT,
		data.data()));

	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			for (int k = 0; k < 3; k++)
			{
				float f = data[3 * (i + (height - j - 1)*width) + k];
				//float f = i / double(width - 1);
				int gi = (int)(f*256.f);
				int fi = std::max(0, std::min(255, gi));
				data2[4 * (i + j*width) + k] = fi;
			}
			data2[4 * (i + j*width) + 3] = 255;
		}
	}

	glChk(glBindTexture(GL_TEXTURE_2D, 0));
	
	bool ret = writePNGFileFromBuffer(png_file.c_str(), data2.data(), width, height);
	return ret;
}