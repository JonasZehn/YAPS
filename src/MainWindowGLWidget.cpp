#include "MainWindowGLWidget.h"

#include "Renderer.h"

#include <iostream>

MainWindowGLWidget::MainWindowGLWidget(Renderer * renderer, QVBoxLayout *settingsLayout, QWidget *window, QWidget *parent)
	:
	QGLWidget(parent),
	m_renderer(renderer),
	settings(window, settingsLayout)
{
}
void MainWindowGLWidget::initializeGL()
{
	QGLWidget::initializeGL();

	if (!gladLoadGL())
	{
		throw std::runtime_error("could not initialize glad");
	}
	GLint drawFboId;
	glGetIntegerv(GL_FRAMEBUFFER_BINDING, &drawFboId);

	m_renderer->init();

	settings.addVariable("Play", m_renderer->m_play);
	settings.addColorVariable("Background color", m_renderer->background_color);
	settings.addVariable("Illumination Mult.", m_renderer->m_illuminationMultiplier);
	settings.addVariable("Shadow multiplier", m_renderer->m_shadowKMultiplier);
	settings.addColorVariable("Base color", m_renderer->m_baseColor);
	settings.addColorVariable("Illuminated Color", m_renderer->m_illuminatedColor);
	settings.addVariable("Radius", m_renderer->m_radius);
	settings.addVariable("Opacity", m_renderer->m_opacity);
	settings.addVariable("Light pos.", m_renderer->m_lightPosition);

	settings.addButton("Output Settings", [this]()
	{
		//std::ofstream outputStream("settings.txt");
		auto& outputStream = std::cout;
		outputStream << "m_lightPosition " << m_renderer->m_lightPosition << std::endl;
		outputStream << "camera Center " << m_renderer->m_camera.computePosition() << std::endl;
		outputStream << "Base Color " << m_renderer->m_baseColor << std::endl;
		outputStream << "Illuminated Color " << m_renderer->m_illuminatedColor << std::endl;
		outputStream << "Illumination Mult. " << m_renderer->m_illuminationMultiplier << std::endl;
		outputStream << "Shadow K Mult. " << m_renderer->m_shadowKMultiplier << std::endl;
		outputStream << "Radius " << m_renderer->m_radius << std::endl;
		outputStream << "Opacity " << m_renderer->m_opacity << std::endl;
	});

	settings.finalize();

	glBindFramebuffer(GL_FRAMEBUFFER, drawFboId);
}
void MainWindowGLWidget::paintGL()
{
	QGLWidget::paintGL();

	m_renderer->step();
	m_renderer->renderToTexture();
	m_renderer->saveToFile();

	int w = this->width();
	int h = this->height();
	glClearColor(1.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glDisable(GL_DEPTH_TEST);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glViewport(0, 0, (GLint)w, (GLint)h);

	m_renderer->renderToScreen(w, h);

	update();
}