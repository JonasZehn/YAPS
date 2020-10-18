#ifndef MAIN_WINDOW_GL_WIDGET_H
#define MAIN_WINDOW_GL_WIDGET_H

#include <glad/glad.h>
#include <QGLWidget>

#include "SettingsHelper.h"

#include <memory>

class Renderer;

class MainWindowGLWidget : public QGLWidget
{
public:

	MainWindowGLWidget(Renderer * renderer, QVBoxLayout *settingsLayout, QWidget *window, QWidget *parent);
	virtual void initializeGL() override;
	virtual void paintGL() override;

	std::unique_ptr<Renderer> m_renderer;
	SettingsHelper settings;
};

#endif