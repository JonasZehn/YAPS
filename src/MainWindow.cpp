#include "MainWindow.h"
#include "MainWindowGLWidget.h"

#include "SettingsHelper.h"

#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QGroupBox>

MainWindow::MainWindow(Renderer * renderer)
{
	QHBoxLayout *mainLayout = new QHBoxLayout();

	QWidget *settingsWidget = new QGroupBox("Settings");
	settingsWidget->setFixedWidth(250);
	QVBoxLayout *settingsLayout = new QVBoxLayout();
	settingsWidget->setLayout(settingsLayout);

	QWidget *glWidget = new MainWindowGLWidget(renderer, settingsLayout, this, nullptr);

	mainLayout->addWidget(settingsWidget);

	mainLayout->addWidget(glWidget);

	QWidget * centralWidget = new QWidget();

	centralWidget->setLayout(mainLayout);

	this->setCentralWidget(centralWidget);

}