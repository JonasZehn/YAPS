#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

class Renderer;

class MainWindow : public QMainWindow
{
public:
	MainWindow(Renderer * renderer);
};

#endif