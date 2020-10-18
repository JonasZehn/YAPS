
#include "MainWindow.h"

#include <QApplication>

#include "Renderer.h"

int main(int argc, char **argv)
{
	//Render example
	Renderer * renderer = new PlumeRenderer(std::string(YAPS_MANTA_FOLDER) + "out/plume_mc_128", "output_plume_mc_128");

	QApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
	QApplication::setAttribute(Qt::AA_UseHighDpiPixmaps);
	QApplication app(argc, argv);
	
	MainWindow mainWin(renderer);
	mainWin.resize(1200, 700);
	mainWin.show();
	return app.exec();
}
