#ifndef SETTINGS_HELPER_H
#define SETTINGS_HELPER_H

#include <Eigen/Core>

#include <QVBoxLayout>
#include <QLabel>
#include <QSpacerItem>
#include <QCheckBox>
#include <QLineEdit>
#include <QDoubleValidator>
#include <QPushButton>
#include <QColorDialog>

inline QColor toQColor(const Eigen::Vector4f &var)
{
	QColor color;
	color.setRedF(var[0]);
	color.setGreenF(var[1]);
	color.setBlueF(var[2]);
	color.setAlphaF(var[3]);
	return color;
}

inline Eigen::Vector4f toEigenVector4f(const QColor &color)
{
	Eigen::Vector4f var;
	var[0] = color.redF();
	var[1] = color.greenF();
	var[2] = color.blueF();
	var[3] = color.alphaF();
	return var;
}

class SettingsHelper
{
public:
	SettingsHelper(QWidget *window, QVBoxLayout *settingsLayout)
		:
		m_window(window),
		m_layout(settingsLayout)
	{

	}


	void addVariable(const char * name, QWidget *varWidget)
	{
		QWidget *mainWidget = new QWidget();
		QHBoxLayout *mainLayout = new QHBoxLayout();
		mainWidget->setLayout(mainLayout);

		QLabel *label = new QLabel(name);

		mainLayout->addWidget(label);
		mainLayout->addWidget(varWidget);

		m_layout->addWidget(mainWidget);
	}

	void addVariable(const char * name, bool &var)
	{
		QCheckBox *cb = new QCheckBox();
		cb->setChecked(var);

		QObject::connect(cb, &QCheckBox::toggled,
			[&var](bool state) {
			var = state;
		});
		addVariable(name, cb);
	}

	void addVariable(const char * name, float &var)
	{
		QLineEdit *cb = new QLineEdit();
		auto validator = new QDoubleValidator(0.0, 1e9, 4);
		validator->setNotation(QDoubleValidator::StandardNotation);
		cb->setValidator(validator);
		cb->setText(QString::number(var));

		QObject::connect(cb, &QLineEdit::textChanged,
			[&var](const QString &s) {
			var = s.toFloat();
		});
		addVariable(name, cb);
	}
	void addVariable(const char * name, Eigen::Vector3f &var)
	{
		std::string sName(name);
		this->addVariable((sName + " x").c_str(), var[0]);
		this->addVariable((sName + " y").c_str(), var[1]);
		this->addVariable((sName + " z").c_str(), var[2]);
	}

	void addColorVariable(const char * name, Eigen::Vector4f &var)
	{
		QPushButton *cb = new QPushButton();
		cb->setText("Choose");

		cb->setFlat(true);
		QPalette palette = cb->palette();
		palette.setColor(QPalette::Button, toQColor(var));
		cb->setAutoFillBackground(true);
		cb->setPalette(palette);

		QString sName(name);

		QObject::connect(cb, &QPushButton::clicked, [this, &var, cb, sName](bool b) {  
			QColor initialColor = toQColor(var);
			QColor color = QColorDialog::getColor(initialColor, m_window, sName, QColorDialog::ShowAlphaChannel);
			if (color.isValid())
			{
				var = toEigenVector4f(color);

				QPalette palette = cb->palette();
				palette.setColor(QPalette::Button, toQColor(var));
				cb->setAutoFillBackground(true);
				cb->setPalette(palette);
			}
		});
		m_layout->addWidget(cb);

		addVariable(name, cb);
	}

	void addButton(const char * name, std::function<void()> f)
	{
		QPushButton *cb = new QPushButton();
		cb->setText(name);
		QObject::connect(cb, &QPushButton::clicked, [f](bool b) {  f(); });
		m_layout->addWidget(cb);
	}

	void finalize()
	{
		m_layout->addItem(new QSpacerItem(20, 40, QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding));
	}

private:
	QWidget *m_window;
	QVBoxLayout *m_layout;
};

#endif