#include "Camera.h"

namespace CameraMatrix
{

void orthographicMatrix(float left, float right, float bottom, float top, float nearVal, float farVal,
	Eigen::Matrix4f & P)
{
	P.setIdentity();
	P(0, 0) = 2.0 / (right - left);
	P(1, 1) = 2.0 / (top - bottom);
	P(2, 2) = -2.0 / (farVal - nearVal);
	P(0, 3) = -(right + left) / (right - left);
	P(1, 3) = -(top + bottom) / (top - bottom);
	P(2, 3) = -(farVal + nearVal) / (farVal - nearVal);
}
void frustumMatrix( float left, float right, float bottom, float top, float nearVal, float farVal,
	Eigen::Matrix4f & P)
{
	P.setConstant(4, 4, 0.);
	P(0, 0) = (2.0 * nearVal) / (right - left);
	P(1, 1) = (2.0 * nearVal) / (top - bottom);
	P(0, 2) = (right + left) / (right - left);
	P(1, 2) = (top + bottom) / (top - bottom);
	P(2, 2) = -(farVal + nearVal) / (farVal - nearVal);
	P(3, 2) = -1.0;
	P(2, 3) = -(2.0 * farVal * nearVal) / (farVal - nearVal);
}
}

Camera::Camera()
{
	orthographic = false;

	trackball_angles.setIdentity();
	trackball_radius = 1.0;
	trackball_center = Eigen::Vector3f::Zero();

	view_angle = 45.0;
	dnear = 1e-2;
	dfar = 1e4;
}
void Camera::setCameraTrackballCenter(const Eigen::Vector3f &center)
{
	trackball_center = center;
}
void Camera::setCameraTrackballRadius(double radius)
{
	trackball_radius = radius;
}
void Camera::computeMatrices(float width, float height, bool ortho, Eigen::Matrix4f &viewMat, Eigen::Matrix4f &projMat)
{
	projMat = Eigen::Matrix4f::Identity();

	// Set projection
	if (ortho)
	{
		float length = trackball_radius;
		float h = tan(view_angle / 360.0 * PR_PI) * (length);
		CameraMatrix::orthographicMatrix(-h*width / height, h*width / height, -h, h, dnear, dfar, projMat);
	}
	else
	{
		float fH = tan(view_angle / 360.0 * PR_PI) * dnear;
		float fW = fH * (double)width / (double)height;
		CameraMatrix::frustumMatrix(-fW, fW, -fH, fH, dnear, dfar, projMat);
	}
	// end projection

	// Set view
	viewMat = Eigen::Matrix4f::Identity();
	viewMat.topLeftCorner(3, 3) = trackball_angles.toRotationMatrix();

	// Why not just use Eigen::Transform<double,3,Projective> for model...?
	//view.topLeftCorner(3, 3) *= camera_zoom;
	Eigen::Vector3f camera_center = trackball_center + trackball_radius*(trackball_angles.inverse()*Eigen::Vector3f(0.0f, 0.0f, 1.0f));
	viewMat.col(3).head(3) = viewMat.topLeftCorner(3, 3)*-camera_center;
}