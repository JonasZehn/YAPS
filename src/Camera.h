#ifndef CAMERA_H
#define CAMERA_H

#define PR_PI 3.14159265358979323846

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace CameraMatrix
{

void frustumMatrix(
	float left,
	float right,
	float bottom,
	float top,
	float nearVal,
	float farVal,
	Eigen::Matrix4f& P);
}

class Camera
{
public:

	Camera();

	void setCameraTrackballRadius(double radius);
	void setCameraTrackballCenter(const Eigen::Vector3f &center);

	void computeMatrices(float width, float height, bool ortho, Eigen::Matrix4f &viewMat, Eigen::Matrix4f &projMat);

	Eigen::Vector3f computePosition() const
	{
		return trackball_center + trackball_radius*(trackball_angles.inverse()*Eigen::Vector3f(0.0f, 0.0f, 1.0f));
	}

	bool orthographic;

	Eigen::Quaternionf trackball_angles;
	float trackball_radius;
	Eigen::Vector3f trackball_center;

	float view_angle;
	float dnear;
	float dfar;

};

#endif