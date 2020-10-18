#ifndef VECTOR_HELPER_H
#define VECTOR_HELPER_H

#include <vector_types.h>
#include <vector_functions.hpp>

inline __host__ __device__ float3 operator+(const float3 & a, const float3 & b)
{
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);

}
inline __host__ __device__ float3 operator*(float a, const float3 & b)
{
	return make_float3(a * b.x, a * b.y, a * b.z);
}
inline __host__ __device__ float3 clamp(const float3 & a, float b, float c)
{
	float3 res = a;
	if (res.x < b) res.x = b;
	else if (res.x > c) res.x = c;
	if (res.y < b) res.y = b;
	else if (res.y > c) res.y = c;
	if (res.z < b) res.z = b;
	else if (res.z > c) res.z = c;
	return res;
}

#endif