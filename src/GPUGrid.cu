#include "GPUGrid.h"

#include <thrust/for_each.h>

#include <zlib.h>

#include <exception>
#include <sstream>


std::ostream& operator<<(std::ostream& str, const int3 &i3)
{
	str << i3.x << ' ' << i3.y << ' ' << i3.z << std::endl;
	return str;
}

struct RK4Kernel
{
	RK4Kernel(float dt, float3 *data, float3 *x, int *types, int3 size, int mStrideZ)
		:dt(dt),
		data(data),
		x(x),
		types(types),
		size(size),
		mStrideZ(mStrideZ)
	{
	}

	float dt;

	float3 *data;
	float3 *x;
	int *types;
	int3 size;
	int mStrideZ;

	/** see interpolMAC in mantaflow, Original license statement: */
/******************************************************************************
 *
 * MantaFlow fluid solver framework
 * Copyright 2011 Tobias Pfaff, Nils Thuerey
 *
 * This program is free software, distributed under the terms of the
 * Apache License, Version 2.0
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Helper functions for interpolation
 *
 ******************************************************************************/
	__host__ __device__ float3 getInterpolated(float3 &pos)
	{
		int Z = mStrideZ;

		float px = pos.x - 0.5f, py = pos.y - 0.5f, pz = pos.z - 0.5f;
		int xi = (int)px;
		int yi = (int)py;
		int zi = (int)pz;
		float s1 = px - (float)xi, s0 = 1. - s1;
		float t1 = py - (float)yi, t0 = 1. - t1;
		float f1 = pz - (float)zi, f0 = 1. - f1;
		/* clamp to border */
		if (px < 0.) { xi = 0; s0 = 1.0; s1 = 0.0; }
		if (py < 0.) { yi = 0; t0 = 1.0; t1 = 0.0; }
		if (pz < 0.) { zi = 0; f0 = 1.0; f1 = 0.0; }
		if (xi >= size.x - 1) { xi = size.x - 2; s0 = 0.0; s1 = 1.0; }
		if (yi >= size.y - 1) { yi = size.y - 2; t0 = 0.0; t1 = 1.0; }
		if (size.z>1) { if (zi >= size.z - 1) { zi = size.z - 2; f0 = 0.0; f1 = 1.0; } }
		const int X = 1;
		const int Y = size.x;


		/* shifted coords */
		int s_xi = (int)pos.x, s_yi = (int)pos.y, s_zi = (int)pos.z;
		float s_s1 = pos.x - (float)s_xi, s_s0 = 1. - s_s1;
		float s_t1 = pos.y - (float)s_yi, s_t0 = 1. - s_t1;
		float s_f1 = pos.z - (float)s_zi, s_f0 = 1. - s_f1;
		/* clamp to border */
		if (pos.x < 0) { s_xi = 0; s_s0 = 1.0; s_s1 = 0.0; }
		if (pos.y < 0) { s_yi = 0; s_t0 = 1.0; s_t1 = 0.0; }
		if (pos.z < 0) { s_zi = 0; s_f0 = 1.0; s_f1 = 0.0; }
		if (s_xi >= size.x - 1) { s_xi = size.x - 2; s_s0 = 0.0; s_s1 = 1.0; }
		if (s_yi >= size.y - 1) { s_yi = size.y - 2; s_t0 = 0.0; s_t1 = 1.0; }
		if (size.z>1) { if (s_zi >= size.z - 1) { s_zi = size.z - 2; s_f0 = 0.0; s_f1 = 1.0; } }

		// process individual components
		float3 ret = make_float3(0.f, 0.f, 0.f);
		{   // X
			const float3* ref = &data[((zi*size.y + yi)*size.x + s_xi)];
			ret.x = f0 * ((ref[0].x  *t0 + ref[Y].x    *t1)*s_s0 +
				(ref[X].x  *t0 + ref[X + Y].x  *t1)*s_s1) +
				f1 * ((ref[Z].x  *t0 + ref[Z + Y].x  *t1)*s_s0 +
				(ref[X + Z].x*t0 + ref[X + Y + Z].x*t1)*s_s1);
		}
		{   // Y
			const float3* ref = &data[((zi*size.y + s_yi)*size.x + xi)];
			ret.y = f0 * ((ref[0].y  *s_t0 + ref[Y].y    *s_t1)*s0 +
				(ref[X].y  *s_t0 + ref[X + Y].y  *s_t1)*s1) +
				f1 * ((ref[Z].y  *s_t0 + ref[Z + Y].y  *s_t1)*s0 +
				(ref[X + Z].y*s_t0 + ref[X + Y + Z].y*s_t1)*s1);
		}
		{   // Z
			const float3* ref = &data[((s_zi*size.y + yi)*size.x + xi)];
			ret.z = s_f0 * ((ref[0].z  *t0 + ref[Y].z    *t1)*s0 +
				(ref[X].z  *t0 + ref[X + Y].z  *t1)*s1) +
				s_f1 * ((ref[Z].z  *t0 + ref[Z + Y].z  *t1)*s0 +
				(ref[X + Z].z*t0 + ref[X + Y + Z].z*t1)*s1);
		}
		return ret;
	}

	__host__ __device__ void operator()(int idx)
	{
		if (types[idx] != 0) return;

		float3 x0 = x[idx];
		float3 u0 = dt * getInterpolated(x0);

		float3 x1 = x0 + 0.5f*u0;
		float3 u1 = dt * getInterpolated(x1);

		float3 x2 = x0 + 0.5f*u1;
		float3 u2 = dt * getInterpolated(x2);

		float3 x3 = x0 + u2;
		float3 u3 = dt * getInterpolated(x3);

		x[idx] = x0 + ((1.f / 6.f)) * (u0 + 2.f * u1 + 2.f * u2 + u3);
	}
};


void GPUGrid::advectParticles(cudaGraphicsResource *particlesResource, cudaGraphicsResource *typeResource, float dt)
{
	float3 * velocityData = thrust::raw_pointer_cast(grid.data());

	cuChk(cudaGraphicsMapResources(1, &particlesResource, NULL));
	float3* devPtr;
	size_t  size;
	cuChk(cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, particlesResource));
	thrust::device_ptr<float3> t_dptr = thrust::device_pointer_cast(devPtr);

	int pointSize = sizeof(float3);
	int numPoints = size / pointSize;

	cuChk(cudaGraphicsMapResources(1, &typeResource, NULL));
	int* types_devPtr;
	size_t  types_size;
	cuChk(cudaGraphicsResourceGetMappedPointer((void**)&types_devPtr, &types_size, typeResource));
	thrust::device_ptr<int> t_types_dptr = thrust::device_pointer_cast(types_devPtr);

	RK4Kernel kernel(dt, velocityData, devPtr, types_devPtr, mSize, mStrideZ);
	thrust::for_each(thrust::make_counting_iterator(0), thrust::make_counting_iterator(numPoints), kernel);
	//thrust::for_each(
	//	thrust::make_zip_iterator(thrust::make_tuple(t_dptr, t_types_dptr)),
	//	thrust::make_zip_iterator(thrust::make_tuple(t_dptr + numPoints, t_types_dptr + numPoints)),
	//	kernel);

	cuChk(cudaGraphicsUnmapResources(1, &typeResource, NULL));
	cuChk(cudaGraphicsUnmapResources(1, &particlesResource, NULL));
}

static const int STR_LEN_GRID = 252;

//! uni file header, v4
typedef struct
{
	int dimX, dimY, dimZ; // grid size
	int gridType, elementType, bytesPerElement; // data type info
	char info[STR_LEN_GRID]; // mantaflow build information
	int dimT;                // optionally store forth dimension for 4d grids
	unsigned long long timestamp; // creation time
} UniHeader;

#define throwError(msg)      { std::ostringstream __s; __s << msg << std::endl << "Error raised in " << __FILE__ << ":" << __LINE__; throw std::logic_error(__s.str()); }
#define assertMsg(cond,msg)  if(!(cond)) throwError(msg)

#define FLOATINGPOINT_PRECISION 1

void GPUGrid::load(const char *name)
{
	std::cout << "Reading grid  from uni file " << name << std::endl;

	gzFile gzf = gzopen(name, "rb");
	if (!gzf)
	{
		std::cout << "can't open file " << name << std::endl;
		return;
	}

	char ID[5] = { 0,0,0,0,0 };
	gzread(gzf, ID, 4);

	if (!strcmp(ID, "MNT3"))
	{
		// current file format
		UniHeader head;
		assertMsg(gzread(gzf, &head, sizeof(UniHeader)) == sizeof(UniHeader), "can't read file, no header present");
		assertMsg(head.dimX == this->getSizeX() && head.dimY == this->getSizeY() && head.dimZ == this->getSizeZ(), "grid dim doesn't match, " << make_int3(head.dimX, head.dimY, head.dimZ) << " vs " << this->getSize());
		//assertMsg(unifyGridType(head.gridType) == unifyGridType(grid->getType()), "grid type doesn't match " << head.gridType << " vs " << grid->getType());
#		if FLOATINGPOINT_PRECISION!=1
		// convert float to double
		Grid<T> temp(grid->getParent());
		void*  ptr = &(temp[0]);
		gridReadConvert<T>(gzf, *grid, ptr, head.bytesPerElement);
#		else
		assertMsg(head.bytesPerElement == sizeof(float3), "grid element size doesn't match " << head.bytesPerElement << " vs " << sizeof(float3));

		thrust::host_vector<float3> hostData(head.dimX * head.dimY * head.dimZ);

		int numBytes = sizeof(float3)*head.dimX*head.dimY*head.dimZ;
		assertMsg(gzread(gzf, hostData.data(), numBytes) == numBytes, "couldn't read all data");
#		endif

		grid.resize(head.dimX*head.dimY*head.dimZ);
		thrust::copy(hostData.begin(), hostData.end(), grid.begin());

		std::cout << "successful read " << name << std::endl;
	}
	else
	{
		std::cout << "Unknown header '" << ID << "' " << std::endl;
	}
}