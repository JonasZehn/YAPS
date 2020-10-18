#ifndef GPU_GRID_H
#define GPU_GRID_H

#include "GLCUDA.h"

#include "VectorHelper.h"

#include <thrust/device_vector.h>

class GPUGrid
{
public:
	GPUGrid(int sizeX, int sizeY, int sizeZ)
	{
		mSize = make_int3(sizeX, sizeY, sizeZ);

		mStrideZ = mSize.x * mSize.y;
	}

	void advectParticles(cudaGraphicsResource *particlesResource, cudaGraphicsResource *typeResource, float dt);
	//void advectParticles2(GPUGrid &grid2, cudaGraphicsResource *particlesResource, float dt);

	void load(const char *file);

	int getSizeX() const
	{
		return mSize.x;
	}
	int getSizeY() const
	{
		return mSize.y;
	}
	int getSizeZ() const
	{
		return mSize.z;
	}
	int3 getSize() const
	{
		return mSize;
	}

private:
	int3 mSize;
	int mStrideZ;
	thrust::device_vector<float3> grid;
};

#endif