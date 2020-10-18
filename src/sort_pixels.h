#ifndef SORT_PIXELS_H
#define SORT_PIXELS_H

#include "GLCUDA.h"
#include <thrust/device_vector.h>


inline void outputGPUMemoryUsage()
{
	size_t free_byte;
	size_t total_byte;
	cuChk(cudaMemGetInfo(&free_byte, &total_byte));

	double free_db = (double)free_byte;
	double total_db = (double)total_byte;
	double used_db = total_db - free_db;

	printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
		used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0, total_db / 1024.0 / 1024.0);
}

void initPixels(cudaGraphicsResource *resource);
void sort_pixels(cudaGraphicsResource *resource);

void sort_points(cudaGraphicsResource *resource, float dx, float dy, float dz);
void sort_points(cudaGraphicsResource *resource, cudaGraphicsResource *idcs_resource, thrust::device_vector<float> &dotProductBuffer, float dx, float dy, float dz);

#endif