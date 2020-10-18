#include "GLCUDA.h"
#include <thrust/fill.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

void fillIntBuffer(cudaGraphicsResource *resource, int val)
{
	cuChk(cudaGraphicsMapResources(1, &resource, NULL));
	int* devPtr;
	size_t  size;
	cuChk(cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, resource));
	thrust::device_ptr<int> t_devPtr = thrust::device_pointer_cast(devPtr);

	int pointSize = sizeof(int);
	int numPoints = size / pointSize;

	thrust::fill(t_devPtr, t_devPtr + numPoints, val);

	cuChk(cudaGraphicsUnmapResources(1, &resource, NULL));
}

void fillIntTexture(cudaGraphicsResource *resource, int width, int height, int val)
{
	cuChk(cudaGraphicsMapResources(1, &resource, NULL));
	cudaArray_t arr;
	cuChk(cudaGraphicsSubResourceGetMappedArray(&arr, resource, 0, 0));

	thrust::device_vector<int> buffer(width*height);
	thrust::fill(buffer.begin(), buffer.end(), val);
	
	int *src = buffer.data().get();
	int spitch = sizeof(int) * width; // number of bytes of a row
	cuChk(cudaMemcpy2DToArray(arr, 0, 0, src, spitch, width, height, cudaMemcpyDeviceToDevice));

	cuChk(cudaGraphicsUnmapResources(1, &resource, NULL));
}
