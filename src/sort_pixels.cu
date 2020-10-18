#include "sort_pixels.h"

#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#define     DIM    512 

// create a green/black pattern
__global__ void kernel(uchar4 *ptr)
{
	// map from threadIdx/BlockIdx to pixel position 
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	// now calculate the value at that position 
	float fx = x / (float)DIM - 0.5f;
	float fy = y / (float)DIM - 0.5f;
	unsigned char   green = 128 + 127 * sin(abs(fx * 100) - abs(fy * 100));

	// accessing uchar4 vs unsigned char* 
	ptr[offset].x = 0;
	ptr[offset].y = green;
	ptr[offset].z = 0;
	ptr[offset].w = 255;
}

void initPixels(cudaGraphicsResource *resource)
{
	cuChk(cudaGraphicsMapResources(1, &resource, NULL));
	uchar4* devPtr;
	size_t  size;

	cuChk(cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, resource));

	dim3    grid(DIM / 16, DIM / 16);
	dim3    threads(16, 16);
	kernel << <grid, threads >> >(devPtr);
	cuChk(cudaGraphicsUnmapResources(1, &resource, NULL));
}


struct sort_functor
{
	__host__ __device__
		bool operator()(uchar4 left, uchar4 right) const
	{
		return (left.y < right.y);
	}
};



void sort_pixels(cudaGraphicsResource *resource)
{
	cuChk(cudaGraphicsMapResources(1, &resource, NULL));
	uchar4* devPtr;
	size_t  size;

	cuChk(cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, resource));

	thrust::device_ptr<uchar4> tptr = thrust::device_pointer_cast(devPtr);
	thrust::sort(tptr, tptr + (DIM*DIM), sort_functor());
	cuChk(cudaGraphicsUnmapResources(1, &resource, NULL));
}

struct Point
{
	float x, y, z;
};

struct point_sort_functor
{
	point_sort_functor(float dx, float dy, float dz)
		:dx(dx), dy(dy), dz(dz)
	{

	}
	__host__ __device__
		bool operator()(const Point& left, const Point& right) const
	{
		float s1 = left.x * dx + left.y * dy + left.z *dz;
		float s2 = right.x * dx + right.y * dy + right.z * dz;
		return s1 < s2;
	}

	float dx;
	float dy;
	float dz;
};

void sort_points(cudaGraphicsResource *resource, float dx, float dy, float dz)
{
	cuChk(cudaGraphicsMapResources(1, &resource, NULL));
	Point* devPtr;
	size_t  size;
	cuChk(cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, resource));
	int pointSize = sizeof(Point);
	int numPoints = size / pointSize;
	thrust::device_ptr<Point> tptr = thrust::device_pointer_cast(devPtr);
	thrust::sort(tptr, tptr + numPoints, point_sort_functor(dx, dy, dz));
	cuChk(cudaGraphicsUnmapResources(1, &resource, NULL));
}

struct indexed_point_sort_functor
{
	indexed_point_sort_functor(Point *points, float dx, float dy, float dz)
		:points(points), dx(dx), dy(dy), dz(dz)
	{

	}
	__host__ __device__
		bool operator()(int l, int r) const
	{
		const Point &left = points[l];
		const Point &right = points[r];
		float s1 = left.x * dx + left.y * dy + left.z *dz;
		float s2 = right.x * dx + right.y * dy + right.z * dz;
		return s1 < s2;
	}

	Point * points;
	float dx;
	float dy;
	float dz;
};
struct DotProduct
{
	DotProduct(float dx, float dy, float dz)
		:dx(dx), dy(dy), dz(dz)
	{

	}
	__host__ __device__
		float operator()(const Point& left) const
	{
		float s1 = left.x * dx + left.y * dy + left.z *dz;
		return s1;
	}

	float dx;
	float dy;
	float dz;
};
struct IndexedDotProduct
{
	IndexedDotProduct(Point * points, float dx, float dy, float dz)
		:points(points), dx(dx), dy(dy), dz(dz)
	{

	}
	__host__ __device__
		float operator()(int l) const
	{
		const Point& left = points[l];
		float s1 = left.x * dx + left.y * dy + left.z *dz;
		return s1;
	}

	Point * points;
	float dx;
	float dy;
	float dz;
};

void sort_points(cudaGraphicsResource *resource, cudaGraphicsResource *idcs_resource, thrust::device_vector<float> &dotProductBuffer, float dx, float dy, float dz)
{
	cuChk(cudaGraphicsMapResources(1, &resource, NULL));
	Point* devPtr;
	size_t  size;
	cuChk(cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, resource));

	int pointSize = sizeof(Point);
	int numPoints = size / pointSize;

	cuChk(cudaGraphicsMapResources(1, &idcs_resource, NULL));
	int* idcs_devPtr;
	size_t  idcs_size;
	cuChk(cudaGraphicsResourceGetMappedPointer((void**)&idcs_devPtr, &idcs_size, idcs_resource));
	thrust::device_ptr<int> idcs_dptr = thrust::device_pointer_cast(idcs_devPtr);

	std::cout << " before resize " << std::endl;
	//thrust::device_vector<float> dotProductBuffer(numPoints);
	if (dotProductBuffer.size() < numPoints)
	{
		//dotProductBuffer.resize((int)(numPoints * 1.5));
		dotProductBuffer.resize(numPoints);
	}
	std::cout << " before transform " << std::endl;
	thrust::transform(idcs_dptr, idcs_dptr + numPoints, dotProductBuffer.begin(), IndexedDotProduct(devPtr, dx, dy, dz));

	std::cout << " before sort " << std::endl;
	outputGPUMemoryUsage();
	thrust::sort_by_key(dotProductBuffer.begin(), dotProductBuffer.begin() + numPoints, idcs_dptr); //cannot use dotProductBuffer.end()! because it is larger than numpoints

	cuChk(cudaGraphicsUnmapResources(1, &idcs_resource, NULL));
	cuChk(cudaGraphicsUnmapResources(1, &resource, NULL));
}
