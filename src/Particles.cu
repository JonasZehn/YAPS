
#include "GLCUDA.h"

#include <thrust/fill.h>
#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <vector_functions.hpp>
#include <thrust/iterator/counting_iterator.h>

#include "Texture.h"
#include "ComputeColorsParameters.h"
#include "VectorHelper.h"


void copyParticles(cudaGraphicsResource *vtcsResource, float *dest)
{
	cuChk(cudaGraphicsMapResources(1, &vtcsResource, NULL));
	float* devPtr;
	size_t  size;
	cuChk(cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, vtcsResource));
	thrust::device_ptr<float> t_devPtr = thrust::device_pointer_cast(devPtr);

	cuChk(cudaMemcpy(dest, devPtr, size, cudaMemcpyDeviceToHost));

	cuChk(cudaGraphicsUnmapResources(1, &vtcsResource, NULL));
}

void fillFloat(cudaGraphicsResource *resource, float val)
{
	cuChk(cudaGraphicsMapResources(1, &resource, NULL));
	float* devPtr;
	size_t  size;
	cuChk(cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, resource));
	thrust::device_ptr<float> t_devPtr = thrust::device_pointer_cast(devPtr);

	int pointSize = sizeof(float);
	int numPoints = size / pointSize;
	
	thrust::fill(t_devPtr, t_devPtr + numPoints, val);

	cuChk(cudaGraphicsUnmapResources(1, &resource, NULL));
}

struct computeColor
{
	computeColor(const ComputeColorsParameters &parameters)
		:parameters(parameters)
	{

	}
	__host__ __device__ float3 operator()(int illu)
	{
		float fillu = float(illu);
		fillu /= 1000.f;
		fillu *= parameters.illuminationMultiplier;
		fillu = fillu > 1.0f ? 1.0f : (fillu < 0.0f ? 0.0f : fillu);
		//float3 v = (1.0 - fillu) * parameters.baseColor + fillu * parameters.illuminatedColor;
		float3 v = parameters.baseColor + fillu * parameters.illuminatedColor;
		v = clamp(v, 0.f, 1.f);
		return v;
		//return (0.4f) * make_float3(0.0f, 0.5f, 0.2f);
	}

	ComputeColorsParameters parameters;
};

void computeColors(TextureBuffer *illuTexture, cudaGraphicsResource *illuminatedResource, const ComputeColorsParameters &parameters)
{
	cuChk(cudaGraphicsMapResources(1, &illuminatedResource, NULL));
	float3* devPtr;
	size_t  size;
	cuChk(cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, illuminatedResource));
	thrust::device_ptr<float3> t_devPtr = thrust::device_pointer_cast(devPtr);

	int pointSize = sizeof(float3);
	int numPoints = size / pointSize;

	cudaGraphicsResource *illuResource = illuTexture->getCUDAResource();

	cuChk(cudaGraphicsMapResources(1, &illuResource, NULL));
	int* illu_devPtr;
	size_t  illu_size;
	cuChk(cudaGraphicsResourceGetMappedPointer((void**)&illu_devPtr, &illu_size, illuResource));
	thrust::device_ptr<int> t_illu_devPtr = thrust::device_pointer_cast(illu_devPtr);

	thrust::transform(t_illu_devPtr, t_illu_devPtr + numPoints, t_devPtr, computeColor(parameters));

	cuChk(cudaGraphicsUnmapResources(1, &illuResource, NULL));
	cuChk(cudaGraphicsUnmapResources(1, &illuminatedResource, NULL));
}

struct computeColor2
{
	computeColor2(float3 * colors, int *illuminated, float3 * illuminatedColor, const ComputeColorsParameters &parameters)
		:
		colors(colors),
		illuminated(illuminated),
		illuminatedColor(illuminatedColor),
		parameters(parameters)
	{

	}
	__host__ __device__ void operator()(int idx) const
	{
		float fillu = float(illuminated[idx]);
		fillu *= parameters.illuminationMultiplier;
		fillu /= 1000.f;
		//float ambient = 0.4f;
		//fillu += ambient;
		//fillu = fillu > 1.0f ? 1.0f : (fillu < 0.0f ? 0.0f : fillu);

		//float3 v = parameters.baseColor + (0.5 + 0.5 * fillu) * colors[idx];
		//1
		//float3 v = colors[idx] + 0.15 * fillu * parameters.illuminatedColor;
		float3 v = parameters.baseColor + 0.15 * fillu * parameters.illuminatedColor;
		//2
		//float3 v = ((1.0 - fillu) * 0.8 + fillu * 1.0) * colors[idx];
		//3

		//float f = fillu > 0.8f ? fillu - 0.8f : 0.0;
		//float3 white = { 1.0f, 1.0f, 1.0f };
		//float3 v = ((1.0 - fillu) * 0.8f) * colors[idx] + fillu * 1.0 * (colors[idx] + 0.2f * white);

		//float3 v = fillu * colors[idx];

		v = clamp(v, 0.f, 1.f);
		illuminatedColor[idx] = v;
	}

	float3 * colors;
	int *illuminated;
	float3 * illuminatedColor;
	ComputeColorsParameters parameters;
};

void computeColors2(cudaGraphicsResource *colorsResource, TextureBuffer *illuTexture, cudaGraphicsResource *illuminatedResource, const ComputeColorsParameters &parameters)
{
	cuChk(cudaGraphicsMapResources(1, &colorsResource, NULL));
	float3* devPtr;
	size_t  size;
	cuChk(cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &size, colorsResource));

	int pointSize = sizeof(float3);
	int numPoints = size / pointSize;

	cudaGraphicsResource *illuResource = illuTexture->getCUDAResource();
	cuChk(cudaGraphicsMapResources(1, &illuResource, NULL));
	int* illu_devPtr;
	size_t  illu_size;
	cuChk(cudaGraphicsResourceGetMappedPointer((void**)&illu_devPtr, &illu_size, illuResource));

	cuChk(cudaGraphicsMapResources(1, &illuminatedResource, NULL));
	float3* illuColor_devPtr;
	size_t  illuColor_size;
	cuChk(cudaGraphicsResourceGetMappedPointer((void**)&illuColor_devPtr, &illuColor_size, illuminatedResource));

	thrust::for_each(thrust::make_counting_iterator(0), thrust::make_counting_iterator(numPoints), computeColor2(devPtr, illu_devPtr, illuColor_devPtr, parameters));

	cuChk(cudaGraphicsUnmapResources(1, &illuminatedResource, NULL));
	cuChk(cudaGraphicsUnmapResources(1, &illuResource, NULL));
	cuChk(cudaGraphicsUnmapResources(1, &colorsResource, NULL));
}

