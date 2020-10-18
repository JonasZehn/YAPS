#ifndef COMPUTE_COLORS_PARAMETERS_H
#define COMPUTE_COLORS_PARAMETERS_H

#include "VectorHelper.h"

struct ComputeColorsParameters
{
	float3 baseColor;
	float3 illuminatedColor;
	float illuminationMultiplier;
};

#endif