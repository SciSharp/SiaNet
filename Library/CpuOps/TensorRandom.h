#pragma once

#include "General.h"
#include "RandomGenerator.h"
#include "TensorRef.h"


OPS_API int TS_NewRNG(UniformGenerator** result);
OPS_API int TS_DeleteRNG(UniformGenerator* rng);
OPS_API int TS_SetRNGSeed(UniformGenerator* rng, int newSeed);

OPS_API int TS_RandomUniform(UniformGenerator* rng, TensorRef* result, float min, float max);
OPS_API int TS_RandomNormal(UniformGenerator* rng, TensorRef* result, float mean, float stdv);
OPS_API int TS_RandomExponential(UniformGenerator* rng, TensorRef* result, float lambda);
OPS_API int TS_RandomCauchy(UniformGenerator* rng, TensorRef* result, float median, float sigma);
OPS_API int TS_RandomLogNormal(UniformGenerator* rng, TensorRef* result, float mean, float stdv);
OPS_API int TS_RandomGeometric(UniformGenerator* rng, TensorRef* result, float p);
OPS_API int TS_RandomBernoulli(UniformGenerator* rng, TensorRef* result, float p);
