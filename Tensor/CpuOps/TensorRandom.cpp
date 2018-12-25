#include "TensorRandom.h"
#include "TensorApply-inl.h"


int TS_NewRNG(UniformGenerator** result)
{
	API_BEGIN()
	*result = new UniformGenerator();
	API_END()
}

int TS_DeleteRNG(UniformGenerator *rng)
{
	API_BEGIN()
	delete rng;
	API_END()
}

int TS_SetRNGSeed(UniformGenerator* rng, int newSeed)
{
	API_BEGIN()
	rng->SetSeed(newSeed);
	API_END()
}

//
// Some utility functions for generating single random values
//

INLINE_FUNC double ScalarUniform(UniformGenerator* rng, double min, double max)
{
	return rng->NextUniform() * (max - min) + min;
}

INLINE_FUNC double ScalarNormal(UniformGenerator* rng, double mean, double stdv)
{
	return rng->NextNormal(mean, stdv);
}

INLINE_FUNC double ScalarExponential(UniformGenerator* rng, double lambda)
{
	return -1.0 / lambda * log(1.0 - rng->NextUniform());
}

INLINE_FUNC double ScalarCauchy(UniformGenerator* rng, double median, double sigma)
{
	return median + sigma * tan(M_PI * (rng->NextUniform() - 0.5));
}

INLINE_FUNC double ScalarLogNormal(UniformGenerator* rng, double mean, double stdv)
{
	Assert(stdv > 0, "LogNormal requires stdv > 0");

	double zm = mean * mean;
	double zs = stdv * stdv;

	double lmean = log(zm / sqrt(zs + zm));
	double lstdv = sqrt(log(zs / zm + 1));

	return exp(ScalarNormal(rng, lmean, lstdv));
}

INLINE_FUNC int ScalarGeometric(UniformGenerator* rng, double p)
{
	Assert(p > 0 && p < 1, "Geometric requires 0 < p < 1");
	return (int)(log(1 - rng->NextUniform()) / log(p)) + 1;
}

INLINE_FUNC int ScalarBernoulli(UniformGenerator* rng, double p)
{
	Assert(p >= 0 && p <= 1, "Bernoulli requires 0 <= p <= 1");
	return rng->NextUniform() <= p ? 1 : 0;
}

//
// Fin
//

template<typename T>
INLINE_FUNC void RandomUniform_Apply(UniformGenerator* rng, TensorRef* result, float min, float max)
{
	auto func = [rng, min, max](T *r) { *r = (T)ScalarUniform(rng, min, max); };
	Apply1<T>(result, func);
}

int TS_RandomUniform(UniformGenerator* rng, TensorRef* result, float min, float max)
{
	API_BEGIN()
	SWITCH_TENSOR_TYPE_FLOAT(result->elementType, RandomUniform_Apply, rng, result, min, max)
	API_END()
}


template<typename T>
INLINE_FUNC void RandomNormal_Apply(UniformGenerator* rng, TensorRef* result, float mean, float stdv)
{
	auto func = [rng, mean, stdv](T *r) { *r = (T)ScalarNormal(rng, mean, stdv); };
	Apply1<T>(result, func);
}

int TS_RandomNormal(UniformGenerator* rng, TensorRef* result, float mean, float stdv)
{
	API_BEGIN()
	SWITCH_TENSOR_TYPE_FLOAT(result->elementType, RandomNormal_Apply, rng, result, mean, stdv)
	API_END()
}


template<typename T>
INLINE_FUNC void RandomExponential_Apply(UniformGenerator* rng, TensorRef* result, float lambda)
{
	auto func = [rng, lambda](T *r) { *r = (T)ScalarExponential(rng, lambda); };
	Apply1<T>(result, func);
}

int TS_RandomExponential(UniformGenerator* rng, TensorRef* result, float lambda)
{
	API_BEGIN()
	SWITCH_TENSOR_TYPE_FLOAT(result->elementType, RandomExponential_Apply, rng, result, lambda)
	API_END()
}


template<typename T>
INLINE_FUNC void RandomCauchy_Apply(UniformGenerator* rng, TensorRef* result, float median, float sigma)
{
	auto func = [rng, median, sigma](T *r) { *r = (T)ScalarCauchy(rng, median, sigma); };
	Apply1<T>(result, func);
}

int TS_RandomCauchy(UniformGenerator* rng, TensorRef* result, float median, float sigma)
{
	API_BEGIN()
	SWITCH_TENSOR_TYPE_FLOAT(result->elementType, RandomCauchy_Apply, rng, result, median, sigma)
	API_END()
}


template<typename T>
INLINE_FUNC void RandomLogNormal_Apply(UniformGenerator* rng, TensorRef* result, float mean, float stdv)
{
	auto func = [rng, mean, stdv](T *r) { *r = (T)ScalarLogNormal(rng, mean, stdv); };
	Apply1<T>(result, func);
}

int TS_RandomLogNormal(UniformGenerator* rng, TensorRef* result, float mean, float stdv)
{
	API_BEGIN()
	SWITCH_TENSOR_TYPE_FLOAT(result->elementType, RandomLogNormal_Apply, rng, result, mean, stdv)
	API_END()
}


template<typename T>
INLINE_FUNC void RandomGeometric_Apply(UniformGenerator* rng, TensorRef* result, float p)
{
	auto func = [rng, p](T *r) { *r = (T)ScalarGeometric(rng, p); };
	Apply1<T>(result, func);
}

int TS_RandomGeometric(UniformGenerator* rng, TensorRef* result, float p)
{
	API_BEGIN()
	SWITCH_TENSOR_TYPE_ALL_CPU(result->elementType, RandomGeometric_Apply, rng, result, p)
	API_END()
}


template<typename T>
INLINE_FUNC void RandomBernoulli_Apply(UniformGenerator* rng, TensorRef* result, float p)
{
	auto func = [rng, p](T *r) { *r = (T)ScalarBernoulli(rng, p); };
	Apply1<T>(result, func);
}

int TS_RandomBernoulli(UniformGenerator* rng, TensorRef* result, float p)
{
	API_BEGIN()
	SWITCH_TENSOR_TYPE_ALL_CPU(result->elementType, RandomBernoulli_Apply, rng, result, p)
	API_END()
}
