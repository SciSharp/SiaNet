
#include "DimensionwiseOps.h"
#include "TensorApplyDim-inl.h"
#include "TensorApply-inl.h"
#include <cmath>
#include <iostream>

template <typename T> INLINE_FUNC T min(T a, T b) {
	if (a < b)
		return a;
	return b;
}

template <typename T> INLINE_FUNC T max(T a, T b) {
	if (a > b)
		return a;
	return b;
}


template<typename T>
INLINE_FUNC void Sum_Apply(TensorRef* result, TensorRef* src, int dimension)
{
	auto func = [](T *r, __int64 rSize, __int64 rStride, T *s, __int64 sSize, __int64 sStride)
	{
		T sum = T(0);
		for (__int64 i = 0; i < sSize; ++i)
		{
			sum += s[i*sStride];
		}
		*r = sum;
	};
	ApplyDim2<T, T>(result, src, dimension, func);
}

int TS_Sum(TensorRef* result, TensorRef* src, int dimension)
{
	API_BEGIN()
	SWITCH_TENSOR_TYPE_ALL_CPU(result->elementType, Sum_Apply, result, src, dimension)
	API_END()
}



template<typename T>
INLINE_FUNC void Prod_Apply(TensorRef* result, TensorRef* src, int dimension)
{
	auto func = [](T *r, __int64 rSize, __int64 rStride, T *s, __int64 sSize, __int64 sStride)
	{
		T value = T(1);
		for (__int64 i = 0; i < sSize; ++i)
		{
			value *= s[i*sStride];
		}
		*r = value;
	};
	ApplyDim2<T, T>(result, src, dimension, func);
}

int TS_Prod(TensorRef* result, TensorRef* src, int dimension)
{
	API_BEGIN()
	SWITCH_TENSOR_TYPE_ALL_CPU(result->elementType, Prod_Apply, result, src, dimension)
	API_END()
}



template<typename T>
INLINE_FUNC void Min_Apply(TensorRef* result, TensorRef* src, int dimension)
{
	auto func = [](T *r, __int64 rSize, __int64 rStride, T *s, __int64 sSize, __int64 sStride)
	{
		T value = s[0];
		for (__int64 i = 1; i < sSize; ++i)
		{
			value = min(value, s[i*sStride]);
		}
		*r = value;
	};
	ApplyDim2<T, T>(result, src, dimension, func);
}

int TS_Min(TensorRef* result, TensorRef* src, int dimension)
{
	API_BEGIN()
		SWITCH_TENSOR_TYPE_ALL_CPU(result->elementType, Min_Apply, result, src, dimension)
		API_END()
}


template<typename T>
INLINE_FUNC void Max_Apply(TensorRef* result, TensorRef* src, int dimension)
{
	auto func = [](T *r, __int64 rSize, __int64 rStride, T *s, __int64 sSize, __int64 sStride)
	{
		T value = s[0];
		for (__int64 i = 1; i < sSize; ++i)
		{
			value = max(value, s[i*sStride]);
		}
		*r = value;
	};
	ApplyDim2<T, T>(result, src, dimension, func);
}

int TS_Max(TensorRef* result, TensorRef* src, int dimension)
{
	API_BEGIN()
	SWITCH_TENSOR_TYPE_ALL_CPU(result->elementType, Max_Apply, result, src, dimension)
	API_END()
}

template<typename T>
INLINE_FUNC void Argmin_Apply(TensorRef* resultIndices, TensorRef* src, int dimension)
{
	auto func = [](
		float *rIndVal, __int64 rIndSize, __int64 rIndStride,
		T *s, __int64 sSize, __int64 sStride)
	{
		T value = s[0];
		float index = 0;
		for (__int64 i = 1; i < sSize; ++i)
		{
			T currentVal = s[i*sStride];
			if (currentVal < value)
			{
				value = currentVal;
				index = (float)i;
			}
		}
		*rIndVal = index;
	};
	ApplyDim2<float, T>(resultIndices, src, dimension, func);
}

OPS_API int TS_Argmin(TensorRef *resultIndices, TensorRef* src, int dimension)
{
	API_BEGIN()

		if (resultIndices->elementType != DType::Float32)
		{
			throw TSError("result indices must have type Float32");
		}

	SWITCH_TENSOR_TYPE_ALL_CPU(src->elementType, Argmin_Apply, resultIndices, src, dimension)
		API_END()
}

template<typename T>
INLINE_FUNC void Argmax_Apply(TensorRef* resultIndices, TensorRef* src, int dimension)
{
	auto func = [](
		float *rIndVal, __int64 rIndSize, __int64 rIndStride,
		T *s, __int64 sSize, __int64 sStride)
	{
		T value = s[0];
		float index = 0;
		for (__int64 i = 1; i < sSize; ++i)
		{
			T currentVal = s[i*sStride];
			if (currentVal > value)
			{
				value = currentVal;
				index = (float)i;
			}
		}
		*rIndVal = index;
	};
	ApplyDim2<float, T>(resultIndices, src, dimension, func);
}

OPS_API int TS_Argmax(TensorRef *resultIndices, TensorRef* src, int dimension)
{
	API_BEGIN()
	
	if(resultIndices->elementType != DType::Float32)
	{
		throw TSError("result indices must have type Float32");
	}

	SWITCH_TENSOR_TYPE_ALL_CPU(src->elementType, Argmax_Apply, resultIndices, src, dimension)
	API_END()
}

template<typename T>
INLINE_FUNC void Mean_Apply(TensorRef* result, TensorRef* src, int dimension)
{
	auto func = [](T *r, __int64 rSize, __int64 rStride, T *s, __int64 sSize, __int64 sStride)
	{
		T sum = T(0);
		for (__int64 i = 0; i < sSize; ++i)
		{
			sum += s[i*sStride];
		}
		*r = sum / sSize;
	};
	ApplyDim2<T, T>(result, src, dimension, func);
}

int TS_Mean(TensorRef* result, TensorRef* src, int dimension)
{
	API_BEGIN()
	SWITCH_TENSOR_TYPE_FLOAT(result->elementType, Mean_Apply, result, src, dimension)
	API_END()
}


template<typename T>
INLINE_FUNC void Norm_Apply(TensorRef* result, TensorRef* src, int dimension, float value)
{
	if (value == 0)
	{
		auto func = [](T *r, __int64 rSize, __int64 rStride, T *s, __int64 sSize, __int64 sStride)
		{
			T sum = T(0);
			for (__int64 i = 0; i < sSize; ++i)
			{
				sum += s[i*sStride] != T(0);
			}
			*r = sum;
		};
		ApplyDim2<T, T>(result, src, dimension, func);
	}
	else
	{
		auto func = [value](T *r, __int64 rSize, __int64 rStride, T *s, __int64 sSize, __int64 sStride)
		{
			T sum = T(0);
			for (__int64 i = 0; i < sSize; ++i)
			{
				sum += pow(fabs(s[i*sStride]), value);
			}
			*r = T(pow(sum, 1.0 / value));
		};
		ApplyDim2<T, T>(result, src, dimension, func);
	}
}

int TS_Norm(TensorRef* result, TensorRef* src, int dimension, float value)
{
	API_BEGIN()
	SWITCH_TENSOR_TYPE_FLOAT(result->elementType, Norm_Apply, result, src, dimension, value)
	API_END()
}

template<typename T>
INLINE_FUNC void Std_Apply(TensorRef* result, TensorRef* src, int dimension, bool normByN)
{
	auto func = [normByN](T *r, __int64 rSize, __int64 rStride, T *s, __int64 sSize, __int64 sStride)
	{
		T sum = T(0);
		T sumOfSquares = T(0);
		for (__int64 i = 0; i < sSize; ++i)
		{
			T item = s[i*sStride];
			sum += item;
			sumOfSquares += item * item;
		}

		if (normByN)
		{
			T mean = sum / sSize;
			T x = (sumOfSquares / sSize) - mean * mean;
			x = x < 0 ? T(0) : x;
			*r = (T)sqrt(x);
		}
		else
		{
			T mean = sum / sSize;
			T factor = T(sSize) / T(sSize - 1);
			T x = (sumOfSquares / sSize) - mean * mean * factor;
			x = x < 0 ? 0 : x;
			*r = (T)sqrt(x);
		}
	};
	ApplyDim2<T, T>(result, src, dimension, func);
}

int TS_Std(TensorRef* result, TensorRef* src, int dimension, bool normByN)
{
	API_BEGIN()
	SWITCH_TENSOR_TYPE_FLOAT(result->elementType, Std_Apply, result, src, dimension, normByN)
	API_END()
}


template<typename T>
INLINE_FUNC void Var_Apply(TensorRef* result, TensorRef* src, int dimension, bool normByN)
{
	auto func = [normByN](T *r, __int64 rSize, __int64 rStride, T *s, __int64 sSize, __int64 sStride)
	{
		T sum = T(0);
		T sumOfSquares = T(0);
		for (__int64 i = 0; i < sSize; ++i)
		{
			T item = s[i*sStride];
			sum += item;
			sumOfSquares += item * item;
		}

		if (normByN)
		{
			T mean = sum / sSize;
			T x = (sumOfSquares / sSize) - mean * mean;
			x = x < 0 ? T(0) : x;
			*r = x;
		}
		else
		{
			T mean = sum / sSize;
			T factor = T(sSize) / T(sSize - 1);
			T x = (sumOfSquares / sSize) - mean * mean * factor;
			x = x < 0 ? 0 : x;
			*r = x;
		}
	};
	ApplyDim2<T, T>(result, src, dimension, func);
}

int TS_Var(TensorRef* result, TensorRef* src, int dimension, bool normByN)
{
	API_BEGIN()
	SWITCH_TENSOR_TYPE_FLOAT(result->elementType, Var_Apply, result, src, dimension, normByN)
	API_END()
}



template<typename T>
INLINE_FUNC void SumAll_Apply(TensorRef* result, TensorRef* src)
{
	auto resultPtr = (T*)result->buffer;
	*resultPtr = 0;

	auto func = [resultPtr](T *s)
	{
		*resultPtr += *s;
	};
	Apply1<T>(src, func);
}

int TS_SumAll(TensorRef* result, TensorRef* src)
{
	API_BEGIN()
	SWITCH_TENSOR_TYPE_ALL_CPU(src->elementType, SumAll_Apply, result, src)
	API_END()
}

template<typename T>
INLINE_FUNC void ProdAll_Apply(TensorRef* result, TensorRef* src)
{
	auto resultPtr = (T*)result->buffer;
	*resultPtr = T(1);

	auto func = [resultPtr](T *s)
	{
		*resultPtr *= *s;
	};
	Apply1<T>(src, func);
}

int TS_ProdAll(TensorRef* result, TensorRef* src)
{
	API_BEGIN()
	SWITCH_TENSOR_TYPE_ALL_CPU(src->elementType, ProdAll_Apply, result, src)
	API_END()
}

template<typename T>
INLINE_FUNC void MinAll_Apply(TensorRef* result, TensorRef* src)
{
	if (src->ElementCount() == 0)
		throw new TSError("MinAll: input must have at least one element");

	auto resultPtr = (T*)result->buffer;
	*resultPtr = ((T*)src->buffer)[0];

	auto func = [resultPtr](T *s)
	{
		*resultPtr = min(*resultPtr, *s);
	};
	Apply1<T>(src, func);
}

int TS_MinAll(TensorRef* result, TensorRef* src)
{
	API_BEGIN()
	SWITCH_TENSOR_TYPE_ALL_CPU(src->elementType, MinAll_Apply, result, src)
	API_END()
}

template<typename T>
INLINE_FUNC void MaxAll_Apply(TensorRef* result, TensorRef* src)
{
	if (src->ElementCount() == 0)
		throw new TSError("MaxAll: input must have at least one element");

	auto resultPtr = (T*)result->buffer;
	*resultPtr = ((T*)src->buffer)[0];

	auto func = [resultPtr](T *s)
	{
		*resultPtr = max(*resultPtr, *s);
	};
	Apply1<T>(src, func);
}

int TS_MaxAll(TensorRef* result, TensorRef* src)
{
	API_BEGIN()
	SWITCH_TENSOR_TYPE_ALL_CPU(src->elementType, MaxAll_Apply, result, src)
	API_END()
}



template<typename T>
INLINE_FUNC T MeanOf(TensorRef* src)
{
	T result = T(0);

	auto func = [&result](T *s)
	{
		result += *s;
	};
	Apply1<T>(src, func);
	result = result / src->ElementCount();
	return result;
}

template<typename T>
INLINE_FUNC void MeanAll_Apply(TensorRef* result, TensorRef* src)
{
	auto resultPtr = (T*)result->buffer;
	*resultPtr = MeanOf<T>(src);
}

int TS_MeanAll(TensorRef* result, TensorRef* src)
{
	API_BEGIN()
	SWITCH_TENSOR_TYPE_FLOAT(src->elementType, MeanAll_Apply, result, src)
	API_END()
}



template<typename T>
INLINE_FUNC T VarOf(TensorRef* src)
{
	T mean = MeanOf<T>(src);
	T sum = T(0);

	auto func = [&sum, mean](T *s)
	{
		sum += (*s - mean) * (*s - mean);
	};
	Apply1<T>(src, func);

	return sum / (src->ElementCount() - 1);
}

template<typename T>
INLINE_FUNC void VarAll_Apply(TensorRef* result, TensorRef* src)
{
	auto resultPtr = (T*)result->buffer;
	*resultPtr = VarOf<T>(src);
}

int TS_VarAll(TensorRef* result, TensorRef* src)
{
	API_BEGIN()
	SWITCH_TENSOR_TYPE_FLOAT(src->elementType, VarAll_Apply, result, src)
	API_END()
}


template<typename T>
INLINE_FUNC void StdAll_Apply(TensorRef* result, TensorRef* src)
{
	auto resultPtr = (T*)result->buffer;
	*resultPtr = sqrt(VarOf<T>(src));
}

int TS_StdAll(TensorRef* result, TensorRef* src)
{
	API_BEGIN()
	SWITCH_TENSOR_TYPE_FLOAT(src->elementType, StdAll_Apply, result, src)
	API_END()
}



template<typename T>
INLINE_FUNC T NormOf(TensorRef* src, T value)
{
	if (value == 0)
	{
		T sum = T(0);
		auto func = [&sum](T *s)
		{
			sum += (*s != T(0));
		};
		Apply1<T>(src, func);
		return sum;
	}
	else
	{
		T sum = T(0);
		auto func = [&sum, value](T *s)
		{
			sum += pow(fabs(*s), value);
		};
		Apply1<T>(src, func);
		return T(pow(sum, 1.0 / value));
	}
}

template<typename T>
INLINE_FUNC void NormAll_Apply(TensorRef* result, TensorRef* src, float value)
{
	auto resultPtr = (T*)result->buffer;
	*resultPtr = NormOf<T>(src, (T)value);
}

int TS_NormAll(TensorRef* result, TensorRef* src, float value)
{
	API_BEGIN()
	SWITCH_TENSOR_TYPE_FLOAT(src->elementType, NormAll_Apply, result, src, value)
	API_END()
}

template<typename T>
INLINE_FUNC void Diag_Apply(TensorRef* result, TensorRef* src)
{
	T* result_p = (T*)result->buffer;
	T* src_p = (T*)src->buffer;
	__int64 sSize = src->ElementCount();
	
	__int64 pos = 0;

#pragma omp parallel for
	for (__int64 i = 0; i < sSize; ++i)
	{
		for (__int64 j = 0; j < sSize; ++j)
		{
			if (i == j)
			{
				result_p[pos] = src_p[i];
			}
			else
			{
				result_p[pos] = T(0);
			}

			pos++;
		}
	}
}

int TS_Diag(TensorRef* result, TensorRef* src)
{
	API_BEGIN()
	SWITCH_TENSOR_TYPE_ALL_CPU(result->elementType, Diag_Apply, result, src)
	API_END()
}