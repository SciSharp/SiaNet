#include "GatherScatterOps.h"
#include "TensorIter-inl.h"
#include "TensorApplyDim-inl.h"


template<typename T>
INLINE_FUNC void Gather_Apply(TensorRef* result, TensorRef* src, int dim, TensorRef* indices)
{
	auto func = [](
		T *rData, __int64 rSize, __int64 rStride,
		T *sData, __int64 sSize, __int64 sStride,
		T *iData, __int64 iSize, __int64 iStride)
	{
		for (int i = 0; i < iSize; ++i)
		{
			long idx = (long)*(iData + i * iStride);
			if (idx < 0 || idx >= sSize) { throw TSError("Invalid index in gather"); }

			*(rData + i*rStride) = sData[idx * sStride];
		}
	};

	ApplyDim3<T, T, T>(result, src, indices, dim, func);
}

int TS_Gather(TensorRef* result, TensorRef* src, int dim, TensorRef* indices)
{
	API_BEGIN()
		SWITCH_TENSOR_TYPE_ALL_CPU(result->elementType, Gather_Apply, result, src, dim, indices)
		API_END()
}


template<typename T>
INLINE_FUNC void Scatter_Apply(TensorRef* result, TensorRef* src, int dim, TensorRef* indices)
{
	auto func = [](
		T *rData, __int64 rSize, __int64 rStride,
		T *sData, __int64 sSize, __int64 sStride,
		T *iData, __int64 iSize, __int64 iStride)
	{
		for (int i = 0; i < iSize; ++i)
		{
			long idx = (long)*(iData + i * iStride);
			if (idx < 0 || idx >= rSize) { throw TSError("Invalid index in gather"); }

			rData[idx*rStride] = *(sData + i*sStride);
		}
	};

	ApplyDim3<T, T, T>(result, src, indices, dim, func);
}

int TS_Scatter(TensorRef* result, TensorRef* src, int dim, TensorRef* indices)
{
	API_BEGIN()
		SWITCH_TENSOR_TYPE_ALL_CPU(result->elementType, Scatter_Apply, result, src, dim, indices)
		API_END()
}


template<typename T>
INLINE_FUNC void ScatterFill_Apply(TensorRef* result, float value, int dim, TensorRef* indices)
{
	auto func = [value](
		T *rData, __int64 rSize, __int64 rStride,
		T *iData, __int64 iSize, __int64 iStride)
	{
		for (int i = 0; i < iSize; ++i)
		{
			long idx = (long)*(iData + i * iStride);
			if (idx < 0 || idx >= rSize) { throw TSError("Invalid index in gather"); }

			rData[idx*rStride] = T(value);
		}
	};

	ApplyDim2<T, T>(result, indices, dim, func);
}

int TS_ScatterFill(TensorRef* result, float value, int dim, TensorRef* indices)
{
	API_BEGIN()
		SWITCH_TENSOR_TYPE_ALL_CPU(result->elementType, ScatterFill_Apply, result, value, dim, indices)
		API_END()
}
