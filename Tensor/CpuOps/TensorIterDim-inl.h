#pragma once

#include "General.h"


template<typename T>
struct TensorDimIterState
{
private:
	__int64* sizes;
	__int64* strides;
	int dimensionCount;
	int iterationDim;
	__int64* counter;

public:
	__int64 stride, size;
	T* data;



	TensorDimIterState(void* buffer, int dimCount, __int64* sizes, __int64* strides, int iterationDim)
	{
		this->sizes = sizes;
		this->strides = strides;
		this->iterationDim = iterationDim;
		this->dimensionCount = dimCount;

		data = (T*)buffer;

		this->size = sizes[iterationDim];
		this->stride = strides[iterationDim];


		counter = new __int64[dimCount];
		for (int i = 0; i < dimCount; ++i)
			counter[i] = 0;
	}

	~TensorDimIterState()
	{
		delete counter;
	}


	// Returns true if there is another block to iterate over,
	// returns false if we are at end of iteration
	INLINE_FUNC bool NextBlock()
	{
		if (dimensionCount == 1)
		{
			return false;
		}

		for (int i = 0; i < dimensionCount; ++i)
		{
			if (i == iterationDim)
			{
				if (i == dimensionCount - 1)
				{
					return false;
				}
				continue;
			}

			counter[i]++;
			data += strides[i];

			if (counter[i] == sizes[i])
			{
				if (i == dimensionCount - 1)
				{
					return false;
				}
				else
				{
					data -= counter[i] * strides[i];
					counter[i] = 0;
				}
			}
			else
			{
				break;
			}
		}

		return true;
	}
};