#pragma once

#include "General.h"


INLINE_FUNC __int64 ElementCount(int dimCount, __int64* sizes)
{
	if (dimCount == 0)
		return 0;

	__int64 total = 1L;
	for (int i = 0; i < dimCount; ++i)
		total *= sizes[i];
	return total;
}

// A port of the algorithm from the Torch7 TH_TENSOR_APPLY macros
template<typename T>
struct TensorIterState
{
private:
	__int64* sizes;
	__int64* strides;

public:
	__int64 elementCount, stride, size;
	int dim;
	__int64* counter;

	__int64 index;
	T* data;


	TensorIterState(void* buffer, int dimCount, __int64* sizes, __int64* strides)
	{
		this->sizes = sizes;
		this->strides = strides;

		index = 0;
		data = (T*)buffer;

		for (dim = dimCount - 1; dim >= 0; dim--)
		{
			if (sizes[dim] != 1)
				break;
		}

		// Get stride for dimension
		stride = (dim == -1 ? 0 : strides[dim]);


		// Find largest contiguous section
		// Note: this updates dim and size
		size = 1;
		for (dim = dimCount - 1; dim >= 0; dim--)
		{
			if (strides[dim] == size)
			{
				size *= sizes[dim];
			}
			else
			{
				break;
			}
		}


		// Counter keeps track of how many iterations have been performed on each dimension
		// that is *not* part of the above contiguous block
		// Iterations are performed from highest dimension index to lowest.
		// When a complete iteration of dimension i is finished, the counter for dim i-1 gets incremented by 1
		counter = new __int64[dim + 1];
		for (int i = 0; i < dim + 1; ++i)
			counter[i] = 0;

		elementCount = ElementCount(dimCount, sizes);
	}

	~TensorIterState()
	{
		delete counter;
	}

	INLINE_FUNC bool ReachedBlockEnd()
	{
		return !(this->index < this->size);
	}

	INLINE_FUNC void BlockStep()
	{
		this->index++;
		this->data += this->stride;
	}

	// Returns true if there is another block to iterate over,
	// returns false if we are at end of iteration
	INLINE_FUNC bool NextBlock()
	{
		// If not at end of current block yet, do nothing
		if (this->index == this->size)
		{
			// If contiguous block encompassed all dimensions, we are done
			if (this->dim == -1)
				return false;

			// Reset data offset
			this->data -= this->size * this->stride;

			// Update counter and data for next contiguous block
			for (__int64 j = this->dim; j >= 0; --j)
			{
				this->counter[j]++;
				this->data += strides[j];

				if (this->counter[j] == sizes[j])
				{
					if (j == 0)
					{
						return false;
					}
					else
					{
						this->data -= this->counter[j] * strides[j];
						this->counter[j] = 0;
					}
				}
				else
				{
					break;
				}
			}

			this->index = 0;
		}

		return true;
	}
};