#pragma once


#include "TensorIterDim-inl.h"

template<typename T1, typename FuncType>
INLINE_FUNC void ApplyDim1(TensorRef* tensor1, int iterationDim, FuncType func)
{
	TensorDimIterState<T1> tensor1Iter(tensor1->buffer, tensor1->dimCount, tensor1->sizes, tensor1->strides, iterationDim);

	do
	{
		func(tensor1Iter.data, tensor1Iter.size, tensor1Iter.stride);

	} while (tensor1Iter.NextBlock());
}


template<typename T1, typename T2, typename FuncType>
INLINE_FUNC void ApplyDim2(TensorRef* tensor1, TensorRef* tensor2, int iterationDim, FuncType func)
{
	TensorDimIterState<T1> tensor1Iter(tensor1->buffer, tensor1->dimCount, tensor1->sizes, tensor1->strides, iterationDim);
	TensorDimIterState<T2> tensor2Iter(tensor2->buffer, tensor2->dimCount, tensor2->sizes, tensor2->strides, iterationDim);

	do
	{
		func(tensor1Iter.data, tensor1Iter.size, tensor1Iter.stride,
			tensor2Iter.data, tensor2Iter.size, tensor2Iter.stride);

	} while (tensor1Iter.NextBlock() && tensor2Iter.NextBlock());
}

template<typename T1, typename T2, typename T3, typename FuncType>
INLINE_FUNC void ApplyDim3(TensorRef* tensor1, TensorRef* tensor2, TensorRef* tensor3, int iterationDim, FuncType func)
{
	TensorDimIterState<T1> tensor1Iter(tensor1->buffer, tensor1->dimCount, tensor1->sizes, tensor1->strides, iterationDim);
	TensorDimIterState<T2> tensor2Iter(tensor2->buffer, tensor2->dimCount, tensor2->sizes, tensor2->strides, iterationDim);
	TensorDimIterState<T3> tensor3Iter(tensor3->buffer, tensor3->dimCount, tensor3->sizes, tensor3->strides, iterationDim);

	do
	{
		func(tensor1Iter.data, tensor1Iter.size, tensor1Iter.stride,
			tensor2Iter.data, tensor2Iter.size, tensor2Iter.stride,
			tensor3Iter.data, tensor3Iter.size, tensor3Iter.stride);

	} while (tensor1Iter.NextBlock() && tensor2Iter.NextBlock() && tensor3Iter.NextBlock());
}
