#pragma once

#include "TensorIter-inl.h"

template<typename T1, typename FuncType>
INLINE_FUNC void Apply1(TensorRef* tensor1, FuncType func)
{
	TensorIterState<T1> tensor1Iter(tensor1->buffer, tensor1->dimCount, tensor1->sizes, tensor1->strides);

	do
	{
		for (; !tensor1Iter.ReachedBlockEnd(); tensor1Iter.BlockStep())
		{
			func(tensor1Iter.data);
		}

	} while (tensor1Iter.NextBlock());
}


template<typename T1, typename T2, typename FuncType>
INLINE_FUNC void Apply2(TensorRef* tensor1, TensorRef* tensor2, FuncType func)
{
	TensorIterState<T1> tensor1Iter(tensor1->buffer, tensor1->dimCount, tensor1->sizes, tensor1->strides);
	TensorIterState<T2> tensor2Iter(tensor2->buffer, tensor2->dimCount, tensor2->sizes, tensor2->strides);

	do
	{
		for (; !tensor1Iter.ReachedBlockEnd() && !tensor2Iter.ReachedBlockEnd(); tensor1Iter.BlockStep(), tensor2Iter.BlockStep())
		{
			func(tensor1Iter.data, tensor2Iter.data);
		}

	} while (tensor1Iter.NextBlock() && tensor2Iter.NextBlock());
}

template<typename T1, typename T2, typename T3, typename FuncType>
INLINE_FUNC void Apply3(TensorRef* tensor1, TensorRef* tensor2, TensorRef* tensor3, FuncType func)
{
	TensorIterState<T1> tensor1Iter(tensor1->buffer, tensor1->dimCount, tensor1->sizes, tensor1->strides);
	TensorIterState<T2> tensor2Iter(tensor2->buffer, tensor2->dimCount, tensor2->sizes, tensor2->strides);
	TensorIterState<T3> tensor3Iter(tensor3->buffer, tensor3->dimCount, tensor3->sizes, tensor3->strides);

	do
	{
		for (; !tensor1Iter.ReachedBlockEnd() && !tensor2Iter.ReachedBlockEnd() && !tensor3Iter.ReachedBlockEnd();
				tensor1Iter.BlockStep(), tensor2Iter.BlockStep(), tensor3Iter.BlockStep())
		{
			func(tensor1Iter.data, tensor2Iter.data, tensor3Iter.data);
		}

	} while (tensor1Iter.NextBlock() && tensor2Iter.NextBlock() && tensor3Iter.NextBlock());
}
