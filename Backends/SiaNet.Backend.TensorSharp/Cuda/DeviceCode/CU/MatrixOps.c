// Compute the offsets into the given tensors for a linear index. For the 't2'
// tensor, dimension 'dim' is skipped. The tensors are assumed to have the same
// size (with the exception of 't2' in dimension 'dim').
// This version uses a static number of dimensions.
// Same as above but using a dynamic number of dimensions.
template <typename IndexType>
struct DiagOffsets<IndexType, -1> {
	static __device__ void compute(
		IndexType linearId, const int dim, const TensorInfo<IndexType>& t, IndexType* tOffset) {
		for (int d = t.dims - 1; d >= 0; d--) {
			IndexType curDimIndex = linearId % t.sizes[d];
			*tOffset += curDimIndex * t.strides[d];
		}

		linearId /= t.sizes[d];
	}
}
};


template <typename IndexType, int Dims>
__global__ void diag_kernel(
	TensorInfo<IndexType> tensor,
	TensorInfo<IndexType> src,
	const IndexType totalElements) {
	for (IndexType i = blockIdx.x * blockDim.x + threadIdx.x; i < totalElements; i += gridDim.x * blockDim.x) {
		for (IndexType j = blockIdx.x * blockDim.x + threadIdx.x; j < totalElements; j += gridDim.x * blockDim.x) {
			IndexType tensorOffset = 0;
			IndexType srcOffset = 0;

			DiagOffsets<IndexType>::compute(i, dim, tensor, &tensorOffset);
			DiagOffsets<IndexType>::compute(i, dim, src, &srcOffset);

			if (i == j)
			{
				IndexType indexValue = (IndexType)src.data[tensorOffset];
				srcOffset += indexValue * src.strides[dim];

				tensor.data[tensorOffset] = src.data[srcOffset];
			}
			else
			{
				tensor.data[tensorOffset] = 0;
			}

		}
	}
};



#define DECLARE_DIAG(KERNEL_NAME, INDEX_TYPE, DIMS) \
    extern "C" {\
        __global__ void KERNEL_NAME(\
                                          TensorInfo<INDEX_TYPE> tensor,\
                                          TensorInfo<INDEX_TYPE> src,\
                                          INDEX_TYPE totalElements)\
        {\
            diag_kernel<INDEX_TYPE, DIMS>(tensor, src, totalElements);\
        }\
    }