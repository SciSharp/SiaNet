// Compute the offsets into the given tensors for a linear index. For the 't2'
// tensor, dimension 'dim' is skipped. The tensors are assumed to have the same
// size (with the exception of 't2' in dimension 'dim').
// This version uses a static number of dimensions.
template <typename IndexType, int Dims>
struct IndexToScatterGatherOffsets {
	static __device__ void compute(
		IndexType linearId, const int dim,
		const TensorInfo<IndexType>& index, IndexType* indexOffset,
		const TensorInfo<IndexType>& t1, IndexType* t1Offset,
		const TensorInfo<IndexType>& t2, IndexType* t2Offset) {
		for (int d = Dims - 1; d >= 0; d--) {
			IndexType curDimIndex = linearId % index.sizes[d];
			*indexOffset += curDimIndex * index.strides[d];
			*t1Offset += curDimIndex * t1.strides[d];
			if (d != dim) {
				*t2Offset += curDimIndex * t2.strides[d];
			}
			linearId /= index.sizes[d];
		}
	}

	static __device__ void compute(
		IndexType linearId, const int dim,
		const TensorInfo<IndexType>& index, IndexType* indexOffset,
		const TensorInfo<IndexType>& t2, IndexType* t2Offset) {
		for (int d = Dims - 1; d >= 0; d--) {
			IndexType curDimIndex = linearId % index.sizes[d];
			*indexOffset += curDimIndex * index.strides[d];
			if (d != dim) {
				*t2Offset += curDimIndex * t2.strides[d];
			}
			linearId /= index.sizes[d];
		}
	}
}

// Same as above but using a dynamic number of dimensions.
template <typename IndexType>
struct IndexToScatterGatherOffsets<IndexType, -1> {
	static __device__ void compute(
		IndexType linearId, const int dim,
		const TensorInfo<IndexType>& index, IndexType* indexOffset,
		const TensorInfo<IndexType>& t1, IndexType* t1Offset,
		const TensorInfo<IndexType>& t2, IndexType* t2Offset) {
		for (int d = index.dims - 1; d >= 0; d--) {
			IndexType curDimIndex = linearId % index.sizes[d];
			*indexOffset += curDimIndex * index.strides[d];
			*t1Offset += curDimIndex * t1.strides[d];
			if (d != dim) {
				*t2Offset += curDimIndex * t2.strides[d];
			}
			linearId /= index.sizes[d];
		}
	}

	static __device__ void compute(
		IndexType linearId, const int dim,
		const TensorInfo<IndexType>& index, IndexType* indexOffset,
		const TensorInfo<IndexType>& t2, IndexType* t2Offset) {
		for (int d = index.dims - 1; d >= 0; d--) {
			IndexType curDimIndex = linearId % index.sizes[d];
			*indexOffset += curDimIndex * index.strides[d];
			if (d != dim) {
				*t2Offset += curDimIndex * t2.strides[d];
			}
			linearId /= index.sizes[d];
		}
	}
}


template <typename IndexType, int Dims>
__global__ void gather_kernel(
	TensorInfo<IndexType> tensor,
	TensorInfo<IndexType> src,
	TensorInfo<IndexType> index,
	const int dim,
	const IndexType totalElements) {
	for (IndexType linearId = blockIdx.x * blockDim.x + threadIdx.x;
		linearId < totalElements;
		linearId += gridDim.x * blockDim.x) {
		IndexType tensorOffset = 0;
		IndexType srcOffset = 0;
		IndexType indexOffset = 0;

		IndexToScatterGatherOffsets<IndexType, Dims>::compute(linearId, dim,
			index, &indexOffset,
			tensor, &tensorOffset,
			src, &srcOffset);

		IndexType indexValue = (IndexType)index.data[indexOffset];
		srcOffset += indexValue * src.strides[dim];

		tensor.data[tensorOffset] = src.data[srcOffset];
	}
}

template <typename IndexType, int Dims>
__global__ void scatter_kernel(
	TensorInfo<IndexType> tensor,
	TensorInfo<IndexType> src,
	TensorInfo<IndexType> index,
	const int dim,
	const IndexType totalElements) {
	for (IndexType linearId = blockIdx.x * blockDim.x + threadIdx.x;
		linearId < totalElements;
		linearId += gridDim.x * blockDim.x) {
		IndexType tensorOffset = 0;
		IndexType srcOffset = 0;
		IndexType indexOffset = 0;

		IndexToScatterGatherOffsets<IndexType, Dims>::compute(linearId, dim,
			index, &indexOffset,
			src, &srcOffset,
			tensor, &tensorOffset);

		IndexType indexValue = (IndexType)index.data[indexOffset];
		tensorOffset += indexValue * tensor.strides[dim];

		tensor.data[tensorOffset] = src.data[srcOffset];
	}
}


template <typename IndexType, int Dims>
__global__ void scatterFill_kernel(
	TensorInfo<IndexType> tensor,
	TensorInfo<IndexType> index,
	float value,
	const int dim,
	const IndexType totalElements) {
	for (IndexType linearId = blockIdx.x * blockDim.x + threadIdx.x;
		linearId < totalElements;
		linearId += gridDim.x * blockDim.x) {
		IndexType tensorOffset = 0;
		IndexType indexOffset = 0;

		IndexToScatterGatherOffsets<IndexType, Dims>::compute(linearId, dim,
			index, &indexOffset,
			tensor, &tensorOffset);

		IndexType indexValue = (IndexType)index.data[indexOffset];
		tensorOffset += indexValue * tensor.strides[dim];

		tensor.data[tensorOffset] = value;
	}
}


#define DECLARE_GATHER(KERNEL_NAME, INDEX_TYPE, DIMS) \
    extern "C" {\
        __global__ void KERNEL_NAME(\
                                          TensorInfo<INDEX_TYPE> tensor,\
                                          TensorInfo<INDEX_TYPE> src,\
                                          TensorInfo<INDEX_TYPE> indices,\
                                          const int dim,\
                                          INDEX_TYPE totalElements)\
        {\
            gather_kernel<INDEX_TYPE, DIMS>(tensor, src, indices, dim, totalElements);\
        }\
    }

#define DECLARE_SCATTER(KERNEL_NAME, INDEX_TYPE, DIMS) \
    extern "C" {\
        __global__ void KERNEL_NAME(\
                                          TensorInfo<INDEX_TYPE> tensor,\
                                          TensorInfo<INDEX_TYPE> src,\
                                          TensorInfo<INDEX_TYPE> indices,\
                                          const int dim,\
                                          INDEX_TYPE totalElements)\
        {\
            scatter_kernel<INDEX_TYPE, DIMS>(tensor, src, indices, dim, totalElements);\
        }\
    }

#define DECLARE_SCATTERFILL(KERNEL_NAME, INDEX_TYPE, DIMS) \
    extern "C" {\
        __global__ void KERNEL_NAME(\
                                          TensorInfo<INDEX_TYPE> tensor,\
                                          TensorInfo<INDEX_TYPE> indices,\
                                          float value,\
                                          const int dim,\
                                          INDEX_TYPE totalElements)\
        {\
            scatterFill_kernel<INDEX_TYPE, DIMS>(tensor, indices, value, dim, totalElements);\
        }\
    }