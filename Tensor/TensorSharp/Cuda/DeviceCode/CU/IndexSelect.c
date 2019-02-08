// We prefer this kernel to avoid reloading index points if the number
// of indices is a small number.
// This kernel in fact works for all choices of problem size, but if
// the number of indices chosen is large, then the
// indexSelectLargeIndex kernel is a better choice to increase
// parallelism.
template <typename IndexType, int DstDim, int SrcDim, int IdxDim>
__device__ void indexSelectSmallIndex(TensorInfo<IndexType> dst,
	TensorInfo<IndexType> src,
	TensorInfo<IndexType> indices,
	int dstSelectDim,
	int srcSelectDim,
	IndexType innerSize,
	__int64 srcSelectDimSize) {
	// In order to avoid reloading the index that we are copying, load
	// it once to handle all of the points that are being selected, so
	// it can be reused as much as possible. This kernel is chosen when
	// this is a good choice (small number of chosen indices), since
	// re-accessing indices in addition to src elements can be slow.
	for (IndexType dstIndex = 0; dstIndex < indices.sizes[0]; ++dstIndex) {

		IndexType srcIndex =
			indices.data[IndexToOffset<IndexType, IdxDim>::get(dstIndex, indices)];

		if (srcIndex < srcSelectDimSize) {
			// We stride over the output ignoring the indexed dimension
			// (innerSize), whose offset calculation is handled differently
			for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
				linearIndex < innerSize;
				linearIndex += gridDim.x * blockDim.x) {
				IndexType dstOffset =
					IndexToOffset<IndexType, DstDim>::get(linearIndex, dst);
				dstOffset += dstIndex * dst.strides[dstSelectDim];

				IndexType srcOffset =
					IndexToOffset<IndexType, SrcDim>::get(linearIndex, src);
				srcOffset += srcIndex * src.strides[srcSelectDim];

				dst.data[dstOffset] = src.data[srcOffset];
			}
		}
	}
}




// We prefer this kernel to balance parallelism across index points,
// if there are a large number of indices.
// This kernel in fact works for all choices of problem size, but if
// the number of indices chosen is small, then the
// indexSelectSmallIndex kernel is a better choice to reduce memory
// accesses.
template <typename IndexType, int DstDim, int SrcDim, int IdxDim>
__device__ void indexSelectLargeIndex(TensorInfo<IndexType> dst,
	TensorInfo<IndexType> src,
	TensorInfo<IndexType> indices,
	int dstSelectDim,
	int srcSelectDim,
	IndexType totalSize,
	IndexType innerSize,
	__int64 srcSelectDimSize) {
	// We stride over the output including the indexed dimension
	// (totalSize), and calculate the destination index point based on that
	for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
		linearIndex < totalSize;
		linearIndex += gridDim.x * blockDim.x) {
		IndexType dstIndex = linearIndex / innerSize;
		IndexType elementInSlice = linearIndex % innerSize;

		IndexType srcIndex =
			indices.data[IndexToOffset<IndexType, IdxDim>::get(dstIndex, indices)];

		if (srcIndex < srcSelectDimSize) {
			IndexType dstOffset =
				IndexToOffset<IndexType, DstDim>::get(elementInSlice, dst);
			dstOffset += dstIndex * dst.strides[dstSelectDim];

			IndexType srcOffset =
				IndexToOffset<IndexType, SrcDim>::get(elementInSlice, src);
			srcOffset += srcIndex * src.strides[srcSelectDim];

			dst.data[dstOffset] = src.data[srcOffset];
		}
	}
}

#define DECLARE_SMALL(KERNEL_NAME, INDEX_TYPE, DST_DIM, SRC_DIM, IDX_DIM) \
    extern "C" {\
        __global__ void KERNEL_NAME(\
                                          TensorInfo<INDEX_TYPE> dst,\
                                          TensorInfo<INDEX_TYPE> src,\
                                          TensorInfo<INDEX_TYPE> indices,\
                                          int dstSelectDim,\
                                          int srcSelectDim,\
                                          INDEX_TYPE innerSize,\
                                          __int64 srcSelectDimSize)\
        {\
            indexSelectSmallIndex<INDEX_TYPE, DST_DIM, SRC_DIM, IDX_DIM>(dst, src, indices, dstSelectDim, srcSelectDim, innerSize, srcSelectDimSize);\
        }\
    }

#define DECLARE_LARGE(KERNEL_NAME, INDEX_TYPE, DST_DIM, SRC_DIM, IDX_DIM) \
    extern "C" {\
        __global__ void KERNEL_NAME(\
                                          TensorInfo<INDEX_TYPE> dst,\
                                          TensorInfo<INDEX_TYPE> src,\
                                          TensorInfo<INDEX_TYPE> indices,\
                                          int dstSelectDim,\
                                          int srcSelectDim,\
                                          INDEX_TYPE totalSize,\
                                          INDEX_TYPE innerSize,\
                                          __int64 srcSelectDimSize)\
        {\
            indexSelectLargeIndex<INDEX_TYPE, DST_DIM, SRC_DIM, IDX_DIM>(dst, src, indices, dstSelectDim, srcSelectDim, totalSize, innerSize, srcSelectDimSize);\
        }\
    }