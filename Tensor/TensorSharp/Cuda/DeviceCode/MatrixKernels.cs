// ***********************************************************************
// Assembly         : TensorSharp.CUDA91
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="GatherScatterKernels.cs" company="TensorSharp.CUDA91">
//     Copyright (c) . All rights reserved.
// </copyright>
// <summary></summary>
// ***********************************************************************
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TensorSharp.Core;
using TensorSharp.CUDA.RuntimeCompiler;
using TensorSharp.Properties;

namespace TensorSharp.CUDA.DeviceCode
{
    /// <summary>
    /// Class GatherScatterKernels.
    /// Implements the <see cref="TensorSharp.CUDA.DeviceCode.CudaCode" />
    /// </summary>
    /// <seealso cref="TensorSharp.CUDA.DeviceCode.CudaCode" />
    [Precompile]
    public class MatrixKernels : CudaCode
    {
        /// <summary>
        /// The code
        /// </summary>
        public static string Code = @"
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
};

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
};


template <typename IndexType, int Dims>
__global__ void diag_kernel(
	TensorInfo<IndexType> tensor,
	TensorInfo<IndexType> src,
	const IndexType totalElements) {
	for (IndexType linearId = blockIdx.x * blockDim.x + threadIdx.x; linearId < totalElements; linearId += gridDim.x * blockDim.x) {
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
};

#define DECLARE_DIAG(KERNEL_NAME, INDEX_TYPE, DIMS) \
    extern ""C"" {\
        __global__ void KERNEL_NAME(\
                                          TensorInfo<INDEX_TYPE> tensor,\
                                          TensorInfo<INDEX_TYPE> src,\
                                          INDEX_TYPE totalElements)\
        {\
            diag_kernel<INDEX_TYPE, DIMS>(tensor, src, totalElements);\
        }\
    }
";

        /// <summary>
        /// The diag matrix base name
        /// </summary>
        private const string DiagBaseName = "diag_";

        /// <summary>
        /// Initializes a new instance of the <see cref="GatherScatterKernels"/> class.
        /// </summary>
        public MatrixKernels() : base(GetCode(), "General", "ReduceApplyUtils")
        {
        }


        /// <summary>
        /// Gets the code.
        /// </summary>
        /// <returns>System.String.</returns>
        private static string GetCode()
        {
            Code = Resources.MatrixOps;
            var sb = new StringBuilder(Code);

            sb.AppendLine(GetMacroInvocations(true, 1));
            sb.AppendLine(GetMacroInvocations(true, 2));
            sb.AppendLine(GetMacroInvocations(true, 3));
            sb.AppendLine(GetMacroInvocations(true, -1));
            sb.AppendLine(GetMacroInvocations(false, -1));
            return sb.ToString();
        }

        /// <summary>
        /// Gets the macro invocations.
        /// </summary>
        /// <param name="is32">if set to <c>true</c> [is32].</param>
        /// <param name="dims">The dims.</param>
        /// <returns>System.String.</returns>
        private static string GetMacroInvocations(bool is32, int dims)
        {
            var indexType = is32 ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;

            return
                string.Format("DECLARE_DIAG({0}, {1}, {2})\n", MakeKernelName(DiagBaseName, is32, dims), indexType, dims);
        }


        /// <summary>
        /// Makes the name of the kernel.
        /// </summary>
        /// <param name="baseName">Name of the base.</param>
        /// <param name="is32">if set to <c>true</c> [is32].</param>
        /// <param name="dims">The dims.</param>
        /// <returns>System.String.</returns>
        private static string MakeKernelName(string baseName, bool is32, int dims)
        {
            return string.Format("{0}{1}_{2}",
                baseName,
                is32 ? "__int32" : "__int64",
                dims.ToString().Replace('-', 'M')
                );
        }

        public Tensor Diag(Tensor src)
        {
            var context = CudaHelpers.TSContextForTensor(src);
            var cudaContext = context.CudaContextForTensor(src);

            var writeTarget = TensorResultBuilder.GetWriteTarget(null, src.Allocator, src.ElementType, false, src.ElementCount(), src.ElementCount());

            var nElement = src.ElementCount();
            var block = ApplyUtils.GetApplyBlock();
            var grid = ApplyUtils.GetApplyGrid(context.DeviceInfoForContext(cudaContext), nElement);

            var kernelName = MakeKernelName(DiagBaseName, false, -1);
            Invoke(context, cudaContext, kernelName, grid, block, 0, CUstream.NullStream, false, writeTarget, src, (long)nElement);

            return writeTarget;
        }

        /// <summary>
        /// Invokes the specified context.
        /// </summary>
        /// <param name="context">The context.</param>
        /// <param name="cudaContext">The cuda context.</param>
        /// <param name="kernelName">Name of the kernel.</param>
        /// <param name="grid">The grid.</param>
        /// <param name="block">The block.</param>
        /// <param name="smemSize">Size of the smem.</param>
        /// <param name="stream">The stream.</param>
        /// <param name="index32">if set to <c>true</c> [index32].</param>
        /// <param name="args">The arguments.</param>
        private void Invoke(TSCudaContext context, CudaContext cudaContext, string kernelName, dim3 grid, dim3 block, uint smemSize, CUstream stream, bool index32, params object[] args)
        {
            ConvertTensorArgs.Convert(cudaContext, index32, args);

            var ptx = GetPtx(context.Compiler);
            var kernel = context.KernelCache.Get(cudaContext, ptx, kernelName);
            kernel.GridDimensions = grid;
            kernel.BlockDimensions = block;
            kernel.DynamicSharedMemory = smemSize;
            kernel.RunAsync(stream, args);
        }
    }
}
