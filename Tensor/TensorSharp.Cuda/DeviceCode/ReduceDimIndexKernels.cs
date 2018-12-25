// ***********************************************************************
// Assembly         : TensorSharp.CUDA91
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="ReduceDimIndexKernels.cs" company="TensorSharp.CUDA91">
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

namespace TensorSharp.CUDA.DeviceCode
{
    /// <summary>
    /// Class ReduceDimIndexKernels.
    /// Implements the <see cref="TensorSharp.CUDA.DeviceCode.CudaCode" />
    /// </summary>
    /// <seealso cref="TensorSharp.CUDA.DeviceCode.CudaCode" />
    [Precompile]
    public class ReduceDimIndexKernels : CudaCode
    {
        /// <summary>
        /// The code
        /// </summary>
        private static readonly string Code = @"

REDUCE_INDEX_KERNELS(argmin, if (a.first < b.first) return a; else return b;)
REDUCE_INDEX_KERNELS(argmax, if (a.first > b.first) return a; else return b;)

";

        /// <summary>
        /// Initializes a new instance of the <see cref="ReduceDimIndexKernels"/> class.
        /// </summary>
        public ReduceDimIndexKernels()
            : base(GetFullCode(), "General", "ReduceApplyUtils", "ReduceBlock", "Reduce", "ReduceMacros", "ReduceIndex", "Math")
        {
        }

        /// <summary>
        /// Gets the full code.
        /// </summary>
        /// <returns>System.String.</returns>
        private static string GetFullCode()
        {
            return Code;
        }

        /// <summary>
        /// Reduces the index outer dim.
        /// </summary>
        /// <param name="context">The context.</param>
        /// <param name="resultValues">The result values.</param>
        /// <param name="resultIndices">The result indices.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <param name="init">The initialize.</param>
        /// <param name="baseKernelName">Name of the base kernel.</param>
        private void ReduceIndexOuterDim(TSCudaContext context, Tensor resultValues, Tensor resultIndices, Tensor src, int dimension, Tuple<float, float> init, string baseKernelName)
        {
            var cudaContext = context.CudaContextForTensor(src);

            var ndim = src.DimensionCount;
            long num_orows = 1;
            for (int dim = 0; dim < dimension; dim++)
            {
                num_orows *= src.Sizes[dim];
            }
            var row_size = src.Sizes[dimension];
            long num_irows = 1;
            for (int dim = dimension + 1; dim < ndim; dim++)
            {
                num_irows *= src.Sizes[dim];
            }

            var threads = new dim3((uint)Math.Min(512, num_irows));
            var maxGridDim = 1024;
            var grid = new dim3((uint)Math.Min(maxGridDim, num_orows), (uint)Math.Min(maxGridDim, ApplyUtils.CeilDiv(num_irows, threads.x)));

            var resultValPtr = CudaHelpers.GetBufferStart(resultValues);
            var resultIdxPtr = CudaHelpers.GetBufferStart(resultIndices);
            var srcPtr = CudaHelpers.GetBufferStart(src);

            var kernelName = "outer_index_" + baseKernelName;

            Invoke(context, cudaContext, kernelName, grid, threads, 0, CUstream.NullStream, resultValPtr, resultIdxPtr, srcPtr, num_orows, num_irows, row_size, init.Item1, init.Item2);
        }

        /// <summary>
        /// Reduces the index innermost dim.
        /// </summary>
        /// <param name="context">The context.</param>
        /// <param name="resultValues">The result values.</param>
        /// <param name="resultIndices">The result indices.</param>
        /// <param name="src">The source.</param>
        /// <param name="init">The initialize.</param>
        /// <param name="baseKernelName">Name of the base kernel.</param>
        private void ReduceIndexInnermostDim(TSCudaContext context, Tensor resultValues, Tensor resultIndices, Tensor src, Tuple<float, float> init, string baseKernelName)
        {
            var cudaContext = context.CudaContextForTensor(src);

            var ndim = src.DimensionCount;
            long num_rows = 1;
            for (int dim = 0; dim < ndim - 1; dim++)
            {
                num_rows *= src.Sizes[dim];
            }
            var row_size = src.Sizes[ndim - 1];

            var threads = new dim3(16, 32);
            var grid = new dim3((uint)Math.Min(1024, ApplyUtils.CeilDiv(num_rows, threads.y)));

            var resultValPtr = CudaHelpers.GetBufferStart(resultValues);
            var resultIdxPtr = CudaHelpers.GetBufferStart(resultIndices);
            var srcPtr = CudaHelpers.GetBufferStart(src);

            var kernelName = "inner_index_" + baseKernelName;

            Invoke(context, cudaContext, kernelName, grid, threads, 0, CUstream.NullStream, resultValPtr, resultIdxPtr, srcPtr, num_rows, row_size, init.Item1, init.Item2);
        }

        /// <summary>
        /// Runs the reduce index op.
        /// </summary>
        /// <param name="resultIndices">The result indices.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <param name="init">The initialize.</param>
        /// <param name="baseKernelName">Name of the base kernel.</param>
        /// <returns>Tensor.</returns>
        private Tensor RunReduceIndexOp(Tensor resultIndices, Tensor src, int dimension, Tuple<float, float> init, string baseKernelName)
        {
            var context = CudaHelpers.TSContextForTensor(src);
            var requiredOutputSize = (long[])src.Sizes.Clone();
            requiredOutputSize[dimension] = 1;
            var writeTarget = TensorResultBuilder.GetWriteTarget(resultIndices, src.Allocator, DType.Float32, true, requiredOutputSize);

            using (var resultValueBuffer = new Tensor(src.Allocator, src.ElementType, requiredOutputSize))
            {
                if (dimension == src.DimensionCount - 1)
                {
                    ReduceIndexInnermostDim(context, resultValueBuffer, writeTarget, src, init, baseKernelName);
                }
                else
                {
                    ReduceIndexOuterDim(context, resultValueBuffer, writeTarget, src, dimension, init, baseKernelName);
                }

            }

            return writeTarget;
        }

        /// <summary>
        /// Arguments the minimum.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <returns>Tensor.</returns>
        public Tensor ArgMin(Tensor result, Tensor src, int dimension)
        {
            return RunReduceIndexOp(result, src, dimension, Tuple.Create(float.MaxValue, 0.0f), "argmin");
        }

        /// <summary>
        /// Arguments the maximum.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <returns>Tensor.</returns>
        public Tensor ArgMax(Tensor result, Tensor src, int dimension)
        {
            return RunReduceIndexOp(result, src, dimension, Tuple.Create(float.MinValue, 0.0f), "argmax");
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
        /// <param name="args">The arguments.</param>
        private void Invoke(TSCudaContext context, CudaContext cudaContext, string kernelName, dim3 grid, dim3 block, uint smemSize, CUstream stream, params object[] args)
        {
            var ptx = GetPtx(context.Compiler);
            var kernel = context.KernelCache.Get(cudaContext, ptx, kernelName);
            kernel.GridDimensions = grid;
            kernel.BlockDimensions = block;
            kernel.DynamicSharedMemory = smemSize;
            kernel.RunAsync(stream, args);
        }
    }
}
