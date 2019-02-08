// ***********************************************************************
// Assembly         : TensorSharp.CUDA91
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="VarStdKernels.cs" company="TensorSharp.CUDA91">
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
using TensorSharp.Properties;

namespace TensorSharp.CUDA.DeviceCode
{
    /// <summary>
    /// Class VarStdKernels.
    /// Implements the <see cref="TensorSharp.CUDA.DeviceCode.CudaCode" />
    /// </summary>
    /// <seealso cref="TensorSharp.CUDA.DeviceCode.CudaCode" />
    [Precompile]
    public class VarStdKernels : CudaCode
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="VarStdKernels"/> class.
        /// </summary>
        public VarStdKernels()
            : base(GetFullCode(), "General", "ReduceApplyUtils", "ReduceBlock", "Reduce", "ReduceMacros", "Math")
        {
        }

        /// <summary>
        /// Gets the full code.
        /// </summary>
        /// <returns>System.String.</returns>
        private static string GetFullCode()
        {
            return Resources.VarStd;
        }

        /// <summary>
        /// Gets the mangled name suffix.
        /// </summary>
        /// <param name="normByN">if set to <c>true</c> [norm by n].</param>
        /// <param name="applySqrt">if set to <c>true</c> [apply SQRT].</param>
        /// <returns>System.String.</returns>
        private static string GetMangledNameSuffix(bool normByN, bool applySqrt)
        {
            return string.Format("_{0}_{1}", normByN, applySqrt).ToLower();
        }


        /// <summary>
        /// Variables the outer dim.
        /// </summary>
        /// <param name="context">The context.</param>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <param name="normByN">if set to <c>true</c> [norm by n].</param>
        /// <param name="applySqrt">if set to <c>true</c> [apply SQRT].</param>
        private void VarOuterDim(TSCudaContext context, Tensor result, Tensor src, int dimension, bool normByN, bool applySqrt)
        {
            var cudaContext = context.CudaContextForTensor(src);

            int ndim = src.DimensionCount;
            long num_orows = 1;
            for (int dim = 0; dim < dimension; dim++)
            {
                num_orows *= src.Shape[dim];
            }
            long row_size = src.Shape[dimension];
            // Treat all inner dimensions (i.e. dim > dimension) as one.
            long num_irows = 1;
            for (int dim = dimension + 1; dim < ndim; dim++)
            {
                num_irows *= src.Shape[dim];
            }

            var threads = new dim3((uint)Math.Min(512, num_irows));
            var maxGridDim = 1024;
            var grid = new dim3((uint)Math.Min(maxGridDim, num_orows), (uint)Math.Min(maxGridDim, ApplyUtils.CeilDiv(num_irows, threads.x)));

            var resultPtr = CudaHelpers.GetBufferStart(result);
            var srcPtr = CudaHelpers.GetBufferStart(src);
            var kernelName = "kernel_varOuterDim" + GetMangledNameSuffix(normByN, applySqrt);

            Invoke(context, cudaContext, kernelName, grid, threads, 0, CUstream.NullStream, resultPtr, srcPtr, num_orows, num_irows, row_size);
        }


        /// <summary>
        /// Variables the innermost dim.
        /// </summary>
        /// <param name="context">The context.</param>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="normByN">if set to <c>true</c> [norm by n].</param>
        /// <param name="applySqrt">if set to <c>true</c> [apply SQRT].</param>
        private void VarInnermostDim(TSCudaContext context, Tensor result, Tensor src, bool normByN, bool applySqrt)
        {
            var cudaContext = context.CudaContextForTensor(src);

            var ndim = src.DimensionCount;
            long num_rows = 1;
            for (var dim = 0; dim < ndim - 1; dim++)
            {
                num_rows *= src.Shape[dim];
            }
            var row_size = src.Shape[ndim - 1];

            // (Comment from cuTorch source): From limited testing, 16x32 seemed a good compromise for handling both long and short dimensions.
            var threads = new dim3(16, 32);
            var grid = new dim3((uint)Math.Min(1024, ApplyUtils.CeilDiv(num_rows, threads.y)));

            var resultPtr = CudaHelpers.GetBufferStart(result);
            var srcPtr = CudaHelpers.GetBufferStart(src);
            var kernelName = "kernel_varInnermostDim" + GetMangledNameSuffix(normByN, applySqrt);

            Invoke(context, cudaContext, kernelName, grid, threads, 0, CUstream.NullStream, resultPtr, srcPtr, num_rows, row_size);
        }

        /// <summary>
        /// Runs the variable op.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <param name="normByN">if set to <c>true</c> [norm by n].</param>
        /// <param name="applySqrt">if set to <c>true</c> [apply SQRT].</param>
        /// <returns>Tensor.</returns>
        private Tensor RunVarOp(Tensor result, Tensor src, int dimension, bool normByN, bool applySqrt)
        {
            var context = CudaHelpers.TSContextForTensor(src);
            var requiredOutputSize = (long[])src.Shape.Clone();
            requiredOutputSize[dimension] = 1;
            var writeTarget = TensorResultBuilder.GetWriteTarget(result, src, true, requiredOutputSize);

            if (dimension == src.DimensionCount - 1)
            {
                VarInnermostDim(context, writeTarget, src, normByN, applySqrt);
            }
            else
            {
                VarOuterDim(context, writeTarget, src, dimension, normByN, applySqrt);
            }

            return writeTarget;
        }

        /// <summary>
        /// Variables the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <param name="normByN">if set to <c>true</c> [norm by n].</param>
        /// <returns>Tensor.</returns>
        public Tensor Var(Tensor result, Tensor src, int dimension, bool normByN)
        {
            return RunVarOp(result, src, dimension, normByN, false);
        }

        /// <summary>
        /// Standards the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <param name="normByN">if set to <c>true</c> [norm by n].</param>
        /// <returns>Tensor.</returns>
        public Tensor Std(Tensor result, Tensor src, int dimension, bool normByN)
        {
            return RunVarOp(result, src, dimension, normByN, true);
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
