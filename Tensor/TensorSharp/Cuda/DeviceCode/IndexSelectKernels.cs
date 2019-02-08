// ***********************************************************************
// Assembly         : TensorSharp.CUDA91
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="IndexSelectKernels.cs" company="TensorSharp.CUDA91">
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
    /// Class IndexSelectKernels.
    /// Implements the <see cref="TensorSharp.CUDA.DeviceCode.CudaCode" />
    /// </summary>
    /// <seealso cref="TensorSharp.CUDA.DeviceCode.CudaCode" />
    [Precompile]
    public class IndexSelectKernels : CudaCode
    {
        /// <summary>
        /// The small base name
        /// </summary>
        private const string SmallBaseName = "kernel_indexSelectSmallIndex_";
        /// <summary>
        /// The large base name
        /// </summary>
        private const string LargeBaseName = "kernel_indexSelectLargeIndex_";

        /// <summary>
        /// Initializes a new instance of the <see cref="IndexSelectKernels"/> class.
        /// </summary>
        public IndexSelectKernels() : base(GetCode(), "General", "ReduceApplyUtils")
        {
        }

        /// <summary>
        /// Gets the code.
        /// </summary>
        /// <returns>System.String.</returns>
        private static string GetCode()
        {
            var sb = new StringBuilder();
            sb.AppendLine(Resources.IndexSelect);
            sb.AppendLine(GetMacroInvocation(true, true, 1, 1, -2));
            sb.AppendLine(GetMacroInvocation(true, true, 2, 2, -2));
            sb.AppendLine(GetMacroInvocation(true, true, 3, 4, -2));
            sb.AppendLine(GetMacroInvocation(true, true, -1, -1, -1));

            sb.AppendLine(GetMacroInvocation(false, true, 1, 1, -2));
            sb.AppendLine(GetMacroInvocation(false, true, 2, 2, -2));
            sb.AppendLine(GetMacroInvocation(false, true, 3, 4, -2));
            sb.AppendLine(GetMacroInvocation(false, true, -1, -1, -1));

            sb.AppendLine(GetMacroInvocation(false, false, -1, -1, -1));
            return sb.ToString();
        }

        /// <summary>
        /// Gets the macro invocation.
        /// </summary>
        /// <param name="isSmall">if set to <c>true</c> [is small].</param>
        /// <param name="is32">if set to <c>true</c> [is32].</param>
        /// <param name="dstDims">The DST dims.</param>
        /// <param name="srcDims">The source dims.</param>
        /// <param name="idxDims">The index dims.</param>
        /// <returns>System.String.</returns>
        private static string GetMacroInvocation(bool isSmall, bool is32, int dstDims, int srcDims, int idxDims)
        {
            var kernelName = MakeKernelName(isSmall, is32, dstDims, srcDims, idxDims);
            return string.Format("{0}({1}, {2}, {3}, {4}, {5})",
                isSmall ? "DECLARE_SMALL" : "DECLARE_LARGE",
                kernelName,
                is32 ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64,
                dstDims,
                srcDims,
                idxDims);
        }

        /// <summary>
        /// Indexes the select.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dim">The dim.</param>
        /// <param name="indices">The indices.</param>
        /// <returns>Tensor.</returns>
        public Tensor IndexSelect(Tensor result, Tensor src, int dim, Tensor indices)
        {
            var context = CudaHelpers.TSContextForTensor(src);
            var cudaContext = context.CudaContextForTensor(src);

            var requiredOutputSize = (long[])src.Shape.Clone();
            requiredOutputSize[dim] = 1;
            var writeTarget = TensorResultBuilder.GetWriteTarget(result, src, true, requiredOutputSize);


            // The `src` is partitioned into two parts:
            // -the size of each slice we are indexing, which is the
            // total size of the tensor ignoring dimension `dim`;
            // -the number of indices we are choosing, which is the total size
            // of the tensor `indices`.
            var numIndices = indices.ElementCount();
            var dstTotalSize = writeTarget.ElementCount();
            var srcSelectDimSize = src.Shape[dim];
            var sliceSize = dstTotalSize / numIndices;

            var mpc = context.DeviceInfoForContext(cudaContext).MultiProcessorCount;
            var smallIndexGrid = new dim3((uint)Math.Min(ApplyUtils.CeilDiv(sliceSize, 128), (mpc * 8)));
            var smallIndexBlock = new dim3((uint)Math.Min(sliceSize, 128));

            var largeIndexGrid = new dim3((uint)Math.Min(ApplyUtils.CeilDiv(dstTotalSize, 128), (mpc * 8)));
            var largeIndexBlock = new dim3((uint)Math.Min(dstTotalSize, 128));


            var newResultSize = (long[])writeTarget.Shape.Clone();
            newResultSize[dim] = 1;
            var resultFlat = new Tensor(newResultSize, writeTarget.Strides, writeTarget.Storage, writeTarget.StorageOffset);

            var newSrcSize = (long[])src.Shape.Clone();
            newSrcSize[dim] = 1;
            var srcFlat = new Tensor(newSrcSize, src.Strides, src.Storage, src.StorageOffset);


            if (ApplyUtils.CanUse32BitIndexMath(writeTarget) &&
                ApplyUtils.CanUse32BitIndexMath(src) &&
                ApplyUtils.CanUse32BitIndexMath(indices))
            {
                // Threshold for small kernel
                var smallKernel = numIndices <= 16;
                string kernelName = "";
                var indContig = indices.IsContiguous();

                if (writeTarget.DimensionCount == src.DimensionCount &&
                    writeTarget.DimensionCount <= 3 &&
                    indContig)
                {
                    kernelName = MakeKernelName(smallKernel, true, writeTarget.DimensionCount, src.DimensionCount, -2);
                }
                else
                {
                    kernelName = MakeKernelName(smallKernel, true, -1, -1, -1);
                }

                var grid = smallKernel ? smallIndexGrid : largeIndexGrid;
                var block = smallKernel ? smallIndexBlock : largeIndexBlock;
                Invoke(context, cudaContext, kernelName, grid, block, 0, CUstream.NullStream, true,
                    writeTarget, src, indices, dim, dim, sliceSize, srcSelectDimSize);
            }
            else
            {
                var kernelName = MakeKernelName(false, false, -1, -1, -1);
                
                Invoke(context, cudaContext, kernelName, largeIndexGrid, largeIndexBlock, 0, CUstream.NullStream, false,
                    writeTarget, src, indices, dim, dim, dstTotalSize, sliceSize, srcSelectDimSize);
            }

            

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

        /// <summary>
        /// Makes the name of the kernel.
        /// </summary>
        /// <param name="isSmall">if set to <c>true</c> [is small].</param>
        /// <param name="is32">if set to <c>true</c> [is32].</param>
        /// <param name="dstDims">The DST dims.</param>
        /// <param name="srcDims">The source dims.</param>
        /// <param name="idxDims">The index dims.</param>
        /// <returns>System.String.</returns>
        private static string MakeKernelName(bool isSmall, bool is32, int dstDims, int srcDims, int idxDims)
        {
            return string.Format("{0}{1}_{2}_{3}_{4}",
                isSmall ? SmallBaseName : LargeBaseName,
                is32 ? "__int32" : "__int64",
                dstDims.ToString().Replace('-', 'M'),
                srcDims.ToString().Replace('-', 'M'),
                idxDims.ToString().Replace('-', 'M')
                );
        }
    }
}
