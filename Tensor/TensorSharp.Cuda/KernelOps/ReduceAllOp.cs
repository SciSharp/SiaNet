// ***********************************************************************
// Assembly         : TensorSharp.CUDA91
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="ReduceAllOp.cs" company="TensorSharp.CUDA91">
//     Copyright (c) . All rights reserved.
// </copyright>
// <summary></summary>
// ***********************************************************************
using ManagedCuda;
using ManagedCuda.VectorTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TensorSharp.Core;
using TensorSharp.CUDA.DeviceCode;
using TensorSharp.CUDA.RuntimeCompiler;

namespace TensorSharp.CUDA.KernelOps
{
    /// <summary>
    /// Class ReduceAllOp.
    /// </summary>
    public static class ReduceAllOp
    {
        /// <summary>
        /// The reduce all block size
        /// </summary>
        private const long ReduceAllBlockSize = 1024;
        /// <summary>
        /// The two pass reduction size
        /// </summary>
        private const long TwoPassReductionSize = 2048;


        /// <summary>
        /// Invokes the specified reduce all kernels.
        /// </summary>
        /// <param name="reduceAllKernels">The reduce all kernels.</param>
        /// <param name="init">The initialize.</param>
        /// <param name="initType">Type of the initialize.</param>
        /// <param name="kernelName">Name of the kernel.</param>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="extraArg">The extra argument.</param>
        /// <returns>Tensor.</returns>
        /// <exception cref="InvalidOperationException">Tensors with dimension count > " + TSCudaContext.MaxDims + " are not supported</exception>
        public static Tensor Invoke(CudaReduceAllKernels reduceAllKernels, float init, ReduceInitType initType, string kernelName, Tensor result, Tensor src, object extraArg = null)
        {
            var deviceId = CudaHelpers.GetDeviceId(src);
            var context = CudaHelpers.TSContextForTensor(src);
            var cudaContext = context.CudaContextForDevice(deviceId);

            if (src.DimensionCount > TSCudaContext.MaxDims)
                throw new InvalidOperationException("Tensors with dimension count > " + TSCudaContext.MaxDims + " are not supported");

            var writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, 1);

            if (src.DimensionCount == 0)
            {
                return result;
            }

            var totalElements = src.ElementCount();
            var config = new ApplySpecialization(src);
            object totalElementsTyped = config.Use32BitIndices ? (uint)totalElements : (ulong)totalElements;
            object initValueTyped = ReduceInitConverter.GetInitValue(init, initType, src.ElementType);

            dim3 grid;
            dim3 block;

            var ptx = reduceAllKernels.GetPtx(context.Compiler);
            var fullKernelName = PermutationGenerator.GetMangledName(kernelName, config);

            var outputDevicePtr = CudaHelpers.GetBufferStart(writeTarget);

            if (isTwoPassReductionSize(totalElements))
            {
                getPass1ReduceBlockGrid(context, deviceId, totalElements, out grid, out block);
                uint smemSize = block.x * sizeof(float);

                var scratchSpace = context.ScratchSpaceForDevice(deviceId).buffer;

                if(extraArg == null)
                    InvokeReduceAll(context, cudaContext, ptx, "twoPassA_" + fullKernelName, grid, block, smemSize, config, src, totalElementsTyped, initValueTyped, scratchSpace);
                else
                    InvokeReduceAll(context, cudaContext, ptx, "twoPassA_" + fullKernelName, grid, block, smemSize, config, src, totalElementsTyped, initValueTyped, scratchSpace, extraArg);


                uint numPass1Blocks = grid.x;
                getPass2ReduceBlockGrid(context, deviceId, totalElements, out grid, out block);
                smemSize = block.x * sizeof(float);

                InvokeReduceAllPass2(context, cudaContext, ptx, "twoPassB_" + fullKernelName, grid, block, smemSize, config.Use32BitIndices, numPass1Blocks, initValueTyped, scratchSpace, outputDevicePtr);

            }
            else {
                getSinglePassReduceBlockGrid(totalElements, out grid, out block);
                uint smemSize = block.x * sizeof(float);

                if(extraArg == null)
                    InvokeReduceAll(context, cudaContext, ptx, "onePass_" + fullKernelName, grid, block, smemSize, config, src, totalElementsTyped, initValueTyped, outputDevicePtr);
                else
                    InvokeReduceAll(context, cudaContext, ptx, "onePass_" + fullKernelName, grid, block, smemSize, config, src, totalElementsTyped, initValueTyped, outputDevicePtr, extraArg);
            }

            return writeTarget;
        }

        /// <summary>
        /// Invokes the reduce all pass2.
        /// </summary>
        /// <param name="context">The context.</param>
        /// <param name="cudaContext">The cuda context.</param>
        /// <param name="ptx">The PTX.</param>
        /// <param name="kernelName">Name of the kernel.</param>
        /// <param name="grid">The grid.</param>
        /// <param name="block">The block.</param>
        /// <param name="smemSize">Size of the smem.</param>
        /// <param name="index32">if set to <c>true</c> [index32].</param>
        /// <param name="args">The arguments.</param>
        public static void InvokeReduceAllPass2(TSCudaContext context, CudaContext cudaContext, byte[] ptx, string kernelName, dim3 grid, dim3 block, uint smemSize, bool index32, params object[] args)
        {
            var config = new ApplySpecialization(index32).GetConfig();

            
            var kernel = context.KernelCache.Get(cudaContext, ptx, kernelName);

            kernel.GridDimensions = grid;
            kernel.BlockDimensions = block;
            kernel.DynamicSharedMemory = smemSize;

            kernel.Run(args);            
        }

        /// <summary>
        /// Invokes the reduce all.
        /// </summary>
        /// <param name="context">The context.</param>
        /// <param name="cudaContext">The cuda context.</param>
        /// <param name="ptx">The PTX.</param>
        /// <param name="kernelName">Name of the kernel.</param>
        /// <param name="grid">The grid.</param>
        /// <param name="block">The block.</param>
        /// <param name="smemSize">Size of the smem.</param>
        /// <param name="spec">The spec.</param>
        /// <param name="args">The arguments.</param>
        public static void InvokeReduceAll(TSCudaContext context, CudaContext cudaContext, byte[] ptx, string kernelName, dim3 grid, dim3 block, uint smemSize, ApplySpecialization spec, params object[] args)
        {
            ConvertTensorArgs.Convert(cudaContext, spec.Use32BitIndices, args);
            
            var kernel = context.KernelCache.Get(cudaContext, ptx, kernelName);

            kernel.GridDimensions = grid;
            kernel.BlockDimensions = block;
            kernel.DynamicSharedMemory = smemSize;

            kernel.Run(args);
            
        }


        /// <summary>
        /// Determines whether [is two pass reduction size] [the specified elements].
        /// </summary>
        /// <param name="elements">The elements.</param>
        /// <returns><c>true</c> if [is two pass reduction size] [the specified elements]; otherwise, <c>false</c>.</returns>
        private static bool isTwoPassReductionSize(long elements)
        {
            return (elements > TwoPassReductionSize);
        }

        /// <summary>
        /// Gets the two pass blocks.
        /// </summary>
        /// <param name="context">The context.</param>
        /// <param name="deviceId">The device identifier.</param>
        /// <param name="elements">The elements.</param>
        /// <returns>System.Int64.</returns>
        /// <exception cref="ApplicationException">Device id " + deviceId + " has no scratch space</exception>
        private static long getTwoPassBlocks(TSCudaContext context, int deviceId, long elements)
        {
            long numBlocks = ApplyUtils.CeilDiv(elements, ReduceAllBlockSize);

            // We can only have as many blocks as there is scratch space
            long scratchSpace =
              context.ScratchSpaceForDevice(deviceId).size / sizeof(float);
            if (scratchSpace <= 0)
                throw new ApplicationException("Device id " + deviceId + " has no scratch space");

            if (numBlocks > scratchSpace)
            {
                numBlocks = scratchSpace;
            }

            return numBlocks;
        }

        /// <summary>
        /// Gets the pass1 reduce block grid.
        /// </summary>
        /// <param name="context">The context.</param>
        /// <param name="deviceId">The device identifier.</param>
        /// <param name="elements">The elements.</param>
        /// <param name="grid">The grid.</param>
        /// <param name="block">The block.</param>
        private static void getPass1ReduceBlockGrid(TSCudaContext context, int deviceId, long elements, out dim3 grid, out dim3 block)
        {
            grid = new dim3((uint)getTwoPassBlocks(context, deviceId, elements));
            block = new dim3((uint)ReduceAllBlockSize);
        }

        /// <summary>
        /// Gets the pass2 reduce block grid.
        /// </summary>
        /// <param name="context">The context.</param>
        /// <param name="deviceId">The device identifier.</param>
        /// <param name="elements">The elements.</param>
        /// <param name="grid">The grid.</param>
        /// <param name="block">The block.</param>
        private static void getPass2ReduceBlockGrid(TSCudaContext context, int deviceId, long elements, out dim3 grid, out dim3 block)
        {
            grid = new dim3(1);
            // We only need as many threads as there were blocks originally
            block = new dim3((uint)getTwoPassBlocks(context, deviceId, elements));
        }

        /// <summary>
        /// Gets the single pass reduce block grid.
        /// </summary>
        /// <param name="elements">The elements.</param>
        /// <param name="grid">The grid.</param>
        /// <param name="block">The block.</param>
        private static void getSinglePassReduceBlockGrid(long elements, out dim3 grid, out dim3 block)
        {
            grid = new dim3(1);
            block = new dim3((uint)ReduceAllBlockSize);
        }
    }
}
