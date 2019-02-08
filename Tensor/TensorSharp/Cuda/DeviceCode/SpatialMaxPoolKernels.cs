// ***********************************************************************
// Assembly         : TensorSharp.CUDA91
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="SpatialMaxPoolKernels.cs" company="TensorSharp.CUDA91">
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
using TensorSharp.Cpu;
using TensorSharp.Properties;

namespace TensorSharp.CUDA.DeviceCode
{
    /// <summary>
    /// Class SpatialMaxPoolKernels.
    /// Implements the <see cref="TensorSharp.CUDA.DeviceCode.CudaCode" />
    /// </summary>
    /// <seealso cref="TensorSharp.CUDA.DeviceCode.CudaCode" />
    [Precompile]
    public class SpatialMaxPoolKernels : CudaCode
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="SpatialMaxPoolKernels"/> class.
        /// </summary>
        public SpatialMaxPoolKernels() : base(Resources.SpatialMaxPool)
        {
        }

        /// <summary>
        /// Spatials the maximum pooling forward.
        /// </summary>
        /// <param name="input">The input.</param>
        /// <param name="output">The output.</param>
        /// <param name="indices">The indices.</param>
        /// <param name="cd">The cd.</param>
        /// <param name="ceilMode">if set to <c>true</c> [ceil mode].</param>
        public void SpatialMaxPoolingForward(Tensor input, Tensor output, Tensor indices, ConvolutionDesc2d cd, bool ceilMode)
        {
            var context = CudaHelpers.TSContextForTensor(input);
            var cudaContext = context.CudaContextForTensor(input);

            var iwidth = input.Shape[3];
            var iheight = input.Shape[2];
            var nInputPlane = input.Shape[1];
            var batchSize = input.Shape[0];

            long owidth;
            long oheight;

            if (ceilMode)
            {
                oheight = (long)(Math.Ceiling((float)(iheight - cd.kH + 2 * cd.padH) / cd.dH)) + 1;
                owidth = (long)(Math.Ceiling((float)(iwidth - cd.kW + 2 * cd.padW) / cd.dW)) + 1;
            }
            else
            {
                oheight = (long)(Math.Floor((float)(iheight - cd.kH + 2 * cd.padH) / cd.dH)) + 1;
                owidth = (long)(Math.Floor((float)(iwidth - cd.kW + 2 * cd.padW) / cd.dW)) + 1;
            }

            if (cd.padW != 0 || cd.padH != 0)
            {
                // ensure that the last pooling starts inside the image
                if ((oheight - 1) * cd.dH >= iheight + cd.padH)
                    --oheight;
                if ((owidth - 1) * cd.dW >= iwidth + cd.padW)
                    --owidth;
            }

            using (var inputContig = Ops.AsContiguous(input))
            {
                var inputPtr = CudaHelpers.GetBufferStart(inputContig);
                var outputPtr = CudaHelpers.GetBufferStart(output);
                var indicesPtr = CudaHelpers.GetBufferStart(indices);

                var count = (int)output.ElementCount();

                Invoke(context, cudaContext, "MaxPoolForward", new dim3(NNThreads.NumBlocks(count)), new dim3(NNThreads.NumThreads), 0, CUstream.NullStream,
                    count, inputPtr, batchSize, nInputPlane, iheight, iwidth, oheight, owidth,
                    cd.kH, cd.kW, cd.dH, cd.dW, cd.padH, cd.padW, outputPtr, indicesPtr);
            }
        }

        /// <summary>
        /// Spatials the maximum pooling backward.
        /// </summary>
        /// <param name="input">The input.</param>
        /// <param name="gradOutput">The grad output.</param>
        /// <param name="gradInput">The grad input.</param>
        /// <param name="indices">The indices.</param>
        /// <param name="cd">The cd.</param>
        /// <param name="ceilMode">if set to <c>true</c> [ceil mode].</param>
        public void SpatialMaxPoolingBackward(Tensor input, Tensor gradOutput, Tensor gradInput, Tensor indices, ConvolutionDesc2d cd, bool ceilMode)
        {
            var context = CudaHelpers.TSContextForTensor(gradOutput);
            var cudaContext = context.CudaContextForTensor(gradOutput);

            var dimw = 3;
            var dimh = 2;
            var dimc = 1;

            var nbatch = input.Shape[0];
            var nslices = input.Shape[dimc];
            var iheight = input.Shape[dimh];
            var iwidth = input.Shape[dimw];
            var owidth = gradOutput.Shape[dimw];
            var oheight = gradOutput.Shape[dimh];


            using (var gradOutputContig = Ops.AsContiguous(gradOutput))
            {
                var gradOutputPtr = CudaHelpers.GetBufferStart(gradOutputContig);
                var indicesPtr = CudaHelpers.GetBufferStart(indices);
                var gradInputPtr = CudaHelpers.GetBufferStart(gradInput);

                var count = (int)input.ElementCount();

                Invoke(context, cudaContext, "MaxPoolBackward", new dim3(NNThreads.NumBlocks(count)), new dim3(NNThreads.NumThreads), 0, CUstream.NullStream,
                    count, gradOutputPtr, indicesPtr, nbatch, nslices, iheight, iwidth, oheight, owidth,
                    cd.kH, cd.kW, cd.dH, cd.dW, cd.padH, cd.padW, gradInputPtr);

            }

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
