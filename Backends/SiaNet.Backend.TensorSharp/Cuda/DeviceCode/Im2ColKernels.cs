// ***********************************************************************
// Assembly         : SiaNet.Backend.TensorSharp.CUDA91
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="Im2ColKernels.cs" company="SiaNet.Backend.TensorSharp.CUDA91">
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
using TensorSharp.Properties;

namespace SiaNet.Backend.TensorSharp.CUDA.DeviceCode
{
    /// <summary>
    /// Class Im2ColKernels.
    /// Implements the <see cref="SiaNet.Backend.TensorSharp.CUDA.DeviceCode.CudaCode" />
    /// </summary>
    /// <seealso cref="SiaNet.Backend.TensorSharp.CUDA.DeviceCode.CudaCode" />
    [Precompile]
    public class Im2ColCuda : CudaCode
    {

        /// <summary>
        /// Initializes a new instance of the <see cref="Im2ColCuda"/> class.
        /// </summary>
        public Im2ColCuda() : base(Resources.Im2Col)
        {
        }


        /// <summary>
        /// Im2s the col.
        /// </summary>
        /// <param name="im">The im.</param>
        /// <param name="col">The col.</param>
        /// <param name="channels">The channels.</param>
        /// <param name="height">The height.</param>
        /// <param name="width">The width.</param>
        /// <param name="ksize_h">The ksize h.</param>
        /// <param name="ksize_w">The ksize w.</param>
        /// <param name="pad_h">The pad h.</param>
        /// <param name="pad_w">The pad w.</param>
        /// <param name="stride_h">The stride h.</param>
        /// <param name="stride_w">The stride w.</param>
        /// <param name="dilation_h">The dilation h.</param>
        /// <param name="dilation_w">The dilation w.</param>
        public void Im2Col(NDArray im, NDArray col, int channels,
            int height, int width,
            int ksize_h, int ksize_w, int pad_h,
            int pad_w, int stride_h, int stride_w,
            int dilation_h, int dilation_w)
        {
            var context = CudaHelpers.TSContextForTensor(im);
            var cudaContext = context.CudaContextForTensor(im);

            // From Torch source:
            // We are going to launch channels * height_col * width_col kernels, each
            // kernel responsible for copying a single-channel grid.
            int height_col = (height + 2 * pad_h - (dilation_h * (ksize_h - 1) + 1))
                             / stride_h + 1;
            int width_col = (width + 2 * pad_w - (dilation_w * (ksize_w - 1) + 1))
                            / stride_w + 1;
            int num_kernels = channels * height_col * width_col;

            var data_im = CudaHelpers.GetBufferStart(im);
            var data_col = CudaHelpers.GetBufferStart(col);

            Invoke(context, cudaContext, "im2col_kernel", new dim3(NNThreads.NumBlocks(num_kernels)), new dim3(NNThreads.NumThreads), 0, CUstream.NullStream,
                num_kernels, data_im, height, width, channels, ksize_h, ksize_w,
                pad_h, pad_w, stride_h, stride_w,
                dilation_h, dilation_w,
                height_col, width_col, data_col);
        }

        /// <summary>
        /// Col2s the im.
        /// </summary>
        /// <param name="col">The col.</param>
        /// <param name="im">The im.</param>
        /// <param name="channels">The channels.</param>
        /// <param name="height">The height.</param>
        /// <param name="width">The width.</param>
        /// <param name="patch_h">The patch h.</param>
        /// <param name="patch_w">The patch w.</param>
        /// <param name="pad_h">The pad h.</param>
        /// <param name="pad_w">The pad w.</param>
        /// <param name="stride_h">The stride h.</param>
        /// <param name="stride_w">The stride w.</param>
        /// <param name="dilation_h">The dilation h.</param>
        /// <param name="dilation_w">The dilation w.</param>
        public void Col2Im(NDArray col, NDArray im, int channels, int height, int width,
            int patch_h, int patch_w, int pad_h,
            int pad_w, int stride_h, int stride_w,
            int dilation_h, int dilation_w)
        {
            var context = CudaHelpers.TSContextForTensor(im);
            var cudaContext = context.CudaContextForTensor(im);


            int height_col = (height + 2 * pad_h - (dilation_h * (patch_h - 1) + 1))
                   / stride_h + 1;
            int width_col = (width + 2 * pad_w - (dilation_w * (patch_w - 1) + 1))
                             / stride_w + 1;
            int num_kernels = channels * height * width;

            var data_im = CudaHelpers.GetBufferStart(im);
            var data_col = CudaHelpers.GetBufferStart(col);

            // From Torch source:
            // To avoid involving atomic operations, we will launch one kernel per
            // bottom dimension, and then in the kernel add up the top dimensions.

            Invoke(context, cudaContext, "col2im_kernel", new dim3(NNThreads.NumBlocks(num_kernels)), new dim3(NNThreads.NumThreads), 0, CUstream.NullStream,
                num_kernels, data_col, height, width, channels, patch_h, patch_w, pad_h, pad_w, stride_h, stride_w,
                dilation_h, dilation_w,
                height_col, width_col, data_im);
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
