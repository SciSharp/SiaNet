// ***********************************************************************
// Assembly         : TensorSharp.CUDA91
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="ConvertTensorArgs.cs" company="TensorSharp.CUDA91">
//     Copyright (c) . All rights reserved.
// </copyright>
// <summary></summary>
// ***********************************************************************
using ManagedCuda;
using ManagedCuda.BasicTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace TensorSharp.CUDA.RuntimeCompiler
{
    /// <summary>
    /// Class ConvertTensorArgs.
    /// </summary>
    public static class ConvertTensorArgs
    {
        /// <summary>
        /// Struct TensorInfoIndex64
        /// </summary>
        unsafe private struct TensorInfoIndex64
        {
            /// <summary>
            /// The data
            /// </summary>
            public ulong data;
            /// <summary>
            /// The sizes
            /// </summary>
            public fixed ulong sizes[TSCudaContext.MaxDims];
            /// <summary>
            /// The strides
            /// </summary>
            public fixed ulong strides[TSCudaContext.MaxDims];
            /// <summary>
            /// The dims
            /// </summary>
            public int dims;
        }

        /// <summary>
        /// Struct TensorInfoIndex32
        /// </summary>
        unsafe private struct TensorInfoIndex32
        {
            /// <summary>
            /// The data
            /// </summary>
            public ulong data;
            /// <summary>
            /// The sizes
            /// </summary>
            public fixed uint sizes[TSCudaContext.MaxDims];
            /// <summary>
            /// The strides
            /// </summary>
            public fixed uint strides[TSCudaContext.MaxDims];
            /// <summary>
            /// The dims
            /// </summary>
            public int dims;
        }


        /// <summary>
        /// Converts the specified context.
        /// </summary>
        /// <param name="context">The context.</param>
        /// <param name="index32">if set to <c>true</c> [index32].</param>
        /// <param name="args">The arguments.</param>
        public static void Convert(CudaContext context, bool index32, object[] args)
        {
            for (int i = 0; i < args.Length; ++i)
            {
                if (args[i] is Tensor)
                {
                    args[i] = MakeTensorInfo(context, (Tensor)args[i], index32);
                }
            }
        }



        /// <summary>
        /// Makes the tensor information.
        /// </summary>
        /// <param name="context">The context.</param>
        /// <param name="tensor">The tensor.</param>
        /// <param name="index32">if set to <c>true</c> [index32].</param>
        /// <param name="flattenDim">The flatten dim.</param>
        /// <returns>System.Object.</returns>
        unsafe public static object MakeTensorInfo(CudaContext context, Tensor tensor, bool index32, int flattenDim = -1)
        {
            if (index32)
            {
                var ti = new TensorInfoIndex32();
                ti.data = CudaHelpers.GetBufferStart(tensor);
                ti.dims = tensor.DimensionCount;
                for (int i = 0; i < tensor.DimensionCount; ++i)
                {
                    ti.sizes[i] = (uint)tensor.Sizes[i];
                    ti.strides[i] = (uint)tensor.Strides[i];
                }

                if (flattenDim != -1)
                {
                    ti.sizes[flattenDim] = 1;
                }

                return ti;
            }
            else
            {
                var ti = new TensorInfoIndex64();
                ti.data = CudaHelpers.GetBufferStart(tensor);
                ti.dims = tensor.DimensionCount;
                for (int i = 0; i < tensor.DimensionCount; ++i)
                {
                    ti.sizes[i] = (ulong)tensor.Sizes[i];
                    ti.strides[i] = (ulong)tensor.Strides[i];
                }

                if (flattenDim != -1)
                {
                    ti.sizes[flattenDim] = 1;
                }

                return ti;
            }
        }
    }
}
