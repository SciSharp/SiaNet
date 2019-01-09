// ***********************************************************************
// Assembly         : TensorSharp.CUDA91
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="CudaKernelCache.cs" company="TensorSharp.CUDA91">
//     Copyright (c) . All rights reserved.
// </copyright>
// <summary></summary>
// ***********************************************************************
using ManagedCuda;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace TensorSharp.CUDA.ContextState
{
    /// <summary>
    /// Class CudaKernelCache.
    /// Implements the <see cref="System.IDisposable" />
    /// </summary>
    /// <seealso cref="System.IDisposable" />
    public class CudaKernelCache : IDisposable
    {
        /// <summary>
        /// The active kernels
        /// </summary>
        private Dictionary<Tuple<CudaContext, byte[], string>, CudaKernel> activeKernels = new Dictionary<Tuple<CudaContext, byte[], string>, CudaKernel>();

        /// <summary>
        /// Initializes a new instance of the <see cref="CudaKernelCache"/> class.
        /// </summary>
        public CudaKernelCache()
        {
        }

        /// <summary>
        /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
        /// </summary>
        public void Dispose()
        {
            foreach (var kvp in activeKernels)
            {
                var ctx = kvp.Key.Item1;
                var kernel = kvp.Value;

                ctx.UnloadKernel(kernel);
            }
        }

        /// <summary>
        /// Gets the specified context.
        /// </summary>
        /// <param name="context">The context.</param>
        /// <param name="ptx">The PTX.</param>
        /// <param name="kernelName">Name of the kernel.</param>
        /// <returns>CudaKernel.</returns>
        public CudaKernel Get(CudaContext context, byte[] ptx, string kernelName)
        {
            CudaKernel value;
            if (activeKernels.TryGetValue(Tuple.Create(context, ptx, kernelName), out value))
            {
                return value;
            }
            else
            {
                value = context.LoadKernelPTX(ptx, kernelName);
                activeKernels.Add(Tuple.Create(context, ptx, kernelName), value);
                return value;
            }
        }
    }

}
