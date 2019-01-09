// ***********************************************************************
// Assembly         : TensorSharp.CUDA91
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="BasicDeviceAllocator.cs" company="TensorSharp.CUDA91">
//     Copyright (c) . All rights reserved.
// </copyright>
// <summary></summary>
// ***********************************************************************
using ManagedCuda;
using ManagedCuda.BasicTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace TensorSharp.CUDA.ContextState
{
    /// <summary>
    /// This allocator simply forwards all alloc/free requests to CUDA. This will generally be slow
    /// because calling cudaMalloc causes GPU synchronization
    /// Implements the <see cref="TensorSharp.CUDA.ContextState.IDeviceAllocator" />
    /// </summary>
    /// <seealso cref="TensorSharp.CUDA.ContextState.IDeviceAllocator" />
    public class BasicDeviceAllocator : IDeviceAllocator
    {
        /// <summary>
        /// The context
        /// </summary>
        private readonly CudaContext context;

        /// <summary>
        /// Initializes a new instance of the <see cref="BasicDeviceAllocator"/> class.
        /// </summary>
        /// <param name="cudaContext">The cuda context.</param>
        public BasicDeviceAllocator(CudaContext cudaContext)
        {
            this.context = cudaContext;
        }

        /// <summary>
        /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
        /// </summary>
        public void Dispose()
        {
        }


        /// <summary>
        /// Allocates the specified byte count.
        /// </summary>
        /// <param name="byteCount">The byte count.</param>
        /// <returns>IDeviceMemory.</returns>
        public IDeviceMemory Allocate(long byteCount)
        {
            var buffer = context.AllocateMemory(byteCount);
            return new BasicDeviceMemory(buffer, () => context.FreeMemory(buffer));
        }
    }

    /// <summary>
    /// Class BasicDeviceMemory.
    /// Implements the <see cref="TensorSharp.CUDA.ContextState.IDeviceMemory" />
    /// </summary>
    /// <seealso cref="TensorSharp.CUDA.ContextState.IDeviceMemory" />
    public class BasicDeviceMemory : IDeviceMemory
    {
        /// <summary>
        /// The pointer
        /// </summary>
        private readonly CUdeviceptr pointer;
        /// <summary>
        /// The free handler
        /// </summary>
        private readonly Action freeHandler;

        /// <summary>
        /// Gets the pointer.
        /// </summary>
        /// <value>The pointer.</value>
        public CUdeviceptr Pointer { get { return pointer; } }


        /// <summary>
        /// Initializes a new instance of the <see cref="BasicDeviceMemory"/> class.
        /// </summary>
        /// <param name="pointer">The pointer.</param>
        /// <param name="freeHandler">The free handler.</param>
        public BasicDeviceMemory(CUdeviceptr pointer, Action freeHandler)
        {
            this.pointer = pointer;
            this.freeHandler = freeHandler;
        }

        /// <summary>
        /// Frees this instance.
        /// </summary>
        public void Free()
        {
            freeHandler();
        }
    }
}
