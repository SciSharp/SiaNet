// ***********************************************************************
// Assembly         : TensorSharp.CUDA91
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="IDeviceAllocator.cs" company="TensorSharp.CUDA91">
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
    /// Interface IDeviceMemory
    /// </summary>
    public interface IDeviceMemory
    {
        /// <summary>
        /// Gets the pointer.
        /// </summary>
        /// <value>The pointer.</value>
        CUdeviceptr Pointer { get; }

        /// <summary>
        /// Frees this instance.
        /// </summary>
        void Free();
    }

    /// <summary>
    /// Interface IDeviceAllocator
    /// Implements the <see cref="System.IDisposable" />
    /// </summary>
    /// <seealso cref="System.IDisposable" />
    public interface IDeviceAllocator : IDisposable
    {
        /// <summary>
        /// Allocates the specified byte count.
        /// </summary>
        /// <param name="byteCount">The byte count.</param>
        /// <returns>IDeviceMemory.</returns>
        IDeviceMemory Allocate(long byteCount);
    }
}
