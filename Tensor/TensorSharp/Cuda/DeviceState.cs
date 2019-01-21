// ***********************************************************************
// Assembly         : TensorSharp.CUDA91
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="DeviceState.cs" company="TensorSharp.CUDA91">
//     Copyright (c) . All rights reserved.
// </copyright>
// <summary></summary>
// ***********************************************************************
using ManagedCuda;
using ManagedCuda.CudaBlas;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TensorSharp.CUDA.ContextState;
using TensorSharp.CUDA.Util;

namespace TensorSharp.CUDA
{
    /// <summary>
    /// Used by TSCudaContext to maintain per-device state
    /// Implements the <see cref="System.IDisposable" />
    /// </summary>
    /// <seealso cref="System.IDisposable" />
    public class DeviceState : IDisposable
    {
        /// <summary>
        /// The scratch space per sm stream
        /// </summary>
        private const int ScratchSpacePerSMStream = 4 * sizeof(float);


        /// <summary>
        /// The cuda context
        /// </summary>
        public readonly CudaContext CudaContext;
        /// <summary>
        /// The device information
        /// </summary>
        public readonly CudaDeviceProperties DeviceInfo;

        /// <summary>
        /// The blas handles
        /// </summary>
        public readonly ObjectPool<CudaBlas> BlasHandles;
        /// <summary>
        /// The DNN handles
        /// </summary>
        public readonly ObjectPool<ManagedCuda.CudaDNN.CudaDNNContext> DnnHandles;

        /// <summary>
        /// The memory allocator
        /// </summary>
        public readonly IDeviceAllocator MemoryAllocator;
        /// <summary>
        /// The scratch space
        /// </summary>
        public readonly ScratchSpace ScratchSpace;


        /// <summary>
        /// Initializes a new instance of the <see cref="DeviceState"/> class.
        /// </summary>
        /// <param name="deviceId">The device identifier.</param>
        public DeviceState(int deviceId)
        {
            this.CudaContext = new CudaContext(deviceId);
            this.DeviceInfo = this.CudaContext.GetDeviceInfo();

            this.BlasHandles = new ObjectPool<CudaBlas>(2, () =>
            {
                CudaContext.SetCurrent();
                return new CudaBlas();
            },
                blas => blas.Dispose());

            this.DnnHandles = new ObjectPool<ManagedCuda.CudaDNN.CudaDNNContext>(2, () =>
            {
                CudaContext.SetCurrent();
                return new ManagedCuda.CudaDNN.CudaDNNContext();
            },
                dnn => dnn.Dispose());

            this.MemoryAllocator = new PoolingDeviceAllocator(CudaContext);
            this.ScratchSpace = AllocScratchSpace(CudaContext, DeviceInfo);
        }

        /// <summary>
        /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
        /// </summary>
        public void Dispose()
        {
            BlasHandles.Dispose();
            CudaContext.Dispose();
            this.MemoryAllocator.Dispose();
        }

        /// <summary>
        /// Allocs the scratch space.
        /// </summary>
        /// <param name="context">The context.</param>
        /// <param name="deviceProps">The device props.</param>
        /// <returns>ScratchSpace.</returns>
        private static ScratchSpace AllocScratchSpace(CudaContext context, CudaDeviceProperties deviceProps)
        {
            var size = ScratchSpacePerSMStream * deviceProps.MultiProcessorCount;
            var buffer = context.AllocateMemory(size);
            return new ScratchSpace() { size = size, buffer = buffer };
        }
    }
}
