// ***********************************************************************
// Assembly         : TensorSharp.CUDA91
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="CudaAllocator.cs" company="TensorSharp.CUDA91">
//     Copyright (c) . All rights reserved.
// </copyright>
// <summary></summary>
// ***********************************************************************
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace TensorSharp.CUDA
{
    /// <summary>
    /// Class CudaAllocator.
    /// Implements the <see cref="TensorSharp.IAllocator" />
    /// </summary>
    /// <seealso cref="TensorSharp.IAllocator" />
    public class CudaAllocator : IAllocator
    {
        /// <summary>
        /// The context
        /// </summary>
        private readonly TSCudaContext context;
        /// <summary>
        /// The device identifier
        /// </summary>
        private readonly int deviceId;

        /// <summary>
        /// Initializes a new instance of the <see cref="CudaAllocator"/> class.
        /// </summary>
        /// <param name="context">The context.</param>
        /// <param name="deviceId">The device identifier.</param>
        public CudaAllocator(TSCudaContext context, int deviceId)
        {
            this.context = context;
            this.deviceId = deviceId;
        }

        /// <summary>
        /// Gets the context.
        /// </summary>
        /// <value>The context.</value>
        public TSCudaContext Context { get { return context; } }
        /// <summary>
        /// Gets the device identifier.
        /// </summary>
        /// <value>The device identifier.</value>
        public int DeviceId { get { return deviceId; } }

        /// <summary>
        /// Allocates the specified element type.
        /// </summary>
        /// <param name="elementType">Type of the element.</param>
        /// <param name="elementCount">The element count.</param>
        /// <returns>Storage.</returns>
        public Storage Allocate(DType elementType, long elementCount)
        {
            return new CudaStorage(this, context, context.CudaContextForDevice(deviceId), elementType, elementCount);
        }

        public void SetCurrent()
        {
            foreach (var item in context.devices)
            {
                item.CudaContext.SetCurrent();
            }
        }
    }
}
