// ***********************************************************************
// Assembly         : TensorSharp.CUDA91
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="CudaHelpers.cs" company="TensorSharp.CUDA91">
//     Copyright (c) . All rights reserved.
// </copyright>
// <summary></summary>
// ***********************************************************************
using ManagedCuda.BasicTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace TensorSharp.CUDA
{
    /// <summary>
    /// Class CudaHelpers.
    /// </summary>
    public static class CudaHelpers
    {
        /// <summary>
        /// Gets the buffer start.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <returns>CUdeviceptr.</returns>
        public static CUdeviceptr GetBufferStart(Tensor tensor)
        {
            return ((CudaStorage)tensor.Storage).DevicePtrAtElement(tensor.StorageOffset);
        }

        /// <summary>
        /// Throws if different devices.
        /// </summary>
        /// <param name="tensors">The tensors.</param>
        /// <exception cref="InvalidOperationException">All tensors must reside on the same device</exception>
        public static void ThrowIfDifferentDevices(params Tensor[] tensors)
        {
            var nonNull = tensors.Where(x => x != null);
            if (!nonNull.Any())
                return;

            var device = CudaHelpers.GetDeviceId(nonNull.First());

            if(nonNull.Any(x => CudaHelpers.GetDeviceId(x) != device))
                throw new InvalidOperationException("All tensors must reside on the same device");
        }

        /// <summary>
        /// Gets the device identifier.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <returns>System.Int32.</returns>
        public static int GetDeviceId(Tensor tensor)
        {
            return ((CudaStorage)tensor.Storage).DeviceId;
        }

        /// <summary>
        /// Tses the context for tensor.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <returns>TSCudaContext.</returns>
        public static TSCudaContext TSContextForTensor(Tensor tensor)
        {
            return ((CudaStorage)tensor.Storage).TSContext;
        }
    }
}
