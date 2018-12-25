// ***********************************************************************
// Assembly         : TensorSharp.CUDA91
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="CudaStorage.cs" company="TensorSharp.CUDA91">
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
using TensorSharp.CUDA.ContextState;

namespace TensorSharp.CUDA
{
    /// <summary>
    /// Class CudaStorage.
    /// Implements the <see cref="TensorSharp.Storage" />
    /// </summary>
    /// <seealso cref="TensorSharp.Storage" />
    public class CudaStorage : Storage
    {
        /// <summary>
        /// The context
        /// </summary>
        private readonly CudaContext context;

        /// <summary>
        /// The buffer handle
        /// </summary>
        private readonly IDeviceMemory bufferHandle;
        /// <summary>
        /// The device buffer
        /// </summary>
        private readonly CUdeviceptr deviceBuffer;


        /// <summary>
        /// Initializes a new instance of the <see cref="CudaStorage"/> class.
        /// </summary>
        /// <param name="allocator">The allocator.</param>
        /// <param name="tsContext">The ts context.</param>
        /// <param name="context">The context.</param>
        /// <param name="ElementType">Type of the element.</param>
        /// <param name="elementCount">The element count.</param>
        public CudaStorage(IAllocator allocator, TSCudaContext tsContext, CudaContext context, DType ElementType, long elementCount)
            : base(allocator, ElementType, elementCount)
        {
            this.TSContext = tsContext;
            this.context = context;

            this.bufferHandle = tsContext.AllocatorForDevice(DeviceId).Allocate(this.ByteLength);
            this.deviceBuffer = this.bufferHandle.Pointer;
        }

        /// <summary>
        /// Gets the ts context.
        /// </summary>
        /// <value>The ts context.</value>
        public TSCudaContext TSContext { get; private set; }

        /// <summary>
        /// This method is called when the reference count reaches zero. It will be called at most once to allow subclasses to release resources.
        /// </summary>
        protected override void Destroy()
        {
            bufferHandle.Free();
        }

        /// <summary>
        /// Locations the description.
        /// </summary>
        /// <returns>System.String.</returns>
        public override string LocationDescription()
        {
            return "CUDA:" + context.DeviceId;
        }

        /// <summary>
        /// Gets the device identifier.
        /// </summary>
        /// <value>The device identifier.</value>
        public int DeviceId
        {
            get { return context.DeviceId; }
        }

        /// <summary>
        /// Devices the PTR at element.
        /// </summary>
        /// <param name="index">The index.</param>
        /// <returns>CUdeviceptr.</returns>
        public CUdeviceptr DevicePtrAtElement(long index)
        {
            var offset = ElementType.Size() * index;
            return new CUdeviceptr(deviceBuffer.Pointer + offset);
        }

        /// <summary>
        /// Gets the element as float.
        /// </summary>
        /// <param name="index">The index.</param>
        /// <returns>System.Single.</returns>
        /// <exception cref="NotSupportedException">Element type " + ElementType + " not supported</exception>
        public override float GetElementAsFloat(long index)
        {
            var ptr = DevicePtrAtElement(index);

            if(ElementType == DType.Float32) { var result = new float[1]; context.CopyToHost(result, ptr); return result[0]; }
            else if (ElementType == DType.Float64) { var result = new double[1]; context.CopyToHost(result, ptr); return (float)result[0]; }
            else if (ElementType == DType.Int32) { var result = new int[1]; context.CopyToHost(result, ptr); return result[0]; }
            else if (ElementType == DType.UInt8) { var result = new byte[1]; context.CopyToHost(result, ptr); return result[0]; }
            else
                throw new NotSupportedException("Element type " + ElementType + " not supported");
        }

        /// <summary>
        /// Sets the element as float.
        /// </summary>
        /// <param name="index">The index.</param>
        /// <param name="value">The value.</param>
        /// <exception cref="NotSupportedException">Element type " + ElementType + " not supported</exception>
        public override void SetElementAsFloat(long index, float value)
        {
            var ptr = DevicePtrAtElement(index);

            if (ElementType == DType.Float32) { context.CopyToDevice(ptr, (float)value); }
            else if (ElementType == DType.Float64) { context.CopyToDevice(ptr, (double)value); }
            else if (ElementType == DType.Int32) { context.CopyToDevice(ptr, (int)value); }
            else if (ElementType == DType.UInt8) { context.CopyToDevice(ptr, (byte)value); }
            else
                throw new NotSupportedException("Element type " + ElementType + " not supported");
        }

        /// <summary>
        /// Copies to storage.
        /// </summary>
        /// <param name="storageIndex">Index of the storage.</param>
        /// <param name="src">The source.</param>
        /// <param name="byteCount">The byte count.</param>
        public override void CopyToStorage(long storageIndex, IntPtr src, long byteCount)
        {
            var dstPtr = DevicePtrAtElement(storageIndex);
            context.CopyToDevice(dstPtr, src, byteCount);
        }

        /// <summary>
        /// Copies from storage.
        /// </summary>
        /// <param name="dst">The DST.</param>
        /// <param name="storageIndex">Index of the storage.</param>
        /// <param name="byteCount">The byte count.</param>
        /// <exception cref="CudaException"></exception>
        public override void CopyFromStorage(IntPtr dst, long storageIndex, long byteCount)
        {
            var srcPtr = DevicePtrAtElement(storageIndex);

            // Call this method directly instead of CudaContext.CopyToHost because this method supports a long byteCount
            // CopyToHost only supports uint byteCount.
            var res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(dst, srcPtr, byteCount);
            if (res != CUResult.Success)
                throw new CudaException(res);
        }
    }
}
