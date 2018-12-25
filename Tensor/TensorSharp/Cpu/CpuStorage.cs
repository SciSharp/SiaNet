// ***********************************************************************
// Assembly         : TensorSharp
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="CpuStorage.cs" company="TensorSharp">
//     Copyright (c) . All rights reserved.
// </copyright>
// <summary></summary>
// ***********************************************************************
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using TensorSharp.Core;

namespace TensorSharp.Cpu
{
    /// <summary>
    /// Class CpuStorage.
    /// Implements the <see cref="TensorSharp.Storage" />
    /// </summary>
    /// <seealso cref="TensorSharp.Storage" />
    public class CpuStorage : Storage
    {
        /// <summary>
        /// The buffer
        /// </summary>
        public IntPtr buffer;


        /// <summary>
        /// Initializes a new instance of the <see cref="CpuStorage"/> class.
        /// </summary>
        /// <param name="allocator">The allocator.</param>
        /// <param name="ElementType">Type of the element.</param>
        /// <param name="elementCount">The element count.</param>
        public CpuStorage(IAllocator allocator, DType ElementType, long elementCount)
            : base(allocator, ElementType, elementCount)
        {
            this.buffer = Marshal.AllocHGlobal(new IntPtr(this.ByteLength));
        }

        /// <summary>
        /// This method is called when the reference count reaches zero. It will be called at most once to allow subclasses to release resources.
        /// </summary>
        protected override void Destroy()
        {
            Marshal.FreeHGlobal(buffer);
            buffer = IntPtr.Zero;
        }

        /// <summary>
        /// Locations the description.
        /// </summary>
        /// <returns>System.String.</returns>
        public override string LocationDescription()
        {
            return "CPU";
        }

        /// <summary>
        /// PTRs at element.
        /// </summary>
        /// <param name="index">The index.</param>
        /// <returns>IntPtr.</returns>
        public IntPtr PtrAtElement(long index)
        {
            return new IntPtr(buffer.ToInt64() + (index * ElementType.Size()));
        }

        /// <summary>
        /// Gets the element as float.
        /// </summary>
        /// <param name="index">The index.</param>
        /// <returns>System.Single.</returns>
        /// <exception cref="NotSupportedException">Element type " + ElementType + " not supported</exception>
        public override float GetElementAsFloat(long index)
        {
            unsafe
            {
                if(ElementType == DType.Float32) return ((float*)buffer.ToPointer())[index];
                else if (ElementType == DType.Float64) return (float)((double*)buffer.ToPointer())[index];
                else if (ElementType == DType.Int32) return (float)((int*)buffer.ToPointer())[index];
                else if (ElementType == DType.UInt8) return (float)((byte*)buffer.ToPointer())[index];
                else
                    throw new NotSupportedException("Element type " + ElementType + " not supported");
            }
        }

        /// <summary>
        /// Sets the element as float.
        /// </summary>
        /// <param name="index">The index.</param>
        /// <param name="value">The value.</param>
        /// <exception cref="NotSupportedException">Element type " + ElementType + " not supported</exception>
        public override void SetElementAsFloat(long index, float value)
        {
            unsafe
            {
                if(ElementType == DType.Float32) ((float*)buffer.ToPointer())[index] = value;
                else if (ElementType == DType.Float64) ((double*)buffer.ToPointer())[index] = value;
                else if (ElementType == DType.Int32) ((int*)buffer.ToPointer())[index] = (int)value;
                else if (ElementType == DType.UInt8) ((byte*)buffer.ToPointer())[index] = (byte)value;
                else
                    throw new NotSupportedException("Element type " + ElementType + " not supported");
            }
        }

        /// <summary>
        /// Copies to storage.
        /// </summary>
        /// <param name="storageIndex">Index of the storage.</param>
        /// <param name="src">The source.</param>
        /// <param name="byteCount">The byte count.</param>
        public override void CopyToStorage(long storageIndex, IntPtr src, long byteCount)
        {
            var dstPtr = PtrAtElement(storageIndex);
            MemoryCopier.Copy(dstPtr, src, (ulong)byteCount);
        }

        /// <summary>
        /// Copies from storage.
        /// </summary>
        /// <param name="dst">The DST.</param>
        /// <param name="storageIndex">Index of the storage.</param>
        /// <param name="byteCount">The byte count.</param>
        public override void CopyFromStorage(IntPtr dst, long storageIndex, long byteCount)
        {
            var srcPtr = PtrAtElement(storageIndex);
            MemoryCopier.Copy(dst, srcPtr, (ulong)byteCount);
        }
    }
}
