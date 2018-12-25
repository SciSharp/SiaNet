// ***********************************************************************
// Assembly         : TensorSharp
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="Storage.cs" company="TensorSharp">
//     Copyright (c) . All rights reserved.
// </copyright>
// <summary></summary>
// ***********************************************************************
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace TensorSharp
{
    /// <summary>
    /// Class Storage.
    /// Implements the <see cref="TensorSharp.RefCounted" />
    /// </summary>
    /// <seealso cref="TensorSharp.RefCounted" />
    public abstract class Storage : RefCounted
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Storage"/> class.
        /// </summary>
        /// <param name="allocator">The allocator.</param>
        /// <param name="elementType">Type of the element.</param>
        /// <param name="elementCount">The element count.</param>
        public Storage(IAllocator allocator, DType elementType, long elementCount)
        {
            this.Allocator = allocator;
            this.ElementType = elementType;
            this.ElementCount = elementCount;
        }

        /// <summary>
        /// Gets a reference to the allocator that constructed this Storage object.
        /// </summary>
        /// <value>The allocator.</value>
        public IAllocator Allocator { get; private set; }

        /// <summary>
        /// Gets the type of the element.
        /// </summary>
        /// <value>The type of the element.</value>
        public DType ElementType { get; private set; }
        /// <summary>
        /// Gets the element count.
        /// </summary>
        /// <value>The element count.</value>
        public long ElementCount { get; private set; }

        /// <summary>
        /// Gets the length of the byte.
        /// </summary>
        /// <value>The length of the byte.</value>
        public long ByteLength { get { return ElementCount * ElementType.Size(); } }

        /// <summary>
        /// Determines whether [is owner exclusive].
        /// </summary>
        /// <returns><c>true</c> if [is owner exclusive]; otherwise, <c>false</c>.</returns>
        public bool IsOwnerExclusive()
        {
            return this.GetCurrentRefCount() == 1;
        }



        /// <summary>
        /// Locations the description.
        /// </summary>
        /// <returns>System.String.</returns>
        public abstract string LocationDescription();

        /// <summary>
        /// Gets the element as float.
        /// </summary>
        /// <param name="index">The index.</param>
        /// <returns>System.Single.</returns>
        public abstract float GetElementAsFloat(long index);
        /// <summary>
        /// Sets the element as float.
        /// </summary>
        /// <param name="index">The index.</param>
        /// <param name="value">The value.</param>
        public abstract void SetElementAsFloat(long index, float value);

        /// <summary>
        /// Copies to storage.
        /// </summary>
        /// <param name="storageIndex">Index of the storage.</param>
        /// <param name="src">The source.</param>
        /// <param name="byteCount">The byte count.</param>
        public abstract void CopyToStorage(long storageIndex, IntPtr src, long byteCount);
        /// <summary>
        /// Copies from storage.
        /// </summary>
        /// <param name="dst">The DST.</param>
        /// <param name="storageIndex">Index of the storage.</param>
        /// <param name="byteCount">The byte count.</param>
        public abstract void CopyFromStorage(IntPtr dst, long storageIndex, long byteCount);
    }
}
