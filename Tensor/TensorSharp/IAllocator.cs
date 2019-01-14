// ***********************************************************************
// Assembly         : TensorSharp
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="IAllocator.cs" company="TensorSharp">
//     Copyright (c) . All rights reserved.
// </copyright>
// <summary></summary>
// ***********************************************************************
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace TensorSharp
{
    /// <summary>
    /// Interface IAllocator
    /// </summary>
    public interface IAllocator
    {
        /// <summary>
        /// Allocates the specified element type.
        /// </summary>
        /// <param name="elementType">Type of the element.</param>
        /// <param name="elementCount">The element count.</param>
        /// <returns>Storage.</returns>
        Storage Allocate(DType elementType, long elementCount);

        void SetCurrent();
    }
}
