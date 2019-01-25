// ***********************************************************************
// Assembly         : TensorSharp
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="TensorDimensionHelpers.cs" company="TensorSharp">
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
    /// Class TensorDimensionHelpers.
    /// </summary>
    public static class TensorDimensionHelpers
    {
        /// <summary>
        /// Elements the count.
        /// </summary>
        /// <param name="sizes">The sizes.</param>
        /// <returns>System.Int64.</returns>
        public static long ElementCount(long[] sizes)
        {
            if (sizes.Length == 0)
                return 0;

            var total = 1L;
            for (int i = 0; i < sizes.Length; ++i)
                total *= sizes[i];
            return total;
        }

        /// <summary>
        /// Gets the size of the storage.
        /// </summary>
        /// <param name="sizes">The sizes.</param>
        /// <param name="strides">The strides.</param>
        /// <returns>System.Int64.</returns>
        public static long GetStorageSize(long[] sizes, long[] strides)
        {
            long offset = 0;
            for (int i = 0; i < sizes.Length; ++i)
            {
                offset += (sizes[i] - 1) * strides[i];
            }
            return offset + 1; // +1 to count last element, which is at *index* equal to offset
        }

        // Returns the stride required for a tensor to be contiguous.
        // If a tensor is contiguous, then the elements in the last dimension are contiguous in memory,
        // with lower numbered dimensions having increasingly large strides.
        /// <summary>
        /// Gets the contiguous stride.
        /// </summary>
        /// <param name="dims">The dims.</param>
        /// <returns>System.Int64[].</returns>
        public static long[] GetContiguousStride(long[] dims)
        {
            long acc = 1;
            var stride = new long[dims.Length];
            for (int i = dims.Length - 1; i >= 0; --i)
            {
                stride[i] = acc;
                acc *= dims[i];
            }
            //if (dims.Last() == 1)
            //{
            //    for (int i = 0; i < dims.Length; i++)
            //    {
            //        stride[i] = acc;
            //        acc *= dims[i];
            //    }
            //}
            //else
            //{
            //    for (int i = dims.Length - 1; i >= 0; --i)
            //    {
            //        stride[i] = acc;
            //        acc *= dims[i];
            //    }
            //}

            return stride;
        }
    }
}
