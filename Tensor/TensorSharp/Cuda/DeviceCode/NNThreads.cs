// ***********************************************************************
// Assembly         : TensorSharp.CUDA91
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="NNThreads.cs" company="TensorSharp.CUDA91">
//     Copyright (c) . All rights reserved.
// </copyright>
// <summary></summary>
// ***********************************************************************
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace TensorSharp.CUDA.DeviceCode
{
    /// <summary>
    /// Class NNThreads.
    /// </summary>
    public static class NNThreads
    {
        /// <summary>
        /// The number threads
        /// </summary>
        public const int NumThreads = 1024;

        /// <summary>
        /// Numbers the blocks.
        /// </summary>
        /// <param name="n">The n.</param>
        /// <returns>System.Int32.</returns>
        public static int NumBlocks(int n)
        {
            return (n + NumThreads - 1) / NumThreads;
        }
    }
}
