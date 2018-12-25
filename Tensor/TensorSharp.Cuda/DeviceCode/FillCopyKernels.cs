// ***********************************************************************
// Assembly         : TensorSharp.CUDA91
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="FillCopyKernels.cs" company="TensorSharp.CUDA91">
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
    /// Class FillCopyKernels.
    /// Implements the <see cref="TensorSharp.CUDA.DeviceCode.CudaCode" />
    /// </summary>
    /// <seealso cref="TensorSharp.CUDA.DeviceCode.CudaCode" />
    [Precompile]
    public class FillCopyKernels : CudaCode
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="FillCopyKernels"/> class.
        /// </summary>
        public FillCopyKernels()
            : base(GetFullCode(), "General", "ReduceApplyUtils", "PointwiseApply", "ApplyMacros")
        {
        }

        /// <summary>
        /// Gets the full code.
        /// </summary>
        /// <returns>System.String.</returns>
        private static string GetFullCode()
        {
            var result = new PermutationGenerator();
            result.AddApplyTS("fill", "*a = b;");

            result.AddApplyTT("copy", "*a = *b;");

            return result.ToString();
        }
    }

}
