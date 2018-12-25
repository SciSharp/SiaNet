// ***********************************************************************
// Assembly         : TensorSharp.CUDA91
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="CudaReduceKernels.cs" company="TensorSharp.CUDA91">
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
    /// Class CudaReduceKernels.
    /// Implements the <see cref="TensorSharp.CUDA.DeviceCode.CudaCode" />
    /// </summary>
    /// <seealso cref="TensorSharp.CUDA.DeviceCode.CudaCode" />
    [Precompile]
    public class CudaReduceKernels : CudaCode
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="CudaReduceKernels"/> class.
        /// </summary>
        public CudaReduceKernels()
            : base(GetFullCode(), "General", "ReduceApplyUtils", "ReduceBlock", "Reduce", "ReduceMacros", "Math")
        {
        }

        /// <summary>
        /// Gets the full code.
        /// </summary>
        /// <returns>System.String.</returns>
        private static string GetFullCode()
        {
            var identity = "return a;";

            var result = new PermutationGenerator();
            result.AddReduce("sum", identity, "return a + b;");
            result.AddReduce("prod", identity, "return a * b;");
            result.AddReduce("min", identity, "return min(a, b);");
            result.AddReduce("max", identity, "return max(a, b);");

            result.AddReduce("e0_norm", "return a != 0 ? 1 : 0;", "return a + b;");
            result.AddReduce("e1_norm", "return fabsf(a);", "return a + b;");
            result.AddReduce("e2_norm", "return a * a;", "return a + b;");
            result.AddReduceNorm("en_norm");

            return result.ToString();
        }
    }

}
