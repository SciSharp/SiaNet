// ***********************************************************************
// Assembly         : SiaNet.Backend.TensorSharp.CUDA91
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="CudaReduceAllKernels.cs" company="SiaNet.Backend.TensorSharp.CUDA91">
//     Copyright (c) . All rights reserved.
// </copyright>
// <summary></summary>
// ***********************************************************************
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using SiaNet.Backend.TensorSharp.CUDA.RuntimeCompiler;

namespace SiaNet.Backend.TensorSharp.CUDA.DeviceCode
{
    /// <summary>
    /// Class CudaReduceAllKernels.
    /// Implements the <see cref="SiaNet.Backend.TensorSharp.CUDA.DeviceCode.CudaCode" />
    /// </summary>
    /// <seealso cref="SiaNet.Backend.TensorSharp.CUDA.DeviceCode.CudaCode" />
    [Precompile]
    public class CudaReduceAllKernels : CudaCode
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="CudaReduceAllKernels"/> class.
        /// </summary>
        public CudaReduceAllKernels()
            : base(GetFullCode(), "General", "ReduceApplyUtils", "ReduceBlock", "ReduceAll", "ReduceAllMacros", "Math")
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
            result.AddReduceAll("sumAll", identity, "return a + b;");
            result.AddReduceAll("prodAll", identity, "return a * b;");
            result.AddReduceAll("minAll", identity, "return min(a, b);");
            result.AddReduceAll("maxAll", identity, "return max(a, b);");

            result.AddReduceAll("e0_norm", "return a != 0 ? 1 : 0;", "return a + b;");
            result.AddReduceAll("e1_norm", "return fabsf(a);", "return a + b;");
            result.AddReduceAll("e2_norm", "return a * a;", "return a + b;");
            result.AddReduceAllNorm("en_norm");

            result.AddReduceAllSubSquare("subSquare");

            return result.ToString();
        }
    }
}
