// ***********************************************************************
// Assembly         : TensorSharp.CUDA91
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="Fp16.cs" company="TensorSharp.CUDA91">
//     Copyright (c) . All rights reserved.
// </copyright>
// <summary></summary>
// ***********************************************************************
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TensorSharp.CUDA.RuntimeCompiler;

namespace TensorSharp.CUDA.DeviceCode.Headers
{
    /// <summary>
    /// Class Fp16.
    /// </summary>
    [CudaInclude("Code", "Fp16")]
    public static class Fp16
    {
        /// <summary>
        /// The code
        /// </summary>
        public static readonly string Code = @"
typedef struct __align__(2) {
   unsigned short x;
} __half;
typedef __half half;
#define FP16_FUNC static __device__ __inline__
FP16_FUNC __half __float2half(const float a);
FP16_FUNC float __half2float(const __half a);

";

    }
}
