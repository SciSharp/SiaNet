// ***********************************************************************
// Assembly         : TensorSharp.CUDA91
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="General.cs" company="TensorSharp.CUDA91">
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
    /// Class KernelGeneral.
    /// </summary>
    [CudaInclude("Code", "General")]
    public static class KernelGeneral
    {
        /// <summary>
        /// The code
        /// </summary>
        public static readonly string Code = @"

#define __int64 long long
#define __int32 int

#define MAX_CUTORCH_DIMS " + TSCudaContext.MaxDims + "\n" + @"

template <typename IndexType>
struct TensorInfo {
  float* data;
  IndexType sizes[MAX_CUTORCH_DIMS];
  IndexType strides[MAX_CUTORCH_DIMS];
  int dims;
};

";

    }
}
