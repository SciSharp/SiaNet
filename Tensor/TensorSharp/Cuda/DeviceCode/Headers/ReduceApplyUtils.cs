// ***********************************************************************
// Assembly         : TensorSharp.CUDA91
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="ReduceApplyUtils.cs" company="TensorSharp.CUDA91">
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
    /// Class ReduceApplyUtils.
    /// </summary>
    [CudaInclude("Code", "ReduceApplyUtils")]
    public static class ReduceApplyUtils
    {
        // ReduceApplyUtils functions from cuTorch

        /// <summary>
        /// The code
        /// </summary>
        public static readonly string Code = @"

// Translate a linear index for the apply to a float* offset;
// specialized on `Dims` to reduce nvcc compilation time
template <typename IndexType, int Dims>
struct IndexToOffset {
  static __host__ __device__ IndexType get(
    IndexType linearId,
    const TensorInfo<IndexType>& info) {
    IndexType offset = 0;

    // Use static dims
    for (int i = Dims - 1; i >= 0; --i) {
      IndexType curDimIndex = linearId % info.sizes[i];
      IndexType curDimOffset = curDimIndex * info.strides[i];
      offset += curDimOffset;

      if (i > 0) {
        linearId /= info.sizes[i];
      }
    }

    return offset;
  }
};

template <typename IndexType>
struct IndexToOffset<IndexType, -2> {
  static __forceinline__ __host__ __device__ IndexType
    get(IndexType linearId, const TensorInfo<IndexType>& info) {
    return linearId;
  }
};

template <typename IndexType>
struct IndexToOffset<IndexType, -1> {
  static __forceinline__ __host__ __device__ IndexType
    get(IndexType linearId, const TensorInfo<IndexType>& info) {
    IndexType offset = 0;

    // Use dynamic dims
    for (int i = info.dims - 1; i >= 0; --i) {
      IndexType curDimIndex = linearId % info.sizes[i];
      IndexType curDimOffset = curDimIndex * info.strides[i];
      offset += curDimOffset;

      linearId /= info.sizes[i];
    }

    return offset;
  }
};

template <typename IndexType>
__device__ __forceinline__ IndexType getLinearBlockId() {
  return blockIdx.z * gridDim.y * gridDim.x +
    blockIdx.y * gridDim.x +
    blockIdx.x;
}

";
    }
}
