// ***********************************************************************
// Assembly         : TensorSharp.CUDA91
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="PointwiseApply.cs" company="TensorSharp.CUDA91">
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
    // Depends on: general, ReduceApplyUtils
    /// <summary>
    /// Class PointwiseApply.
    /// </summary>
    [CudaInclude("Code", "PointwiseApply")]
    public static class PointwiseApply
    {
        // Apply functions from cuTorch

        /// <summary>
        /// The code
        /// </summary>
        public static readonly string Code = @"

template <typename Op, typename IndexType, int ADims>
#if __CUDA_ARCH__ >= 350
__launch_bounds__(32 * 16, 4)
#endif
__global__ void
pointwiseApply1(TensorInfo<IndexType> a,
                             IndexType totalElements,
                             Op op) {
  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalElements;
       linearIndex += gridDim.x * blockDim.x) {
    // Convert `linearIndex` into an offset of `a`
    const IndexType aOffset =
      IndexToOffset<IndexType, ADims>::get(linearIndex, a);

    op(&a.data[aOffset]);
  }
}

template <typename Op, typename IndexType, int ADims, int BDims>
#if __CUDA_ARCH__ >= 350
__launch_bounds__(32 * 16, 4)
#endif
__global__ void
pointwiseApply2(TensorInfo<IndexType> a,
                             TensorInfo<IndexType> b,
                             IndexType totalElements,
                             Op op) {
  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalElements;
       linearIndex += gridDim.x * blockDim.x) {
    // Convert `linearIndex` into an offset of `a`
    const IndexType aOffset =
      IndexToOffset<IndexType, ADims>::get(linearIndex, a);

    // Convert `linearIndex` into an offset of `b`
    const IndexType bOffset =
      IndexToOffset<IndexType, BDims>::get(linearIndex, b);

    op(&a.data[aOffset], &b.data[bOffset]);
  }
}

template <typename Op, typename IndexType, int ADims, int BDims, int CDims>
#if __CUDA_ARCH__ >= 350
__launch_bounds__(32 * 16, 4)
#endif
__global__ void
pointwiseApply3(TensorInfo<IndexType> a,
                             TensorInfo<IndexType> b,
                             TensorInfo<IndexType> c,
                             IndexType totalElements,
                             Op op) {
  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalElements;
       linearIndex += gridDim.x * blockDim.x) {
    // Convert `linearIndex` into an offset of `a`
    const IndexType aOffset =
      IndexToOffset<IndexType, ADims>::get(linearIndex, a);

    // Convert `linearIndex` into an offset of `b`
    const IndexType bOffset =
      IndexToOffset<IndexType, BDims>::get(linearIndex, b);

    // Convert `linearIndex` into an offset of `c`
    const IndexType cOffset =
      IndexToOffset<IndexType, CDims>::get(linearIndex, c);

    op(&a.data[aOffset], &b.data[bOffset], &c.data[cOffset]);
  }
}

";
    }
}
