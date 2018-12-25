// ***********************************************************************
// Assembly         : TensorSharp.CUDA91
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="ReduceBlock.cs" company="TensorSharp.CUDA91">
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
    /// Class ReduceBlock.
    /// </summary>
    [CudaInclude("Code", "ReduceBlock")]
    public static class ReduceBlock
    {
        // reduceBlock function from cuTorch
        /// <summary>
        /// The code
        /// </summary>
        public const string Code = @"

// Block-wide reduction in shared memory helper; only threadIdx.x == 0 will
// return the reduced value
template <typename T, typename ReduceOp>
__device__ T reduceBlock(T* smem,
                         int numVals,
                         T threadVal,
                         ReduceOp reduceOp,
                         T init) {
  if (numVals == 0) {
    return init;
  }

  if (threadIdx.x < numVals) {
    smem[threadIdx.x] = threadVal;
  }

  // First warp will perform reductions across warps
  __syncthreads();
  if ((threadIdx.x / warpSize) == 0) {
    T r = threadIdx.x < numVals ? smem[threadIdx.x] : init;

    for (int i = warpSize + threadIdx.x; i < numVals; i += warpSize) {
      r = reduceOp(r, smem[i]);
    }

    smem[threadIdx.x] = r;
  }

  // First thread will perform reductions across the block
  __syncthreads();

  T r = init;
  if (threadIdx.x == 0) {
    r = smem[0];

    int numLanesParticipating = min(numVals, warpSize);

    if (numLanesParticipating == 32) {
      // Unroll for warpSize == 32 and numVals >= 32
#pragma unroll
      for (int i = 1; i < 32; ++i) {
        r = reduceOp(r, smem[i]);
      }
    } else {
      for (int i = 1; i < numLanesParticipating; ++i) {
        r = reduceOp(r, smem[i]);
      }
    }
  }

  return r;
}

";
    }
}
