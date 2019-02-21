// ***********************************************************************
// Assembly         : SiaNet.Backend.TensorSharp
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="CpuIndexingOps.cs" company="SiaNet.Backend.TensorSharp">
//     Copyright (c) . All rights reserved.
// </copyright>
// <summary></summary>
// ***********************************************************************
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using SiaNet.Backend.TensorSharp.Core;

namespace SiaNet.Backend.TensorSharp.Cpu
{
    /// <summary>
    /// Class CpuIndexingOps.
    /// </summary>
    [OpsClass]
    public class CpuIndexingOps
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="CpuIndexingOps"/> class.
        /// </summary>
        public CpuIndexingOps()
        {
        }

        /// <summary>
        /// The gather function
        /// </summary>
        private MethodInfo gather_func = NativeWrapper.GetMethod("TS_Gather");
        /// <summary>
        /// The scatter function
        /// </summary>
        private MethodInfo scatter_func = NativeWrapper.GetMethod("TS_Scatter");
        /// <summary>
        /// The scatter fill function
        /// </summary>
        private MethodInfo scatterFill_func = NativeWrapper.GetMethod("TS_ScatterFill");


        /// <summary>
        /// Gathers the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dim">The dim.</param>
        /// <param name="indices">The indices.</param>
        /// <returns>Tensor.</returns>
        /// <exception cref="InvalidOperationException">
        /// result and src must have same number of dimensions
        /// or
        /// src and indices must have same number of dimensions
        /// or
        /// result and indices must be the same size
        /// or
        /// result and src must be the same size except in dimension dim
        /// </exception>
        /// <exception cref="ArgumentOutOfRangeException">dim</exception>
        [RegisterOpStorageType("gather", typeof(CpuStorage))]
        public NDArray Gather(NDArray result, NDArray src, int dim, NDArray indices)
        {
            if (result != null && result.DimensionCount != src.DimensionCount) throw new InvalidOperationException("result and src must have same number of dimensions");
            if (result != null && dim < 0 && dim >= result.DimensionCount) throw new ArgumentOutOfRangeException("dim");
            if (indices.DimensionCount != src.DimensionCount) throw new InvalidOperationException("src and indices must have same number of dimensions");
            if (result != null && !result.IsSameSizeAs(indices)) throw new InvalidOperationException("result and indices must be the same size");
            if (result != null && !TensorResultBuilder.ArrayEqualExcept(src.Shape, result.Shape, dim)) throw new InvalidOperationException("result and src must be the same size except in dimension dim");

            var writeTarget = TensorResultBuilder.GetWriteTarget(result, indices.Allocator, src.ElementType, false, indices.Shape);

            NativeWrapper.InvokeTypeMatch(gather_func, writeTarget, src, dim, indices);
            return writeTarget;
        }

        /// <summary>
        /// Scatters the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dim">The dim.</param>
        /// <param name="indices">The indices.</param>
        /// <returns>Tensor.</returns>
        /// <exception cref="ArgumentNullException">result</exception>
        /// <exception cref="InvalidOperationException">
        /// result and src must have same number of dimensions
        /// or
        /// src and indices must have same number of dimensions
        /// or
        /// src and indices must be the same size
        /// or
        /// result and src must be the same size except in dimension dim
        /// </exception>
        /// <exception cref="ArgumentOutOfRangeException">dim</exception>
        [RegisterOpStorageType("scatter", typeof(CpuStorage))]
        public NDArray Scatter(NDArray result, NDArray src, int dim, NDArray indices)
        {
            if (result == null) throw new ArgumentNullException("result");

            if (result.DimensionCount != src.DimensionCount) throw new InvalidOperationException("result and src must have same number of dimensions");
            if (dim < 0 && dim >= result.DimensionCount) throw new ArgumentOutOfRangeException("dim");
            if (indices.DimensionCount != src.DimensionCount) throw new InvalidOperationException("src and indices must have same number of dimensions");
            if (!src.IsSameSizeAs(indices)) throw new InvalidOperationException("src and indices must be the same size");
            if (!TensorResultBuilder.ArrayEqualExcept(src.Shape, result.Shape, dim)) throw new InvalidOperationException("result and src must be the same size except in dimension dim");

            var writeTarget = result;

            NativeWrapper.InvokeTypeMatch(scatter_func, writeTarget, src, dim, indices);
            return writeTarget;
        }

        /// <summary>
        /// Scatters the fill.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="value">The value.</param>
        /// <param name="dim">The dim.</param>
        /// <param name="indices">The indices.</param>
        /// <returns>Tensor.</returns>
        /// <exception cref="ArgumentNullException">result</exception>
        /// <exception cref="ArgumentOutOfRangeException">dim</exception>
        /// <exception cref="InvalidOperationException">
        /// result and indices must have same number of dimensions
        /// or
        /// result and indices must be the same size except in dimension dim
        /// </exception>
        [RegisterOpStorageType("scatter_fill", typeof(CpuStorage))]
        public NDArray ScatterFill(NDArray result, float value, int dim, NDArray indices)
        {
            if (result == null) throw new ArgumentNullException("result");

            if (dim < 0 && dim >= result.DimensionCount) throw new ArgumentOutOfRangeException("dim");
            if (indices.DimensionCount != result.DimensionCount) throw new InvalidOperationException("result and indices must have same number of dimensions");
            if (!TensorResultBuilder.ArrayEqualExcept(indices.Shape, result.Shape, dim)) throw new InvalidOperationException("result and indices must be the same size except in dimension dim");

            var writeTarget = result;

            NativeWrapper.InvokeTypeMatch(scatterFill_func, writeTarget, value, dim, indices);
            return writeTarget;
        }
    }
}
