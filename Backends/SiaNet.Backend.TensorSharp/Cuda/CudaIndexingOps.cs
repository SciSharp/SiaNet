// ***********************************************************************
// Assembly         : SiaNet.Backend.TensorSharp.CUDA91
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="CudaIndexingOps.cs" company="SiaNet.Backend.TensorSharp.CUDA91">
//     Copyright (c) . All rights reserved.
// </copyright>
// <summary></summary>
// ***********************************************************************
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using SiaNet.Backend.TensorSharp.CUDA.DeviceCode;

namespace SiaNet.Backend.TensorSharp.CUDA
{
    /// <summary>
    /// Class CudaIndexingOps.
    /// </summary>
    [OpsClass]
    public class CudaIndexingOps
    {
        /// <summary>
        /// The index select
        /// </summary>
        private readonly IndexSelectKernels indexSelect = new IndexSelectKernels();
        /// <summary>
        /// The gather
        /// </summary>
        private readonly GatherScatterKernels gather = new GatherScatterKernels();


        /// <summary>
        /// Initializes a new instance of the <see cref="CudaIndexingOps"/> class.
        /// </summary>
        public CudaIndexingOps()
        {
        }

        /// <summary>
        /// Indexes the select.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <param name="indices">The indices.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("index_select", typeof(CudaStorage))]
        public NDArray IndexSelect(NDArray result, NDArray src, int dimension, NDArray indices) { return indexSelect.IndexSelect(result, src, dimension, indices); }

        /// <summary>
        /// Gathers the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <param name="indices">The indices.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("gather", typeof(CudaStorage))]
        public NDArray Gather(NDArray result, NDArray src, int dimension, NDArray indices) { return gather.Gather(result, src, dimension, indices); }

        /// <summary>
        /// Scatters the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <param name="indices">The indices.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("scatter", typeof(CudaStorage))]
        public NDArray Scatter(NDArray result, NDArray src, int dimension, NDArray indices) { return gather.Scatter(result, src, dimension, indices); }

        /// <summary>
        /// Scatters the fill.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="value">The value.</param>
        /// <param name="dimension">The dimension.</param>
        /// <param name="indices">The indices.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("scatter_fill", typeof(CudaStorage))]
        public NDArray ScatterFill(NDArray result, float value, int dimension, NDArray indices) { return gather.ScatterFill(result, value, dimension, indices); }

        [RegisterOpStorageType("diag", typeof(CudaStorage))]
        public NDArray Diag(NDArray result, NDArray src) { return src; }
    }
}
