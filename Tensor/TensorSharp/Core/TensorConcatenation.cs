// ***********************************************************************
// Assembly         : TensorSharp
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="TensorConcatenation.cs" company="TensorSharp">
//     Copyright (c) . All rights reserved.
// </copyright>
// <summary></summary>
// ***********************************************************************
namespace TensorSharp.Core
{
    using System;
    using System.Linq;

    /// <summary>
    /// Class Tensor Concatenation.
    /// </summary>
    public static class TensorConcatenation
    {
        // Requires an implementation of *copy* for the given tensor types
        /// <summary>
        /// Concats the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="dimension">The dimension.</param>
        /// <param name="inputs">The inputs.</param>
        /// <returns>Tensor.</returns>
        /// <exception cref="ArgumentException">Concat: at least two tensors required - inputs</exception>
        public static Tensor Concat(Tensor result, int dimension, params Tensor[] inputs)
        {
            if (inputs.Length < 2) throw new ArgumentException("Concat: at least two tensors required", "inputs");

            var ndim = Math.Max(dimension, inputs.Max(x => x.DimensionCount));
            var size = ConcatTensorSize(ndim, dimension, inputs);

            var writeTarget = TensorResultBuilder.GetWriteTarget(result, inputs[0], false, size);


            // Select each region of the result corresponding to each input tensor,
            // and copy into the result
            long offset = 0;
            for (int j = 0; j < inputs.Length; ++j)
            {
                var dimSize = GetDimSize(inputs[j], dimension);
                using (var nt = writeTarget.Narrow(dimension, offset, dimSize))
                {
                    Ops.Copy(nt, inputs[j]);
                }
                offset += dimSize;
            }

            return writeTarget;
        }



        /// <summary>
        /// Gets the size of the dim.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <param name="dim">The dim.</param>
        /// <returns>System.Int64.</returns>
        private static long GetDimSize(Tensor tensor, int dim)
        {
            return dim < tensor.DimensionCount ? tensor.Shape[dim] : 1;
        }

        /// <summary>
        /// Concats the size of the tensor.
        /// </summary>
        /// <param name="ndim">The ndim.</param>
        /// <param name="dimension">The dimension.</param>
        /// <param name="tensors">The tensors.</param>
        /// <returns>System.Int64[].</returns>
        /// <exception cref="InvalidOperationException">Inconsistent tensor sizes</exception>
        private static long[] ConcatTensorSize(int ndim, int dimension, Tensor[] tensors)
        {
            var result = new long[ndim];
            for (int i = 0; i < ndim; ++i)
            {
                var dimSize = GetDimSize(tensors[0], i);
                if (i == dimension)
                {
                    for (int j = 1; j < tensors.Length; ++j)
                    {
                        dimSize += GetDimSize(tensors[j], i);
                    }
                }
                else
                {
                    for (int j = 1; j < tensors.Length; ++j)
                    {
                        if (dimSize != GetDimSize(tensors[j], i))
                        {
                            throw new InvalidOperationException("Inconsistent tensor sizes");
                        }
                    }
                }
                result[i] = dimSize;
            }
            return result;
        }

    }
}
