// ***********************************************************************
// Assembly         : SiaNet.Backend.TensorSharp.CUDA91
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="ApplySpecialization.cs" company="SiaNet.Backend.TensorSharp.CUDA91">
//     Copyright (c) . All rights reserved.
// </copyright>
// <summary></summary>
// ***********************************************************************
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using SiaNet.Backend.TensorSharp.CUDA.RuntimeCompiler;

namespace SiaNet.Backend.TensorSharp.CUDA
{
    // Represents a compile-time specialization of ApplyN.
    // If all tensors are small enough, the kernel will use 32-bit indices
    // The kernels are also specialized for contiguous tensors, tensors with a
    // small number of dimensions, and a totally generic 'specialization'.
    // If TensorDims[i] == -2, then tensor i is entirely contiguous
    // If TensorDims[i] == -1, a totally generic kernel should be generated for that tensor.
    /// <summary>
    /// Class ApplySpecialization.
    /// </summary>
    public class ApplySpecialization
    {
        /// <summary>
        /// The index type32
        /// </summary>
        public const string IndexType32 = "unsigned __int32";
        /// <summary>
        /// The index type64
        /// </summary>
        public const string IndexType64 = "unsigned __int64";


        /// <summary>
        /// Gets a value indicating whether [use32 bit indices].
        /// </summary>
        /// <value><c>true</c> if [use32 bit indices]; otherwise, <c>false</c>.</value>
        public bool Use32BitIndices { get; private set; }
        /// <summary>
        /// Gets the tensor dims.
        /// </summary>
        /// <value>The tensor dims.</value>
        public int[] TensorDims { get; private set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="ApplySpecialization"/> class.
        /// </summary>
        /// <param name="tensors">The tensors.</param>
        public ApplySpecialization(params NDArray[] tensors)
        {
            if (tensors.All(ApplyUtils.CanUse32BitIndexMath))
            {
                this.Use32BitIndices = true;

                // Specialize each tensor dimenionality independently
                this.TensorDims = tensors.Select(tensor =>
                {
                    if (tensor.IsContiguous())
                        return -2;
                    return tensor.DimensionCount > 3 ? -1 : tensor.DimensionCount;
                })
                .ToArray();
            }
            else
            {
                this.Use32BitIndices = false;
                // For 64-bit index case (ie. large tensors), only specalize on totally contiguous
                // or totally generic
                if (tensors.All(x => x.IsContiguous()))
                {
                    // All tensors are contiguous
                    TensorDims = Enumerable.Repeat(-2, tensors.Length).ToArray();
                }
                else
                {
                    // Not all tensors are contiguous - just generate a completely generic kernel
                    TensorDims = Enumerable.Repeat(-1, tensors.Length).ToArray();
                }
            }

        }

        /// <summary>
        /// Initializes a new instance of the <see cref="ApplySpecialization"/> class.
        /// </summary>
        /// <param name="use32BitIndices">if set to <c>true</c> [use32 bit indices].</param>
        /// <param name="tensorDims">The tensor dims.</param>
        public ApplySpecialization(bool use32BitIndices, params int[] tensorDims)
        {
            this.Use32BitIndices = use32BitIndices;
            this.TensorDims = tensorDims;
        }



        /// <summary>
        /// Gets the configuration.
        /// </summary>
        /// <returns>KernelConfig.</returns>
        public KernelConfig GetConfig()
        {
            var result = new KernelConfig();

            result.Set("INDEX_TYPE", Use32BitIndices ? IndexType32 : IndexType64);

            for (int i = 0; i < TensorDims.Length; ++i)
            {
                var tensorName = (char)('A' + i);
                result.Set("DIMS" + tensorName, this.TensorDims[i].ToString());
            }

            return result;
        }

        /// <summary>
        /// Alls the specializations.
        /// </summary>
        /// <param name="tensorCount">The tensor count.</param>
        /// <returns>IEnumerable&lt;ApplySpecialization&gt;.</returns>
        public static IEnumerable<ApplySpecialization> AllSpecializations(int tensorCount)
        {
            yield return new ApplySpecialization(false, Enumerable.Repeat(-2, tensorCount).ToArray());
            yield return new ApplySpecialization(false, Enumerable.Repeat(-1, tensorCount).ToArray());

            foreach (var combination in CombinationsOf(All32BitTensorDims, tensorCount))
            {
                yield return new ApplySpecialization(true, combination);
            }
        }

        /// <summary>
        /// The all32 bit tensor dims
        /// </summary>
        private static readonly int[] All32BitTensorDims = new int[] { -2, -1, 1, 2, 3 };

        /// <summary>
        /// Combinationses the of.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="possibleValues">The possible values.</param>
        /// <param name="count">The count.</param>
        /// <returns>IEnumerable&lt;T[]&gt;.</returns>
        /// <exception cref="ArgumentOutOfRangeException">count</exception>
        private static IEnumerable<T[]> CombinationsOf<T>(T[] possibleValues, int count)
        {
            if (count < 1) throw new ArgumentOutOfRangeException("count");

            if (count == 1)
            {
                foreach (var item in possibleValues)
                {
                    yield return new T[] { item };
                }
            }
            else
            {
                foreach (var item in possibleValues)
                {
                    var restCombinations = CombinationsOf(possibleValues, count - 1);
                    foreach (var restItems in restCombinations)
                    {
                        var result = new List<T>(count);
                        result.AddRange(restItems);
                        result.Add(item);
                        yield return result.ToArray();
                    }
                }
            }
        }
    }
}
