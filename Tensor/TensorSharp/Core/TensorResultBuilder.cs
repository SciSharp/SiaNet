// ***********************************************************************
// Assembly         : TensorSharp
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="TensorResultBuilder.cs" company="TensorSharp">
//     Copyright (c) . All rights reserved.
// </copyright>
// <summary></summary>
// ***********************************************************************

namespace TensorSharp.Core
{
    using System;

    /// <summary>
    /// Class TensorResultBuilder.
    /// </summary>
    public static class TensorResultBuilder
    {
        // If a maybeResult is null, a new tensor will be constructed using the device id and element type of newTemplate
        /// <summary>
        /// Gets the write target.
        /// </summary>
        /// <param name="maybeResult">The maybe result.</param>
        /// <param name="newTemplate">The new template.</param>
        /// <param name="requireContiguous">if set to <c>true</c> [require contiguous].</param>
        /// <param name="requiredSizes">The required sizes.</param>
        /// <returns>Tensor.</returns>
        public static Tensor GetWriteTarget(Tensor maybeResult, Tensor newTemplate, bool requireContiguous, params long[] requiredSizes)
        {
            return GetWriteTarget(maybeResult, newTemplate.Allocator, newTemplate.ElementType, requireContiguous, requiredSizes);
        }

        /// <summary>
        /// Gets the write target.
        /// </summary>
        /// <param name="maybeResult">The maybe result.</param>
        /// <param name="allocatorForNew">The allocator for new.</param>
        /// <param name="elementTypeForNew">The element type for new.</param>
        /// <param name="requireContiguous">if set to <c>true</c> [require contiguous].</param>
        /// <param name="requiredSizes">The required sizes.</param>
        /// <returns>Tensor.</returns>
        /// <exception cref="InvalidOperationException"></exception>
        public static Tensor GetWriteTarget(Tensor maybeResult, IAllocator allocatorForNew, DType elementTypeForNew, bool requireContiguous, params long[] requiredSizes)
        {
            if (maybeResult != null)
            {
                if (!MatchesRequirements(maybeResult, requireContiguous, requiredSizes))
                {
                    var message = string.Format("output tensor does not match requirements. Tensor must have sizes {0}{1}",
                        string.Join(", ", requiredSizes),
                        requireContiguous ? "; and must be contiguous" : "");

                    throw new InvalidOperationException(message);
                }
                return maybeResult;
            }
            else
            {
                return new Tensor(allocatorForNew, elementTypeForNew, requiredSizes);
            }
        }

        /// <summary>
        /// Matcheses the requirements.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <param name="requireContiguous">if set to <c>true</c> [require contiguous].</param>
        /// <param name="requiredSizes">The required sizes.</param>
        /// <returns><c>true</c> if XXXX, <c>false</c> otherwise.</returns>
        private static bool MatchesRequirements(Tensor tensor, bool requireContiguous, params long[] requiredSizes)
        {
            if (requireContiguous && !tensor.IsContiguous())
            {
                return false;
            }

            return ArrayEqual(tensor.Shape, requiredSizes);
        }

        /// <summary>
        /// Arrays the equal.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="a">a.</param>
        /// <param name="b">The b.</param>
        /// <returns><c>true</c> if XXXX, <c>false</c> otherwise.</returns>
        public static bool ArrayEqual<T>(T[] a, T[] b)
        {
            if (a.Length != b.Length)
                return false;

            for(int i = 0; i < a.Length; ++i)
            {
                if (!a[i].Equals(b[i]))
                    return false;
            }

            return true;
        }

        /// <summary>
        /// Arrays the equal except.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="a">a.</param>
        /// <param name="b">The b.</param>
        /// <param name="ignoreIndex">Index of the ignore.</param>
        /// <returns><c>true</c> if XXXX, <c>false</c> otherwise.</returns>
        public static bool ArrayEqualExcept<T>(T[] a, T[] b, int ignoreIndex)
        {
            if (a.Length != b.Length)
                return false;

            for (int i = 0; i < a.Length; ++i)
            {
                if (i == ignoreIndex)
                    continue;

                if (!a[i].Equals(b[i]))
                    return false;
            }

            return true;
        }
    }
}
