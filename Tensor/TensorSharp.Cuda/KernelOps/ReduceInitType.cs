// ***********************************************************************
// Assembly         : TensorSharp.CUDA91
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="ReduceInitType.cs" company="TensorSharp.CUDA91">
//     Copyright (c) . All rights reserved.
// </copyright>
// <summary></summary>
// ***********************************************************************
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace TensorSharp.CUDA.KernelOps
{
    /// <summary>
    /// Enum ReduceInitType
    /// </summary>
    public enum ReduceInitType
    {
        /// <summary>
        /// The given value
        /// </summary>
        GivenValue,
        /// <summary>
        /// The minimum value
        /// </summary>
        MinValue,
        /// <summary>
        /// The maximum value
        /// </summary>
        MaxValue,
    }

    /// <summary>
    /// Class ReduceInitConverter.
    /// </summary>
    public static class ReduceInitConverter
    {
        /// <summary>
        /// Gets the initialize value.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <param name="initType">Type of the initialize.</param>
        /// <param name="elementType">Type of the element.</param>
        /// <returns>System.Object.</returns>
        /// <exception cref="NotSupportedException"></exception>
        public static object GetInitValue(float value, ReduceInitType initType, DType elementType)
        {
            switch (initType)
            {
                case ReduceInitType.GivenValue: return FloatAsType(value, elementType);
                case ReduceInitType.MinValue: return GetMinValue(elementType);
                case ReduceInitType.MaxValue: return GetMaxValue(elementType);
                default:
                    throw new NotSupportedException();
            }
        }

        /// <summary>
        /// Floats as type.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <param name="elementType">Type of the element.</param>
        /// <returns>System.Object.</returns>
        /// <exception cref="NotSupportedException">casting value to type " + elementType + " not supported</exception>
        private static object FloatAsType(float value, DType elementType)
        {
            if (elementType == DType.Float32) return value;
            else if (elementType == DType.Float64) return (double)value;
            else if (elementType == DType.Int32) return (int)value;
            else if (elementType == DType.UInt8) return (byte)value;
            else
                throw new NotSupportedException("casting value to type " + elementType + " not supported");
        }

        /// <summary>
        /// Gets the minimum value.
        /// </summary>
        /// <param name="elementType">Type of the element.</param>
        /// <returns>System.Object.</returns>
        /// <exception cref="NotSupportedException">getting min value of type " + elementType + " not supported</exception>
        private static object GetMinValue(DType elementType)
        {
            if (elementType == DType.Float32) return float.MinValue;
            else if (elementType == DType.Float64) return double.MinValue;
            else if (elementType == DType.Int32) return int.MinValue;
            else if (elementType == DType.UInt8) return byte.MinValue;
            else
                throw new NotSupportedException("getting min value of type " + elementType + " not supported");
        }

        /// <summary>
        /// Gets the maximum value.
        /// </summary>
        /// <param name="elementType">Type of the element.</param>
        /// <returns>System.Object.</returns>
        /// <exception cref="NotSupportedException">getting max value of type " + elementType + " not supported</exception>
        private static object GetMaxValue(DType elementType)
        {
            if (elementType == DType.Float32) return float.MaxValue;
            else if (elementType == DType.Float64) return double.MaxValue;
            else if (elementType == DType.Int32) return int.MaxValue;
            else if (elementType == DType.UInt8) return byte.MaxValue;
            else
                throw new NotSupportedException("getting max value of type " + elementType + " not supported");
        }
    }
}
