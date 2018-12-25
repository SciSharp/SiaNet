// ***********************************************************************
// Assembly         : TensorSharp
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="DType.cs" company="TensorSharp">
//     Copyright (c) . All rights reserved.
// </copyright>
// <summary></summary>
// ***********************************************************************
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace TensorSharp
{
    /// <summary>
    /// Enum DType
    /// </summary>
    public enum DType
    {
        /// <summary>
        /// The float32
        /// </summary>
        Float32 = 0,
        /// <summary>
        /// The float16
        /// </summary>
        Float16 = 1,
        /// <summary>
        /// The float64
        /// </summary>
        Float64 = 2,
        /// <summary>
        /// The int32
        /// </summary>
        Int32 = 3,
        /// <summary>
        /// The u int8
        /// </summary>
        UInt8 = 4,
    }

    /// <summary>
    /// Struct Half
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct Half
    {
        /// <summary>
        /// The value
        /// </summary>
        public ushort value;
    }


    /// <summary>
    /// Class DTypeExtensions.
    /// </summary>
    public static class DTypeExtensions
    {
        /// <summary>
        /// Sizes the specified value.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <returns>System.Int32.</returns>
        /// <exception cref="NotSupportedException">Element type " + value + " not supported.</exception>
        public static int Size(this DType value)
        {
            switch (value)
            {
                case DType.Float16: return 2;
                case DType.Float32: return 4;
                case DType.Float64: return 8;
                case DType.Int32: return 4;
                case DType.UInt8: return 1;
                default:
                    throw new NotSupportedException("Element type " + value + " not supported.");
            }
        }

        /// <summary>
        /// Converts to clrtype.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <returns>Type.</returns>
        /// <exception cref="NotSupportedException">Element type " + value + " not supported.</exception>
        public static Type ToCLRType(this DType value)
        {
            switch (value)
            {
                case DType.Float16: return typeof(Half);
                case DType.Float32: return typeof(float);
                case DType.Float64: return typeof(double);
                case DType.Int32: return typeof(int);
                case DType.UInt8: return typeof(byte);
                default:
                    throw new NotSupportedException("Element type " + value + " not supported.");
            }
        }
    }

    /// <summary>
    /// Class DTypeBuilder.
    /// </summary>
    public static class DTypeBuilder
    {
        /// <summary>
        /// Froms the type of the color.
        /// </summary>
        /// <param name="type">The type.</param>
        /// <returns>DType.</returns>
        /// <exception cref="NotSupportedException">No corresponding DType value for CLR type " + type</exception>
        public static DType FromCLRType(Type type)
        {
            if (type == typeof(Half)) return DType.Float16;
            else if (type == typeof(float)) return DType.Float32;
            else if (type == typeof(double)) return DType.Float64;
            else if (type == typeof(int)) return DType.Int32;
            else if (type == typeof(byte)) return DType.UInt8;
            else
                throw new NotSupportedException("No corresponding DType value for CLR type " + type);
        }
    }
}
