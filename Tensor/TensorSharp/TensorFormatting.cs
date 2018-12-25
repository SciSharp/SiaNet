// ***********************************************************************
// Assembly         : TensorSharp
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="TensorFormatting.cs" company="TensorSharp">
//     Copyright (c) . All rights reserved.
// </copyright>
// <summary></summary>
// ***********************************************************************
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace TensorSharp
{
    /// <summary>
    /// Class TensorFormatting.
    /// </summary>
    internal static class TensorFormatting
    {
        /// <summary>
        /// Repeats the character.
        /// </summary>
        /// <param name="c">The c.</param>
        /// <param name="count">The count.</param>
        /// <returns>System.String.</returns>
        private static string RepeatChar(char c, int count)
        {
            var builder = new StringBuilder();
            for (int i = 0; i < count; ++i)
            {
                builder.Append(c);
            }
            return builder.ToString();
        }

        /// <summary>
        /// Gets the int format.
        /// </summary>
        /// <param name="length">The length.</param>
        /// <returns>System.String.</returns>
        private static string GetIntFormat(int length)
        {
            var padding = RepeatChar('#', length - 1);
            return string.Format(" {0}0;-{0}0", padding);
        }

        /// <summary>
        /// Gets the float format.
        /// </summary>
        /// <param name="length">The length.</param>
        /// <returns>System.String.</returns>
        private static string GetFloatFormat(int length)
        {
            var padding = RepeatChar('#', length - 1);
            return string.Format(" {0}0.0000;-{0}0.0000", padding);
        }

        /// <summary>
        /// Gets the scientific format.
        /// </summary>
        /// <param name="length">The length.</param>
        /// <returns>System.String.</returns>
        private static string GetScientificFormat(int length)
        {
            var padCount = length - 6;
            var padding = RepeatChar('0', padCount);
            return string.Format(" {0}.0000e+00;-0.{0}e+00", padding);
        }




        /// <summary>
        /// Determines whether [is int only] [the specified storage].
        /// </summary>
        /// <param name="storage">The storage.</param>
        /// <param name="tensor">The tensor.</param>
        /// <returns><c>true</c> if [is int only] [the specified storage]; otherwise, <c>false</c>.</returns>
        private static bool IsIntOnly(Storage storage, Tensor tensor)
        {
            // HACK this is a hacky way of iterating over the elements of the tensor.
            // if the tensor has holes, this will incorrectly include those elements
            // in the iteration.
            var minOffset = tensor.StorageOffset;
            var maxOffset = minOffset + TensorDimensionHelpers.GetStorageSize(tensor.Sizes, tensor.Strides) - 1;
            for (long i = minOffset; i <= maxOffset; ++i)
            {
                var value = Convert.ToDouble((object)storage.GetElementAsFloat(i));
                if (value != Math.Ceiling(value))
                {
                    return false;
                }
            }

            return true;
        }

        /// <summary>
        /// Abses the minimum maximum.
        /// </summary>
        /// <param name="storage">The storage.</param>
        /// <param name="tensor">The tensor.</param>
        /// <returns>Tuple&lt;System.Double, System.Double&gt;.</returns>
        private static Tuple<double, double> AbsMinMax(Storage storage, Tensor tensor)
        {
            if (storage.ElementCount == 0)
                return Tuple.Create(0.0, 0.0);

            double min = storage.GetElementAsFloat(0);
            double max = storage.GetElementAsFloat(0);

            // HACK this is a hacky way of iterating over the elements of the tensor.
            // if the tensor has holes, this will incorrectly include those elements
            // in the iteration.
            var minOffset = tensor.StorageOffset;
            var maxOffset = minOffset + TensorDimensionHelpers.GetStorageSize(tensor.Sizes, tensor.Strides) - 1;

            for (long i = minOffset; i <= maxOffset; ++i)
            {
                var item = storage.GetElementAsFloat(i);
                if (item < min)
                    min = item;
                if (item > max)
                    max = item;
            }

            return Tuple.Create(Math.Abs(min), Math.Abs(max));
        }

        /// <summary>
        /// Enum FormatType
        /// </summary>
        private enum FormatType
        {
            /// <summary>
            /// The int
            /// </summary>
            Int,
            /// <summary>
            /// The scientific
            /// </summary>
            Scientific,
            /// <summary>
            /// The float
            /// </summary>
            Float,
        }
        /// <summary>
        /// Gets the size of the format.
        /// </summary>
        /// <param name="minMax">The minimum maximum.</param>
        /// <param name="intMode">if set to <c>true</c> [int mode].</param>
        /// <returns>Tuple&lt;FormatType, System.Double, System.Int32&gt;.</returns>
        private static Tuple<FormatType, double, int> GetFormatSize(Tuple<double, double> minMax, bool intMode)
        {
            var expMin = minMax.Item1 != 0 ?
                    (int)Math.Floor(Math.Log10(minMax.Item1)) + 1 :
                    1;
            var expMax = minMax.Item2 != 0 ?
                    (int)Math.Floor(Math.Log10(minMax.Item2)) + 1 :
                    1;

            if (intMode)
            {
                if (expMax > 9)
                    return Tuple.Create(FormatType.Scientific, 1.0, 11);
                else
                    return Tuple.Create(FormatType.Int, 1.0, expMax + 1);
            }
            else
            {
                if (expMax - expMin > 4)
                {
                    var sz = Math.Abs(expMax) > 99 || Math.Abs(expMin) > 99 ?
                        12 : 11;
                    return Tuple.Create(FormatType.Scientific, 1.0, sz);
                }
                else
                {
                    if (expMax > 5 || expMax < 0)
                    {
                        return Tuple.Create(FormatType.Float,
                            Math.Pow(10, expMax - 1), 7);
                    }
                    else
                    {
                        return Tuple.Create(FormatType.Float, 1.0,
                            expMax == 0 ? 7 : expMax + 6);
                    }
                }
            }
        }

        /// <summary>
        /// Builds the format string.
        /// </summary>
        /// <param name="type">The type.</param>
        /// <param name="size">The size.</param>
        /// <returns>System.String.</returns>
        /// <exception cref="InvalidOperationException">Invalid format type " + type</exception>
        private static string BuildFormatString(FormatType type, int size)
        {
            switch (type)
            {
                case FormatType.Int: return GetIntFormat(size);
                case FormatType.Float: return GetFloatFormat(size);
                case FormatType.Scientific: return GetScientificFormat(size);
                default: throw new InvalidOperationException("Invalid format type " + type);
            }
        }

        /// <summary>
        /// Gets the storage format.
        /// </summary>
        /// <param name="storage">The storage.</param>
        /// <param name="tensor">The tensor.</param>
        /// <returns>Tuple&lt;System.String, System.Double, System.Int32&gt;.</returns>
        private static Tuple<string, double, int> GetStorageFormat(Storage storage, Tensor tensor)
        {
            if (storage.ElementCount == 0)
                return Tuple.Create("", 1.0, 0);

            bool intMode = IsIntOnly(storage, tensor);
            var minMax = AbsMinMax(storage, tensor);

            var formatSize = GetFormatSize(minMax, intMode);
            var formatString = BuildFormatString(formatSize.Item1, formatSize.Item3);

            return Tuple.Create("{0:" + formatString + "}", formatSize.Item2, formatSize.Item3);
        }

        /// <summary>
        /// Formats the size of the tensor type and.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <returns>System.String.</returns>
        public static string FormatTensorTypeAndSize(Tensor tensor)
        {
            var result = new StringBuilder();
            result
                .Append("[")
                .Append(tensor.ElementType)
                .Append(" tensor");

            if (tensor.DimensionCount == 0)
            {
                result.Append(" with no dimension");
            }
            else
            {
                result
                .Append(" of size ")
                .Append(tensor.Sizes[0]);

                for (int i = 1; i < tensor.DimensionCount; ++i)
                {
                    result.Append("x").Append(tensor.Sizes[i]);
                }
            }

            result.Append(" on ").Append(tensor.Storage.LocationDescription());
            result.Append("]");
            return result.ToString();
        }

        /// <summary>
        /// Formats the vector.
        /// </summary>
        /// <param name="builder">The builder.</param>
        /// <param name="tensor">The tensor.</param>
        private static void FormatVector(StringBuilder builder, Tensor tensor)
        {
            var storageFormat = GetStorageFormat(tensor.Storage, tensor);
            var format = storageFormat.Item1;
            var scale = storageFormat.Item2;

            if (scale != 1)
            {
                builder.AppendLine(scale + " *");
                for (int i = 0; i < tensor.Sizes[0]; ++i)
                {
                    var value = Convert.ToDouble((object)tensor.GetElementAsFloat(i)) / scale;
                    builder.AppendLine(string.Format(format, value));
                }
            }
            else
            {
                for (int i = 0; i < tensor.Sizes[0]; ++i)
                {
                    var value = Convert.ToDouble((object)tensor.GetElementAsFloat(i));
                    builder.AppendLine(string.Format(format, value));
                }
            }
        }

        /// <summary>
        /// Formats the matrix.
        /// </summary>
        /// <param name="builder">The builder.</param>
        /// <param name="tensor">The tensor.</param>
        /// <param name="indent">The indent.</param>
        private static void FormatMatrix(StringBuilder builder, Tensor tensor, string indent)
        {
            var storageFormat = GetStorageFormat(tensor.Storage, tensor);
            var format = storageFormat.Item1;
            var scale = storageFormat.Item2;
            var sz = storageFormat.Item3;

            builder.Append(indent);

            var nColumnPerLine = (int)Math.Floor((80 - indent.Length) / (double)(sz + 1));
            long firstColumn = 0;
            long lastColumn = -1;
            while (firstColumn < tensor.Sizes[1])
            {
                if (firstColumn + nColumnPerLine - 2 < tensor.Sizes[1])
                {
                    lastColumn = firstColumn + nColumnPerLine - 2;
                }
                else
                {
                    lastColumn = tensor.Sizes[1] - 1;
                }

                if (nColumnPerLine < tensor.Sizes[1])
                {
                    if (firstColumn != 1)
                    {
                        builder.AppendLine();
                    }
                    builder.Append("Columns ").Append(firstColumn).Append(" to ").Append(lastColumn).AppendLine();
                }

                if (scale != 1)
                {
                    builder.Append(scale).AppendLine(" *");
                }

                for (long l = 0; l < tensor.Sizes[0]; ++l)
                {
                    using (var row = tensor.Select(0, l))
                    {
                        for (long c = firstColumn; c <= lastColumn; ++c)
                        {
                            var value = Convert.ToDouble((object)row.GetElementAsFloat(c)) / scale;
                            builder.Append(string.Format(format, value));
                            if (c == lastColumn)
                            {
                                builder.AppendLine();
                                if (l != tensor.Sizes[0])
                                {
                                    builder.Append(scale != 1 ? indent + " " : indent);
                                }
                            }
                            else
                            {
                                builder.Append(' ');
                            }
                        }
                    }
                }
                firstColumn = lastColumn + 1;
            }
        }

        /// <summary>
        /// Formats the tensor.
        /// </summary>
        /// <param name="builder">The builder.</param>
        /// <param name="tensor">The tensor.</param>
        private static void FormatTensor(StringBuilder builder, Tensor tensor)
        {
            var storageFormat = GetStorageFormat(tensor.Storage, tensor);
            var format = storageFormat.Item1;
            var scale = storageFormat.Item2;
            var sz = storageFormat.Item3;

            var startingLength = builder.Length;
            var counter = Enumerable.Repeat((long)0, tensor.DimensionCount - 2).ToArray();
            bool finished = false;
            counter[0] = -1;
            while (true)
            {
                for (int i = 0; i < tensor.DimensionCount - 2; ++i)
                {
                    counter[i]++;
                    if (counter[i] >= tensor.Sizes[i])
                    {
                        if (i == tensor.DimensionCount - 3)
                        {
                            finished = true;
                            break;
                        }
                        counter[i] = 1;
                    }
                    else
                    {
                        break;
                    }
                }

                if (finished)
                    break;

                if (builder.Length - startingLength > 1)
                {
                    builder.AppendLine();
                }

                builder.Append('(');
                var tensorCopy = tensor.CopyRef();
                for (int i = 0; i < tensor.DimensionCount - 2; ++i)
                {
                    var newCopy = tensorCopy.Select(0, counter[i]);
                    tensorCopy.Dispose();
                    tensorCopy = newCopy;
                    builder.Append(counter[i]).Append(',');
                }

                builder.AppendLine(".,.) = ");
                FormatMatrix(builder, tensorCopy, " ");

                tensorCopy.Dispose();
            }
        }

        /// <summary>
        /// Formats the specified tensor.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <returns>System.String.</returns>
        public static string Format(Tensor tensor)
        {
            var result = new StringBuilder();
            if (tensor.DimensionCount == 0)
            {
            }
            else if (tensor.DimensionCount == 1)
            {
                FormatVector(result, tensor);
            }
            else if (tensor.DimensionCount == 2)
            {
                FormatMatrix(result, tensor, "");
            }
            else
            {
                FormatTensor(result, tensor);
            }

            result.AppendLine(FormatTensorTypeAndSize(tensor));
            return result.ToString();
        }
    }
}
