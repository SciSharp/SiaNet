// ***********************************************************************
// Assembly         : TensorSharp
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="BitmapExtensions.cs" company="TensorSharp">
//     Copyright (c) . All rights reserved.
// </copyright>
// <summary></summary>
// ***********************************************************************
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Text;

namespace TensorSharp
{
    /// <summary>
    /// Class BitmapExtensions.
    /// </summary>
    public static class BitmapExtensions
    {
        /// <summary>
        /// Returns a Tensor constructed from the data in the Bitmap. The Tensor's dimensions are
        /// ordered CHW (channel x height x width). The color channel dimension is in the same order
        /// as the original Bitmap data. For 24bit bitmaps, this will be BGR. For 32bit bitmaps this
        /// will be BGRA.
        /// </summary>
        /// <param name="bitmap">The bitmap.</param>
        /// <param name="allocator">The allocator.</param>
        /// <returns>Tensor.</returns>
        /// <exception cref="InvalidOperationException">Bitmap must be 24bit or 32bit</exception>
        public static Tensor ToTensor(this Bitmap bitmap, IAllocator allocator)
        {
            var cpuAllocator = new Cpu.CpuAllocator();

            int bytesPerPixel = 0;

            if (bitmap.PixelFormat == PixelFormat.Format24bppRgb)
                bytesPerPixel = 3;
            else if (bitmap.PixelFormat == PixelFormat.Format32bppArgb ||
                bitmap.PixelFormat == PixelFormat.Format32bppPArgb ||
                bitmap.PixelFormat == PixelFormat.Format32bppRgb)
                bytesPerPixel = 4;
            else
                throw new InvalidOperationException("Bitmap must be 24bit or 32bit");

            var lockData = bitmap.LockBits(
                new Rectangle(0, 0, bitmap.Width, bitmap.Height),
                ImageLockMode.ReadOnly,
                bitmap.PixelFormat);

            try
            {

                var sizes = new long[] { bitmap.Height, bitmap.Width, bytesPerPixel };
                var strides = new long[] { lockData.Stride, bytesPerPixel, 1 };
                using (var cpuByteTensor = new Tensor(cpuAllocator, DType.UInt8, sizes, strides))
                {
                    cpuByteTensor.Storage.CopyToStorage(cpuByteTensor.StorageOffset, lockData.Scan0, cpuByteTensor.Storage.ByteLength);
                    using (var permutedTensor = cpuByteTensor.Transpose(2, 0, 1))
                    {
                        using (var cpuFloatTensor = new Tensor(cpuAllocator, DType.Float32, permutedTensor.Shape))
                        {
                            Ops.Copy(cpuFloatTensor, permutedTensor);

                            // TODO this could be made more efficient by skipping a the following copy if allocator is a CpuAllocator,
                            // but make sure that in that case the result tensor is not disposed before returning.

                            var result = new Tensor(allocator, DType.Float32, permutedTensor.Shape);
                            Ops.Copy(result, cpuFloatTensor);
                            Ops.Div(result, result, 255);
                            return result;
                        }
                    }
                }
            }
            finally
            {
                bitmap.UnlockBits(lockData);
            }

        }
    }
}
