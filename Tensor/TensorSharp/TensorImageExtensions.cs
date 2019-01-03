// ***********************************************************************
// Assembly         : TensorSharp
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="TensorImageExtensions.cs" company="TensorSharp">
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
    /// Class TensorImageExtensions.
    /// </summary>
    public static class TensorImageExtensions
    {
        /// <summary>
        /// Converts a Tensor to a Bitmap. Elements of the tensor are assumed to be normalized in the range [0, 1]
        /// The tensor must have one of the following structures:
        /// * 2D tensor - output is a 24bit BGR bitmap in greyscale
        /// * 3D tensor where first dimension has length 1 - output is 24bit BGR bitmap in greyscale
        /// * 3D tensor where first dimension has length 3 - output is 24bit BGR bitmap
        /// * 3D tensor where first dimension has length 4 - output is 32bit BGRA bitmap
        /// 2D tensors must be in HW (height x width) order;
        /// 3D tensors must be in CHW (channel x height x width) order.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <returns>Bitmap.</returns>
        /// <exception cref="InvalidOperationException">
        /// tensor must have 2 or 3 dimensions
        /// or
        /// 3D tensor's first dimension (color channels) must be of length 1, 3 or 4
        /// </exception>
        public static Bitmap ToBitmap(this Tensor tensor)
        {
            if (tensor.DimensionCount != 2 && tensor.DimensionCount != 3)
                throw new InvalidOperationException("tensor must have 2 or 3 dimensions");

            if (tensor.DimensionCount == 3 &&
                (tensor.Shape[0] != 1 && tensor.Shape[0] != 3 && tensor.Shape[0] != 4))
                throw new InvalidOperationException("3D tensor's first dimension (color channels) must be of length 1, 3 or 4");

            Tensor src;
            if (tensor.DimensionCount == 2)
                src = tensor.RepeatTensor(3, 1, 1);
            else if (tensor.DimensionCount == 3 && tensor.Shape[0] == 1)
                src = tensor.RepeatTensor(3, 1, 1);
            else
                src = tensor.CopyRef();

            var cpuAllocator = new Cpu.CpuAllocator();
            var bytesPerPixel = src.Shape[0];

            try
            {
                using (var cpuFloatTensor = new Tensor(cpuAllocator, DType.Float32, src.Shape))
                using (var permutedFloatTensor = cpuFloatTensor.Transpose(1, 2, 0))
                {
                    Ops.Copy(cpuFloatTensor, src);
                    Ops.Mul(cpuFloatTensor, cpuFloatTensor, 255);

                    var resultFormat = bytesPerPixel == 3 ? PixelFormat.Format24bppRgb : PixelFormat.Format32bppArgb;
                    var result = new Bitmap((int)src.Shape[2], (int)src.Shape[1], resultFormat);



                    var lockData = result.LockBits(
                        new Rectangle(0, 0, result.Width, result.Height),
                        ImageLockMode.WriteOnly,
                        result.PixelFormat);

                    var sizes = new long[] { result.Height, result.Width, bytesPerPixel };
                    var strides = new long[] { lockData.Stride, bytesPerPixel, 1 };
                    var resultTensor = new Tensor(cpuAllocator, DType.UInt8, sizes, strides);

                    // Re-order tensor and convert to bytes
                    Ops.Copy(resultTensor, permutedFloatTensor);

                    var byteLength = lockData.Stride * lockData.Height;
                    resultTensor.Storage.CopyFromStorage(lockData.Scan0, resultTensor.StorageOffset, byteLength);

                    result.UnlockBits(lockData);
                    return result;
                }
            }
            finally
            {
                src.Dispose();
            }
        }
    }
}
