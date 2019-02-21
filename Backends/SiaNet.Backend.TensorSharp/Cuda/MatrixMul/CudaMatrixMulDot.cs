// ***********************************************************************
// Assembly         : SiaNet.Backend.TensorSharp.CUDA91
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="CudaMatrixMulDot.cs" company="SiaNet.Backend.TensorSharp.CUDA91">
//     Copyright (c) . All rights reserved.
// </copyright>
// <summary></summary>
// ***********************************************************************
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.CudaBlas;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using SiaNet.Backend.TensorSharp.Core;

namespace SiaNet.Backend.TensorSharp.CUDA.MatrixMul
{
    /// <summary>
    /// Class CudaMatrixMulDot.
    /// </summary>
    public static class CudaMatrixMulDot
    {
        /// <summary>
        /// Dots the specified context.
        /// </summary>
        /// <param name="context">The context.</param>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        /// <exception cref="InvalidOperationException">All tensors must have the same element type</exception>
        /// <exception cref="ArgumentException">
        /// result must be a CUDA tensor - result
        /// or
        /// lhs must be a CUDA tensor - lhs
        /// or
        /// rhs must be a CUDA tensor - rhs
        /// or
        /// lhs must have 1 dimension (ie. be a vector) - lhs
        /// or
        /// rhs must have 1 dimension (ie. be a vector) - rhs
        /// </exception>
        /// <exception cref="NotSupportedException">CUDA vector dot product with element type " + result.ElementType + " not supported</exception>
        public static NDArray Dot(TSCudaContext context, NDArray result, NDArray lhs, NDArray rhs)
        {
            if (lhs.ElementType != rhs.ElementType || (result != null && result.ElementType != lhs.ElementType))
                throw new InvalidOperationException("All tensors must have the same element type");

            CudaHelpers.ThrowIfDifferentDevices(result, lhs, rhs);

            if (result != null && !(result.Storage is CudaStorage)) throw new ArgumentException("result must be a CUDA tensor", "result");
            if (!(lhs.Storage is CudaStorage)) throw new ArgumentException("lhs must be a CUDA tensor", "lhs");
            if (!(rhs.Storage is CudaStorage)) throw new ArgumentException("rhs must be a CUDA tensor", "rhs");

            if (lhs.DimensionCount != 1) throw new ArgumentException("lhs must have 1 dimension (ie. be a vector)", "lhs");
            if (rhs.DimensionCount != 1) throw new ArgumentException("rhs must have 1 dimension (ie. be a vector)", "rhs");


            var writeTarget = TensorResultBuilder.GetWriteTarget(result, lhs, false, 1);

            if (writeTarget.ElementType == DType.Float32) Run_Dot_float(context, writeTarget, lhs, rhs);
            else if (writeTarget.ElementType == DType.Float64) Run_Dot_double(context, writeTarget, lhs, rhs);
            else
                throw new NotSupportedException("CUDA vector dot product with element type " + result.ElementType + " not supported");

            return writeTarget;
        }

        /// <summary>
        /// Runs the dot float.
        /// </summary>
        /// <param name="context">The context.</param>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <exception cref="CudaBlasException"></exception>
        private static void Run_Dot_float(TSCudaContext context, NDArray result, NDArray lhs, NDArray rhs)
        {
            using (var blas = context.BlasForTensor(lhs))
            {
                //var resultPtr = CudaNativeHelpers.GetBufferStart(result);
                var lhsPtr = CudaHelpers.GetBufferStart(lhs);
                var rhsPtr = CudaHelpers.GetBufferStart(rhs);

                int n = (int)lhs.Shape[0];
                int incx = (int)lhs.Strides[0];
                int incy = (int)rhs.Strides[0];

                float resultVal = 0;
                var _status = CudaBlasNativeMethods.cublasSdot_v2(blas.Value.CublasHandle, n, lhsPtr, incx, rhsPtr, incy, ref resultVal);
                if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
                result.Storage.SetElementAsFloat(result.StorageOffset, resultVal);
            }
        }

        /// <summary>
        /// Runs the dot double.
        /// </summary>
        /// <param name="context">The context.</param>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <exception cref="CudaBlasException"></exception>
        private static void Run_Dot_double(TSCudaContext context, NDArray result, NDArray lhs, NDArray rhs)
        {
            using (var blas = context.BlasForTensor(lhs))
            {
                //var resultPtr = CudaNativeHelpers.GetBufferStart(result);
                var lhsPtr = CudaHelpers.GetBufferStart(lhs);
                var rhsPtr = CudaHelpers.GetBufferStart(rhs);

                int n = (int)lhs.Shape[0];
                int incx = (int)lhs.Strides[0];
                int incy = (int)rhs.Strides[0];

                // TODO add SetElementAsDouble to prevent need to round to float here
                double resultVal = 0;
                var _status = CudaBlasNativeMethods.cublasDdot_v2(blas.Value.CublasHandle, n, lhsPtr, incx, rhsPtr, incy, ref resultVal);
                if (_status != CublasStatus.Success) throw new CudaBlasException(_status);
                result.Storage.SetElementAsFloat(result.StorageOffset, (float)resultVal);
            }
        }
    }
}
