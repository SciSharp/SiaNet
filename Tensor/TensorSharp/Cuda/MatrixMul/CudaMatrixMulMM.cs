// ***********************************************************************
// Assembly         : TensorSharp.CUDA91
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="CudaMatrixMulMM.cs" company="TensorSharp.CUDA91">
//     Copyright (c) . All rights reserved.
// </copyright>
// <summary></summary>
// ***********************************************************************
using ManagedCuda.BasicTypes;
using ManagedCuda.CudaBlas;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TensorSharp.Core;
using TensorSharp.Cpu;

namespace TensorSharp.CUDA.MatrixMul
{
    /// <summary>
    /// Class CudaMatrixMulMM.
    /// </summary>
    public static class CudaMatrixMulMM
    {
        // Computes  c := alpha * a * b  +  beta * c
        /// <summary>
        /// Gemms the specified context.
        /// </summary>
        /// <param name="context">The context.</param>
        /// <param name="alpha">The alpha.</param>
        /// <param name="a">a.</param>
        /// <param name="b">The b.</param>
        /// <param name="beta">The beta.</param>
        /// <param name="c">The c.</param>
        /// <exception cref="InvalidOperationException">Size mismatch</exception>
        public static void Gemm(TSCudaContext context, float alpha, Tensor a, Tensor b, float beta, Tensor c)
        {
            if (a.Shape[0] != c.Shape[0] || b.Shape[1] != c.Shape[1] || a.Shape[1] != b.Shape[0])
                throw new InvalidOperationException("Size mismatch");

            BlasOp aOp = default(BlasOp);
            BlasOp bOp = default(BlasOp);
            bool copyC = false;

            Tensor aClone = null;
            Tensor bClone = null;
            Tensor cClone = null;


            if (c.Strides[0] == 1 &&
                c.Strides[1] != 0)
            {
                // If c is contiguous in dimension 0 (column-major)
                aClone = a.CopyRef();
                bClone = b.CopyRef();
                cClone = c.CopyRef();
            }
            else if (c.Strides[1] == 1 &&
                c.Strides[0] != 0)
            {
                // If c is contiguous in dimension 1 (row-major)
                // using (a * b)' == b' * a'
                // we can pass row-major matrices to BLAS functions that expect column-major by swapping A and B,
                // and transposing all 3 matrices

                cClone = c.Transpose();
                aClone = b.Transpose(); // Note swap of a and b
                bClone = a.Transpose();
            }
            else
            {
                var cNew = new Tensor(c.Allocator, c.ElementType, c.Shape[1], c.Shape[0]);
                cClone = cNew.Transpose();
                Ops.Copy(cClone, c);
                cNew.Dispose();
                copyC = true;

                aClone = a.CopyRef();
                bClone = b.CopyRef();
            }

            try
            {
                if (aClone.Strides[0] == 1 &&
                    aClone.Strides[1] != 0)
                {
                    // If a is contiguous in dimension 0 (column-major)
                    aOp = BlasOp.NonTranspose;
                }
                else if (aClone.Strides[1] == 1 &&
                    aClone.Strides[0] != 0)
                {
                    aOp = BlasOp.Transpose;
                    var aNew = aClone.Transpose();
                    aClone.Dispose();
                    aClone = aNew;
                }
                else
                {
                    var aNew = new Tensor(aClone.Allocator, aClone.ElementType, aClone.Shape[1], aClone.Shape[0]);
                    var aClone2 = aNew.Transpose();
                    Ops.Copy(aClone2, aClone);
                    aClone.Dispose();
                    aClone = aClone2;
                    aNew.Dispose();
                }

                if (bClone.Strides[0] == 1 &&
                    bClone.Strides[1] != 0)
                {
                    // If a is contiguous in dimension 0 (column-major)
                    bOp = BlasOp.NonTranspose;
                }
                else if (bClone.Strides[1] == 1 &&
                    bClone.Strides[0] != 0)
                {
                    bOp = BlasOp.Transpose;
                    var bNew = bClone.Transpose();
                    bClone.Dispose();
                    bClone = bNew;
                }
                else
                {
                    var bNew = new Tensor(bClone.Allocator, bClone.ElementType, bClone.Shape[1], bClone.Shape[0]);
                    var bClone2 = bNew.Transpose();
                    Ops.Copy(bClone2, bClone);
                    bClone.Dispose();
                    bClone = bClone2;
                    bNew.Dispose();
                }

                GemmOp(context, aOp, bOp, alpha, aClone, bClone, beta, cClone);

                if (copyC)
                {
                    Ops.Copy(c, cClone);
                }
            }
            finally
            {
                aClone.Dispose();
                bClone.Dispose();
                cClone.Dispose();
            }
        }


        /// <summary>
        /// Gets the cuda blas op.
        /// </summary>
        /// <param name="op">The op.</param>
        /// <returns>Operation.</returns>
        /// <exception cref="InvalidOperationException">BlasOp not supported: " + op</exception>
        private static Operation GetCudaBlasOp(BlasOp op)
        {
            switch(op)
            {
                case BlasOp.NonTranspose: return Operation.NonTranspose;
                case BlasOp.Transpose: return Operation.Transpose;
                case BlasOp.ConjugateTranspose: return Operation.ConjugateTranspose;
                default:
                    throw new InvalidOperationException("BlasOp not supported: " + op);
            }
        }

        /// <summary>
        /// Gemms the op.
        /// </summary>
        /// <param name="context">The context.</param>
        /// <param name="transA">The trans a.</param>
        /// <param name="transB">The trans b.</param>
        /// <param name="alpha">The alpha.</param>
        /// <param name="a">a.</param>
        /// <param name="b">The b.</param>
        /// <param name="beta">The beta.</param>
        /// <param name="c">The c.</param>
        /// <exception cref="ArgumentException">
        /// a must be contiguous in the first dimension (column major / fortran order)
        /// or
        /// b must be contiguous in the first dimension (column major / fortran order)
        /// or
        /// c must be contiguous in the first dimension (column major / fortran order)
        /// </exception>
        /// <exception cref="CudaBlasException">
        /// </exception>
        /// <exception cref="NotSupportedException">CUDA GEMM with element type " + c.ElementType + " not supported</exception>
        private static void GemmOp(TSCudaContext context, BlasOp transA, BlasOp transB, float alpha, Tensor a, Tensor b, float beta, Tensor c)
        {
            if (a.Strides[0] != 1) throw new ArgumentException("a must be contiguous in the first dimension (column major / fortran order)");
            if (b.Strides[0] != 1) throw new ArgumentException("b must be contiguous in the first dimension (column major / fortran order)");
            if (c.Strides[0] != 1) throw new ArgumentException("c must be contiguous in the first dimension (column major / fortran order)");

            using (var blas = context.BlasForTensor(c))
            {
                bool nta = transA == BlasOp.NonTranspose;
                bool ntb = transB == BlasOp.NonTranspose;
                Operation transa = GetCudaBlasOp(transA);
                Operation transb = GetCudaBlasOp(transB);
                int m = (int)a.Shape[nta ? 0 : 1];
                int k = (int)b.Shape[ntb ? 0 : 1];
                int n = (int)b.Shape[ntb ? 1 : 0];
                int lda = (int)a.Strides[1];
                int ldb = (int)b.Strides[1];
                int ldc = (int)c.Strides[1];



                if (c.ElementType == DType.Float32)
                {
                    var aPtrSingle = CudaHelpers.GetBufferStart(a);
                    var bPtrSingle = CudaHelpers.GetBufferStart(b);
                    var cPtrSingle = CudaHelpers.GetBufferStart(c);

                    var _statusF32 = CudaBlasNativeMethods.cublasSgemm_v2(blas.Value.CublasHandle,
                        transa, transb, m, n, k, ref alpha, aPtrSingle, lda, bPtrSingle, ldb, ref beta, cPtrSingle, ldc);
                    if (_statusF32 != CublasStatus.Success) throw new CudaBlasException(_statusF32);
                }
                else if (c.ElementType == DType.Float64)
                {
                    var aPtrDouble = CudaHelpers.GetBufferStart(a);
                    var bPtrDouble = CudaHelpers.GetBufferStart(b);
                    var cPtrDouble = CudaHelpers.GetBufferStart(c);
                    var alphaDouble = (double)alpha;
                    var betaDouble = (double)beta;
                    var _statusF64 = CudaBlasNativeMethods.cublasDgemm_v2(blas.Value.CublasHandle,
                        transa, transb, m, n, k, ref alphaDouble, aPtrDouble, lda, bPtrDouble, ldb, ref betaDouble, cPtrDouble, ldc);
                    if (_statusF64 != CublasStatus.Success) throw new CudaBlasException(_statusF64);
                }
                else
                {
                    throw new NotSupportedException("CUDA GEMM with element type " + c.ElementType + " not supported");
                }
            }
        }


        /// <summary>
        /// Muls the m m.
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
        /// </exception>
        public static Tensor Mul_M_M(TSCudaContext context, Tensor result, Tensor lhs, Tensor rhs)
        {
            if (lhs.ElementType != rhs.ElementType || (result != null && result.ElementType != lhs.ElementType))
                throw new InvalidOperationException("All tensors must have the same element type");
            CudaHelpers.ThrowIfDifferentDevices(result, lhs, rhs);
            if (result != null && !(result.Storage is CudaStorage)) throw new ArgumentException("result must be a CUDA tensor", "result");
            if (!(lhs.Storage is CudaStorage)) throw new ArgumentException("lhs must be a CUDA tensor", "lhs");
            if (!(rhs.Storage is CudaStorage)) throw new ArgumentException("rhs must be a CUDA tensor", "rhs");

            
            var writeTarget = TensorResultBuilder.GetWriteTarget(result, lhs, false, lhs.Shape[0], rhs.Shape[1]);
            
            Gemm(context, 1, lhs, rhs, 0, writeTarget);
            
            return writeTarget;
        }
    }
}
