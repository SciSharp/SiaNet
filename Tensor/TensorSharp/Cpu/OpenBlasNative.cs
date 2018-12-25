// ***********************************************************************
// Assembly         : TensorSharp
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-26-2018
// ***********************************************************************
// <copyright file="OpenBlasNative.cs" company="TensorSharp">
//     Copyright (c) . All rights reserved.
// </copyright>
// <summary></summary>
// ***********************************************************************
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace TensorSharp.Cpu
{
    // When used with 64bit openblas, this interface requires that it is compiled with 32-bit ints
    /// <summary>
    /// Class OpenBlasNative.
    /// </summary>
    public static class OpenBlasNative
    {
        /// <summary>
        /// The DLL
        /// </summary>
        private const string dll = "lib/libopenblas.dll";
        /// <summary>
        /// The cc
        /// </summary>
        private const CallingConvention cc = CallingConvention.Cdecl;

        /// <summary>
        /// Sgemms the specified transa.
        /// </summary>
        /// <param name="transa">The transa.</param>
        /// <param name="transb">The transb.</param>
        /// <param name="m">The m.</param>
        /// <param name="n">The n.</param>
        /// <param name="k">The k.</param>
        /// <param name="alpha">The alpha.</param>
        /// <param name="a">a.</param>
        /// <param name="lda">The lda.</param>
        /// <param name="b">The b.</param>
        /// <param name="ldb">The LDB.</param>
        /// <param name="beta">The beta.</param>
        /// <param name="c">The c.</param>
        /// <param name="ldc">The LDC.</param>
        [DllImport(dll, CallingConvention = cc)]
        public static unsafe extern void sgemm_(byte* transa, byte* transb, int *m, int *n, int *k,
            float *alpha, float *a, int *lda, float *b, int *ldb, float *beta, float *c, int *ldc);

        /// <summary>
        /// Dgemms the specified transa.
        /// </summary>
        /// <param name="transa">The transa.</param>
        /// <param name="transb">The transb.</param>
        /// <param name="m">The m.</param>
        /// <param name="n">The n.</param>
        /// <param name="k">The k.</param>
        /// <param name="alpha">The alpha.</param>
        /// <param name="a">a.</param>
        /// <param name="lda">The lda.</param>
        /// <param name="b">The b.</param>
        /// <param name="ldb">The LDB.</param>
        /// <param name="beta">The beta.</param>
        /// <param name="c">The c.</param>
        /// <param name="ldc">The LDC.</param>
        [DllImport(dll, CallingConvention = cc)]
        public static unsafe extern void dgemm_(byte* transa, byte* transb, int* m, int* n, int* k,
            double* alpha, double* a, int* lda, double* b, int* ldb, double* beta, double* c, int* ldc);


        /// <summary>
        /// Sgemvs the specified trans.
        /// </summary>
        /// <param name="trans">The trans.</param>
        /// <param name="m">The m.</param>
        /// <param name="n">The n.</param>
        /// <param name="alpha">The alpha.</param>
        /// <param name="a">a.</param>
        /// <param name="lda">The lda.</param>
        /// <param name="x">The x.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="beta">The beta.</param>
        /// <param name="y">The y.</param>
        /// <param name="incy">The incy.</param>
        [DllImport(dll, CallingConvention = cc)]
        public static unsafe extern void sgemv_(byte* trans, int* m, int* n,
            float* alpha, float* a, int* lda, float* x, int* incx, float* beta, float* y, int* incy);

        /// <summary>
        /// Dgemvs the specified trans.
        /// </summary>
        /// <param name="trans">The trans.</param>
        /// <param name="m">The m.</param>
        /// <param name="n">The n.</param>
        /// <param name="alpha">The alpha.</param>
        /// <param name="a">a.</param>
        /// <param name="lda">The lda.</param>
        /// <param name="x">The x.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="beta">The beta.</param>
        /// <param name="y">The y.</param>
        /// <param name="incy">The incy.</param>
        [DllImport(dll, CallingConvention = cc)]
        public static unsafe extern void dgemv_(byte* trans, int* m, int* n,
            double* alpha, double* a, int* lda, double* x, int* incx, double* beta, double* y, int* incy);


        /// <summary>
        /// Sdots the specified n.
        /// </summary>
        /// <param name="n">The n.</param>
        /// <param name="x">The x.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="y">The y.</param>
        /// <param name="incy">The incy.</param>
        /// <returns>System.Single.</returns>
        [DllImport(dll, CallingConvention = cc)]
        public static unsafe extern float sdot_(int* n, float* x, int* incx, float* y, int* incy);

        /// <summary>
        /// Ddots the specified n.
        /// </summary>
        /// <param name="n">The n.</param>
        /// <param name="x">The x.</param>
        /// <param name="incx">The incx.</param>
        /// <param name="y">The y.</param>
        /// <param name="incy">The incy.</param>
        /// <returns>System.Double.</returns>
        [DllImport(dll, CallingConvention = cc)]
        public static unsafe extern double ddot_(int* n, double* x, int* incx, double* y, int* incy);
    }
}
