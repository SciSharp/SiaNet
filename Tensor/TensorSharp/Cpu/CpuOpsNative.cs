// ***********************************************************************
// Assembly         : TensorSharp
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-26-2018
// ***********************************************************************
// <copyright file="CpuOpsNative.cs" company="TensorSharp">
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
    public enum CpuDType : int
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
    /// Struct TensorRef64
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct TensorRef64
    {
        /// <summary>
        /// The buffer
        /// </summary>
        public IntPtr buffer;
        /// <summary>
        /// The sizes
        /// </summary>
        public IntPtr sizes;
        /// <summary>
        /// The strides
        /// </summary>
        public IntPtr strides;
        /// <summary>
        /// The dim count
        /// </summary>
        public int dimCount;
        /// <summary>
        /// The element type
        /// </summary>
        public CpuDType elementType;
    }


    /// <summary>
    /// Class CpuOpsNative.
    /// </summary>
    public static class CpuOpsNative
    {
        /// <summary>
        /// The DLL
        /// </summary>
        private const string dll = "lib/CpuOps.dll";
        /// <summary>
        /// The cc
        /// </summary>
        private const CallingConvention cc = CallingConvention.Cdecl;

        /// <summary>
        /// Tses the get last error.
        /// </summary>
        /// <returns>IntPtr.</returns>
        [DllImport(dll, CallingConvention = cc)]
        public static extern IntPtr TS_GetLastError();

        /// <summary>
        /// Tses the fill.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="value">The value.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Fill(IntPtr result, float value);
        /// <summary>
        /// Tses the copy.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Copy(IntPtr result, IntPtr src);

        /// <summary>
        /// Tses the abs.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Abs(IntPtr result, IntPtr src);
        /// <summary>
        /// Tses the neg.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Neg(IntPtr result, IntPtr src);
        /// <summary>
        /// Tses the sign.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Sign(IntPtr result, IntPtr src);


        /// <summary>
        /// Tses the SQRT.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Sqrt(IntPtr result, IntPtr src);
        /// <summary>
        /// Tses the exp.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Exp(IntPtr result, IntPtr src);
        /// <summary>
        /// Tses the log.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Log(IntPtr result, IntPtr src);
        /// <summary>
        /// Tses the log1p.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Log1p(IntPtr result, IntPtr src);
        /// <summary>
        /// Tses the floor.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Floor(IntPtr result, IntPtr src);
        /// <summary>
        /// Tses the ceil.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Ceil(IntPtr result, IntPtr src);
        /// <summary>
        /// Tses the round.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Round(IntPtr result, IntPtr src);
        /// <summary>
        /// Tses the trunc.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Trunc(IntPtr result, IntPtr src);
        /// <summary>
        /// Tses the frac.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Frac(IntPtr result, IntPtr src);

        /// <summary>
        /// Tses the sin.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Sin(IntPtr result, IntPtr src);
        /// <summary>
        /// Tses the cos.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Cos(IntPtr result, IntPtr src);
        /// <summary>
        /// Tses the tan.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Tan(IntPtr result, IntPtr src);
        /// <summary>
        /// Tses the asin.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Asin(IntPtr result, IntPtr src);
        /// <summary>
        /// Tses the acos.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Acos(IntPtr result, IntPtr src);
        /// <summary>
        /// Tses the atan.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Atan(IntPtr result, IntPtr src);
        /// <summary>
        /// Tses the sinh.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Sinh(IntPtr result, IntPtr src);
        /// <summary>
        /// Tses the cosh.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Cosh(IntPtr result, IntPtr src);
        /// <summary>
        /// Tses the tanh.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Tanh(IntPtr result, IntPtr src);

        /// <summary>
        /// Tses the sigmoid.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Sigmoid(IntPtr result, IntPtr src);

        /// <summary>
        /// Tses the atan2.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="srcY">The source y.</param>
        /// <param name="srcX">The source x.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Atan2(IntPtr result, IntPtr srcY, IntPtr srcX);
        /// <summary>
        /// Tses the pow.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="value">The value.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Pow(IntPtr result, IntPtr src, float value);
        /// <summary>
        /// Tses the tpow.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="value">The value.</param>
        /// <param name="src">The source.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Tpow(IntPtr result, float value, IntPtr src);
        /// <summary>
        /// Tses the lerp.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="srcA">The source a.</param>
        /// <param name="srcB">The source b.</param>
        /// <param name="weight">The weight.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Lerp(IntPtr result, IntPtr srcA, IntPtr srcB, float weight);
        /// <summary>
        /// Tses the clamp.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="min">The minimum.</param>
        /// <param name="max">The maximum.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Clamp(IntPtr result, IntPtr src, float min, float max);

        /// <summary>
        /// Tses the add.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Add(IntPtr result, IntPtr lhs, float rhs);
        /// <summary>
        /// Tses the sub.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Sub(IntPtr result, IntPtr lhs, float rhs);
        /// <summary>
        /// Tses the rsub.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Rsub(IntPtr result, IntPtr lhs, float rhs);
        /// <summary>
        /// Tses the mul.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Mul(IntPtr result, IntPtr lhs, float rhs);
        /// <summary>
        /// Tses the div.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Div(IntPtr result, IntPtr lhs, float rhs);
        /// <summary>
        /// Tses the rdiv.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Rdiv(IntPtr result, IntPtr lhs, float rhs);
        /// <summary>
        /// Tses the mod.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Mod(IntPtr result, IntPtr lhs, float rhs);

        /// <summary>
        /// Tses the gt value.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_gtValue(IntPtr result, IntPtr lhs, float rhs);
        /// <summary>
        /// Tses the lt value.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_ltValue(IntPtr result, IntPtr lhs, float rhs);
        /// <summary>
        /// Tses the ge value.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_geValue(IntPtr result, IntPtr lhs, float rhs);
        /// <summary>
        /// Tses the le value.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_leValue(IntPtr result, IntPtr lhs, float rhs);
        /// <summary>
        /// Tses the eq value.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_eqValue(IntPtr result, IntPtr lhs, float rhs);
        /// <summary>
        /// Tses the ne value.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_neValue(IntPtr result, IntPtr lhs, float rhs);


        /// <summary>
        /// Tses the c add.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_CAdd(IntPtr result, IntPtr lhs, IntPtr rhs);
        /// <summary>
        /// Tses the c sub.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_CSub(IntPtr result, IntPtr lhs, IntPtr rhs);
        /// <summary>
        /// Tses the c mul.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_CMul(IntPtr result, IntPtr lhs, IntPtr rhs);
        /// <summary>
        /// Tses the c div.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_CDiv(IntPtr result, IntPtr lhs, IntPtr rhs);
        /// <summary>
        /// Tses the c mod.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_CMod(IntPtr result, IntPtr lhs, IntPtr rhs);

        /// <summary>
        /// Tses the gt tensor.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_gtTensor(IntPtr result, IntPtr lhs, IntPtr rhs);
        /// <summary>
        /// Tses the lt tensor.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_ltTensor(IntPtr result, IntPtr lhs, IntPtr rhs);
        /// <summary>
        /// Tses the ge tensor.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_geTensor(IntPtr result, IntPtr lhs, IntPtr rhs);
        /// <summary>
        /// Tses the le tensor.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_leTensor(IntPtr result, IntPtr lhs, IntPtr rhs);
        /// <summary>
        /// Tses the eq tensor.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_eqTensor(IntPtr result, IntPtr lhs, IntPtr rhs);
        /// <summary>
        /// Tses the ne tensor.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_neTensor(IntPtr result, IntPtr lhs, IntPtr rhs);


        /// <summary>
        /// Tses the sum.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Sum(IntPtr result, IntPtr src, int dimension);
        /// <summary>
        /// Tses the product.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Prod(IntPtr result, IntPtr src, int dimension);
        /// <summary>
        /// Tses the minimum.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Min(IntPtr result, IntPtr src, int dimension);
        /// <summary>
        /// Tses the maximum.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Max(IntPtr result, IntPtr src, int dimension);

        /// <summary>
        /// Tses the argmin.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Argmin(IntPtr result, IntPtr src, int dimension);
        /// <summary>
        /// Tses the argmax.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Argmax(IntPtr result, IntPtr src, int dimension);

        /// <summary>
        /// Tses the mean.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Mean(IntPtr result, IntPtr src, int dimension);
        /// <summary>
        /// Tses the norm.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <param name="value">The value.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Norm(IntPtr result, IntPtr src, int dimension, float value);
        /// <summary>
        /// Tses the standard.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <param name="normByN">if set to <c>true</c> [norm by n].</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Std(IntPtr result, IntPtr src, int dimension, bool normByN);
        /// <summary>
        /// Tses the variable.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <param name="normByN">if set to <c>true</c> [norm by n].</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Var(IntPtr result, IntPtr src, int dimension, bool normByN);

        /// <summary>
        /// Tses the sum all.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_SumAll(IntPtr result, IntPtr src);
        /// <summary>
        /// Tses the product all.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_ProdAll(IntPtr result, IntPtr src);
        /// <summary>
        /// Tses the minimum all.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_MinAll(IntPtr result, IntPtr src);
        /// <summary>
        /// Tses the maximum all.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_MaxAll(IntPtr result, IntPtr src);

        /// <summary>
        /// Tses the mean all.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_MeanAll(IntPtr result, IntPtr src);
        /// <summary>
        /// Tses the variable all.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_VarAll(IntPtr result, IntPtr src);
        /// <summary>
        /// Tses the standard all.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_StdAll(IntPtr result, IntPtr src);
        /// <summary>
        /// Tses the norm all.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="value">The value.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_NormAll(IntPtr result, IntPtr src, float value);


        /// <summary>
        /// Tses the new RNG.
        /// </summary>
        /// <param name="rng">The RNG.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_NewRNG(out IntPtr rng);
        /// <summary>
        /// Tses the delete RNG.
        /// </summary>
        /// <param name="rng">The RNG.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_DeleteRNG(IntPtr rng);
        /// <summary>
        /// Tses the set RNG seed.
        /// </summary>
        /// <param name="rng">The RNG.</param>
        /// <param name="newSeed">The new seed.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_SetRNGSeed(IntPtr rng, int newSeed);

        /// <summary>
        /// Tses the random uniform.
        /// </summary>
        /// <param name="rng">The RNG.</param>
        /// <param name="result">The result.</param>
        /// <param name="min">The minimum.</param>
        /// <param name="max">The maximum.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_RandomUniform(IntPtr rng, IntPtr result, float min, float max);
        /// <summary>
        /// Tses the random normal.
        /// </summary>
        /// <param name="rng">The RNG.</param>
        /// <param name="result">The result.</param>
        /// <param name="mean">The mean.</param>
        /// <param name="stdv">The STDV.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_RandomNormal(IntPtr rng, IntPtr result, float mean, float stdv);
        /// <summary>
        /// Tses the random exponential.
        /// </summary>
        /// <param name="rng">The RNG.</param>
        /// <param name="result">The result.</param>
        /// <param name="lambda">The lambda.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_RandomExponential(IntPtr rng, IntPtr result, float lambda);
        /// <summary>
        /// Tses the random cauchy.
        /// </summary>
        /// <param name="rng">The RNG.</param>
        /// <param name="result">The result.</param>
        /// <param name="median">The median.</param>
        /// <param name="sigma">The sigma.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_RandomCauchy(IntPtr rng, IntPtr result, float median, float sigma);
        /// <summary>
        /// Tses the random log normal.
        /// </summary>
        /// <param name="rng">The RNG.</param>
        /// <param name="result">The result.</param>
        /// <param name="mean">The mean.</param>
        /// <param name="stdv">The STDV.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_RandomLogNormal(IntPtr rng, IntPtr result, float mean, float stdv);
        /// <summary>
        /// Tses the random geometric.
        /// </summary>
        /// <param name="rng">The RNG.</param>
        /// <param name="result">The result.</param>
        /// <param name="p">The p.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_RandomGeometric(IntPtr rng, IntPtr result, float p);
        /// <summary>
        /// Tses the random bernoulli.
        /// </summary>
        /// <param name="rng">The RNG.</param>
        /// <param name="result">The result.</param>
        /// <param name="p">The p.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_RandomBernoulli(IntPtr rng, IntPtr result, float p);


        /// <summary>
        /// Tses the unfolded acc.
        /// </summary>
        /// <param name="finput">The finput.</param>
        /// <param name="input">The input.</param>
        /// <param name="kW">The k w.</param>
        /// <param name="kH">The k h.</param>
        /// <param name="dW">The d w.</param>
        /// <param name="dH">The d h.</param>
        /// <param name="padW">The pad w.</param>
        /// <param name="padH">The pad h.</param>
        /// <param name="nInputPlane">The n input plane.</param>
        /// <param name="inputWidth">Width of the input.</param>
        /// <param name="inputHeight">Height of the input.</param>
        /// <param name="outputWidth">Width of the output.</param>
        /// <param name="outputHeight">Height of the output.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)]
        public static extern int TS_Unfolded_Acc(IntPtr finput, IntPtr input, int kW, int kH, int dW, int dH, int padW, int padH, int nInputPlane, int inputWidth, int inputHeight, int outputWidth, int outputHeight);
        /// <summary>
        /// Tses the unfolded copy.
        /// </summary>
        /// <param name="finput">The finput.</param>
        /// <param name="input">The input.</param>
        /// <param name="kW">The k w.</param>
        /// <param name="kH">The k h.</param>
        /// <param name="dW">The d w.</param>
        /// <param name="dH">The d h.</param>
        /// <param name="padW">The pad w.</param>
        /// <param name="padH">The pad h.</param>
        /// <param name="nInputPlane">The n input plane.</param>
        /// <param name="inputWidth">Width of the input.</param>
        /// <param name="inputHeight">Height of the input.</param>
        /// <param name="outputWidth">Width of the output.</param>
        /// <param name="outputHeight">Height of the output.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)]
        public static extern int TS_Unfolded_Copy(IntPtr finput, IntPtr input, int kW, int kH, int dW, int dH, int padW, int padH, int nInputPlane, int inputWidth, int inputHeight, int outputWidth, int outputHeight);


        /// <summary>
        /// Tses the spatial maximum pooling update output frame.
        /// </summary>
        /// <param name="input_p">The input p.</param>
        /// <param name="output_p">The output p.</param>
        /// <param name="ind_p">The ind p.</param>
        /// <param name="nslices">The nslices.</param>
        /// <param name="iwidth">The iwidth.</param>
        /// <param name="iheight">The iheight.</param>
        /// <param name="owidth">The owidth.</param>
        /// <param name="oheight">The oheight.</param>
        /// <param name="kW">The k w.</param>
        /// <param name="kH">The k h.</param>
        /// <param name="dW">The d w.</param>
        /// <param name="dH">The d h.</param>
        /// <param name="padW">The pad w.</param>
        /// <param name="padH">The pad h.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)]
        public static extern int TS_SpatialMaxPooling_updateOutput_frame(IntPtr input_p, IntPtr output_p, IntPtr ind_p, long nslices, long iwidth, long iheight, long owidth, long oheight, int kW, int kH, int dW, int dH, int padW, int padH);

        /// <summary>
        /// Tses the spatial maximum pooling update grad input frame.
        /// </summary>
        /// <param name="gradInput">The grad input.</param>
        /// <param name="gradOutput">The grad output.</param>
        /// <param name="ind">The ind.</param>
        /// <param name="nslices">The nslices.</param>
        /// <param name="iwidth">The iwidth.</param>
        /// <param name="iheight">The iheight.</param>
        /// <param name="owidth">The owidth.</param>
        /// <param name="oheight">The oheight.</param>
        /// <param name="dW">The d w.</param>
        /// <param name="dH">The d h.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)]
        public static extern int TS_SpatialMaxPooling_updateGradInput_frame(IntPtr gradInput, IntPtr gradOutput, IntPtr ind, long nslices, long iwidth, long iheight, long owidth, long oheight, int dW, int dH);


        /// <summary>
        /// Tses the gather.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dim">The dim.</param>
        /// <param name="indices">The indices.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Gather(IntPtr result, IntPtr src, int dim, IntPtr indices);
        /// <summary>
        /// Tses the scatter.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dim">The dim.</param>
        /// <param name="indices">The indices.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Scatter(IntPtr result, IntPtr src, int dim, IntPtr indices);
        /// <summary>
        /// Tses the scatter fill.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="value">The value.</param>
        /// <param name="dim">The dim.</param>
        /// <param name="indices">The indices.</param>
        /// <returns>System.Int32.</returns>
        [DllImport(dll, CallingConvention = cc)] public static extern int TS_ScatterFill(IntPtr result, float value, int dim, IntPtr indices);

        [DllImport(dll, CallingConvention = cc)]
        public static extern int TS_Im2Cols(IntPtr data_im, int height, int width, int channels,int ksize_h, int ksize_w, 
                                            int pad_h, int pad_w, int stride_h, int stride_w, int dilation_h, 
                                            int dilation_w, int height_col, int width_col, IntPtr data_col);

        [DllImport(dll, CallingConvention = cc)]
        public static extern int TS_Cols2Im(IntPtr data_col, int height, int width, int channels, int ksize_h, int ksize_w,
                                            int pad_h, int pad_w, int stride_h, int stride_w, int dilation_h,
                                            int dilation_w, int height_col, int width_col, IntPtr data_im);


        [DllImport(dll, CallingConvention = cc)] public static extern int TS_Diag(IntPtr result, IntPtr src);

    }
}
