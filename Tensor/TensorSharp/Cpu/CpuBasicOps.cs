// ***********************************************************************
// Assembly         : TensorSharp
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="CpuBasicOps.cs" company="TensorSharp">
//     Copyright (c) . All rights reserved.
// </copyright>
// <summary></summary>
// ***********************************************************************
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using TensorSharp.Core;

namespace TensorSharp.Cpu
{
    /// <summary>
    /// Class CpuBasicOps.
    /// </summary>
    [OpsClass]
    public class CpuBasicOps
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="CpuBasicOps"/> class.
        /// </summary>
        public CpuBasicOps()
        {
        }


        /// <summary>
        /// Dots the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        /// <exception cref="NotSupportedException"></exception>
        [RegisterOpStorageType("dot", typeof(CpuStorage))]
        public Tensor Dot(Tensor result, Tensor lhs, Tensor rhs)
        {
            if (lhs.DimensionCount == 1 && rhs.DimensionCount == 1)
            {
                return MatrixMultiplication.Dot(result, lhs, rhs);
            }
            else if (lhs.DimensionCount == 2 && (rhs.DimensionCount == 1 || rhs.PossibleVector))
            {
                return MatrixMultiplication.Mul_M_V(result, lhs, rhs.Ravel()).Reshape(lhs.Shape[0], 1);
            }
            else if (lhs.DimensionCount == 2 && rhs.DimensionCount == 2)
            {
                return MatrixMultiplication.Mul_M_M(result, lhs, rhs);
            }
            else
            {
                throw new NotSupportedException(string.Format("Multiplication of {0}D with {1}D tensor is not supported"));
            }
        }

        /// <summary>
        /// Addmms the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="beta">The beta.</param>
        /// <param name="src">The source.</param>
        /// <param name="alpha">The alpha.</param>
        /// <param name="m1">The m1.</param>
        /// <param name="m2">The m2.</param>
        /// <returns>Tensor.</returns>
        /// <exception cref="InvalidOperationException">
        /// All tensors must have the same element type
        /// or
        /// Size mismatch
        /// </exception>
        /// <exception cref="ArgumentException">
        /// result must be a CPU tensor - result
        /// or
        /// m1 must be a CPU tensor - m1
        /// or
        /// m2 must be a CPU tensor - m2
        /// or
        /// src must be a matrix - src
        /// or
        /// m1 must be a matrix - m1
        /// or
        /// m2 must be a matrix - m2
        /// </exception>
        [RegisterOpStorageType("addmm", typeof(CpuStorage))]
        public Tensor Addmm(Tensor result, float beta, Tensor src, float alpha, Tensor m1, Tensor m2)
        {
            if (src.ElementType != m1.ElementType || src.ElementType != m2.ElementType || (result != null && result.ElementType != src.ElementType))
                throw new InvalidOperationException("All tensors must have the same element type");
            if (result != null && !(result.Storage is CpuStorage)) throw new ArgumentException("result must be a CPU tensor", "result");
            if (!(m1.Storage is CpuStorage)) throw new ArgumentException("m1 must be a CPU tensor", "m1");
            if (!(m2.Storage is CpuStorage)) throw new ArgumentException("m2 must be a CPU tensor", "m2");

            if (src.DimensionCount != 2) throw new ArgumentException("src must be a matrix", "src");
            if (m1.DimensionCount != 2) throw new ArgumentException("m1 must be a matrix", "m1");
            if (m2.DimensionCount != 2) throw new ArgumentException("m2 must be a matrix", "m2");

            if (src.Shape[0] != m1.Shape[0] || src.Shape[1] != m2.Shape[1] || m1.Shape[1] != m2.Shape[0])
                throw new InvalidOperationException("Size mismatch");

            var writeTarget = TensorResultBuilder.GetWriteTarget(result, src, true, src.Shape);

            if (writeTarget != src)
            {
                Ops.Copy(writeTarget, src);
            }

            
            MatrixMultiplication.Gemm(alpha, m1, m2, beta, writeTarget);
            

            return writeTarget;
        }





        /// <summary>
        /// The abs function
        /// </summary>
        private MethodInfo abs_func = NativeWrapper.GetMethod("TS_Abs");
        /// <summary>
        /// Abses the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("abs", typeof(CpuStorage))]
        public Tensor Abs(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(abs_func, result, src); }

        /// <summary>
        /// The neg function
        /// </summary>
        private MethodInfo neg_func = NativeWrapper.GetMethod("TS_Neg");
        /// <summary>
        /// Negs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("neg", typeof(CpuStorage))]
        public Tensor Neg(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(neg_func, result, src); }

        /// <summary>
        /// The sign function
        /// </summary>
        private MethodInfo sign_func = NativeWrapper.GetMethod("TS_Sign");
        /// <summary>
        /// Signs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("sign", typeof(CpuStorage))]
        public Tensor Sign(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(sign_func, result, src); }


        /// <summary>
        /// The SQRT function
        /// </summary>
        private MethodInfo sqrt_func = NativeWrapper.GetMethod("TS_Sqrt");
        /// <summary>
        /// SQRTs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("sqrt", typeof(CpuStorage))]
        public Tensor Sqrt(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(sqrt_func, result, src); }

        /// <summary>
        /// The exp function
        /// </summary>
        private MethodInfo exp_func = NativeWrapper.GetMethod("TS_Exp");
        /// <summary>
        /// Exps the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("exp", typeof(CpuStorage))]
        public Tensor Exp(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(exp_func, result, src); }

        /// <summary>
        /// The log function
        /// </summary>
        private MethodInfo log_func = NativeWrapper.GetMethod("TS_Log");
        /// <summary>
        /// Logs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("log", typeof(CpuStorage))]
        public Tensor Log(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(log_func, result, src); }

        /// <summary>
        /// The log1p function
        /// </summary>
        private MethodInfo log1p_func = NativeWrapper.GetMethod("TS_Log1p");
        /// <summary>
        /// Log1ps the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("log1p", typeof(CpuStorage))]
        public Tensor Log1p(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(log1p_func, result, src); }

        /// <summary>
        /// The floor function
        /// </summary>
        private MethodInfo floor_func = NativeWrapper.GetMethod("TS_Floor");
        /// <summary>
        /// Floors the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("floor", typeof(CpuStorage))]
        public Tensor Floor(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(floor_func, result, src); }

        /// <summary>
        /// The ceil function
        /// </summary>
        private MethodInfo ceil_func = NativeWrapper.GetMethod("TS_Ceil");
        /// <summary>
        /// Ceils the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("ceil", typeof(CpuStorage))]
        public Tensor Ceil(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(ceil_func, result, src); }

        /// <summary>
        /// The round function
        /// </summary>
        private MethodInfo round_func = NativeWrapper.GetMethod("TS_Round");
        /// <summary>
        /// Rounds the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("round", typeof(CpuStorage))]
        public Tensor Round(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(round_func, result, src); }

        /// <summary>
        /// The trunc function
        /// </summary>
        private MethodInfo trunc_func = NativeWrapper.GetMethod("TS_Trunc");
        /// <summary>
        /// Truncs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("trunc", typeof(CpuStorage))]
        public Tensor Trunc(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(trunc_func, result, src); }

        /// <summary>
        /// The frac function
        /// </summary>
        private MethodInfo frac_func = NativeWrapper.GetMethod("TS_Frac");
        /// <summary>
        /// Fracs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("frac", typeof(CpuStorage))]
        public Tensor Frac(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(frac_func, result, src); }


        /// <summary>
        /// The sin function
        /// </summary>
        private MethodInfo sin_func = NativeWrapper.GetMethod("TS_Sin");
        /// <summary>
        /// Sins the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("sin", typeof(CpuStorage))]
        public Tensor Sin(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(sin_func, result, src); }

        /// <summary>
        /// The cos function
        /// </summary>
        private MethodInfo cos_func = NativeWrapper.GetMethod("TS_Cos");
        /// <summary>
        /// Coses the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("cos", typeof(CpuStorage))]
        public Tensor Cos(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(cos_func, result, src); }

        /// <summary>
        /// The tan function
        /// </summary>
        private MethodInfo tan_func = NativeWrapper.GetMethod("TS_Tan");
        /// <summary>
        /// Tans the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("tan", typeof(CpuStorage))]
        public Tensor Tan(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(tan_func, result, src); }


        /// <summary>
        /// The asin function
        /// </summary>
        private MethodInfo asin_func = NativeWrapper.GetMethod("TS_Asin");
        /// <summary>
        /// Asins the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("asin", typeof(CpuStorage))]
        public Tensor Asin(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(asin_func, result, src); }

        /// <summary>
        /// The acos function
        /// </summary>
        private MethodInfo acos_func = NativeWrapper.GetMethod("TS_Acos");
        /// <summary>
        /// Acoses the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("acos", typeof(CpuStorage))]
        public Tensor Acos(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(acos_func, result, src); }

        /// <summary>
        /// The atan function
        /// </summary>
        private MethodInfo atan_func = NativeWrapper.GetMethod("TS_Atan");
        /// <summary>
        /// Atans the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("atan", typeof(CpuStorage))]
        public Tensor Atan(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(atan_func, result, src); }


        /// <summary>
        /// The sinh function
        /// </summary>
        private MethodInfo sinh_func = NativeWrapper.GetMethod("TS_Sinh");
        /// <summary>
        /// Sinhes the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("sinh", typeof(CpuStorage))]
        public Tensor Sinh(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(sinh_func, result, src); }

        /// <summary>
        /// The cosh function
        /// </summary>
        private MethodInfo cosh_func = NativeWrapper.GetMethod("TS_Cosh");
        /// <summary>
        /// Coshes the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("cosh", typeof(CpuStorage))]
        public Tensor Cosh(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(cosh_func, result, src); }

        /// <summary>
        /// The tanh function
        /// </summary>
        private MethodInfo tanh_func = NativeWrapper.GetMethod("TS_Tanh");
        /// <summary>
        /// Tanhes the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("tanh", typeof(CpuStorage))]
        public Tensor Tanh(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(tanh_func, result, src); }


        /// <summary>
        /// [ERROR: invalid expression FieldName.Words.TheAndAllAsSentence]
        /// </summary>
        private MethodInfo sigmoid_func = NativeWrapper.GetMethod("TS_Sigmoid");
        /// <summary>
        /// Sigmoids the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("sigmoid", typeof(CpuStorage))]
        public Tensor Sigmoid(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(sigmoid_func, result, src); }


        /// <summary>
        /// The atan2 function
        /// </summary>
        private MethodInfo atan2_func = NativeWrapper.GetMethod("TS_Atan2");
        /// <summary>
        /// Atan2s the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="srcY">The source y.</param>
        /// <param name="srcX">The source x.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("atan2", typeof(CpuStorage))]
        public Tensor Atan2(Tensor result, Tensor srcY, Tensor srcX) { return NativeWrapper.InvokeNullableResultElementwise(atan2_func, result, srcY, srcX); }

        /// <summary>
        /// The pow function
        /// </summary>
        private MethodInfo pow_func = NativeWrapper.GetMethod("TS_Pow");
        /// <summary>
        /// Pows the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="value">The value.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("pow", typeof(CpuStorage))]
        public Tensor Pow(Tensor result, Tensor src, float value) { return NativeWrapper.InvokeNullableResultElementwise(pow_func, result, src, value); }

        /// <summary>
        /// The tpow function
        /// </summary>
        private MethodInfo tpow_func = NativeWrapper.GetMethod("TS_Tpow");
        /// <summary>
        /// Tpows the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="value">The value.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("tpow", typeof(CpuStorage))]
        public Tensor Tpow(Tensor result, float value, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(tpow_func, result, value, src); }

        /// <summary>
        /// The lerp function
        /// </summary>
        private MethodInfo lerp_func = NativeWrapper.GetMethod("TS_Lerp");
        /// <summary>
        /// Lerps the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="srcA">The source a.</param>
        /// <param name="srcB">The source b.</param>
        /// <param name="weight">The weight.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("lerp", typeof(CpuStorage))]
        public Tensor Lerp(Tensor result, Tensor srcA, Tensor srcB, float weight) { return NativeWrapper.InvokeNullableResultElementwise(lerp_func, result, srcA, srcB, weight); }

        /// <summary>
        /// The clamp function
        /// </summary>
        private MethodInfo clamp_func = NativeWrapper.GetMethod("TS_Clamp");
        /// <summary>
        /// Clamps the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="min">The minimum.</param>
        /// <param name="max">The maximum.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("clamp", typeof(CpuStorage))]
        public Tensor Clamp(Tensor result, Tensor src, float min, float max) { return NativeWrapper.InvokeNullableResultElementwise(clamp_func, result, src, min, max); }



        /// <summary>
        /// The add function
        /// </summary>
        private MethodInfo add_func = NativeWrapper.GetMethod("TS_Add");
        /// <summary>
        /// Adds the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("addv", typeof(CpuStorage))]
        public Tensor Add(Tensor result, Tensor lhs, float rhs) { return NativeWrapper.InvokeNullableResultElementwise(add_func, result, lhs, rhs); }

        /// <summary>
        /// The sub function
        /// </summary>
        private MethodInfo sub_func = NativeWrapper.GetMethod("TS_Sub");
        /// <summary>
        /// Subs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("subv", typeof(CpuStorage))]
        public Tensor Sub(Tensor result, Tensor lhs, float rhs) { return NativeWrapper.InvokeNullableResultElementwise(sub_func, result, lhs, rhs); }

        /// <summary>
        /// The rsub function
        /// </summary>
        private MethodInfo rsub_func = NativeWrapper.GetMethod("TS_Rsub");
        /// <summary>
        /// Subs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("rsubv", typeof(CpuStorage))]
        public Tensor Sub(Tensor result, float lhs, Tensor rhs) { return NativeWrapper.InvokeNullableResultElementwise(rsub_func, result, rhs, lhs); }

        /// <summary>
        /// The mul function
        /// </summary>
        private MethodInfo mul_func = NativeWrapper.GetMethod("TS_Mul");
        /// <summary>
        /// Muls the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("mulv", typeof(CpuStorage))]
        public Tensor Mul(Tensor result, Tensor lhs, float rhs) { return NativeWrapper.InvokeNullableResultElementwise(mul_func, result, lhs, rhs); }

        /// <summary>
        /// The div function
        /// </summary>
        private MethodInfo div_func = NativeWrapper.GetMethod("TS_Div");
        /// <summary>
        /// Divs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("divv", typeof(CpuStorage))]
        public Tensor Div(Tensor result, Tensor lhs, float rhs) { return NativeWrapper.InvokeNullableResultElementwise(div_func, result, lhs, rhs); }

        /// <summary>
        /// The rdiv function
        /// </summary>
        private MethodInfo rdiv_func = NativeWrapper.GetMethod("TS_Rdiv");
        /// <summary>
        /// Divs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("rdivv", typeof(CpuStorage))]
        public Tensor Div(Tensor result, float lhs, Tensor rhs) { return NativeWrapper.InvokeNullableResultElementwise(rdiv_func, result, rhs, lhs); }

        /// <summary>
        /// The mod function
        /// </summary>
        private MethodInfo mod_func = NativeWrapper.GetMethod("TS_Mod");
        /// <summary>
        /// Mods the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("modv", typeof(CpuStorage))]
        public Tensor Mod(Tensor result, Tensor lhs, float rhs) { return NativeWrapper.InvokeNullableResultElementwise(mod_func, result, lhs, rhs); }


        /// <summary>
        /// The gt value function
        /// </summary>
        private MethodInfo gtValue_func = NativeWrapper.GetMethod("TS_gtValue");
        /// <summary>
        /// Greaters the than.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("gtValue", typeof(CpuStorage))]
        public Tensor GreaterThan(Tensor result, Tensor lhs, float rhs) { return NativeWrapper.InvokeNullableResultElementwise(gtValue_func, result, lhs, rhs); }

        /// <summary>
        /// The lt value function
        /// </summary>
        private MethodInfo ltValue_func = NativeWrapper.GetMethod("TS_gtValue");
        /// <summary>
        /// Lesses the than.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("ltValue", typeof(CpuStorage))]
        public Tensor LessThan(Tensor result, Tensor lhs, float rhs) { return NativeWrapper.InvokeNullableResultElementwise(ltValue_func, result, lhs, rhs); }

        /// <summary>
        /// The ge value function
        /// </summary>
        private MethodInfo geValue_func = NativeWrapper.GetMethod("TS_gtValue");
        /// <summary>
        /// Greaters the or equal.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("geValue", typeof(CpuStorage))]
        public Tensor GreaterOrEqual(Tensor result, Tensor lhs, float rhs) { return NativeWrapper.InvokeNullableResultElementwise(geValue_func, result, lhs, rhs); }

        /// <summary>
        /// The le value function
        /// </summary>
        private MethodInfo leValue_func = NativeWrapper.GetMethod("TS_gtValue");
        /// <summary>
        /// Lesses the or equal.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("leValue", typeof(CpuStorage))]
        public Tensor LessOrEqual(Tensor result, Tensor lhs, float rhs) { return NativeWrapper.InvokeNullableResultElementwise(leValue_func, result, lhs, rhs); }

        /// <summary>
        /// The eq value function
        /// </summary>
        private MethodInfo eqValue_func = NativeWrapper.GetMethod("TS_gtValue");
        /// <summary>
        /// Equals to.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("eqValue", typeof(CpuStorage))]
        public Tensor EqualTo(Tensor result, Tensor lhs, float rhs) { return NativeWrapper.InvokeNullableResultElementwise(eqValue_func, result, lhs, rhs); }

        /// <summary>
        /// The ne value function
        /// </summary>
        private MethodInfo neValue_func = NativeWrapper.GetMethod("TS_gtValue");
        /// <summary>
        /// Nots the equal.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("neValue", typeof(CpuStorage))]
        public Tensor NotEqual(Tensor result, Tensor lhs, float rhs) { return NativeWrapper.InvokeNullableResultElementwise(neValue_func, result, lhs, rhs); }



        /// <summary>
        /// The cadd function
        /// </summary>
        private MethodInfo cadd_func = NativeWrapper.GetMethod("TS_CAdd");
        /// <summary>
        /// Adds the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("addt", typeof(CpuStorage))]
        public Tensor Add(Tensor result, Tensor lhs, Tensor rhs) { return NativeWrapper.InvokeNullableResultElementwise(cadd_func, result, lhs, rhs); }

        /// <summary>
        /// The csub function
        /// </summary>
        private MethodInfo csub_func = NativeWrapper.GetMethod("TS_CSub");
        /// <summary>
        /// Subs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("subt", typeof(CpuStorage))]
        public Tensor Sub(Tensor result, Tensor lhs, Tensor rhs) { return NativeWrapper.InvokeNullableResultElementwise(csub_func, result, lhs, rhs); }

        /// <summary>
        /// The cmul function
        /// </summary>
        private MethodInfo cmul_func = NativeWrapper.GetMethod("TS_CMul");
        /// <summary>
        /// Muls the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("mult", typeof(CpuStorage))]
        public Tensor Mul(Tensor result, Tensor lhs, Tensor rhs) { return NativeWrapper.InvokeNullableResultElementwise(cmul_func, result, lhs, rhs); }

        /// <summary>
        /// The cdiv function
        /// </summary>
        private MethodInfo cdiv_func = NativeWrapper.GetMethod("TS_CDiv");
        /// <summary>
        /// Divs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("divt", typeof(CpuStorage))]
        public Tensor Div(Tensor result, Tensor lhs, Tensor rhs) { return NativeWrapper.InvokeNullableResultElementwise(cdiv_func, result, lhs, rhs); }

        /// <summary>
        /// The cmod function
        /// </summary>
        private MethodInfo cmod_func = NativeWrapper.GetMethod("TS_CMod");
        /// <summary>
        /// Mods the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("modt", typeof(CpuStorage))]
        public Tensor Mod(Tensor result, Tensor lhs, Tensor rhs) { return NativeWrapper.InvokeNullableResultElementwise(cmod_func, result, lhs, rhs); }


        /// <summary>
        /// The gt tensor function
        /// </summary>
        private MethodInfo gtTensor_func = NativeWrapper.GetMethod("TS_gtTensor");
        /// <summary>
        /// Greaters the than.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("gtTensor", typeof(CpuStorage))]
        public Tensor GreaterThan(Tensor result, Tensor lhs, Tensor rhs) { return NativeWrapper.InvokeNullableResultElementwise(gtTensor_func, result, lhs, rhs); }

        /// <summary>
        /// The lt tensor function
        /// </summary>
        private MethodInfo ltTensor_func = NativeWrapper.GetMethod("TS_ltTensor");
        /// <summary>
        /// Lesses the than.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("gtTensor", typeof(CpuStorage))]
        public Tensor LessThan(Tensor result, Tensor lhs, Tensor rhs) { return NativeWrapper.InvokeNullableResultElementwise(ltTensor_func, result, lhs, rhs); }

        /// <summary>
        /// The ge tensor function
        /// </summary>
        private MethodInfo geTensor_func = NativeWrapper.GetMethod("TS_geTensor");
        /// <summary>
        /// Greaters the or equal.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("geTensor", typeof(CpuStorage))]
        public Tensor GreaterOrEqual(Tensor result, Tensor lhs, Tensor rhs) { return NativeWrapper.InvokeNullableResultElementwise(geTensor_func, result, lhs, rhs); }

        /// <summary>
        /// The le tensor function
        /// </summary>
        private MethodInfo leTensor_func = NativeWrapper.GetMethod("TS_leTensor");
        /// <summary>
        /// Lesses the or equal.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("leTensor", typeof(CpuStorage))]
        public Tensor LessOrEqual(Tensor result, Tensor lhs, Tensor rhs) { return NativeWrapper.InvokeNullableResultElementwise(leTensor_func, result, lhs, rhs); }

        /// <summary>
        /// The eq tensor function
        /// </summary>
        private MethodInfo eqTensor_func = NativeWrapper.GetMethod("TS_eqTensor");
        /// <summary>
        /// Equals to.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("eqTensor", typeof(CpuStorage))]
        public Tensor EqualTo(Tensor result, Tensor lhs, Tensor rhs) { return NativeWrapper.InvokeNullableResultElementwise(eqTensor_func, result, lhs, rhs); }

        /// <summary>
        /// The ne tensor function
        /// </summary>
        private MethodInfo neTensor_func = NativeWrapper.GetMethod("TS_neTensor");
        /// <summary>
        /// Nots the equal.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("neTensor", typeof(CpuStorage))]
        public Tensor NotEqual(Tensor result, Tensor lhs, Tensor rhs) { return NativeWrapper.InvokeNullableResultElementwise(neTensor_func, result, lhs, rhs); }


        /// <summary>
        /// The sum function
        /// </summary>
        private MethodInfo sum_func = NativeWrapper.GetMethod("TS_Sum");
        /// <summary>
        /// Sums the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("sum", typeof(CpuStorage))]
        public Tensor Sum(Tensor result, Tensor src, int dimension) { return NativeWrapper.InvokeNullableResultDimensionwise(sum_func, result, src, dimension); }

        /// <summary>
        /// The product function
        /// </summary>
        private MethodInfo prod_func = NativeWrapper.GetMethod("TS_Prod");
        /// <summary>
        /// Products the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("prod", typeof(CpuStorage))]
        public Tensor Prod(Tensor result, Tensor src, int dimension) { return NativeWrapper.InvokeNullableResultDimensionwise(prod_func, result, src, dimension); }

        /// <summary>
        /// The minimum function
        /// </summary>
        private MethodInfo min_func = NativeWrapper.GetMethod("TS_Min");
        /// <summary>
        /// Determines the minimum of the parameters.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("min", typeof(CpuStorage))]
        public Tensor Min(Tensor result, Tensor src, int dimension) { return NativeWrapper.InvokeNullableResultDimensionwise(min_func, result, src, dimension); }

        /// <summary>
        /// The maximum function
        /// </summary>
        private MethodInfo max_func = NativeWrapper.GetMethod("TS_Max");
        /// <summary>
        /// Determines the maximun of the parameters.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("max", typeof(CpuStorage))]
        public Tensor Max(Tensor result, Tensor src, int dimension) { return NativeWrapper.InvokeNullableResultDimensionwise(max_func, result, src, dimension); }


        /// <summary>
        /// The argmin function
        /// </summary>
        private MethodInfo argmin_func = NativeWrapper.GetMethod("TS_Argmin");
        /// <summary>
        /// Argmins the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("argmin", typeof(CpuStorage))]
        public Tensor Argmin(Tensor result, Tensor src, int dimension) { return NativeWrapper.InvokeNullableResultDimensionwise(argmax_func, result, src, dimension); }

        /// <summary>
        /// The argmax function
        /// </summary>
        private MethodInfo argmax_func = NativeWrapper.GetMethod("TS_Argmax");
        /// <summary>
        /// Argmaxes the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("argmax", typeof(CpuStorage))]
        public Tensor Argmax(Tensor result, Tensor src, int dimension) { return NativeWrapper.InvokeNullableResultDimensionwise(argmax_func, result, src, dimension); }



        /// <summary>
        /// The mean function
        /// </summary>
        private MethodInfo mean_func = NativeWrapper.GetMethod("TS_Mean");
        /// <summary>
        /// Means the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("mean", typeof(CpuStorage))]
        public Tensor Mean(Tensor result, Tensor src, int dimension) { return NativeWrapper.InvokeNullableResultDimensionwise(mean_func, result, src, dimension); }

        /// <summary>
        /// The norm function
        /// </summary>
        private MethodInfo norm_func = NativeWrapper.GetMethod("TS_Norm");
        /// <summary>
        /// Norms the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <param name="value">The value.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("norm", typeof(CpuStorage))]
        public Tensor Norm(Tensor result, Tensor src, int dimension, float value) { return NativeWrapper.InvokeNullableResultDimensionwise(norm_func, result, src, dimension, value); }

        /// <summary>
        /// The standard function
        /// </summary>
        private MethodInfo std_func = NativeWrapper.GetMethod("TS_Std");
        /// <summary>
        /// Standards the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <param name="normByN">if set to <c>true</c> [norm by n].</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("std", typeof(CpuStorage))]
        public Tensor Std(Tensor result, Tensor src, int dimension, bool normByN) { return NativeWrapper.InvokeNullableResultDimensionwise(std_func, result, src, dimension, normByN); }

        /// <summary>
        /// The variable function
        /// </summary>
        private MethodInfo var_func = NativeWrapper.GetMethod("TS_Var");
        /// <summary>
        /// Variables the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <param name="normByN">if set to <c>true</c> [norm by n].</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("var", typeof(CpuStorage))]
        public Tensor Var(Tensor result, Tensor src, int dimension, bool normByN) { return NativeWrapper.InvokeNullableResultDimensionwise(var_func, result, src, dimension, normByN); }



        /// <summary>
        /// The sumall function
        /// </summary>
        private MethodInfo sumall_func = NativeWrapper.GetMethod("TS_SumAll");
        /// <summary>
        /// Sums all.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("sumall", typeof(CpuStorage))]
        public Tensor SumAll(Tensor result, Tensor src)
        {
            var writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, 1);
            NativeWrapper.InvokeTypeMatch(sumall_func, writeTarget, src);
            return writeTarget;
        }

        /// <summary>
        /// The prodall function
        /// </summary>
        private MethodInfo prodall_func = NativeWrapper.GetMethod("TS_ProdAll");
        /// <summary>
        /// Products all.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("prodall", typeof(CpuStorage))]
        public Tensor ProdAll(Tensor result, Tensor src)
        {
            var writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, 1);
            NativeWrapper.InvokeTypeMatch(prodall_func, writeTarget, src);
            return writeTarget;
        }

        /// <summary>
        /// The minall function
        /// </summary>
        private MethodInfo minall_func = NativeWrapper.GetMethod("TS_MinAll");
        /// <summary>
        /// Minimums all.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("prodall", typeof(CpuStorage))]
        public Tensor MinAll(Tensor result, Tensor src)
        {
            var writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, 1);
            NativeWrapper.InvokeTypeMatch(minall_func, writeTarget, src);
            return writeTarget;
        }

        /// <summary>
        /// The maxall function
        /// </summary>
        private MethodInfo maxall_func = NativeWrapper.GetMethod("TS_MaxAll");
        /// <summary>
        /// Maximums all.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("maxall", typeof(CpuStorage))]
        public Tensor MaxAll(Tensor result, Tensor src)
        {
            var writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, 1);
            NativeWrapper.InvokeTypeMatch(maxall_func, writeTarget, src);
            return writeTarget;
        }


        /// <summary>
        /// The meanall function
        /// </summary>
        private MethodInfo meanall_func = NativeWrapper.GetMethod("TS_MeanAll");
        /// <summary>
        /// Means all.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("meanall", typeof(CpuStorage))]
        public Tensor MeanAll(Tensor result, Tensor src)
        {
            var writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, 1);
            NativeWrapper.InvokeTypeMatch(meanall_func, writeTarget, src);
            return writeTarget;
        }

        /// <summary>
        /// The varall function
        /// </summary>
        private MethodInfo varall_func = NativeWrapper.GetMethod("TS_VarAll");
        /// <summary>
        /// Variables all.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("varall", typeof(CpuStorage))]
        public Tensor VarAll(Tensor result, Tensor src)
        {
            var writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, 1);
            NativeWrapper.InvokeTypeMatch(varall_func, writeTarget, src);
            return writeTarget;
        }

        /// <summary>
        /// The stdall function
        /// </summary>
        private MethodInfo stdall_func = NativeWrapper.GetMethod("TS_StdAll");
        /// <summary>
        /// Standards all.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("stdall", typeof(CpuStorage))]
        public Tensor StdAll(Tensor result, Tensor src)
        {
            var writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, 1);
            NativeWrapper.InvokeTypeMatch(stdall_func, writeTarget, src);
            return writeTarget;
        }

        /// <summary>
        /// The normall function
        /// </summary>
        private MethodInfo normall_func = NativeWrapper.GetMethod("TS_NormAll");
        /// <summary>
        /// Norms all.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="value">The value.</param>
        /// <returns>Tensor.</returns>
        [RegisterOpStorageType("normall", typeof(CpuStorage))]
        public Tensor NormAll(Tensor result, Tensor src, float value)
        {
            var writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, 1);
            NativeWrapper.InvokeTypeMatch(normall_func, writeTarget, src, value);
            return writeTarget;
        }
    }
}
