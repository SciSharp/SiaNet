// ***********************************************************************
// Assembly         : TensorSharp
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="Ops.cs" company="TensorSharp">
//     Copyright (c) . All rights reserved.
// </copyright>
// <summary></summary>
// ***********************************************************************
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TensorSharp.Core;

namespace TensorSharp
{
    /// <summary>
    /// Class Ops.
    /// </summary>
    public static class Ops
    {
        /// <summary>
        /// Creates new contiguous.
        /// </summary>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static Tensor NewContiguous(Tensor src)
        {
            var result = new Tensor(src.Allocator, src.ElementType, (long[])src.Sizes.Clone());
            Copy(result, src);
            return result;
        }

        /// <summary>
        /// Ases the contiguous.
        /// </summary>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static Tensor AsContiguous(Tensor src)
        {
            if (src.IsContiguous())
                return src.CopyRef();
            else
                return NewContiguous(src);
        }

        /// <summary>
        /// Concats the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="dimension">The dimension.</param>
        /// <param name="inputs">The inputs.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Concat(Tensor result, int dimension, params Tensor[] inputs)
        {
            return TensorConcatenation.Concat(result, dimension, inputs);
        }

        /// <summary>
        /// Fills the one hot.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="labelCount">The label count.</param>
        /// <param name="labels">The labels.</param>
        public static void FillOneHot(Tensor result, int labelCount, int[] labels)
        {
            if (result.Storage is Cpu.CpuStorage)
            {
                DoFillOneHot(result, labelCount, labels);
            }
            else
            {
                //If the result is not on the CPU, it is much faster to build the tensor on the CPU and then copy
                //An alternative to this would be building a specific GPU kernel for this operation
                var cpuAlloc = new Cpu.CpuAllocator();
                using (var cpuResult = new Tensor(cpuAlloc, result.ElementType, result.Sizes))
                {
                    DoFillOneHot(cpuResult, labelCount, labels);
                    Ops.Copy(result, cpuResult);
                }
            }
        }

        /// <summary>
        /// Does the fill one hot.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="labelCount">The label count.</param>
        /// <param name="labels">The labels.</param>
        /// <exception cref="InvalidOperationException">
        /// result must be a 2D tensor
        /// or
        /// first dimension of result must equal the number of samples
        /// or
        /// second dimension of result must be at least as large as labelCount
        /// or
        /// label at index " + i + " is out of range 0 <= x < labelCount
        /// </exception>
        private static void DoFillOneHot(Tensor result, int labelCount, int[] labels)
        {
            if (result.DimensionCount != 2) throw new InvalidOperationException("result must be a 2D tensor");
            if (result.Sizes[0] != labels.Length) throw new InvalidOperationException("first dimension of result must equal the number of samples");
            if (result.Sizes[1] > labelCount) throw new InvalidOperationException("second dimension of result must be at least as large as labelCount");

            Ops.Fill(result, 0);
            for (int i = 0; i < labels.Length; ++i)
            {
                if (labels[i] < 0 || labels[i] >= labelCount)
                    throw new InvalidOperationException("label at index " + i + " is out of range 0 <= x < labelCount");

                result.SetElementAsFloat(1.0f, i, labels[i]);
            }
        }


        /// <summary>
        /// Copies the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        public static void Copy(Tensor result, Tensor src) { OpRegistry.Invoke("copy", result, src); }

        /// <summary>
        /// Fills the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="value">The value.</param>
        public static void Fill(Tensor result, float value) { OpRegistry.Invoke("fill", result, value); }

        /// <summary>
        /// Dots the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Dot(Tensor result, Tensor lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("dot", result, lhs, rhs); }
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
        public static Tensor Addmm(Tensor result, float beta, Tensor src, float alpha, Tensor m1, Tensor m2) { return (Tensor)OpRegistry.Invoke("addmm", result, beta, src, alpha, m1, m2); }

        /// <summary>
        /// Abses the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Abs(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("abs", result, src); }

        /// <summary>
        /// Negs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Neg(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("neg", result, src); }

        /// <summary>
        /// Signs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Sign(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("sign", result, src); }

        /// <summary>
        /// SQRTs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Sqrt(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("sqrt", result, src); }
        /// <summary>
        /// Exps the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Exp(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("exp", result, src); }
        /// <summary>
        /// Logs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Log(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("log", result, src); }
        /// <summary>
        /// Log1ps the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Log1p(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("log1p", result, src); }
        /// <summary>
        /// Floors the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Floor(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("floor", result, src); }
        /// <summary>
        /// Ceils the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Ceil(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("ceil", result, src); }
        /// <summary>
        /// Rounds the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Round(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("round", result, src); }
        /// <summary>
        /// Truncs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Trunc(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("trunc", result, src); }
        /// <summary>
        /// Fracs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Frac(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("frac", result, src); }

        /// <summary>
        /// Sins the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Sin(Tensor src) { return (Tensor)OpRegistry.Invoke("sin", null, src); }
        /// <summary>
        /// Coses the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Cos(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("cos", result, src); }
        /// <summary>
        /// Tans the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Tan(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("tan", result, src); }

        /// <summary>
        /// Asins the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Asin(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("asin", result, src); }
        /// <summary>
        /// Acoses the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Acos(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("acos", result, src); }
        /// <summary>
        /// Atans the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Atan(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("atan", result, src); }

        /// <summary>
        /// Sinhes the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Sinh(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("sinh", result, src); }
        /// <summary>
        /// Coshes the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Cosh(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("cosh", result, src); }
        /// <summary>
        /// Tanhes the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Tanh(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("tanh", result, src); }

        /// <summary>
        /// Sigmoids the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Sigmoid(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("sigmoid", result, src); }


        /// <summary>
        /// Atan2s the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="srcY">The source y.</param>
        /// <param name="srcX">The source x.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Atan2(Tensor result, Tensor srcY, Tensor srcX) { return (Tensor)OpRegistry.Invoke("atan2", result, srcY, srcX); }
        /// <summary>
        /// Pows the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="value">The value.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Pow(Tensor result, Tensor src, float value) { return (Tensor)OpRegistry.Invoke("pow", result, src, value); }
        /// <summary>
        /// Tpows the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="value">The value.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Tpow(Tensor result, float value, Tensor src) { return (Tensor)OpRegistry.Invoke("tpow", result, value, src); }
        /// <summary>
        /// Lerps the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="srcA">The source a.</param>
        /// <param name="srcB">The source b.</param>
        /// <param name="weight">The weight.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Lerp(Tensor result, Tensor srcA, Tensor srcB, float weight) { return (Tensor)OpRegistry.Invoke("lerp", result, srcA, srcB); }
        /// <summary>
        /// Clamps the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="min">The minimum.</param>
        /// <param name="max">The maximum.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Clamp(Tensor result, Tensor src, float min, float max) { return (Tensor)OpRegistry.Invoke("clamp", result, src, min, max); }


        /// <summary>
        /// Adds the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Add(Tensor result, Tensor lhs, float rhs) { return (Tensor)OpRegistry.Invoke("addv", result, lhs, rhs); }
        /// <summary>
        /// Subs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Sub(Tensor result, Tensor lhs, float rhs) { return (Tensor)OpRegistry.Invoke("subv", result, lhs, rhs); }
        /// <summary>
        /// Subs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Sub(Tensor result, float lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("rsubv", result, lhs, rhs); }
        /// <summary>
        /// Muls the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Mul(Tensor result, Tensor lhs, float rhs) { return (Tensor)OpRegistry.Invoke("mulv", result, lhs, rhs); }
        /// <summary>
        /// Divs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Div(Tensor result, Tensor lhs, float rhs) { return (Tensor)OpRegistry.Invoke("divv", result, lhs, rhs); }
        /// <summary>
        /// Divs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Div(Tensor result, float lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("rdivv", result, lhs, rhs); }
        /// <summary>
        /// Mods the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Mod(Tensor result, Tensor lhs, float rhs) { return (Tensor)OpRegistry.Invoke("modv", result, lhs, rhs); }

        /// <summary>
        /// Greaters the than.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static Tensor GreaterThan(Tensor result, Tensor lhs, float rhs) { return (Tensor)OpRegistry.Invoke("gtValue", result, lhs, rhs); }
        /// <summary>
        /// Lesses the than.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static Tensor LessThan(Tensor result, Tensor lhs, float rhs) { return (Tensor)OpRegistry.Invoke("ltValue", result, lhs, rhs); }
        /// <summary>
        /// Greaters the or equal.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static Tensor GreaterOrEqual(Tensor result, Tensor lhs, float rhs) { return (Tensor)OpRegistry.Invoke("geValue", result, lhs, rhs); }
        /// <summary>
        /// Lesses the or equal.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static Tensor LessOrEqual(Tensor result, Tensor lhs, float rhs) { return (Tensor)OpRegistry.Invoke("leValue", result, lhs, rhs); }
        /// <summary>
        /// Equals to.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static Tensor EqualTo(Tensor result, Tensor lhs, float rhs) { return (Tensor)OpRegistry.Invoke("eqValue", result, lhs, rhs); }
        /// <summary>
        /// Nots the equal.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static Tensor NotEqual(Tensor result, Tensor lhs, float rhs) { return (Tensor)OpRegistry.Invoke("neValue", result, lhs, rhs); }

        /// <summary>
        /// Adds the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Add(Tensor result, Tensor lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("addt", result, lhs, rhs); }
        /// <summary>
        /// Subs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Sub(Tensor result, Tensor lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("subt", result, lhs, rhs); }
        /// <summary>
        /// Muls the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Mul(Tensor result, Tensor lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("mult", result, lhs, rhs); }
        /// <summary>
        /// Divs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Div(Tensor result, Tensor lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("divt", result, lhs, rhs); }
        /// <summary>
        /// Mods the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Mod(Tensor result, Tensor lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("modt", result, lhs, rhs); }

        /// <summary>
        /// Greaters the than.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static Tensor GreaterThan(Tensor result, Tensor lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("gtTensor", result, lhs, rhs); }
        /// <summary>
        /// Lesses the than.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static Tensor LessThan(Tensor result, Tensor lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("ltTensor", result, lhs, rhs); }
        /// <summary>
        /// Greaters the or equal.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static Tensor GreaterOrEqual(Tensor result, Tensor lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("geTensor", result, lhs, rhs); }
        /// <summary>
        /// Lesses the or equal.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static Tensor LessOrEqual(Tensor result, Tensor lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("leTensor", result, lhs, rhs); }
        /// <summary>
        /// Equals to.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static Tensor EqualTo(Tensor result, Tensor lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("eqTensor", result, lhs, rhs); }
        /// <summary>
        /// Nots the equal.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static Tensor NotEqual(Tensor result, Tensor lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("neTensor", result, lhs, rhs); }


        /// <summary>
        /// Sums the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Sum(Tensor result, Tensor src, int dimension) { return (Tensor)OpRegistry.Invoke("sum", result, src, dimension); }
        /// <summary>
        /// Products the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Prod(Tensor result, Tensor src, int dimension) { return (Tensor)OpRegistry.Invoke("prod", result, src, dimension); }
        /// <summary>
        /// Determines the minimum of the parameters.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Min(Tensor result, Tensor src, int dimension) { return (Tensor)OpRegistry.Invoke("min", result, src, dimension); }
        /// <summary>
        /// Determines the maximun of the parameters.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Max(Tensor result, Tensor src, int dimension) { return (Tensor)OpRegistry.Invoke("max", result, src, dimension); }
        /// <summary>
        /// Argmins the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Argmin(Tensor result, Tensor src, int dimension) { return (Tensor)OpRegistry.Invoke("argmin", result, src, dimension); }
        /// <summary>
        /// Argmaxes the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Argmax(Tensor result, Tensor src, int dimension) { return (Tensor)OpRegistry.Invoke("argmax", result, src, dimension); }

        /// <summary>
        /// Means the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Mean(Tensor result, Tensor src, int dimension) { return (Tensor)OpRegistry.Invoke("mean", result, src, dimension); }
        /// <summary>
        /// Norms the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <param name="value">The value.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Norm(Tensor result, Tensor src, int dimension, float value) { return (Tensor)OpRegistry.Invoke("norm", result, src, dimension, value); }
        /// <summary>
        /// Standards the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <param name="normByN">if set to <c>true</c> [norm by n].</param>
        /// <returns>Tensor.</returns>
        public static Tensor Std(Tensor result, Tensor src, int dimension, bool normByN) { return (Tensor)OpRegistry.Invoke("std", result, src, dimension, normByN); }
        /// <summary>
        /// Variables the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <param name="normByN">if set to <c>true</c> [norm by n].</param>
        /// <returns>Tensor.</returns>
        public static Tensor Var(Tensor result, Tensor src, int dimension, bool normByN) { return (Tensor)OpRegistry.Invoke("var", result, src, dimension, normByN); }

        /// <summary>
        /// Sums all.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static Tensor SumAll(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("sumall", result, src); }
        /// <summary>
        /// Products all.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static Tensor ProdAll(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("prodall", result, src); }
        /// <summary>
        /// Minimums all.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static Tensor MinAll(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("minall", result, src); }
        /// <summary>
        /// Maximums all.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static Tensor MaxAll(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("maxall", result, src); }

        /// <summary>
        /// Means all.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static Tensor MeanAll(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("meanall", result, src); }
        /// <summary>
        /// Norms all.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="value">The value.</param>
        /// <returns>Tensor.</returns>
        public static Tensor NormAll(Tensor result, Tensor src, float value) { return (Tensor)OpRegistry.Invoke("normall", result, src, value); }
        /// <summary>
        /// Standards all.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static Tensor StdAll(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("stdall", result, src); }
        /// <summary>
        /// Variables all.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static Tensor VarAll(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("varall", result, src); }


        /// <summary>
        /// Sums all.
        /// </summary>
        /// <param name="src">The source.</param>
        /// <returns>System.Single.</returns>
        public static float SumAll(Tensor src) { using (var resultTensor = SumAll(null, src)) { return resultTensor.GetElementAsFloat(0); } }
        /// <summary>
        /// Products all.
        /// </summary>
        /// <param name="src">The source.</param>
        /// <returns>System.Single.</returns>
        public static float ProdAll(Tensor src) { using (var resultTensor = ProdAll(null, src)) { return resultTensor.GetElementAsFloat(0); } }
        /// <summary>
        /// Minimums all.
        /// </summary>
        /// <param name="src">The source.</param>
        /// <returns>System.Single.</returns>
        public static float MinAll(Tensor src) { using (var resultTensor = MinAll(null, src)) { return resultTensor.GetElementAsFloat(0); } }
        /// <summary>
        /// Maximums all.
        /// </summary>
        /// <param name="src">The source.</param>
        /// <returns>System.Single.</returns>
        public static float MaxAll(Tensor src) { using (var resultTensor = MaxAll(null, src)) { return resultTensor.GetElementAsFloat(0); } }

        /// <summary>
        /// Means all.
        /// </summary>
        /// <param name="src">The source.</param>
        /// <returns>System.Single.</returns>
        public static float MeanAll(Tensor src) { using (var resultTensor = MeanAll(null, src)) { return resultTensor.GetElementAsFloat(0); } }
        /// <summary>
        /// Variables all.
        /// </summary>
        /// <param name="src">The source.</param>
        /// <returns>System.Single.</returns>
        public static float VarAll(Tensor src) { using (var resultTensor = VarAll(null, src)) { return resultTensor.GetElementAsFloat(0); } }
        /// <summary>
        /// Standards all.
        /// </summary>
        /// <param name="src">The source.</param>
        /// <returns>System.Single.</returns>
        public static float StdAll(Tensor src) { using (var resultTensor = StdAll(null, src)) { return resultTensor.GetElementAsFloat(0); } }
        /// <summary>
        /// Norms all.
        /// </summary>
        /// <param name="src">The source.</param>
        /// <param name="value">The value.</param>
        /// <returns>System.Single.</returns>
        public static float NormAll(Tensor src, float value) { using (var resultTensor = NormAll(null, src, value)) { return resultTensor.GetElementAsFloat(0); } }


        /// <summary>
        /// Indexes the select.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dim">The dim.</param>
        /// <param name="indices">The indices.</param>
        /// <returns>Tensor.</returns>
        public static Tensor IndexSelect(Tensor result, Tensor src, int dim, Tensor indices) { return (Tensor)OpRegistry.Invoke("index_select", result, src, dim, indices); }
        /// <summary>
        /// Gathers the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dim">The dim.</param>
        /// <param name="indices">The indices.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Gather(Tensor result, Tensor src, int dim, Tensor indices) { return (Tensor)OpRegistry.Invoke("gather", result, src, dim, indices); }
        /// <summary>
        /// Scatters the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dim">The dim.</param>
        /// <param name="indices">The indices.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Scatter(Tensor result, Tensor src, int dim, Tensor indices) { return (Tensor)OpRegistry.Invoke("scatter", result, src, dim, indices); }
        /// <summary>
        /// Scatters the fill.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="value">The value.</param>
        /// <param name="dim">The dim.</param>
        /// <param name="indices">The indices.</param>
        /// <returns>Tensor.</returns>
        public static Tensor ScatterFill(Tensor result, float value, int dim, Tensor indices) { return (Tensor)OpRegistry.Invoke("scatter_fill", result, value, dim, indices); }


        /// <summary>
        /// Gets the seed.
        /// </summary>
        /// <param name="src">The source.</param>
        /// <returns>System.Nullable&lt;System.Int32&gt;.</returns>
        private static int? GetSeed(SeedSource src)
        {
            return src == null ? (int?)null : src.NextSeed();
        }

        /// <summary>
        /// Randoms the uniform.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="seedSource">The seed source.</param>
        /// <param name="min">The minimum.</param>
        /// <param name="max">The maximum.</param>
        public static void RandomUniform(Tensor result, SeedSource seedSource, float min, float max) { OpRegistry.Invoke("random_uniform", result, GetSeed(seedSource), min, max); }
        /// <summary>
        /// Randoms the normal.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="seedSource">The seed source.</param>
        /// <param name="mean">The mean.</param>
        /// <param name="stdv">The STDV.</param>
        public static void RandomNormal(Tensor result, SeedSource seedSource, float mean, float stdv) { OpRegistry.Invoke("random_normal", result, GetSeed(seedSource), mean, stdv); }
        /// <summary>
        /// Randoms the exponential.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="seedSource">The seed source.</param>
        /// <param name="lambda">The lambda.</param>
        public static void RandomExponential(Tensor result, SeedSource seedSource, float lambda) { OpRegistry.Invoke("random_exponential", result, GetSeed(seedSource), lambda); }
        /// <summary>
        /// Randoms the cauchy.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="seedSource">The seed source.</param>
        /// <param name="median">The median.</param>
        /// <param name="sigma">The sigma.</param>
        public static void RandomCauchy(Tensor result, SeedSource seedSource, float median, float sigma) { OpRegistry.Invoke("random_cauchy", result, GetSeed(seedSource), median, sigma); }
        /// <summary>
        /// Randoms the log normal.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="seedSource">The seed source.</param>
        /// <param name="mean">The mean.</param>
        /// <param name="stdv">The STDV.</param>
        public static void RandomLogNormal(Tensor result, SeedSource seedSource, float mean, float stdv) { OpRegistry.Invoke("random_lognormal", result, GetSeed(seedSource), mean, stdv); }
        /// <summary>
        /// Randoms the geometric.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="seedSource">The seed source.</param>
        /// <param name="p">The p.</param>
        public static void RandomGeometric(Tensor result, SeedSource seedSource, float p) { OpRegistry.Invoke("random_geometric", result, GetSeed(seedSource), p); }
        /// <summary>
        /// Randoms the bernoulli.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="seedSource">The seed source.</param>
        /// <param name="p">The p.</param>
        public static void RandomBernoulli(Tensor result, SeedSource seedSource, float p) { OpRegistry.Invoke("random_bernoulli", result, GetSeed(seedSource), p); }
    }
}
