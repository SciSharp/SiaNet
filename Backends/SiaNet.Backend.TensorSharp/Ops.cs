// ***********************************************************************
// Assembly         : SiaNet.Backend.TensorSharp
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="Ops.cs" company="SiaNet.Backend.TensorSharp">
//     Copyright (c) . All rights reserved.
// </copyright>
// <summary></summary>
// ***********************************************************************
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using SiaNet.Backend.TensorSharp.Core;

namespace SiaNet.Backend.TensorSharp
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
        public static NDArray NewContiguous(NDArray src)
        {
            var result = new NDArray(src.Allocator, src.ElementType, (long[])src.Shape.Clone());
            Copy(result, src);
            return result;
        }

        /// <summary>
        /// Ases the contiguous.
        /// </summary>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static NDArray AsContiguous(NDArray src)
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
        public static NDArray Concat(NDArray result, int dimension, params NDArray[] inputs)
        {
            return TensorConcatenation.Concat(result, dimension, inputs);
        }

        /// <summary>
        /// Fills the one hot.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="labelCount">The label count.</param>
        /// <param name="labels">The labels.</param>
        public static void FillOneHot(NDArray result, int labelCount, int[] labels)
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
                using (var cpuResult = new NDArray(cpuAlloc, result.ElementType, result.Shape))
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
        private static void DoFillOneHot(NDArray result, int labelCount, int[] labels)
        {
            if (result.DimensionCount != 2) throw new InvalidOperationException("result must be a 2D tensor");
            if (result.Shape[0] != labels.Length) throw new InvalidOperationException("first dimension of result must equal the number of samples");
            if (result.Shape[1] > labelCount) throw new InvalidOperationException("second dimension of result must be at least as large as labelCount");

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
        public static void Copy(NDArray result, NDArray src) { OpRegistry.Invoke("copy", result, src); }

        /// <summary>
        /// Fills the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="value">The value.</param>
        public static void Fill(NDArray result, float value) { OpRegistry.Invoke("fill", result, value); }

        /// <summary>
        /// Dots the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Dot(NDArray result, NDArray lhs, NDArray rhs) { return (NDArray)OpRegistry.Invoke("dot", result, lhs, rhs); }
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
        public static NDArray Addmm(NDArray result, float beta, NDArray src, float alpha, NDArray m1, NDArray m2) { return (NDArray)OpRegistry.Invoke("addmm", result, beta, src, alpha, m1, m2); }

        /// <summary>
        /// Abses the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Abs(NDArray result, NDArray src) { return (NDArray)OpRegistry.Invoke("abs", result, src); }

        /// <summary>
        /// Negs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Neg(NDArray result, NDArray src) { return (NDArray)OpRegistry.Invoke("neg", result, src); }

        /// <summary>
        /// Signs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Sign(NDArray result, NDArray src) { return (NDArray)OpRegistry.Invoke("sign", result, src); }

        /// <summary>
        /// SQRTs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Sqrt(NDArray result, NDArray src) { return (NDArray)OpRegistry.Invoke("sqrt", result, src); }
        /// <summary>
        /// Exps the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Exp(NDArray result, NDArray src) { return (NDArray)OpRegistry.Invoke("exp", result, src); }
        /// <summary>
        /// Logs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Log(NDArray result, NDArray src) { return (NDArray)OpRegistry.Invoke("log", result, src); }
        /// <summary>
        /// Log1ps the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Log1p(NDArray result, NDArray src) { return (NDArray)OpRegistry.Invoke("log1p", result, src); }
        /// <summary>
        /// Floors the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Floor(NDArray result, NDArray src) { return (NDArray)OpRegistry.Invoke("floor", result, src); }
        /// <summary>
        /// Ceils the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Ceil(NDArray result, NDArray src) { return (NDArray)OpRegistry.Invoke("ceil", result, src); }
        /// <summary>
        /// Rounds the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Round(NDArray result, NDArray src) { return (NDArray)OpRegistry.Invoke("round", result, src); }
        /// <summary>
        /// Truncs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Trunc(NDArray result, NDArray src) { return (NDArray)OpRegistry.Invoke("trunc", result, src); }
        /// <summary>
        /// Fracs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Frac(NDArray result, NDArray src) { return (NDArray)OpRegistry.Invoke("frac", result, src); }

        /// <summary>
        /// Sins the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Sin(NDArray src) { return (NDArray)OpRegistry.Invoke("sin", null, src); }
        /// <summary>
        /// Coses the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Cos(NDArray result, NDArray src) { return (NDArray)OpRegistry.Invoke("cos", result, src); }
        /// <summary>
        /// Tans the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Tan(NDArray result, NDArray src) { return (NDArray)OpRegistry.Invoke("tan", result, src); }

        /// <summary>
        /// Asins the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Asin(NDArray result, NDArray src) { return (NDArray)OpRegistry.Invoke("asin", result, src); }
        /// <summary>
        /// Acoses the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Acos(NDArray result, NDArray src) { return (NDArray)OpRegistry.Invoke("acos", result, src); }
        /// <summary>
        /// Atans the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Atan(NDArray result, NDArray src) { return (NDArray)OpRegistry.Invoke("atan", result, src); }

        /// <summary>
        /// Sinhes the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Sinh(NDArray result, NDArray src) { return (NDArray)OpRegistry.Invoke("sinh", result, src); }
        /// <summary>
        /// Coshes the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Cosh(NDArray result, NDArray src) { return (NDArray)OpRegistry.Invoke("cosh", result, src); }
        /// <summary>
        /// Tanhes the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Tanh(NDArray result, NDArray src) { return (NDArray)OpRegistry.Invoke("tanh", result, src); }

        /// <summary>
        /// Sigmoids the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Sigmoid(NDArray result, NDArray src) { return (NDArray)OpRegistry.Invoke("sigmoid", result, src); }


        /// <summary>
        /// Atan2s the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="srcY">The source y.</param>
        /// <param name="srcX">The source x.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Atan2(NDArray result, NDArray srcY, NDArray srcX) { return (NDArray)OpRegistry.Invoke("atan2", result, srcY, srcX); }
        /// <summary>
        /// Pows the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="value">The value.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Pow(NDArray result, NDArray src, float value) { return (NDArray)OpRegistry.Invoke("pow", result, src, value); }
        /// <summary>
        /// Tpows the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="value">The value.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Tpow(NDArray result, float value, NDArray src) { return (NDArray)OpRegistry.Invoke("tpow", result, value, src); }
        /// <summary>
        /// Lerps the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="srcA">The source a.</param>
        /// <param name="srcB">The source b.</param>
        /// <param name="weight">The weight.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Lerp(NDArray result, NDArray srcA, NDArray srcB, float weight) { return (NDArray)OpRegistry.Invoke("lerp", result, srcA, srcB); }
        /// <summary>
        /// Clamps the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="min">The minimum.</param>
        /// <param name="max">The maximum.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Clamp(NDArray result, NDArray src, float min, float max) { return (NDArray)OpRegistry.Invoke("clamp", result, src, min, max); }


        /// <summary>
        /// Adds the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Add(NDArray result, NDArray lhs, float rhs) { return (NDArray)OpRegistry.Invoke("addv", result, lhs, rhs); }
        /// <summary>
        /// Subs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Sub(NDArray result, NDArray lhs, float rhs) { return (NDArray)OpRegistry.Invoke("subv", result, lhs, rhs); }
        /// <summary>
        /// Subs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Sub(NDArray result, float lhs, NDArray rhs) { return (NDArray)OpRegistry.Invoke("rsubv", result, lhs, rhs); }
        /// <summary>
        /// Muls the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Mul(NDArray result, NDArray lhs, float rhs) { return (NDArray)OpRegistry.Invoke("mulv", result, lhs, rhs); }
        /// <summary>
        /// Divs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Div(NDArray result, NDArray lhs, float rhs) { return (NDArray)OpRegistry.Invoke("divv", result, lhs, rhs); }
        /// <summary>
        /// Divs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Div(NDArray result, float lhs, NDArray rhs) { return (NDArray)OpRegistry.Invoke("rdivv", result, lhs, rhs); }
        /// <summary>
        /// Mods the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Mod(NDArray result, NDArray lhs, float rhs) { return (NDArray)OpRegistry.Invoke("modv", result, lhs, rhs); }

        /// <summary>
        /// Greaters the than.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static NDArray GreaterThan(NDArray result, NDArray lhs, float rhs) { return (NDArray)OpRegistry.Invoke("gtValue", result, lhs, rhs); }
        /// <summary>
        /// Lesses the than.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static NDArray LessThan(NDArray result, NDArray lhs, float rhs) { return (NDArray)OpRegistry.Invoke("ltValue", result, lhs, rhs); }
        /// <summary>
        /// Greaters the or equal.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static NDArray GreaterOrEqual(NDArray result, NDArray lhs, float rhs) { return (NDArray)OpRegistry.Invoke("geValue", result, lhs, rhs); }
        /// <summary>
        /// Lesses the or equal.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static NDArray LessOrEqual(NDArray result, NDArray lhs, float rhs) { return (NDArray)OpRegistry.Invoke("leValue", result, lhs, rhs); }
        /// <summary>
        /// Equals to.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static NDArray EqualTo(NDArray result, NDArray lhs, float rhs) { return (NDArray)OpRegistry.Invoke("eqValue", result, lhs, rhs); }
        /// <summary>
        /// Nots the equal.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static NDArray NotEqual(NDArray result, NDArray lhs, float rhs) { return (NDArray)OpRegistry.Invoke("neValue", result, lhs, rhs); }

        /// <summary>
        /// Adds the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Add(NDArray result, NDArray lhs, NDArray rhs) { return (NDArray)OpRegistry.Invoke("addt", result, lhs, rhs); }
        /// <summary>
        /// Subs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Sub(NDArray result, NDArray lhs, NDArray rhs) { return (NDArray)OpRegistry.Invoke("subt", result, lhs, rhs); }
        /// <summary>
        /// Muls the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Mul(NDArray result, NDArray lhs, NDArray rhs) { return (NDArray)OpRegistry.Invoke("mult", result, lhs, rhs); }
        /// <summary>
        /// Divs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Div(NDArray result, NDArray lhs, NDArray rhs) { return (NDArray)OpRegistry.Invoke("divt", result, lhs, rhs); }
        /// <summary>
        /// Mods the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Mod(NDArray result, NDArray lhs, NDArray rhs) { return (NDArray)OpRegistry.Invoke("modt", result, lhs, rhs); }

        /// <summary>
        /// Greaters the than.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static NDArray GreaterThan(NDArray result, NDArray lhs, NDArray rhs) { return (NDArray)OpRegistry.Invoke("gtTensor", result, lhs, rhs); }
        /// <summary>
        /// Lesses the than.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static NDArray LessThan(NDArray result, NDArray lhs, NDArray rhs) { return (NDArray)OpRegistry.Invoke("ltTensor", result, lhs, rhs); }
        /// <summary>
        /// Greaters the or equal.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static NDArray GreaterOrEqual(NDArray result, NDArray lhs, NDArray rhs) { return (NDArray)OpRegistry.Invoke("geTensor", result, lhs, rhs); }
        /// <summary>
        /// Lesses the or equal.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static NDArray LessOrEqual(NDArray result, NDArray lhs, NDArray rhs) { return (NDArray)OpRegistry.Invoke("leTensor", result, lhs, rhs); }
        /// <summary>
        /// Equals to.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static NDArray EqualTo(NDArray result, NDArray lhs, NDArray rhs) { return (NDArray)OpRegistry.Invoke("eqTensor", result, lhs, rhs); }
        /// <summary>
        /// Nots the equal.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static NDArray NotEqual(NDArray result, NDArray lhs, NDArray rhs) { return (NDArray)OpRegistry.Invoke("neTensor", result, lhs, rhs); }


        /// <summary>
        /// Sums the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Sum(NDArray result, NDArray src, int dimension) { return (NDArray)OpRegistry.Invoke("sum", result, src, dimension); }
        /// <summary>
        /// Products the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Prod(NDArray result, NDArray src, int dimension) { return (NDArray)OpRegistry.Invoke("prod", result, src, dimension); }
        /// <summary>
        /// Determines the minimum of the parameters.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Min(NDArray result, NDArray src, int dimension) { return (NDArray)OpRegistry.Invoke("min", result, src, dimension); }
        /// <summary>
        /// Determines the maximun of the parameters.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Max(NDArray result, NDArray src, int dimension) { return (NDArray)OpRegistry.Invoke("max", result, src, dimension); }
        /// <summary>
        /// Argmins the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Argmin(NDArray result, NDArray src, int dimension) { return (NDArray)OpRegistry.Invoke("argmin", result, src, dimension); }
        /// <summary>
        /// Argmaxes the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Argmax(NDArray result, NDArray src, int dimension) { return (NDArray)OpRegistry.Invoke("argmax", result, src, dimension); }

        /// <summary>
        /// Means the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Mean(NDArray result, NDArray src, int dimension) { return (NDArray)OpRegistry.Invoke("mean", result, src, dimension); }
        /// <summary>
        /// Norms the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <param name="value">The value.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Norm(NDArray result, NDArray src, int dimension, float value) { return (NDArray)OpRegistry.Invoke("norm", result, src, dimension, value); }
        /// <summary>
        /// Standards the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <param name="normByN">if set to <c>true</c> [norm by n].</param>
        /// <returns>Tensor.</returns>
        public static NDArray Std(NDArray result, NDArray src, int dimension, bool normByN) { return (NDArray)OpRegistry.Invoke("std", result, src, dimension, normByN); }
        /// <summary>
        /// Variables the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <param name="normByN">if set to <c>true</c> [norm by n].</param>
        /// <returns>Tensor.</returns>
        public static NDArray Var(NDArray result, NDArray src, int dimension, bool normByN) { return (NDArray)OpRegistry.Invoke("var", result, src, dimension, normByN); }

        /// <summary>
        /// Sums all.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static NDArray SumAll(NDArray result, NDArray src) { return (NDArray)OpRegistry.Invoke("sumall", result, src); }
        /// <summary>
        /// Products all.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static NDArray ProdAll(NDArray result, NDArray src) { return (NDArray)OpRegistry.Invoke("prodall", result, src); }
        /// <summary>
        /// Minimums all.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static NDArray MinAll(NDArray result, NDArray src) { return (NDArray)OpRegistry.Invoke("minall", result, src); }
        /// <summary>
        /// Maximums all.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static NDArray MaxAll(NDArray result, NDArray src) { return (NDArray)OpRegistry.Invoke("maxall", result, src); }

        /// <summary>
        /// Means all.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static NDArray MeanAll(NDArray result, NDArray src) { return (NDArray)OpRegistry.Invoke("meanall", result, src); }
        /// <summary>
        /// Norms all.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="value">The value.</param>
        /// <returns>Tensor.</returns>
        public static NDArray NormAll(NDArray result, NDArray src, float value) { return (NDArray)OpRegistry.Invoke("normall", result, src, value); }
        /// <summary>
        /// Standards all.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static NDArray StdAll(NDArray result, NDArray src) { return (NDArray)OpRegistry.Invoke("stdall", result, src); }
        /// <summary>
        /// Variables all.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static NDArray VarAll(NDArray result, NDArray src) { return (NDArray)OpRegistry.Invoke("varall", result, src); }


        /// <summary>
        /// Sums all.
        /// </summary>
        /// <param name="src">The source.</param>
        /// <returns>System.Single.</returns>
        public static float SumAll(NDArray src) { using (var resultTensor = SumAll(null, src)) { return resultTensor.GetElementAsFloat(0); } }
        /// <summary>
        /// Products all.
        /// </summary>
        /// <param name="src">The source.</param>
        /// <returns>System.Single.</returns>
        public static float ProdAll(NDArray src) { using (var resultTensor = ProdAll(null, src)) { return resultTensor.GetElementAsFloat(0); } }
        /// <summary>
        /// Minimums all.
        /// </summary>
        /// <param name="src">The source.</param>
        /// <returns>System.Single.</returns>
        public static float MinAll(NDArray src) { using (var resultTensor = MinAll(null, src)) { return resultTensor.GetElementAsFloat(0); } }
        /// <summary>
        /// Maximums all.
        /// </summary>
        /// <param name="src">The source.</param>
        /// <returns>System.Single.</returns>
        public static float MaxAll(NDArray src) { using (var resultTensor = MaxAll(null, src)) { return resultTensor.GetElementAsFloat(0); } }

        /// <summary>
        /// Means all.
        /// </summary>
        /// <param name="src">The source.</param>
        /// <returns>System.Single.</returns>
        public static float MeanAll(NDArray src) { using (var resultTensor = MeanAll(null, src)) { return resultTensor.GetElementAsFloat(0); } }
        /// <summary>
        /// Variables all.
        /// </summary>
        /// <param name="src">The source.</param>
        /// <returns>System.Single.</returns>
        public static float VarAll(NDArray src) { using (var resultTensor = VarAll(null, src)) { return resultTensor.GetElementAsFloat(0); } }
        /// <summary>
        /// Standards all.
        /// </summary>
        /// <param name="src">The source.</param>
        /// <returns>System.Single.</returns>
        public static float StdAll(NDArray src) { using (var resultTensor = StdAll(null, src)) { return resultTensor.GetElementAsFloat(0); } }
        /// <summary>
        /// Norms all.
        /// </summary>
        /// <param name="src">The source.</param>
        /// <param name="value">The value.</param>
        /// <returns>System.Single.</returns>
        public static float NormAll(NDArray src, float value) { using (var resultTensor = NormAll(null, src, value)) { return resultTensor.GetElementAsFloat(0); } }


        /// <summary>
        /// Indexes the select.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dim">The dim.</param>
        /// <param name="indices">The indices.</param>
        /// <returns>Tensor.</returns>
        public static NDArray IndexSelect(NDArray result, NDArray src, int dim, NDArray indices) { return (NDArray)OpRegistry.Invoke("index_select", result, src, dim, indices); }
        /// <summary>
        /// Gathers the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dim">The dim.</param>
        /// <param name="indices">The indices.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Gather(NDArray result, NDArray src, int dim, NDArray indices) { return (NDArray)OpRegistry.Invoke("gather", result, src, dim, indices); }
        /// <summary>
        /// Scatters the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dim">The dim.</param>
        /// <param name="indices">The indices.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Scatter(NDArray result, NDArray src, int dim, NDArray indices)
        {
            if (result == null)
                result = src;
            return (NDArray)OpRegistry.Invoke("scatter", result, src, dim, indices);
        }
        /// <summary>
        /// Scatters the fill.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="value">The value.</param>
        /// <param name="dim">The dim.</param>
        /// <param name="indices">The indices.</param>
        /// <returns>Tensor.</returns>
        public static NDArray ScatterFill(NDArray result, float value, int dim, NDArray indices) { return (NDArray)OpRegistry.Invoke("scatter_fill", result, value, dim, indices); }


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
        public static void RandomUniform(NDArray result, SeedSource seedSource, float min, float max) { OpRegistry.Invoke("random_uniform", result, GetSeed(seedSource), min, max); }
        /// <summary>
        /// Randoms the normal.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="seedSource">The seed source.</param>
        /// <param name="mean">The mean.</param>
        /// <param name="stdv">The STDV.</param>
        public static void RandomNormal(NDArray result, SeedSource seedSource, float mean, float stdv) { OpRegistry.Invoke("random_normal", result, GetSeed(seedSource), mean, stdv); }
        /// <summary>
        /// Randoms the exponential.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="seedSource">The seed source.</param>
        /// <param name="lambda">The lambda.</param>
        public static void RandomExponential(NDArray result, SeedSource seedSource, float lambda) { OpRegistry.Invoke("random_exponential", result, GetSeed(seedSource), lambda); }
        /// <summary>
        /// Randoms the cauchy.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="seedSource">The seed source.</param>
        /// <param name="median">The median.</param>
        /// <param name="sigma">The sigma.</param>
        public static void RandomCauchy(NDArray result, SeedSource seedSource, float median, float sigma) { OpRegistry.Invoke("random_cauchy", result, GetSeed(seedSource), median, sigma); }
        /// <summary>
        /// Randoms the log normal.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="seedSource">The seed source.</param>
        /// <param name="mean">The mean.</param>
        /// <param name="stdv">The STDV.</param>
        public static void RandomLogNormal(NDArray result, SeedSource seedSource, float mean, float stdv) { OpRegistry.Invoke("random_lognormal", result, GetSeed(seedSource), mean, stdv); }
        /// <summary>
        /// Randoms the geometric.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="seedSource">The seed source.</param>
        /// <param name="p">The p.</param>
        public static void RandomGeometric(NDArray result, SeedSource seedSource, float p) { OpRegistry.Invoke("random_geometric", result, GetSeed(seedSource), p); }
        /// <summary>
        /// Randoms the bernoulli.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="seedSource">The seed source.</param>
        /// <param name="p">The p.</param>
        public static void RandomBernoulli(NDArray result, SeedSource seedSource, float p) { OpRegistry.Invoke("random_bernoulli", result, GetSeed(seedSource), p); }
    }
}
