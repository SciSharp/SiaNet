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
using SiaNet.Backend.TensorSharp.Expression;

namespace SiaNet.Backend.TensorSharp
{
    /// <summary>
    /// Class Ops.
    /// </summary>
    public class TOps
    {
        [NonSerialized]
        public float EPSILON = 1e-07f;

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
        public static NDArray Concat( int dimension, params NDArray[] inputs)
        {
            return TensorConcatenation.Concat(null, dimension, inputs);
        }

        /// <summary>
        /// Fills the one hot.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="labelCount">The label count.</param>
        /// <param name="labels">The labels.</param>
        public static void FillOneHot(NDArray src, int labelCount, int[] labels)
        {
            if (src.Storage is Cpu.CpuStorage)
            {
                DoFillOneHot(src, labelCount, labels);
            }
            else
            {
                //If the result is not on the CPU, it is much faster to build the tensor on the CPU and then copy
                //An alternative to this would be building a specific GPU kernel for this operation
                var cpuAlloc = new Cpu.CpuAllocator();
                using (var cpuResult = new NDArray(cpuAlloc, src.ElementType, src.Shape))
                {
                    DoFillOneHot(cpuResult, labelCount, labels);
                    Ops.Copy(src, cpuResult);
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
        private static void DoFillOneHot(NDArray src, int labelCount, int[] labels)
        {
            if (src.DimensionCount != 2) throw new InvalidOperationException("result must be a 2D tensor");
            if (src.Shape[0] != labels.Length) throw new InvalidOperationException("first dimension of result must equal the number of samples");
            if (src.Shape[1] > labelCount) throw new InvalidOperationException("second dimension of result must be at least as large as labelCount");

            Ops.Fill(src, 0);
            for (int i = 0; i < labels.Length; ++i)
            {
                if (labels[i] < 0 || labels[i] >= labelCount)
                    throw new InvalidOperationException("label at index " + i + " is out of range 0 <= x < labelCount");

                src.SetElementAsFloat(1.0f, i, labels[i]);
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
        public static void Fill(NDArray src, float value) { OpRegistry.Invoke("fill", src, value); }

        /// <summary>
        /// Dots the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Dot( NDArray lhs, NDArray rhs) { return (NDArray)OpRegistry.Invoke("dot", null, lhs, rhs); }
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
        public static NDArray Addmm( float beta, NDArray src, float alpha, NDArray m1, NDArray m2) { return (NDArray)OpRegistry.Invoke("addmm", null, beta, src, alpha, m1, m2); }

        /// <summary>
        /// Abses the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Abs( NDArray src) { return (NDArray)OpRegistry.Invoke("abs", null, src); }

        /// <summary>
        /// Negs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Neg( NDArray src) { return (NDArray)OpRegistry.Invoke("neg", null, src); }

        /// <summary>
        /// Signs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Sign( NDArray src) { return (NDArray)OpRegistry.Invoke("sign", null, src); }

        /// <summary>
        /// SQRTs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Sqrt( NDArray src) { return (NDArray)OpRegistry.Invoke("sqrt", null, src); }
        /// <summary>
        /// Exps the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Exp( NDArray src) { return (NDArray)OpRegistry.Invoke("exp", null, src); }
        /// <summary>
        /// Logs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Log( NDArray src) { return (NDArray)OpRegistry.Invoke("log", null, src); }
        /// <summary>
        /// Log1ps the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Log1p( NDArray src) { return (NDArray)OpRegistry.Invoke("log1p", null, src); }
        /// <summary>
        /// Floors the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Floor( NDArray src) { return (NDArray)OpRegistry.Invoke("floor", null, src); }
        /// <summary>
        /// Ceils the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Ceil( NDArray src) { return (NDArray)OpRegistry.Invoke("ceil", null, src); }
        /// <summary>
        /// Rounds the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Round( NDArray src) { return (NDArray)OpRegistry.Invoke("round", null, src); }
        /// <summary>
        /// Truncs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Trunc( NDArray src) { return (NDArray)OpRegistry.Invoke("trunc", null, src); }
        /// <summary>
        /// Fracs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Frac( NDArray src) { return (NDArray)OpRegistry.Invoke("frac", null, src); }

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
        public static NDArray Cos( NDArray src) { return (NDArray)OpRegistry.Invoke("cos", null, src); }
        /// <summary>
        /// Tans the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Tan( NDArray src) { return (NDArray)OpRegistry.Invoke("tan", null, src); }

        /// <summary>
        /// Asins the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Asin( NDArray src) { return (NDArray)OpRegistry.Invoke("asin", null, src); }
        /// <summary>
        /// Acoses the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Acos( NDArray src) { return (NDArray)OpRegistry.Invoke("acos", null, src); }
        /// <summary>
        /// Atans the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Atan( NDArray src) { return (NDArray)OpRegistry.Invoke("atan", null, src); }

        /// <summary>
        /// Sinhes the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Sinh( NDArray src) { return (NDArray)OpRegistry.Invoke("sinh", null, src); }
        /// <summary>
        /// Coshes the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Cosh( NDArray src) { return (NDArray)OpRegistry.Invoke("cosh", null, src); }
        /// <summary>
        /// Tanhes the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Tanh( NDArray src) { return (NDArray)OpRegistry.Invoke("tanh", null, src); }

        /// <summary>
        /// Sigmoids the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Sigmoid( NDArray src) { return (NDArray)OpRegistry.Invoke("sigmoid", null, src); }


        /// <summary>
        /// Atan2s the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="srcY">The source y.</param>
        /// <param name="srcX">The source x.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Atan2( NDArray srcY, NDArray srcX) { return (NDArray)OpRegistry.Invoke("atan2", null, srcY, srcX); }
        /// <summary>
        /// Pows the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="value">The value.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Pow( NDArray src, float value) { return (NDArray)OpRegistry.Invoke("pow", null, src, value); }

        /// <summary>
        /// Squares the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="value">The value.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Square(NDArray src) { return (NDArray)OpRegistry.Invoke("pow", null, src, 2); }

        /// <summary>
        /// Tpows the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="value">The value.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Tpow( float value, NDArray src) { return (NDArray)OpRegistry.Invoke("tpow", null, value, src); }


        /// <summary>
        /// Lerps the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="srcA">The source a.</param>
        /// <param name="srcB">The source b.</param>
        /// <param name="weight">The weight.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Lerp( NDArray srcA, NDArray srcB, float weight) { return (NDArray)OpRegistry.Invoke("lerp", null, srcA, srcB); }
        /// <summary>
        /// Clamps the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="min">The minimum.</param>
        /// <param name="max">The maximum.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Clip( NDArray src, float min, float max) { return (NDArray)OpRegistry.Invoke("clamp", null, src, min, max); }


        /// <summary>
        /// Adds the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Add( NDArray lhs, float rhs) { return (NDArray)OpRegistry.Invoke("addv", null, lhs, rhs); }
        /// <summary>
        /// Subs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Sub( NDArray lhs, float rhs) { return (NDArray)OpRegistry.Invoke("subv", null, lhs, rhs); }
        /// <summary>
        /// Subs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Sub( float lhs, NDArray rhs) { return (NDArray)OpRegistry.Invoke("rsubv", null, lhs, rhs); }
        /// <summary>
        /// Muls the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Mul( NDArray lhs, float rhs) { return (NDArray)OpRegistry.Invoke("mulv", null, lhs, rhs); }
        /// <summary>
        /// Divs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Div( NDArray lhs, float rhs) { return (NDArray)OpRegistry.Invoke("divv", null, lhs, rhs); }
        /// <summary>
        /// Divs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Div( float lhs, NDArray rhs) { return (NDArray)OpRegistry.Invoke("rdivv", null, lhs, rhs); }
        /// <summary>
        /// Mods the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Mod( NDArray lhs, float rhs) { return (NDArray)OpRegistry.Invoke("modv", null, lhs, rhs); }

        /// <summary>
        /// Greaters the than.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static NDArray GreaterThan( NDArray lhs, float rhs) { return (NDArray)OpRegistry.Invoke("gtValue", null, lhs, rhs); }
        /// <summary>
        /// Lesses the than.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static NDArray LessThan( NDArray lhs, float rhs) { return (NDArray)OpRegistry.Invoke("ltValue", null, lhs, rhs); }
        /// <summary>
        /// Greaters the or equal.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static NDArray GreaterOrEqual( NDArray lhs, float rhs) { return (NDArray)OpRegistry.Invoke("geValue", null, lhs, rhs); }
        /// <summary>
        /// Lesses the or equal.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static NDArray LessOrEqual( NDArray lhs, float rhs) { return (NDArray)OpRegistry.Invoke("leValue", null, lhs, rhs); }
        /// <summary>
        /// Equals to.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static NDArray EqualTo( NDArray lhs, float rhs) { return (NDArray)OpRegistry.Invoke("eqValue", null, lhs, rhs); }
        /// <summary>
        /// Nots the equal.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static NDArray NotEqual( NDArray lhs, float rhs) { return (NDArray)OpRegistry.Invoke("neValue", null, lhs, rhs); }

        /// <summary>
        /// Adds the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Add(NDArray lhs, NDArray rhs) { return (NDArray)OpRegistry.Invoke("addt", null, lhs, rhs); }
        /// <summary>
        /// Subs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Sub(NDArray lhs, NDArray rhs) { return (NDArray)OpRegistry.Invoke("subt", null, lhs, rhs); }
        /// <summary>
        /// Muls the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Mul(NDArray lhs, NDArray rhs) { return (NDArray)OpRegistry.Invoke("mult", null, lhs, rhs); }
        /// <summary>
        /// Divs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Div(NDArray lhs, NDArray rhs) { return (NDArray)OpRegistry.Invoke("divt", null, lhs, rhs); }
        /// <summary>
        /// Mods the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Mod(NDArray lhs, NDArray rhs) { return (NDArray)OpRegistry.Invoke("modt", null, lhs, rhs); }

        /// <summary>
        /// Greaters the than.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static NDArray GreaterThan( NDArray lhs, NDArray rhs) { return (NDArray)OpRegistry.Invoke("gtTensor", null, lhs, rhs); }
        /// <summary>
        /// Lesses the than.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static NDArray LessThan( NDArray lhs, NDArray rhs) { return (NDArray)OpRegistry.Invoke("ltTensor", null, lhs, rhs); }
        /// <summary>
        /// Greaters the or equal.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static NDArray GreaterOrEqual( NDArray lhs, NDArray rhs) { return (NDArray)OpRegistry.Invoke("geTensor", null, lhs, rhs); }
        /// <summary>
        /// Lesses the or equal.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static NDArray LessOrEqual( NDArray lhs, NDArray rhs) { return (NDArray)OpRegistry.Invoke("leTensor", null, lhs, rhs); }
        /// <summary>
        /// Equals to.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static NDArray EqualTo( NDArray lhs, NDArray rhs) { return (NDArray)OpRegistry.Invoke("eqTensor", null, lhs, rhs); }
        /// <summary>
        /// Nots the equal.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static NDArray NotEqual( NDArray lhs, NDArray rhs) { return (NDArray)OpRegistry.Invoke("neTensor", null, lhs, rhs); }


        /// <summary>
        /// Sums the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Sum(NDArray src, int dimension)
        {
            dimension = dimension < 0 ? src.DimensionCount + dimension : dimension;
            return (NDArray)OpRegistry.Invoke("sum", null, src, dimension);
        }

        public static NDArray Sum(NDArray src, params int[] dimension)
        {
            foreach (int dim in dimension)
            {
                src = Sum(src, dim);
            }

            return src;
        }

        /// <summary>
        /// Products the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Prod( NDArray src, int dimension) { dimension = dimension < 0 ? src.DimensionCount + dimension : dimension; return (NDArray)OpRegistry.Invoke("prod", null, src, dimension); }
        /// <summary>
        /// Determines the minimum of the parameters.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Min( NDArray src, int dimension) { dimension = dimension < 0 ? src.DimensionCount + dimension : dimension; return (NDArray)OpRegistry.Invoke("min", null, src, dimension); }
        /// <summary>
        /// Determines the maximun of the parameters.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Max( NDArray src, int dimension)
        {
            dimension = dimension < 0 ? src.DimensionCount + dimension : dimension;

            return (NDArray)OpRegistry.Invoke("max", null, src, dimension);
        }

        public static NDArray Max(NDArray src, params int[] dimension)
        {
            var shape = src.Shape;
            
            foreach (var item in dimension)
            {
                src = Max(src, item);
            }

            return src;
        }

        /// <summary>
        /// Argmins the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Argmin(NDArray src, int dimension) { dimension = dimension < 0 ? src.DimensionCount + dimension : dimension; return (NDArray)OpRegistry.Invoke("argmin", null, src, dimension); }
        /// <summary>
        /// Argmaxes the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Argmax(NDArray src, int dimension) { dimension = dimension < 0 ? src.DimensionCount + dimension : dimension; return (NDArray)OpRegistry.Invoke("argmax", null, src, dimension); }

        /// <summary>
        /// Means the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Mean(NDArray src, int dimension)
        {
            dimension = dimension < 0 ? src.DimensionCount + dimension : dimension;

            return (NDArray)OpRegistry.Invoke("mean", null, src, dimension);
        }

        public static NDArray Mean(NDArray src, params int[] dimension)
        {
            foreach (var item in dimension)
            {
                src = Mean(src, item);
            }

            return src;
        }

        /// <summary>
        /// Norms the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <param name="value">The value.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Norm( NDArray src, int dimension, float value) { dimension = dimension < 0 ? src.DimensionCount + dimension : dimension; return (NDArray)OpRegistry.Invoke("norm", null, src, dimension, value); }
        /// <summary>
        /// Standards the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <param name="normByN">if set to <c>true</c> [norm by n].</param>
        /// <returns>Tensor.</returns>
        public static NDArray Std( NDArray src, int dimension, bool normByN) { dimension = dimension < 0 ? src.DimensionCount + dimension : dimension; return (NDArray)OpRegistry.Invoke("std", null, src, dimension, normByN); }
        /// <summary>
        /// Variables the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <param name="normByN">if set to <c>true</c> [norm by n].</param>
        /// <returns>Tensor.</returns>
        public static NDArray Var( NDArray src, int dimension, bool normByN) { dimension = dimension < 0 ? src.DimensionCount + dimension : dimension; return (NDArray)OpRegistry.Invoke("var", null, src, dimension, normByN); }

        /// <summary>
        /// Sums all.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Sum( NDArray src) { return (NDArray)OpRegistry.Invoke("sumall", null, src); }
        /// <summary>
        /// Products all.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Prod( NDArray src) { return (NDArray)OpRegistry.Invoke("prodall", null, src); }
        /// <summary>
        /// Minimums all.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Min( NDArray src) { return (NDArray)OpRegistry.Invoke("minall", null, src); }
        /// <summary>
        /// Maximums all.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Max( NDArray src) { return (NDArray)OpRegistry.Invoke("maxall", null, src); }

        /// <summary>
        /// Means all.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Mean( NDArray src) { return (NDArray)OpRegistry.Invoke("meanall", null, src); }
        /// <summary>
        /// Norms all.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="value">The value.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Norm( NDArray src, float value) { return (NDArray)OpRegistry.Invoke("normall", null, src, value); }
        /// <summary>
        /// Standards all.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Std( NDArray src) { return (NDArray)OpRegistry.Invoke("stdall", null, src); }
        /// <summary>
        /// Variables all.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Var( NDArray src) { return (NDArray)OpRegistry.Invoke("varall", null, src); }


        /// <summary>
        /// Sums all.
        /// </summary>
        /// <param name="src">The source.</param>
        /// <returns>System.Single.</returns>
        public static float SumF(NDArray src) { using (var resultTensor = Sum(src)) { return resultTensor.GetElementAsFloat(0); } }
        /// <summary>
        /// Products all.
        /// </summary>
        /// <param name="src">The source.</param>
        /// <returns>System.Single.</returns>
        public static float ProdF(NDArray src) { using (var resultTensor = Prod(src)) { return resultTensor.GetElementAsFloat(0); } }
        /// <summary>
        /// Minimums all.
        /// </summary>
        /// <param name="src">The source.</param>
        /// <returns>System.Single.</returns>
        public static float MinF(NDArray src) { using (var resultTensor = Min(src)) { return resultTensor.GetElementAsFloat(0); } }
        /// <summary>
        /// Maximums all.
        /// </summary>
        /// <param name="src">The source.</param>
        /// <returns>System.Single.</returns>
        public static float MaxF(NDArray src) { using (var resultTensor = Max(src)) { return resultTensor.GetElementAsFloat(0); } }

        /// <summary>
        /// Means all.
        /// </summary>
        /// <param name="src">The source.</param>
        /// <returns>System.Single.</returns>
        public static float MeanF(NDArray src) { using (var resultTensor = Mean(src)) { return resultTensor.GetElementAsFloat(0); } }
        /// <summary>
        /// Variables all.
        /// </summary>
        /// <param name="src">The source.</param>
        /// <returns>System.Single.</returns>
        public static float VarF(NDArray src) { using (var resultTensor = Var(src)) { return resultTensor.GetElementAsFloat(0); } }
        /// <summary>
        /// Standards all.
        /// </summary>
        /// <param name="src">The source.</param>
        /// <returns>System.Single.</returns>
        public static float StdF(NDArray src) { using (var resultTensor = Std(src)) { return resultTensor.GetElementAsFloat(0); } }
        /// <summary>
        /// Norms all.
        /// </summary>
        /// <param name="src">The source.</param>
        /// <param name="value">The value.</param>
        /// <returns>System.Single.</returns>
        public static float NormF(NDArray src, float value) { using (var resultTensor = Norm(src, value)) { return resultTensor.GetElementAsFloat(0); } }


        /// <summary>
        /// Indexes the select.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dim">The dim.</param>
        /// <param name="indices">The indices.</param>
        /// <returns>Tensor.</returns>
        public static NDArray IndexSelect( NDArray src, int dim, NDArray indices) { return (NDArray)OpRegistry.Invoke("index_select", null, src, dim, indices); }
        /// <summary>
        /// Gathers the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dim">The dim.</param>
        /// <param name="indices">The indices.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Gather( NDArray src, int dim, NDArray indices) { return (NDArray)OpRegistry.Invoke("gather", null, src, dim, indices); }
        /// <summary>
        /// Scatters the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dim">The dim.</param>
        /// <param name="indices">The indices.</param>
        /// <returns>Tensor.</returns>
        public static NDArray Scatter( NDArray src, int dim, NDArray indices)
        {
            var result = src;
            result = (NDArray)OpRegistry.Invoke("scatter", result, src, dim, indices);
            return result;
        }
        /// <summary>
        /// Scatters the fill.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="value">The value.</param>
        /// <param name="dim">The dim.</param>
        /// <param name="indices">The indices.</param>
        /// <returns>Tensor.</returns>
        public static NDArray ScatterFill( float value, int dim, NDArray indices) { return (NDArray)OpRegistry.Invoke("scatter_fill", null, value, dim, indices); }


        public static NDArray Maximum(NDArray a, NDArray b)
        {
            var t1 = (a >= b);
            var t2 = (a > b);

            return (t1 * a + t2 * b);
        }

        public static NDArray Maximum(NDArray a, float b)
        {
            var b_t = new NDArray(a.Allocator, a.ElementType, a.Shape);
            TOps.Fill(b_t, b);
            return Maximum(a, b_t);
        }

        public static NDArray Maximum(float a, NDArray b)
        {
            var a_t = new NDArray(b.Allocator, b.ElementType, b.Shape);
            TOps.Fill(a_t, a);
            return Maximum(a_t, b);
        }

        public static NDArray Softplus(NDArray x)
        {
            return Log((Exp(x) + 1));
        }

        public static NDArray Softmax(NDArray x, int axis=-1)
        {
            var e = Exp(x - Max(x, axis));
            var s = Sum(e, axis);
            return e / s;
        }

        public static NDArray L2Normalize(NDArray x, int axis = -1)
        {
            NDArray y = Max(Sum(Square(x), axis), axis);

            return x / Sqrt(y);
        }

        public static NDArray Tile(NDArray x, long repetitions)
        {
            long[] shape = new long[x.DimensionCount];
            for (int i = 0; i < shape.Length; i++)
            {
                shape[i] = 1;
            }

            shape[shape.Length - 1] = repetitions;

            return x.RepeatTensor(shape);
        }

        public static NDArray Repeat(NDArray x, int reps)
        {
            x = x.View(x.ElementCount(), 1).Tile(reps);
            return x.View(1, x.ElementCount());
        }

        public static NDArray Diag(NDArray x)
        {
            return (NDArray)OpRegistry.Invoke("diag", x);
        }

        public static ValueTuple<NDArray, NDArray> BroadcastTensor(NDArray lhs, NDArray rhs)
        {
            if (!lhs.IsSameSizeAs(rhs))
            {
                if (lhs.Shape[0] == rhs.Shape[0] && (lhs.Shape[1] == 1 || rhs.Shape[1] == 1))
                {
                    if (lhs.Shape[1] == 1)
                    {
                        lhs = lhs.RepeatTensor(1, rhs.Shape[1]);
                    }

                    if (rhs.Shape[1] == 1)
                    {
                        rhs = rhs.RepeatTensor(1, lhs.Shape[1]);
                    }
                }

                if (lhs.Shape[1] == rhs.Shape[1] && (lhs.Shape[0] == 1 || rhs.Shape[0] == 1))
                {
                    if (lhs.Shape[0] == 1)
                    {
                        lhs = lhs.RepeatTensor(rhs.Shape[0], 1);
                    }

                    if (rhs.Shape[0] == 1)
                    {
                        rhs = rhs.RepeatTensor(lhs.Shape[0], 1);
                    }
                }

                if (lhs.Shape[1] == 1 && rhs.Shape[0] == 1)
                {
                    if (lhs.Shape[1] == 1)
                    {
                        lhs = lhs.RepeatTensor(1, rhs.Shape[1]);
                    }

                    if (rhs.Shape[0] == 1)
                    {
                        rhs = rhs.RepeatTensor(lhs.Shape[0], 1);
                    }
                }

                if (lhs.Shape[0] == 1 || rhs.Shape[1] == 1)
                {
                    if (lhs.Shape[0] == 1)
                    {
                        lhs = lhs.RepeatTensor(rhs.Shape[0], 1);
                    }

                    if (rhs.Shape[1] == 1)
                    {
                        rhs = rhs.RepeatTensor(1, lhs.Shape[1]);
                    }
                }
            }

            return (lhs, rhs);
        }

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
