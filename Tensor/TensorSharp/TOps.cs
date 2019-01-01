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
    public class TOps
    {
        /// <summary>
        /// Creates new contiguous.
        /// </summary>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static Tensor NewContiguous(Tensor src)
        {
            var result = new Tensor(src.Allocator, src.ElementType, (long[])src.Shape.Clone());
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
        public static Tensor Concat( int dimension, params Tensor[] inputs)
        {
            return TensorConcatenation.Concat(null, dimension, inputs);
        }

        /// <summary>
        /// Fills the one hot.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="labelCount">The label count.</param>
        /// <param name="labels">The labels.</param>
        public static void FillOneHot(Tensor src, int labelCount, int[] labels)
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
                using (var cpuResult = new Tensor(cpuAlloc, src.ElementType, src.Shape))
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
        private static void DoFillOneHot(Tensor src, int labelCount, int[] labels)
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
        public static void Copy(Tensor result, Tensor src) { OpRegistry.Invoke("copy", result, src); }

        /// <summary>
        /// Fills the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="value">The value.</param>
        public static void Fill(Tensor src, float value) { OpRegistry.Invoke("fill", src, value); }

        /// <summary>
        /// Dots the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Dot( Tensor lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("dot", null, lhs, rhs); }
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
        public static Tensor Addmm( float beta, Tensor src, float alpha, Tensor m1, Tensor m2) { return (Tensor)OpRegistry.Invoke("addmm", null, beta, src, alpha, m1, m2); }

        /// <summary>
        /// Abses the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Abs( Tensor src) { return (Tensor)OpRegistry.Invoke("abs", null, src); }

        /// <summary>
        /// Negs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Neg( Tensor src) { return (Tensor)OpRegistry.Invoke("neg", null, src); }

        /// <summary>
        /// Signs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Sign( Tensor src) { return (Tensor)OpRegistry.Invoke("sign", null, src); }

        /// <summary>
        /// SQRTs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Sqrt( Tensor src) { return (Tensor)OpRegistry.Invoke("sqrt", null, src); }
        /// <summary>
        /// Exps the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Exp( Tensor src) { return (Tensor)OpRegistry.Invoke("exp", null, src); }
        /// <summary>
        /// Logs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Log( Tensor src) { return (Tensor)OpRegistry.Invoke("log", null, src); }
        /// <summary>
        /// Log1ps the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Log1p( Tensor src) { return (Tensor)OpRegistry.Invoke("log1p", null, src); }
        /// <summary>
        /// Floors the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Floor( Tensor src) { return (Tensor)OpRegistry.Invoke("floor", null, src); }
        /// <summary>
        /// Ceils the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Ceil( Tensor src) { return (Tensor)OpRegistry.Invoke("ceil", null, src); }
        /// <summary>
        /// Rounds the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Round( Tensor src) { return (Tensor)OpRegistry.Invoke("round", null, src); }
        /// <summary>
        /// Truncs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Trunc( Tensor src) { return (Tensor)OpRegistry.Invoke("trunc", null, src); }
        /// <summary>
        /// Fracs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Frac( Tensor src) { return (Tensor)OpRegistry.Invoke("frac", null, src); }

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
        public static Tensor Cos( Tensor src) { return (Tensor)OpRegistry.Invoke("cos", null, src); }
        /// <summary>
        /// Tans the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Tan( Tensor src) { return (Tensor)OpRegistry.Invoke("tan", null, src); }

        /// <summary>
        /// Asins the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Asin( Tensor src) { return (Tensor)OpRegistry.Invoke("asin", null, src); }
        /// <summary>
        /// Acoses the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Acos( Tensor src) { return (Tensor)OpRegistry.Invoke("acos", null, src); }
        /// <summary>
        /// Atans the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Atan( Tensor src) { return (Tensor)OpRegistry.Invoke("atan", null, src); }

        /// <summary>
        /// Sinhes the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Sinh( Tensor src) { return (Tensor)OpRegistry.Invoke("sinh", null, src); }
        /// <summary>
        /// Coshes the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Cosh( Tensor src) { return (Tensor)OpRegistry.Invoke("cosh", null, src); }
        /// <summary>
        /// Tanhes the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Tanh( Tensor src) { return (Tensor)OpRegistry.Invoke("tanh", null, src); }

        /// <summary>
        /// Sigmoids the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Sigmoid( Tensor src) { return (Tensor)OpRegistry.Invoke("sigmoid", null, src); }


        /// <summary>
        /// Atan2s the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="srcY">The source y.</param>
        /// <param name="srcX">The source x.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Atan2( Tensor srcY, Tensor srcX) { return (Tensor)OpRegistry.Invoke("atan2", null, srcY, srcX); }
        /// <summary>
        /// Pows the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="value">The value.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Pow( Tensor src, float value) { return (Tensor)OpRegistry.Invoke("pow", null, src, value); }

        /// <summary>
        /// Squares the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="value">The value.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Square(Tensor src) { return (Tensor)OpRegistry.Invoke("pow", null, src, 2); }

        /// <summary>
        /// Tpows the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="value">The value.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Tpow( float value, Tensor src) { return (Tensor)OpRegistry.Invoke("tpow", null, value, src); }


        /// <summary>
        /// Lerps the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="srcA">The source a.</param>
        /// <param name="srcB">The source b.</param>
        /// <param name="weight">The weight.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Lerp( Tensor srcA, Tensor srcB, float weight) { return (Tensor)OpRegistry.Invoke("lerp", null, srcA, srcB); }
        /// <summary>
        /// Clamps the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="min">The minimum.</param>
        /// <param name="max">The maximum.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Clip( Tensor src, float min, float max) { return (Tensor)OpRegistry.Invoke("clamp", null, src, min, max); }


        /// <summary>
        /// Adds the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Add( Tensor lhs, float rhs) { return (Tensor)OpRegistry.Invoke("addv", null, lhs, rhs); }
        /// <summary>
        /// Subs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Sub( Tensor lhs, float rhs) { return (Tensor)OpRegistry.Invoke("subv", null, lhs, rhs); }
        /// <summary>
        /// Subs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Sub( float lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("rsubv", null, lhs, rhs); }
        /// <summary>
        /// Muls the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Mul( Tensor lhs, float rhs) { return (Tensor)OpRegistry.Invoke("mulv", null, lhs, rhs); }
        /// <summary>
        /// Divs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Div( Tensor lhs, float rhs) { return (Tensor)OpRegistry.Invoke("divv", null, lhs, rhs); }
        /// <summary>
        /// Divs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Div( float lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("rdivv", null, lhs, rhs); }
        /// <summary>
        /// Mods the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Mod( Tensor lhs, float rhs) { return (Tensor)OpRegistry.Invoke("modv", null, lhs, rhs); }

        /// <summary>
        /// Greaters the than.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static Tensor GreaterThan( Tensor lhs, float rhs) { return (Tensor)OpRegistry.Invoke("gtValue", null, lhs, rhs); }
        /// <summary>
        /// Lesses the than.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static Tensor LessThan( Tensor lhs, float rhs) { return (Tensor)OpRegistry.Invoke("ltValue", null, lhs, rhs); }
        /// <summary>
        /// Greaters the or equal.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static Tensor GreaterOrEqual( Tensor lhs, float rhs) { return (Tensor)OpRegistry.Invoke("geValue", null, lhs, rhs); }
        /// <summary>
        /// Lesses the or equal.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static Tensor LessOrEqual( Tensor lhs, float rhs) { return (Tensor)OpRegistry.Invoke("leValue", null, lhs, rhs); }
        /// <summary>
        /// Equals to.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static Tensor EqualTo( Tensor lhs, float rhs) { return (Tensor)OpRegistry.Invoke("eqValue", null, lhs, rhs); }
        /// <summary>
        /// Nots the equal.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static Tensor NotEqual( Tensor lhs, float rhs) { return (Tensor)OpRegistry.Invoke("neValue", null, lhs, rhs); }

        /// <summary>
        /// Adds the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Add( Tensor lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("addt", null, lhs, rhs); }
        /// <summary>
        /// Subs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Sub( Tensor lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("subt", null, lhs, rhs); }
        /// <summary>
        /// Muls the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Mul( Tensor lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("mult", null, lhs, rhs); }
        /// <summary>
        /// Divs the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Div( Tensor lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("divt", null, lhs, rhs); }
        /// <summary>
        /// Mods the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Mod( Tensor lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("modt", null, lhs, rhs); }

        /// <summary>
        /// Greaters the than.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static Tensor GreaterThan( Tensor lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("gtTensor", null, lhs, rhs); }
        /// <summary>
        /// Lesses the than.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static Tensor LessThan( Tensor lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("ltTensor", null, lhs, rhs); }
        /// <summary>
        /// Greaters the or equal.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static Tensor GreaterOrEqual( Tensor lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("geTensor", null, lhs, rhs); }
        /// <summary>
        /// Lesses the or equal.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static Tensor LessOrEqual( Tensor lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("leTensor", null, lhs, rhs); }
        /// <summary>
        /// Equals to.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static Tensor EqualTo( Tensor lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("eqTensor", null, lhs, rhs); }
        /// <summary>
        /// Nots the equal.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>Tensor.</returns>
        public static Tensor NotEqual( Tensor lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("neTensor", null, lhs, rhs); }


        /// <summary>
        /// Sums the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Sum( Tensor src, int dimension) { return (Tensor)OpRegistry.Invoke("sum", null, src, dimension); }
        /// <summary>
        /// Products the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Prod( Tensor src, int dimension) { return (Tensor)OpRegistry.Invoke("prod", null, src, dimension); }
        /// <summary>
        /// Determines the minimum of the parameters.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Min( Tensor src, int dimension) { return (Tensor)OpRegistry.Invoke("min", null, src, dimension); }
        /// <summary>
        /// Determines the maximun of the parameters.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Max( Tensor src, int dimension) { return (Tensor)OpRegistry.Invoke("max", null, src, dimension); }
        /// <summary>
        /// Argmins the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Argmin( Tensor src, int dimension) { return (Tensor)OpRegistry.Invoke("argmin", null, src, dimension); }
        /// <summary>
        /// Argmaxes the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Argmax( Tensor src, int dimension) { return (Tensor)OpRegistry.Invoke("argmax", null, src, dimension); }

        /// <summary>
        /// Means the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Mean( Tensor src, int dimension) { return (Tensor)OpRegistry.Invoke("mean", null, src, dimension); }
        /// <summary>
        /// Norms the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <param name="value">The value.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Norm( Tensor src, int dimension, float value) { return (Tensor)OpRegistry.Invoke("norm", null, src, dimension, value); }
        /// <summary>
        /// Standards the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <param name="normByN">if set to <c>true</c> [norm by n].</param>
        /// <returns>Tensor.</returns>
        public static Tensor Std( Tensor src, int dimension, bool normByN) { return (Tensor)OpRegistry.Invoke("std", null, src, dimension, normByN); }
        /// <summary>
        /// Variables the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dimension">The dimension.</param>
        /// <param name="normByN">if set to <c>true</c> [norm by n].</param>
        /// <returns>Tensor.</returns>
        public static Tensor Var( Tensor src, int dimension, bool normByN) { return (Tensor)OpRegistry.Invoke("var", null, src, dimension, normByN); }

        /// <summary>
        /// Sums all.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Sum( Tensor src) { return (Tensor)OpRegistry.Invoke("sumall", null, src); }
        /// <summary>
        /// Products all.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Prod( Tensor src) { return (Tensor)OpRegistry.Invoke("prodall", null, src); }
        /// <summary>
        /// Minimums all.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Min( Tensor src) { return (Tensor)OpRegistry.Invoke("minall", null, src); }
        /// <summary>
        /// Maximums all.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Max( Tensor src) { return (Tensor)OpRegistry.Invoke("maxall", null, src); }

        /// <summary>
        /// Means all.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Mean( Tensor src) { return (Tensor)OpRegistry.Invoke("meanall", null, src); }
        /// <summary>
        /// Norms all.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="value">The value.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Norm( Tensor src, float value) { return (Tensor)OpRegistry.Invoke("normall", null, src, value); }
        /// <summary>
        /// Standards all.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Std( Tensor src) { return (Tensor)OpRegistry.Invoke("stdall", null, src); }
        /// <summary>
        /// Variables all.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Var( Tensor src) { return (Tensor)OpRegistry.Invoke("varall", null, src); }


        /// <summary>
        /// Sums all.
        /// </summary>
        /// <param name="src">The source.</param>
        /// <returns>System.Single.</returns>
        public static float SumF(Tensor src) { using (var resultTensor = Sum(src)) { return resultTensor.GetElementAsFloat(0); } }
        /// <summary>
        /// Products all.
        /// </summary>
        /// <param name="src">The source.</param>
        /// <returns>System.Single.</returns>
        public static float ProdF(Tensor src) { using (var resultTensor = Prod(src)) { return resultTensor.GetElementAsFloat(0); } }
        /// <summary>
        /// Minimums all.
        /// </summary>
        /// <param name="src">The source.</param>
        /// <returns>System.Single.</returns>
        public static float MinF(Tensor src) { using (var resultTensor = Min(src)) { return resultTensor.GetElementAsFloat(0); } }
        /// <summary>
        /// Maximums all.
        /// </summary>
        /// <param name="src">The source.</param>
        /// <returns>System.Single.</returns>
        public static float MaxF(Tensor src) { using (var resultTensor = Max(src)) { return resultTensor.GetElementAsFloat(0); } }

        /// <summary>
        /// Means all.
        /// </summary>
        /// <param name="src">The source.</param>
        /// <returns>System.Single.</returns>
        public static float MeanF(Tensor src) { using (var resultTensor = Mean(src)) { return resultTensor.GetElementAsFloat(0); } }
        /// <summary>
        /// Variables all.
        /// </summary>
        /// <param name="src">The source.</param>
        /// <returns>System.Single.</returns>
        public static float VarF(Tensor src) { using (var resultTensor = Var(src)) { return resultTensor.GetElementAsFloat(0); } }
        /// <summary>
        /// Standards all.
        /// </summary>
        /// <param name="src">The source.</param>
        /// <returns>System.Single.</returns>
        public static float StdF(Tensor src) { using (var resultTensor = Std(src)) { return resultTensor.GetElementAsFloat(0); } }
        /// <summary>
        /// Norms all.
        /// </summary>
        /// <param name="src">The source.</param>
        /// <param name="value">The value.</param>
        /// <returns>System.Single.</returns>
        public static float NormF(Tensor src, float value) { using (var resultTensor = Norm(src, value)) { return resultTensor.GetElementAsFloat(0); } }


        /// <summary>
        /// Indexes the select.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dim">The dim.</param>
        /// <param name="indices">The indices.</param>
        /// <returns>Tensor.</returns>
        public static Tensor IndexSelect( Tensor src, int dim, Tensor indices) { return (Tensor)OpRegistry.Invoke("index_select", null, src, dim, indices); }
        /// <summary>
        /// Gathers the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dim">The dim.</param>
        /// <param name="indices">The indices.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Gather( Tensor src, int dim, Tensor indices) { return (Tensor)OpRegistry.Invoke("gather", null, src, dim, indices); }
        /// <summary>
        /// Scatters the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="src">The source.</param>
        /// <param name="dim">The dim.</param>
        /// <param name="indices">The indices.</param>
        /// <returns>Tensor.</returns>
        public static Tensor Scatter( Tensor src, int dim, Tensor indices) { return (Tensor)OpRegistry.Invoke("scatter", null, src, dim, indices); }
        /// <summary>
        /// Scatters the fill.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="value">The value.</param>
        /// <param name="dim">The dim.</param>
        /// <param name="indices">The indices.</param>
        /// <returns>Tensor.</returns>
        public static Tensor ScatterFill( float value, int dim, Tensor indices) { return (Tensor)OpRegistry.Invoke("scatter_fill", null, value, dim, indices); }


        public static Tensor Maximum(Tensor a, Tensor b)
        {
            var t1 = (a >= b);
            var t2 = (a > b);

            return (t1 * a + t2 * b);
        }

        public static Tensor Maximum(Tensor a, float b)
        {
            var b_t = new Tensor(a.Allocator, a.ElementType, a.Shape);
            TOps.Fill(b_t, b);
            return Maximum(a, b_t);
        }

        public static Tensor Maximum(float a, Tensor b)
        {
            var a_t = new Tensor(b.Allocator, b.ElementType, b.Shape);
            TOps.Fill(a_t, a);
            return Maximum(a_t, b);
        }

        public static Tensor Softplus(Tensor x)
        {
            return Log((Exp(x) + 1));
        }

        public static Tensor Softmax(Tensor x)
        {
            long[] shape = x.Shape;
            List<float> data = new List<float>();
            for (long i = 0; i < shape[0]; i++)
            {
                var s_x = x.Select(0, i);
                var exp = Exp(s_x);
                var sum = SumF(exp);
                var s_t = (exp / sum).View(1, shape[1]);
                data.AddRange(s_t.ToArray().Cast<float>());
            }

            x.CopyFrom(data.ToArray());
            return x;
        }

        public static Tensor L2Normalize(Tensor x, int axis = -1)
        {
            Tensor y = null;
            if (axis == -1)
            {
                y = Max(Sum(Square(x)));
            }
            else
            {
                y = Max(Sum(Square(x), axis), axis);
            }

            return x / Sqrt(y);
        }

        public static Tensor Tile(Tensor x, long repetitions)
        {
            long[] shape = new long[x.DimensionCount];
            for (int i = 0; i < shape.Length; i++)
            {
                shape[i] = 1;
            }

            shape[shape.Length - 1] = repetitions;

            return x.RepeatTensor(shape);
        }

        public static Tensor Repeat(Tensor x, int reps)
        {
            x = x.View(x.ElementCount(), 1).Tile(reps);
            return x.View(1, x.ElementCount());
        }

        public static ValueTuple<Tensor, Tensor> Broadcast(Tensor lhs, Tensor rhs)
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
        public static void RandomUniform( SeedSource seedSource, float min, float max) { OpRegistry.Invoke("random_uniform", null, GetSeed(seedSource), min, max); }
        /// <summary>
        /// Randoms the normal.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="seedSource">The seed source.</param>
        /// <param name="mean">The mean.</param>
        /// <param name="stdv">The STDV.</param>
        public static void RandomNormal( SeedSource seedSource, float mean, float stdv) { OpRegistry.Invoke("random_normal", null, GetSeed(seedSource), mean, stdv); }
        /// <summary>
        /// Randoms the exponential.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="seedSource">The seed source.</param>
        /// <param name="lambda">The lambda.</param>
        public static void RandomExponential( SeedSource seedSource, float lambda) { OpRegistry.Invoke("random_exponential", null, GetSeed(seedSource), lambda); }
        /// <summary>
        /// Randoms the cauchy.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="seedSource">The seed source.</param>
        /// <param name="median">The median.</param>
        /// <param name="sigma">The sigma.</param>
        public static void RandomCauchy( SeedSource seedSource, float median, float sigma) { OpRegistry.Invoke("random_cauchy", null, GetSeed(seedSource), median, sigma); }
        /// <summary>
        /// Randoms the log normal.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="seedSource">The seed source.</param>
        /// <param name="mean">The mean.</param>
        /// <param name="stdv">The STDV.</param>
        public static void RandomLogNormal( SeedSource seedSource, float mean, float stdv) { OpRegistry.Invoke("random_lognormal", null, GetSeed(seedSource), mean, stdv); }
        /// <summary>
        /// Randoms the geometric.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="seedSource">The seed source.</param>
        /// <param name="p">The p.</param>
        public static void RandomGeometric( SeedSource seedSource, float p) { OpRegistry.Invoke("random_geometric", null, GetSeed(seedSource), p); }
        /// <summary>
        /// Randoms the bernoulli.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <param name="seedSource">The seed source.</param>
        /// <param name="p">The p.</param>
        public static void RandomBernoulli( SeedSource seedSource, float p) { OpRegistry.Invoke("random_bernoulli", null, GetSeed(seedSource), p); }

        
    }
}
