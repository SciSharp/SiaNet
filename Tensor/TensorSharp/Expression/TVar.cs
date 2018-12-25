// ***********************************************************************
// Assembly         : TensorSharp
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="TVar.cs" company="TensorSharp">
//     Copyright (c) . All rights reserved.
// </copyright>
// <summary></summary>
// ***********************************************************************
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace TensorSharp.Expression
{
    /// <summary>
    /// Class TVar.
    /// </summary>
    public class TVar
    {
        /// <summary>
        /// The expression
        /// </summary>
        private TExpression expression;

        /// <summary>
        /// Initializes a new instance of the <see cref="TVar"/> class.
        /// </summary>
        /// <param name="expression">The expression.</param>
        public TVar(TExpression expression)
        {
            this.expression = expression;
        }

        // Note that this is not compatible with the implementation of operator ==
        // This merely checks for reference equality
        /// <summary>
        /// Determines whether the specified <see cref="System.Object" /> is equal to this instance.
        /// </summary>
        /// <param name="obj">The object to compare with the current object.</param>
        /// <returns><c>true</c> if the specified <see cref="System.Object" /> is equal to this instance; otherwise, <c>false</c>.</returns>
        public override bool Equals(object obj)
        {
            return base.Equals(obj);
        }

        /// <summary>
        /// Returns a hash code for this instance.
        /// </summary>
        /// <returns>A hash code for this instance, suitable for use in hashing algorithms and data structures like a hash table.</returns>
        public override int GetHashCode()
        {
            return base.GetHashCode();
        }


        /// <summary>
        /// Gets the expression.
        /// </summary>
        /// <value>The expression.</value>
        public TExpression Expression { get { return expression; } }

        /// <summary>
        /// Performs an implicit conversion from <see cref="Tensor"/> to <see cref="TVar"/>.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <returns>The result of the conversion.</returns>
        public static implicit operator TVar(Tensor value)
        {
            return new TVar(new TensorValueExpression(value));
        }

        /// <summary>
        /// Converts to scalar.
        /// </summary>
        /// <returns>SVar.</returns>
        public SVar ToScalar()
        {
            return new SVar(new DelegateScalarExpression(() =>
            {
                using (var result = this.Expression.Evaluate(null))
                {
                    return result.GetElementAsFloat(0);
                }
            }));
        }


        /// <summary>
        /// Fills the specified value.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <param name="allocator">The allocator.</param>
        /// <param name="type">The type.</param>
        /// <param name="sizes">The sizes.</param>
        /// <returns>TVar.</returns>
        public static TVar Fill(SVar value, IAllocator allocator, DType type, params long[] sizes) { return new TVar(new FillExpression(allocator, type, sizes, res => Ops.Fill(res, value.Evaluate()))); }


        /// <summary>
        /// Implements the - operator.
        /// </summary>
        /// <param name="src">The source.</param>
        /// <returns>The result of the operator.</returns>
        public static TVar operator -(TVar src) { return new TVar(new UnaryTensorExpression(src.Expression, Ops.Neg)); }

        /// <summary>
        /// Implements the + operator.
        /// </summary>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>The result of the operator.</returns>
        public static TVar operator +(TVar lhs, SVar rhs) { return new TVar(new BinaryTensorScalarExpression(lhs.Expression, rhs.Expression, Ops.Add)); }
        /// <summary>
        /// Implements the + operator.
        /// </summary>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>The result of the operator.</returns>
        public static TVar operator +(SVar lhs, TVar rhs) { return new TVar(new BinaryTensorScalarExpression(rhs.Expression, lhs.Expression, Ops.Add)); }
        /// <summary>
        /// Implements the * operator.
        /// </summary>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>The result of the operator.</returns>
        public static TVar operator *(TVar lhs, SVar rhs) { return new TVar(new BinaryTensorScalarExpression(lhs.Expression, rhs.Expression, Ops.Mul)); }
        /// <summary>
        /// Implements the * operator.
        /// </summary>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>The result of the operator.</returns>
        public static TVar operator *(SVar lhs, TVar rhs) { return new TVar(new BinaryTensorScalarExpression(rhs.Expression, lhs.Expression, Ops.Mul)); }

        /// <summary>
        /// Implements the - operator.
        /// </summary>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>The result of the operator.</returns>
        public static TVar operator -(SVar lhs, TVar rhs) { return new TVar(new BinaryScalarTensorExpression(lhs.Expression, rhs.Expression, Ops.Sub)); }
        /// <summary>
        /// Implements the - operator.
        /// </summary>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>The result of the operator.</returns>
        public static TVar operator -(TVar lhs, SVar rhs) { return new TVar(new BinaryTensorScalarExpression(lhs.Expression, rhs.Expression, Ops.Sub)); }
        /// <summary>
        /// Implements the / operator.
        /// </summary>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>The result of the operator.</returns>
        public static TVar operator /(SVar lhs, TVar rhs) { return new TVar(new BinaryScalarTensorExpression(lhs.Expression, rhs.Expression, Ops.Div)); }
        /// <summary>
        /// Implements the / operator.
        /// </summary>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>The result of the operator.</returns>
        public static TVar operator /(TVar lhs, SVar rhs) { return new TVar(new BinaryTensorScalarExpression(lhs.Expression, rhs.Expression, Ops.Div)); }

        /// <summary>
        /// Implements the + operator.
        /// </summary>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>The result of the operator.</returns>
        public static TVar operator +(TVar lhs, TVar rhs) { return new TVar(new BinaryTensorTensorExpression(lhs.Expression, rhs.Expression, Ops.Add)); }
        /// <summary>
        /// Implements the - operator.
        /// </summary>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>The result of the operator.</returns>
        public static TVar operator -(TVar lhs, TVar rhs) { return new TVar(new BinaryTensorTensorExpression(lhs.Expression, rhs.Expression, Ops.Sub)); }

        /// <summary>
        /// Implements the &gt; operator.
        /// </summary>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>The result of the operator.</returns>
        public static TVar operator >(TVar lhs, TVar rhs) { return new TVar(new BinaryTensorTensorExpression(lhs.Expression, rhs.Expression, Ops.GreaterThan)); }
        /// <summary>
        /// Implements the &lt; operator.
        /// </summary>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>The result of the operator.</returns>
        public static TVar operator <(TVar lhs, TVar rhs) { return new TVar(new BinaryTensorTensorExpression(lhs.Expression, rhs.Expression, Ops.LessThan)); }
        /// <summary>
        /// Implements the &gt;= operator.
        /// </summary>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>The result of the operator.</returns>
        public static TVar operator >=(TVar lhs, TVar rhs) { return new TVar(new BinaryTensorTensorExpression(lhs.Expression, rhs.Expression, Ops.GreaterOrEqual)); }
        /// <summary>
        /// Implements the &lt;= operator.
        /// </summary>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>The result of the operator.</returns>
        public static TVar operator <=(TVar lhs, TVar rhs) { return new TVar(new BinaryTensorTensorExpression(lhs.Expression, rhs.Expression, Ops.LessOrEqual)); }
        /// <summary>
        /// Implements the == operator.
        /// </summary>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>The result of the operator.</returns>
        public static TVar operator ==(TVar lhs, TVar rhs) { return new TVar(new BinaryTensorTensorExpression(lhs.Expression, rhs.Expression, Ops.EqualTo)); }
        /// <summary>
        /// Implements the != operator.
        /// </summary>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>The result of the operator.</returns>
        public static TVar operator !=(TVar lhs, TVar rhs) { return new TVar(new BinaryTensorTensorExpression(lhs.Expression, rhs.Expression, Ops.NotEqual)); }

        /// <summary>
        /// Implements the &gt; operator.
        /// </summary>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>The result of the operator.</returns>
        public static TVar operator >(TVar lhs, SVar rhs) { return new TVar(new BinaryTensorScalarExpression(lhs.Expression, rhs.Expression, Ops.GreaterThan)); }
        /// <summary>
        /// Implements the &lt; operator.
        /// </summary>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>The result of the operator.</returns>
        public static TVar operator <(TVar lhs, SVar rhs) { return new TVar(new BinaryTensorScalarExpression(lhs.Expression, rhs.Expression, Ops.LessThan)); }
        /// <summary>
        /// Implements the &gt;= operator.
        /// </summary>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>The result of the operator.</returns>
        public static TVar operator >=(TVar lhs, SVar rhs) { return new TVar(new BinaryTensorScalarExpression(lhs.Expression, rhs.Expression, Ops.GreaterOrEqual)); }
        /// <summary>
        /// Implements the &lt;= operator.
        /// </summary>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>The result of the operator.</returns>
        public static TVar operator <=(TVar lhs, SVar rhs) { return new TVar(new BinaryTensorScalarExpression(lhs.Expression, rhs.Expression, Ops.LessOrEqual)); }
        /// <summary>
        /// Implements the == operator.
        /// </summary>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>The result of the operator.</returns>
        public static TVar operator ==(TVar lhs, SVar rhs) { return new TVar(new BinaryTensorScalarExpression(lhs.Expression, rhs.Expression, Ops.EqualTo)); }
        /// <summary>
        /// Implements the != operator.
        /// </summary>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>The result of the operator.</returns>
        public static TVar operator !=(TVar lhs, SVar rhs) { return new TVar(new BinaryTensorScalarExpression(lhs.Expression, rhs.Expression, Ops.NotEqual)); }


        // Use symmetry of these Scalar/Tensor ops to share kernels with the Tensor/Scalar versions
        /// <summary>
        /// Implements the &gt; operator.
        /// </summary>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>The result of the operator.</returns>
        public static TVar operator >(SVar lhs, TVar rhs) { return new TVar(new BinaryTensorScalarExpression(rhs.Expression, lhs.Expression, Ops.LessThan)); }
        /// <summary>
        /// Implements the &lt; operator.
        /// </summary>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>The result of the operator.</returns>
        public static TVar operator <(SVar lhs, TVar rhs) { return new TVar(new BinaryTensorScalarExpression(rhs.Expression, lhs.Expression, Ops.GreaterThan)); }
        /// <summary>
        /// Implements the &gt;= operator.
        /// </summary>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>The result of the operator.</returns>
        public static TVar operator >=(SVar lhs, TVar rhs) { return new TVar(new BinaryTensorScalarExpression(rhs.Expression, lhs.Expression, Ops.LessOrEqual)); }
        /// <summary>
        /// Implements the &lt;= operator.
        /// </summary>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>The result of the operator.</returns>
        public static TVar operator <=(SVar lhs, TVar rhs) { return new TVar(new BinaryTensorScalarExpression(rhs.Expression, lhs.Expression, Ops.GreaterOrEqual)); }
        /// <summary>
        /// Implements the == operator.
        /// </summary>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>The result of the operator.</returns>
        public static TVar operator ==(SVar lhs, TVar rhs) { return new TVar(new BinaryTensorScalarExpression(rhs.Expression, lhs.Expression, Ops.EqualTo)); }
        /// <summary>
        /// Implements the != operator.
        /// </summary>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>The result of the operator.</returns>
        public static TVar operator !=(SVar lhs, TVar rhs) { return new TVar(new BinaryTensorScalarExpression(rhs.Expression, lhs.Expression, Ops.NotEqual)); }


        /// <summary>
        /// Dots the specified RHS.
        /// </summary>
        /// <param name="rhs">The RHS.</param>
        /// <returns>TVar.</returns>
        public TVar Dot(TVar rhs) { return new TVar(new BinaryTensorTensorExpression(this.Expression, rhs.Expression, Ops.Dot)); }

        // Returns beta * this + alpha * m1 * m2
        /// <summary>
        /// Addmms the specified beta.
        /// </summary>
        /// <param name="beta">The beta.</param>
        /// <param name="alpha">The alpha.</param>
        /// <param name="m1">The m1.</param>
        /// <param name="m2">The m2.</param>
        /// <returns>TVar.</returns>
        public TVar Addmm(float beta, float alpha, TVar m1, TVar m2) { return new TVar(new AddmmExpression(beta, this.Expression, alpha, m1.Expression, m2.Expression)); }

        /// <summary>
        /// cs the mul.
        /// </summary>
        /// <param name="rhs">The RHS.</param>
        /// <returns>TVar.</returns>
        public TVar CMul(TVar rhs) { return new TVar(new BinaryTensorTensorExpression(this.Expression, rhs.Expression, Ops.Mul)); }
        /// <summary>
        /// cs the div.
        /// </summary>
        /// <param name="rhs">The RHS.</param>
        /// <returns>TVar.</returns>
        public TVar CDiv(TVar rhs) { return new TVar(new BinaryTensorTensorExpression(this.Expression, rhs.Expression, Ops.Div)); }

        /// <summary>
        /// Divs the specified RHS.
        /// </summary>
        /// <param name="rhs">The RHS.</param>
        /// <returns>TVar.</returns>
        public TVar Div(SVar rhs) { return new TVar(new BinaryTensorScalarExpression(this.Expression, rhs.Expression, Ops.Div)); }


        /// <summary>
        /// Abses this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public TVar Abs() { return new TVar(new UnaryTensorExpression(this.Expression, Ops.Abs)); }
        /// <summary>
        /// Signs this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public TVar Sign() { return new TVar(new UnaryTensorExpression(this.Expression, Ops.Sign)); }

        /// <summary>
        /// SQRTs this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public TVar Sqrt() { return new TVar(new UnaryTensorExpression(this.Expression, Ops.Abs)); }
        /// <summary>
        /// Exps this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public TVar Exp() { return new TVar(new UnaryTensorExpression(this.Expression, Ops.Exp)); }
        /// <summary>
        /// Logs this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public TVar Log() { return new TVar(new UnaryTensorExpression(this.Expression, Ops.Log)); }
        /// <summary>
        /// Log1ps this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public TVar Log1p() { return new TVar(new UnaryTensorExpression(this.Expression, Ops.Log1p)); }
        /// <summary>
        /// Floors this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public TVar Floor() { return new TVar(new UnaryTensorExpression(this.Expression, Ops.Floor)); }
        /// <summary>
        /// Ceils this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public TVar Ceil() { return new TVar(new UnaryTensorExpression(this.Expression, Ops.Ceil)); }
        /// <summary>
        /// Rounds this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public TVar Round() { return new TVar(new UnaryTensorExpression(this.Expression, Ops.Round)); }
        /// <summary>
        /// Truncs this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public TVar Trunc() { return new TVar(new UnaryTensorExpression(this.Expression, Ops.Trunc)); }
        /// <summary>
        /// Fracs this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public TVar Frac() { return new TVar(new UnaryTensorExpression(this.Expression, Ops.Frac)); }

        /// <summary>
        /// Sins this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public TVar Sin() { return new TVar(new UnaryTensorExpression(this.Expression, Ops.Abs)); }
        /// <summary>
        /// Coses this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public TVar Cos() { return new TVar(new UnaryTensorExpression(this.Expression, Ops.Cos)); }
        /// <summary>
        /// Tans this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public TVar Tan() { return new TVar(new UnaryTensorExpression(this.Expression, Ops.Tan)); }

        /// <summary>
        /// Asins this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public TVar Asin() { return new TVar(new UnaryTensorExpression(this.Expression, Ops.Asin)); }
        /// <summary>
        /// Acoses this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public TVar Acos() { return new TVar(new UnaryTensorExpression(this.Expression, Ops.Acos)); }
        /// <summary>
        /// Atans this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public TVar Atan() { return new TVar(new UnaryTensorExpression(this.Expression, Ops.Atan)); }

        /// <summary>
        /// Sinhes this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public TVar Sinh() { return new TVar(new UnaryTensorExpression(this.Expression, Ops.Sinh)); }
        /// <summary>
        /// Coshes this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public TVar Cosh() { return new TVar(new UnaryTensorExpression(this.Expression, Ops.Cosh)); }
        /// <summary>
        /// Tanhes this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public TVar Tanh() { return new TVar(new UnaryTensorExpression(this.Expression, Ops.Tanh)); }

        /// <summary>
        /// Sigmoids this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public TVar Sigmoid() { return new TVar(new UnaryTensorExpression(this.Expression, Ops.Sigmoid)); }

        /// <summary>
        /// Pows the specified y.
        /// </summary>
        /// <param name="y">The y.</param>
        /// <returns>TVar.</returns>
        public TVar Pow(SVar y) { return new TVar(new BinaryTensorScalarExpression(this.Expression, y.Expression, Ops.Pow)); }
        /// <summary>
        /// Clamps the specified minimum.
        /// </summary>
        /// <param name="min">The minimum.</param>
        /// <param name="max">The maximum.</param>
        /// <returns>TVar.</returns>
        public TVar Clamp(SVar min, SVar max) { return new TVar(new UnaryTensorExpression(this.Expression, (res, src) => Ops.Clamp(res, src, min.Evaluate(), max.Evaluate()))); }

        /// <summary>
        /// Atan2s the specified y.
        /// </summary>
        /// <param name="y">The y.</param>
        /// <param name="x">The x.</param>
        /// <returns>TVar.</returns>
        public static TVar Atan2(TVar y, TVar x) { return new TVar(new BinaryTensorTensorExpression(x.Expression, y.Expression, Ops.Atan2)); }
        /// <summary>
        /// Lerps the specified a.
        /// </summary>
        /// <param name="a">a.</param>
        /// <param name="b">The b.</param>
        /// <param name="weight">The weight.</param>
        /// <returns>TVar.</returns>
        public static TVar Lerp(TVar a, TVar b, SVar weight) { return new TVar(new BinaryTensorTensorExpression(a.Expression, b.Expression, (res, aVal, bVal) => Ops.Lerp(res, aVal, bVal, weight.Evaluate()))); }


        /// <summary>
        /// Sums the specified dimension.
        /// </summary>
        /// <param name="dimension">The dimension.</param>
        /// <returns>TVar.</returns>
        public TVar Sum(int dimension) { return new TVar(new UnaryTensorExpression(this.Expression, (result, src) => Ops.Sum(result, src, dimension))); }
        /// <summary>
        /// Products the specified dimension.
        /// </summary>
        /// <param name="dimension">The dimension.</param>
        /// <returns>TVar.</returns>
        public TVar Prod(int dimension) { return new TVar(new UnaryTensorExpression(this.Expression, (result, src) => Ops.Prod(result, src, dimension))); }
        /// <summary>
        /// Determines the minimum of the parameters.
        /// </summary>
        /// <param name="dimension">The dimension.</param>
        /// <returns>TVar.</returns>
        public TVar Min(int dimension) { return new TVar(new UnaryTensorExpression(this.Expression, (result, src) => Ops.Min(result, src, dimension))); }
        /// <summary>
        /// Determines the maximun of the parameters.
        /// </summary>
        /// <param name="dimension">The dimension.</param>
        /// <returns>TVar.</returns>
        public TVar Max(int dimension) { return new TVar(new UnaryTensorExpression(this.Expression, (result, src) => Ops.Max(result, src, dimension))); }
        /// <summary>
        /// Argmins the specified dimension.
        /// </summary>
        /// <param name="dimension">The dimension.</param>
        /// <returns>TVar.</returns>
        public TVar Argmin(int dimension) { return new TVar(new UnaryTensorExpression(this.Expression, (result, src) => Ops.Argmin(result, src, dimension))); }
        /// <summary>
        /// Argmaxes the specified dimension.
        /// </summary>
        /// <param name="dimension">The dimension.</param>
        /// <returns>TVar.</returns>
        public TVar Argmax(int dimension) { return new TVar(new UnaryTensorExpression(this.Expression, (result, src) => Ops.Argmax(result, src, dimension))); }

        /// <summary>
        /// Means the specified dimension.
        /// </summary>
        /// <param name="dimension">The dimension.</param>
        /// <returns>TVar.</returns>
        public TVar Mean(int dimension) { return new TVar(new UnaryTensorExpression(this.Expression, (result, src) => Ops.Mean(result, src, dimension))); }
        /// <summary>
        /// Norms the specified dimension.
        /// </summary>
        /// <param name="dimension">The dimension.</param>
        /// <param name="value">The value.</param>
        /// <returns>TVar.</returns>
        public TVar Norm(int dimension, float value) { return new TVar(new UnaryTensorExpression(this.Expression, (result, src) => Ops.Norm(result, src, dimension, value))); }
        /// <summary>
        /// Standards the specified dimension.
        /// </summary>
        /// <param name="dimension">The dimension.</param>
        /// <param name="normByN">if set to <c>true</c> [norm by n].</param>
        /// <returns>TVar.</returns>
        public TVar Std(int dimension, bool normByN = false) { return new TVar(new UnaryTensorExpression(this.Expression, (result, src) => Ops.Std(result, src, dimension, normByN))); }
        /// <summary>
        /// Variables the specified dimension.
        /// </summary>
        /// <param name="dimension">The dimension.</param>
        /// <param name="normByN">if set to <c>true</c> [norm by n].</param>
        /// <returns>TVar.</returns>
        public TVar Var(int dimension, bool normByN = false) { return new TVar(new UnaryTensorExpression(this.Expression, (result, src) => Ops.Var(result, src, dimension, normByN))); }


        /// <summary>
        /// Sums all.
        /// </summary>
        /// <returns>TVar.</returns>
        public TVar SumAll() { return new TVar(new UnaryTensorExpression(this.Expression, (result, src) => Ops.SumAll(result, src))); }
        /// <summary>
        /// Products all.
        /// </summary>
        /// <returns>TVar.</returns>
        public TVar ProdAll() { return new TVar(new UnaryTensorExpression(this.Expression, (result, src) => Ops.ProdAll(result, src))); }
        /// <summary>
        /// Minimums all.
        /// </summary>
        /// <returns>TVar.</returns>
        public TVar MinAll() { return new TVar(new UnaryTensorExpression(this.Expression, (result, src) => Ops.MinAll(result, src))); }
        /// <summary>
        /// Maximums all.
        /// </summary>
        /// <returns>TVar.</returns>
        public TVar MaxAll() { return new TVar(new UnaryTensorExpression(this.Expression, (result, src) => Ops.MaxAll(result, src))); }

        /// <summary>
        /// Means all.
        /// </summary>
        /// <returns>TVar.</returns>
        public TVar MeanAll() { return new TVar(new UnaryTensorExpression(this.Expression, (result, src) => Ops.MeanAll(result, src))); }
        /// <summary>
        /// Variables all.
        /// </summary>
        /// <returns>TVar.</returns>
        public TVar VarAll() { return new TVar(new UnaryTensorExpression(this.Expression, (result, src) => Ops.VarAll(result, src))); }
        /// <summary>
        /// Standards all.
        /// </summary>
        /// <returns>TVar.</returns>
        public TVar StdAll() { return new TVar(new UnaryTensorExpression(this.Expression, (result, src) => Ops.StdAll(result, src))); }
        /// <summary>
        /// Norms all.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <returns>TVar.</returns>
        public TVar NormAll(float value) { return new TVar(new UnaryTensorExpression(this.Expression, (result, src) => Ops.NormAll(result, src, value))); }


        /// <summary>
        /// Gathers the specified dimension.
        /// </summary>
        /// <param name="dimension">The dimension.</param>
        /// <param name="indices">The indices.</param>
        /// <returns>TVar.</returns>
        public TVar Gather(int dimension, TVar indices) { return new TVar(new BinaryTensorTensorExpression(this.Expression, indices.Expression, (res, src, ind) => Ops.Gather(res, src, dimension, ind))); }
        /// <summary>
        /// Scatters the specified dimension.
        /// </summary>
        /// <param name="dimension">The dimension.</param>
        /// <param name="indices">The indices.</param>
        /// <returns>TVar.</returns>
        public TVar Scatter(int dimension, TVar indices) { return new TVar(new BinaryTensorTensorExpression(this.Expression, indices.Expression, (res, src, ind) => Ops.Scatter(res, src, dimension, ind))); }

        // Returns a copy of this tensor, with the given indices filled with the given value.
        // If, when this op is evaluated, the write target is the same tensor as this, then the copy is unnecessary and is skipped.
        /// <summary>
        /// Scatters the fill.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <param name="dimension">The dimension.</param>
        /// <param name="indices">The indices.</param>
        /// <returns>TVar.</returns>
        public TVar ScatterFill(SVar value, int dimension, TVar indices) { return new TVar(new ScatterFillExpression(this.Expression, value, dimension, indices.Expression)); }



        /// <summary>
        /// Randoms the uniform.
        /// </summary>
        /// <param name="seedSource">The seed source.</param>
        /// <param name="min">The minimum.</param>
        /// <param name="max">The maximum.</param>
        /// <param name="allocator">The allocator.</param>
        /// <param name="type">The type.</param>
        /// <param name="sizes">The sizes.</param>
        /// <returns>TVar.</returns>
        public static TVar RandomUniform(SeedSource seedSource, SVar min, SVar max, IAllocator allocator, DType type, params long[] sizes)
        {
            return new TVar(new FillExpression(allocator, type, sizes, res => Ops.RandomUniform(res, seedSource, min.Evaluate(), max.Evaluate())));
        }

        /// <summary>
        /// Randoms the normal.
        /// </summary>
        /// <param name="seedSource">The seed source.</param>
        /// <param name="mean">The mean.</param>
        /// <param name="stdv">The STDV.</param>
        /// <param name="allocator">The allocator.</param>
        /// <param name="type">The type.</param>
        /// <param name="sizes">The sizes.</param>
        /// <returns>TVar.</returns>
        public static TVar RandomNormal(SeedSource seedSource, SVar mean, SVar stdv, IAllocator allocator, DType type, params long[] sizes)
        {
            return new TVar(new FillExpression(allocator, type, sizes, res => Ops.RandomNormal(res, seedSource, mean.Evaluate(), stdv.Evaluate())));
        }

        /// <summary>
        /// Randoms the exponential.
        /// </summary>
        /// <param name="seedSource">The seed source.</param>
        /// <param name="lambda">The lambda.</param>
        /// <param name="allocator">The allocator.</param>
        /// <param name="type">The type.</param>
        /// <param name="sizes">The sizes.</param>
        /// <returns>TVar.</returns>
        public static TVar RandomExponential(SeedSource seedSource, SVar lambda, IAllocator allocator, DType type, params long[] sizes)
        {
            return new TVar(new FillExpression(allocator, type, sizes, res => Ops.RandomExponential(res, seedSource, lambda.Evaluate())));
        }

        /// <summary>
        /// Randoms the cauchy.
        /// </summary>
        /// <param name="seedSource">The seed source.</param>
        /// <param name="median">The median.</param>
        /// <param name="sigma">The sigma.</param>
        /// <param name="allocator">The allocator.</param>
        /// <param name="type">The type.</param>
        /// <param name="sizes">The sizes.</param>
        /// <returns>TVar.</returns>
        public static TVar RandomCauchy(SeedSource seedSource, SVar median, SVar sigma, IAllocator allocator, DType type, params long[] sizes)
        {
            return new TVar(new FillExpression(allocator, type, sizes, res => Ops.RandomCauchy(res, seedSource, median.Evaluate(), sigma.Evaluate())));
        }

        /// <summary>
        /// Randoms the log normal.
        /// </summary>
        /// <param name="seedSource">The seed source.</param>
        /// <param name="mean">The mean.</param>
        /// <param name="stdv">The STDV.</param>
        /// <param name="allocator">The allocator.</param>
        /// <param name="type">The type.</param>
        /// <param name="sizes">The sizes.</param>
        /// <returns>TVar.</returns>
        public static TVar RandomLogNormal(SeedSource seedSource, SVar mean, SVar stdv, IAllocator allocator, DType type, params long[] sizes)
        {
            return new TVar(new FillExpression(allocator, type, sizes, res => Ops.RandomLogNormal(res, seedSource, mean.Evaluate(), stdv.Evaluate())));
        }

        /// <summary>
        /// Randoms the geometric.
        /// </summary>
        /// <param name="seedSource">The seed source.</param>
        /// <param name="p">The p.</param>
        /// <param name="allocator">The allocator.</param>
        /// <param name="type">The type.</param>
        /// <param name="sizes">The sizes.</param>
        /// <returns>TVar.</returns>
        public static TVar RandomGeometric(SeedSource seedSource, SVar p, IAllocator allocator, DType type, params long[] sizes)
        {
            return new TVar(new FillExpression(allocator, type, sizes, res => Ops.RandomGeometric(res, seedSource, p.Evaluate())));
        }

        /// <summary>
        /// Randoms the bernoulli.
        /// </summary>
        /// <param name="seedSource">The seed source.</param>
        /// <param name="p">The p.</param>
        /// <param name="allocator">The allocator.</param>
        /// <param name="type">The type.</param>
        /// <param name="sizes">The sizes.</param>
        /// <returns>TVar.</returns>
        public static TVar RandomBernoulli(SeedSource seedSource, SVar p, IAllocator allocator, DType type, params long[] sizes)
        {
            return new TVar(new FillExpression(allocator, type, sizes, res => Ops.RandomBernoulli(res, seedSource, p.Evaluate())));
        }



        /// <summary>
        /// Ases the type.
        /// </summary>
        /// <param name="elementType">Type of the element.</param>
        /// <returns>TVar.</returns>
        public TVar AsType(DType elementType)
        {
            return new TVar(new AsTypeExpression(this.Expression, elementType));
        }

        /// <summary>
        /// Converts to device.
        /// </summary>
        /// <param name="device">The device.</param>
        /// <returns>TVar.</returns>
        public TVar ToDevice(IAllocator device)
        {
            return new TVar(new ToDeviceExpression(this.Expression, device));
        }

        /// <summary>
        /// Evaluates this instance.
        /// </summary>
        /// <returns>Tensor.</returns>
        public Tensor Evaluate()
        {
            return expression.Evaluate(null);
        }

        /// <summary>
        /// Evaluates the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <exception cref="InvalidOperationException">cannot write to given result - it is not a valid lvalue</exception>
        public void Evaluate(TVar result)
        {
            if (!result.Expression.IsValidLvalue)
                throw new InvalidOperationException("cannot write to given result - it is not a valid lvalue");

            using (var res = result.Expression.Evaluate(null))
            {
                this.expression.Evaluate(res);
            }
        }

        /// <summary>
        /// Froms the array.
        /// </summary>
        /// <param name="array">The array.</param>
        /// <param name="allocator">The allocator.</param>
        /// <returns>TVar.</returns>
        public static TVar FromArray(Array array, IAllocator allocator)
        {
            return new TVar(new FromArrayExpression(allocator, array));
        }



        /// <summary>
        /// Selects the specified dimension.
        /// </summary>
        /// <param name="dimension">The dimension.</param>
        /// <param name="index">The index.</param>
        /// <returns>TVar.</returns>
        public TVar Select(int dimension, long index) { return new TVar(new ViewExpression(this.Expression, src => src.Select(dimension, index))); }
        /// <summary>
        /// Narrows the specified dimension.
        /// </summary>
        /// <param name="dimension">The dimension.</param>
        /// <param name="startIndex">The start index.</param>
        /// <param name="size">The size.</param>
        /// <returns>TVar.</returns>
        public TVar Narrow(int dimension, long startIndex, long size) { return new TVar(new ViewExpression(this.Expression, src => src.Narrow(dimension, startIndex, size))); }
        /// <summary>
        /// Transposes this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public TVar Transpose() { return new TVar(new ViewExpression(this.Expression, src => src.Transpose())); }
        /// <summary>
        /// Transposes the specified dim1.
        /// </summary>
        /// <param name="dim1">The dim1.</param>
        /// <param name="dim2">The dim2.</param>
        /// <returns>TVar.</returns>
        public TVar Transpose(int dim1, int dim2) { return new TVar(new ViewExpression(this.Expression, src => src.Transpose(dim1, dim2))); }
        /// <summary>
        /// Permutes the specified dims.
        /// </summary>
        /// <param name="dims">The dims.</param>
        /// <returns>TVar.</returns>
        public TVar Permute(params int[] dims) { return new TVar(new ViewExpression(this.Expression, src => src.Permute(dims))); }
        /// <summary>
        /// Views the specified sizes.
        /// </summary>
        /// <param name="sizes">The sizes.</param>
        /// <returns>TVar.</returns>
        public TVar View(params long[] sizes) { return new TVar(new ViewExpression(this.Expression, src => src.View(sizes))); }
        /// <summary>
        /// Expands the specified sizes.
        /// </summary>
        /// <param name="sizes">The sizes.</param>
        /// <returns>TVar.</returns>
        public TVar Expand(params long[] sizes) { return new TVar(new ViewExpression(this.Expression, src => src.Expand(sizes))); }
        /// <summary>
        /// Squeezes this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public TVar Squeeze() { return new TVar(new ViewExpression(this.Expression, src => src.Squeeze())); }

        public void Print()
        {
            Console.WriteLine(Evaluate().Format());
        }
    }

    /// <summary>
    /// Class TensorVarExtensions.
    /// </summary>
    public static class TensorVarExtensions
    {
        /// <summary>
        /// ts the variable.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <returns>TVar.</returns>
        public static TVar TVar(this Tensor value)
        {
            return new Expression.TVar(new TensorValueExpression(value));
        }

      
    }
}
