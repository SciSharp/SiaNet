// ***********************************************************************
// Assembly         : SiaNet.Backend.TensorSharp
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="TVar.cs" company="SiaNet.Backend.TensorSharp">
//     Copyright (c) . All rights reserved.
// </copyright>
// <summary></summary>
// ***********************************************************************
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace SiaNet.Backend.TensorSharp.Expression
{
    /// <summary>
    /// Class TVar.
    /// </summary>
    public class Variable
    {
        /// <summary>
        /// The expression
        /// </summary>
        private VariableExpression expression;

        public long[] Shape
        {
            get
            {
                return Evaluate().Shape;
            }
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Variable"/> class.
        /// </summary>
        /// <param name="expression">The expression.</param>
        public Variable(VariableExpression expression)
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
        public VariableExpression Expression { get { return expression; } }

        /// <summary>
        /// Performs an implicit conversion from <see cref="NDArray"/> to <see cref="Variable"/>.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <returns>The result of the conversion.</returns>
        public static implicit operator Variable(NDArray value)
        {
            return new Variable(new TensorValueExpression(value));
        }

        /// <summary>
        /// Converts to scalar.
        /// </summary>
        /// <returns>SVar.</returns>
        public ScalarVar ToScalar()
        {
            return new ScalarVar(new DelegateScalarExpression(() =>
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
        public static Variable Fill(ScalarVar value, IAllocator allocator, DType type, params long[] sizes) { return new Variable(new FillExpression(allocator, type, sizes, res => Ops.Fill(res, value.Evaluate()))); }

        /// <summary>
        /// Implements the - operator.
        /// </summary>
        /// <param name="src">The source.</param>
        /// <returns>The result of the operator.</returns>
        public static Variable operator -(Variable src) { return new Variable(new UnaryTensorExpression(src.Expression, Ops.Neg)); }

        /// <summary>
        /// Implements the + operator.
        /// </summary>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>The result of the operator.</returns>
        public static Variable operator +(Variable lhs, ScalarVar rhs) { return new Variable(new BinaryTensorScalarExpression(lhs.Expression, rhs.Expression, Ops.Add)); }
        /// <summary>
        /// Implements the + operator.
        /// </summary>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>The result of the operator.</returns>
        public static Variable operator +(ScalarVar lhs, Variable rhs) { return new Variable(new BinaryTensorScalarExpression(rhs.Expression, lhs.Expression, Ops.Add)); }

        /// <summary>
        /// Implements the * operator.
        /// </summary>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>The result of the operator.</returns>
        public static Variable operator *(Variable lhs, ScalarVar rhs) { return new Variable(new BinaryTensorScalarExpression(lhs.Expression, rhs.Expression, Ops.Mul)); }

        /// <summary>
        /// Implements the * operator.
        /// </summary>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>The result of the operator.</returns>
        public static Variable operator *(ScalarVar lhs, Variable rhs) { return new Variable(new BinaryTensorScalarExpression(rhs.Expression, lhs.Expression, Ops.Mul)); }

        public static Variable operator *(Variable lhs, Variable rhs) { return new Variable(new BinaryTensorTensorExpression(lhs.Expression, rhs.Expression, Ops.Mul)); }

        /// <summary>
        /// Implements the - operator.
        /// </summary>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>The result of the operator.</returns>
        public static Variable operator -(ScalarVar lhs, Variable rhs) { return new Variable(new BinaryScalarTensorExpression(lhs.Expression, rhs.Expression, Ops.Sub)); }
        /// <summary>
        /// Implements the - operator.
        /// </summary>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>The result of the operator.</returns>
        public static Variable operator -(Variable lhs, ScalarVar rhs) { return new Variable(new BinaryTensorScalarExpression(lhs.Expression, rhs.Expression, Ops.Sub)); }
        /// <summary>
        /// Implements the / operator.
        /// </summary>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>The result of the operator.</returns>
        public static Variable operator /(ScalarVar lhs, Variable rhs) { return new Variable(new BinaryScalarTensorExpression(lhs.Expression, rhs.Expression, Ops.Div)); }
        /// <summary>
        /// Implements the / operator.
        /// </summary>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>The result of the operator.</returns>
        public static Variable operator /(Variable lhs, ScalarVar rhs) { return new Variable(new BinaryTensorScalarExpression(lhs.Expression, rhs.Expression, Ops.Div)); }

        /// <summary>
        /// Implements the + operator.
        /// </summary>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>The result of the operator.</returns>
        public static Variable operator +(Variable lhs, Variable rhs) { return new Variable(new BinaryTensorTensorExpression(lhs.Expression, rhs.Expression, Ops.Add)); }
        /// <summary>
        /// Implements the - operator.
        /// </summary>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>The result of the operator.</returns>
        public static Variable operator -(Variable lhs, Variable rhs) { return new Variable(new BinaryTensorTensorExpression(lhs.Expression, rhs.Expression, Ops.Sub)); }

        /// <summary>
        /// Implements the &gt; operator.
        /// </summary>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>The result of the operator.</returns>
        public static Variable operator >(Variable lhs, Variable rhs) { return new Variable(new BinaryTensorTensorExpression(lhs.Expression, rhs.Expression, Ops.GreaterThan)); }
        /// <summary>
        /// Implements the &lt; operator.
        /// </summary>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>The result of the operator.</returns>
        public static Variable operator <(Variable lhs, Variable rhs) { return new Variable(new BinaryTensorTensorExpression(lhs.Expression, rhs.Expression, Ops.LessThan)); }
        /// <summary>
        /// Implements the &gt;= operator.
        /// </summary>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>The result of the operator.</returns>
        public static Variable operator >=(Variable lhs, Variable rhs) { return new Variable(new BinaryTensorTensorExpression(lhs.Expression, rhs.Expression, Ops.GreaterOrEqual)); }
        /// <summary>
        /// Implements the &lt;= operator.
        /// </summary>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>The result of the operator.</returns>
        public static Variable operator <=(Variable lhs, Variable rhs) { return new Variable(new BinaryTensorTensorExpression(lhs.Expression, rhs.Expression, Ops.LessOrEqual)); }
        /// <summary>
        /// Implements the == operator.
        /// </summary>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>The result of the operator.</returns>
        public static Variable operator ==(Variable lhs, Variable rhs) { return new Variable(new BinaryTensorTensorExpression(lhs.Expression, rhs.Expression, Ops.EqualTo)); }
        /// <summary>
        /// Implements the != operator.
        /// </summary>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>The result of the operator.</returns>
        public static Variable operator !=(Variable lhs, Variable rhs) { return new Variable(new BinaryTensorTensorExpression(lhs.Expression, rhs.Expression, Ops.NotEqual)); }

        /// <summary>
        /// Implements the &gt; operator.
        /// </summary>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>The result of the operator.</returns>
        public static Variable operator >(Variable lhs, ScalarVar rhs) { return new Variable(new BinaryTensorScalarExpression(lhs.Expression, rhs.Expression, Ops.GreaterThan)); }
        /// <summary>
        /// Implements the &lt; operator.
        /// </summary>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>The result of the operator.</returns>
        public static Variable operator <(Variable lhs, ScalarVar rhs) { return new Variable(new BinaryTensorScalarExpression(lhs.Expression, rhs.Expression, Ops.LessThan)); }
        /// <summary>
        /// Implements the &gt;= operator.
        /// </summary>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>The result of the operator.</returns>
        public static Variable operator >=(Variable lhs, ScalarVar rhs) { return new Variable(new BinaryTensorScalarExpression(lhs.Expression, rhs.Expression, Ops.GreaterOrEqual)); }
        /// <summary>
        /// Implements the &lt;= operator.
        /// </summary>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>The result of the operator.</returns>
        public static Variable operator <=(Variable lhs, ScalarVar rhs) { return new Variable(new BinaryTensorScalarExpression(lhs.Expression, rhs.Expression, Ops.LessOrEqual)); }
        /// <summary>
        /// Implements the == operator.
        /// </summary>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>The result of the operator.</returns>
        public static Variable operator ==(Variable lhs, ScalarVar rhs) { return new Variable(new BinaryTensorScalarExpression(lhs.Expression, rhs.Expression, Ops.EqualTo)); }
        /// <summary>
        /// Implements the != operator.
        /// </summary>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>The result of the operator.</returns>
        public static Variable operator !=(Variable lhs, ScalarVar rhs) { return new Variable(new BinaryTensorScalarExpression(lhs.Expression, rhs.Expression, Ops.NotEqual)); }


        // Use symmetry of these Scalar/Tensor ops to share kernels with the Tensor/Scalar versions
        /// <summary>
        /// Implements the &gt; operator.
        /// </summary>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>The result of the operator.</returns>
        public static Variable operator >(ScalarVar lhs, Variable rhs) { return new Variable(new BinaryTensorScalarExpression(rhs.Expression, lhs.Expression, Ops.LessThan)); }
        /// <summary>
        /// Implements the &lt; operator.
        /// </summary>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>The result of the operator.</returns>
        public static Variable operator <(ScalarVar lhs, Variable rhs) { return new Variable(new BinaryTensorScalarExpression(rhs.Expression, lhs.Expression, Ops.GreaterThan)); }
        /// <summary>
        /// Implements the &gt;= operator.
        /// </summary>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>The result of the operator.</returns>
        public static Variable operator >=(ScalarVar lhs, Variable rhs) { return new Variable(new BinaryTensorScalarExpression(rhs.Expression, lhs.Expression, Ops.LessOrEqual)); }
        /// <summary>
        /// Implements the &lt;= operator.
        /// </summary>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>The result of the operator.</returns>
        public static Variable operator <=(ScalarVar lhs, Variable rhs) { return new Variable(new BinaryTensorScalarExpression(rhs.Expression, lhs.Expression, Ops.GreaterOrEqual)); }
        /// <summary>
        /// Implements the == operator.
        /// </summary>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>The result of the operator.</returns>
        public static Variable operator ==(ScalarVar lhs, Variable rhs) { return new Variable(new BinaryTensorScalarExpression(rhs.Expression, lhs.Expression, Ops.EqualTo)); }
        /// <summary>
        /// Implements the != operator.
        /// </summary>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>The result of the operator.</returns>
        public static Variable operator !=(ScalarVar lhs, Variable rhs) { return new Variable(new BinaryTensorScalarExpression(rhs.Expression, lhs.Expression, Ops.NotEqual)); }


        /// <summary>
        /// Dots the specified RHS.
        /// </summary>
        /// <param name="rhs">The RHS.</param>
        /// <returns>TVar.</returns>
        public Variable Dot(Variable rhs) { return new Variable(new BinaryTensorTensorExpression(this.Expression, rhs.Expression, Ops.Dot)); }

        // Returns beta * this + alpha * m1 * m2
        /// <summary>
        /// Addmms the specified beta.
        /// </summary>
        /// <param name="beta">The beta.</param>
        /// <param name="alpha">The alpha.</param>
        /// <param name="m1">The m1.</param>
        /// <param name="m2">The m2.</param>
        /// <returns>TVar.</returns>
        public Variable Addmm(float beta, float alpha, Variable m1, Variable m2) { return new Variable(new AddmmExpression(beta, this.Expression, alpha, m1.Expression, m2.Expression)); }

        /// <summary>
        /// cs the mul.
        /// </summary>
        /// <param name="rhs">The RHS.</param>
        /// <returns>TVar.</returns>
        public Variable CMul(Variable rhs) { return new Variable(new BinaryTensorTensorExpression(this.Expression, rhs.Expression, Ops.Mul)); }
        /// <summary>
        /// cs the div.
        /// </summary>
        /// <param name="rhs">The RHS.</param>
        /// <returns>TVar.</returns>
        public Variable CDiv(Variable rhs) { return new Variable(new BinaryTensorTensorExpression(this.Expression, rhs.Expression, Ops.Div)); }

        /// <summary>
        /// Divs the specified RHS.
        /// </summary>
        /// <param name="rhs">The RHS.</param>
        /// <returns>TVar.</returns>
        public Variable Div(ScalarVar rhs) { return new Variable(new BinaryTensorScalarExpression(this.Expression, rhs.Expression, Ops.Div)); }


        /// <summary>
        /// Abses this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public Variable Abs() { return new Variable(new UnaryTensorExpression(this.Expression, Ops.Abs)); }
        /// <summary>
        /// Signs this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public Variable Sign() { return new Variable(new UnaryTensorExpression(this.Expression, Ops.Sign)); }

        /// <summary>
        /// SQRTs this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public Variable Sqrt() { return new Variable(new UnaryTensorExpression(this.Expression, Ops.Abs)); }
        /// <summary>
        /// Exps this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public Variable Exp() { return new Variable(new UnaryTensorExpression(this.Expression, Ops.Exp)); }
        /// <summary>
        /// Logs this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public Variable Log() { return new Variable(new UnaryTensorExpression(this.Expression, Ops.Log)); }
        /// <summary>
        /// Log1ps this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public Variable Log1p() { return new Variable(new UnaryTensorExpression(this.Expression, Ops.Log1p)); }
        /// <summary>
        /// Floors this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public Variable Floor() { return new Variable(new UnaryTensorExpression(this.Expression, Ops.Floor)); }
        /// <summary>
        /// Ceils this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public Variable Ceil() { return new Variable(new UnaryTensorExpression(this.Expression, Ops.Ceil)); }
        /// <summary>
        /// Rounds this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public Variable Round() { return new Variable(new UnaryTensorExpression(this.Expression, Ops.Round)); }
        /// <summary>
        /// Truncs this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public Variable Trunc() { return new Variable(new UnaryTensorExpression(this.Expression, Ops.Trunc)); }
        /// <summary>
        /// Fracs this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public Variable Frac() { return new Variable(new UnaryTensorExpression(this.Expression, Ops.Frac)); }

        /// <summary>
        /// Sins this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public Variable Sin() { return new Variable(new UnaryTensorExpression(this.Expression, Ops.Abs)); }
        /// <summary>
        /// Coses this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public Variable Cos() { return new Variable(new UnaryTensorExpression(this.Expression, Ops.Cos)); }
        /// <summary>
        /// Tans this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public Variable Tan() { return new Variable(new UnaryTensorExpression(this.Expression, Ops.Tan)); }

        /// <summary>
        /// Asins this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public Variable Asin() { return new Variable(new UnaryTensorExpression(this.Expression, Ops.Asin)); }
        /// <summary>
        /// Acoses this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public Variable Acos() { return new Variable(new UnaryTensorExpression(this.Expression, Ops.Acos)); }
        /// <summary>
        /// Atans this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public Variable Atan() { return new Variable(new UnaryTensorExpression(this.Expression, Ops.Atan)); }

        /// <summary>
        /// Sinhes this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public Variable Sinh() { return new Variable(new UnaryTensorExpression(this.Expression, Ops.Sinh)); }
        /// <summary>
        /// Coshes this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public Variable Cosh() { return new Variable(new UnaryTensorExpression(this.Expression, Ops.Cosh)); }
        /// <summary>
        /// Tanhes this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public Variable Tanh() { return new Variable(new UnaryTensorExpression(this.Expression, Ops.Tanh)); }

        /// <summary>
        /// Sigmoids this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public Variable Sigmoid() { return new Variable(new UnaryTensorExpression(this.Expression, Ops.Sigmoid)); }

        /// <summary>
        /// Pows the specified y.
        /// </summary>
        /// <param name="y">The y.</param>
        /// <returns>TVar.</returns>
        public Variable Pow(ScalarVar y) { return new Variable(new BinaryTensorScalarExpression(this.Expression, y.Expression, Ops.Pow)); }
        /// <summary>
        /// Clamps the specified minimum.
        /// </summary>
        /// <param name="min">The minimum.</param>
        /// <param name="max">The maximum.</param>
        /// <returns>TVar.</returns>
        public Variable Clamp(ScalarVar min, ScalarVar max) { return new Variable(new UnaryTensorExpression(this.Expression, (res, src) => Ops.Clamp(res, src, min.Evaluate(), max.Evaluate()))); }

        /// <summary>
        /// Atan2s the specified y.
        /// </summary>
        /// <param name="y">The y.</param>
        /// <param name="x">The x.</param>
        /// <returns>TVar.</returns>
        public static Variable Atan2(Variable y, Variable x) { return new Variable(new BinaryTensorTensorExpression(x.Expression, y.Expression, Ops.Atan2)); }
        /// <summary>
        /// Lerps the specified a.
        /// </summary>
        /// <param name="a">a.</param>
        /// <param name="b">The b.</param>
        /// <param name="weight">The weight.</param>
        /// <returns>TVar.</returns>
        public static Variable Lerp(Variable a, Variable b, ScalarVar weight) { return new Variable(new BinaryTensorTensorExpression(a.Expression, b.Expression, (res, aVal, bVal) => Ops.Lerp(res, aVal, bVal, weight.Evaluate()))); }


        /// <summary>
        /// Sums the specified dimension.
        /// </summary>
        /// <param name="dimension">The dimension.</param>
        /// <returns>TVar.</returns>
        public Variable Sum(int dimension) { return new Variable(new UnaryTensorExpression(this.Expression, (result, src) => Ops.Sum(result, src, dimension))); }
        /// <summary>
        /// Products the specified dimension.
        /// </summary>
        /// <param name="dimension">The dimension.</param>
        /// <returns>TVar.</returns>
        public Variable Prod(int dimension) { return new Variable(new UnaryTensorExpression(this.Expression, (result, src) => Ops.Prod(result, src, dimension))); }
        /// <summary>
        /// Determines the minimum of the parameters.
        /// </summary>
        /// <param name="dimension">The dimension.</param>
        /// <returns>TVar.</returns>
        public Variable Min(int dimension) { return new Variable(new UnaryTensorExpression(this.Expression, (result, src) => Ops.Min(result, src, dimension))); }
        /// <summary>
        /// Determines the maximun of the parameters.
        /// </summary>
        /// <param name="dimension">The dimension.</param>
        /// <returns>TVar.</returns>
        public Variable Max(int dimension) { return new Variable(new UnaryTensorExpression(this.Expression, (result, src) => Ops.Max(result, src, dimension))); }
        /// <summary>
        /// Argmins the specified dimension.
        /// </summary>
        /// <param name="dimension">The dimension.</param>
        /// <returns>TVar.</returns>
        public Variable Argmin(int dimension) { return new Variable(new UnaryTensorExpression(this.Expression, (result, src) => Ops.Argmin(result, src, dimension))); }
        /// <summary>
        /// Argmaxes the specified dimension.
        /// </summary>
        /// <param name="dimension">The dimension.</param>
        /// <returns>TVar.</returns>
        public Variable Argmax(int dimension) { return new Variable(new UnaryTensorExpression(this.Expression, (result, src) => Ops.Argmax(result, src, dimension))); }

        /// <summary>
        /// Means the specified dimension.
        /// </summary>
        /// <param name="dimension">The dimension.</param>
        /// <returns>TVar.</returns>
        public Variable Mean(int dimension) { return new Variable(new UnaryTensorExpression(this.Expression, (result, src) => Ops.Mean(result, src, dimension))); }
        /// <summary>
        /// Norms the specified dimension.
        /// </summary>
        /// <param name="dimension">The dimension.</param>
        /// <param name="value">The value.</param>
        /// <returns>TVar.</returns>
        public Variable Norm(int dimension, float value) { return new Variable(new UnaryTensorExpression(this.Expression, (result, src) => Ops.Norm(result, src, dimension, value))); }
        /// <summary>
        /// Standards the specified dimension.
        /// </summary>
        /// <param name="dimension">The dimension.</param>
        /// <param name="normByN">if set to <c>true</c> [norm by n].</param>
        /// <returns>TVar.</returns>
        public Variable Std(int dimension, bool normByN = false) { return new Variable(new UnaryTensorExpression(this.Expression, (result, src) => Ops.Std(result, src, dimension, normByN))); }
        /// <summary>
        /// Variables the specified dimension.
        /// </summary>
        /// <param name="dimension">The dimension.</param>
        /// <param name="normByN">if set to <c>true</c> [norm by n].</param>
        /// <returns>TVar.</returns>
        public Variable Var(int dimension, bool normByN = false) { return new Variable(new UnaryTensorExpression(this.Expression, (result, src) => Ops.Var(result, src, dimension, normByN))); }


        /// <summary>
        /// Sums all.
        /// </summary>
        /// <returns>TVar.</returns>
        public Variable SumAll() { return new Variable(new UnaryTensorExpression(this.Expression, (result, src) => Ops.SumAll(result, src))); }
        /// <summary>
        /// Products all.
        /// </summary>
        /// <returns>TVar.</returns>
        public Variable ProdAll() { return new Variable(new UnaryTensorExpression(this.Expression, (result, src) => Ops.ProdAll(result, src))); }
        /// <summary>
        /// Minimums all.
        /// </summary>
        /// <returns>TVar.</returns>
        public Variable MinAll() { return new Variable(new UnaryTensorExpression(this.Expression, (result, src) => Ops.MinAll(result, src))); }
        /// <summary>
        /// Maximums all.
        /// </summary>
        /// <returns>TVar.</returns>
        public Variable MaxAll() { return new Variable(new UnaryTensorExpression(this.Expression, (result, src) => Ops.MaxAll(result, src))); }

        /// <summary>
        /// Means all.
        /// </summary>
        /// <returns>TVar.</returns>
        public Variable MeanAll() { return new Variable(new UnaryTensorExpression(this.Expression, (result, src) => Ops.MeanAll(result, src))); }
        /// <summary>
        /// Variables all.
        /// </summary>
        /// <returns>TVar.</returns>
        public Variable VarAll() { return new Variable(new UnaryTensorExpression(this.Expression, (result, src) => Ops.VarAll(result, src))); }
        /// <summary>
        /// Standards all.
        /// </summary>
        /// <returns>TVar.</returns>
        public Variable StdAll() { return new Variable(new UnaryTensorExpression(this.Expression, (result, src) => Ops.StdAll(result, src))); }
        /// <summary>
        /// Norms all.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <returns>TVar.</returns>
        public Variable NormAll(float value) { return new Variable(new UnaryTensorExpression(this.Expression, (result, src) => Ops.NormAll(result, src, value))); }


        /// <summary>
        /// Gathers the specified dimension.
        /// </summary>
        /// <param name="dimension">The dimension.</param>
        /// <param name="indices">The indices.</param>
        /// <returns>TVar.</returns>
        public Variable Gather(int dimension, Variable indices) { return new Variable(new BinaryTensorTensorExpression(this.Expression, indices.Expression, (res, src, ind) => Ops.Gather(res, src, dimension, ind))); }
        /// <summary>
        /// Scatters the specified dimension.
        /// </summary>
        /// <param name="dimension">The dimension.</param>
        /// <param name="indices">The indices.</param>
        /// <returns>TVar.</returns>
        public Variable Scatter(int dimension, Variable indices) { return new Variable(new BinaryTensorTensorExpression(this.Expression, indices.Expression, (res, src, ind) => Ops.Scatter(res, src, dimension, ind))); }

        // Returns a copy of this tensor, with the given indices filled with the given value.
        // If, when this op is evaluated, the write target is the same tensor as this, then the copy is unnecessary and is skipped.
        /// <summary>
        /// Scatters the fill.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <param name="dimension">The dimension.</param>
        /// <param name="indices">The indices.</param>
        /// <returns>TVar.</returns>
        public Variable ScatterFill(ScalarVar value, int dimension, Variable indices) { return new Variable(new ScatterFillExpression(this.Expression, value, dimension, indices.Expression)); }



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
        public static Variable RandomUniform(SeedSource seedSource, ScalarVar min, ScalarVar max, IAllocator allocator, DType type, params long[] sizes)
        {
            return new Variable(new FillExpression(allocator, type, sizes, res => Ops.RandomUniform(res, seedSource, min.Evaluate(), max.Evaluate())));
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
        public static Variable RandomNormal(SeedSource seedSource, ScalarVar mean, ScalarVar stdv, IAllocator allocator, DType type, params long[] sizes)
        {
            return new Variable(new FillExpression(allocator, type, sizes, res => Ops.RandomNormal(res, seedSource, mean.Evaluate(), stdv.Evaluate())));
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
        public static Variable RandomExponential(SeedSource seedSource, ScalarVar lambda, IAllocator allocator, DType type, params long[] sizes)
        {
            return new Variable(new FillExpression(allocator, type, sizes, res => Ops.RandomExponential(res, seedSource, lambda.Evaluate())));
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
        public static Variable RandomCauchy(SeedSource seedSource, ScalarVar median, ScalarVar sigma, IAllocator allocator, DType type, params long[] sizes)
        {
            return new Variable(new FillExpression(allocator, type, sizes, res => Ops.RandomCauchy(res, seedSource, median.Evaluate(), sigma.Evaluate())));
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
        public static Variable RandomLogNormal(SeedSource seedSource, ScalarVar mean, ScalarVar stdv, IAllocator allocator, DType type, params long[] sizes)
        {
            return new Variable(new FillExpression(allocator, type, sizes, res => Ops.RandomLogNormal(res, seedSource, mean.Evaluate(), stdv.Evaluate())));
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
        public static Variable RandomGeometric(SeedSource seedSource, ScalarVar p, IAllocator allocator, DType type, params long[] sizes)
        {
            return new Variable(new FillExpression(allocator, type, sizes, res => Ops.RandomGeometric(res, seedSource, p.Evaluate())));
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
        public static Variable RandomBernoulli(SeedSource seedSource, ScalarVar p, IAllocator allocator, DType type, params long[] sizes)
        {
            return new Variable(new FillExpression(allocator, type, sizes, res => Ops.RandomBernoulli(res, seedSource, p.Evaluate())));
        }



        /// <summary>
        /// Ases the type.
        /// </summary>
        /// <param name="elementType">Type of the element.</param>
        /// <returns>TVar.</returns>
        public Variable AsType(DType elementType)
        {
            return new Variable(new AsTypeExpression(this.Expression, elementType));
        }

        /// <summary>
        /// Converts to device.
        /// </summary>
        /// <param name="device">The device.</param>
        /// <returns>TVar.</returns>
        public Variable ToDevice(IAllocator device)
        {
            return new Variable(new ToDeviceExpression(this.Expression, device));
        }

        /// <summary>
        /// Evaluates this instance.
        /// </summary>
        /// <returns>Tensor.</returns>
        public NDArray Evaluate()
        {
            return expression.Evaluate(null);
        }

        /// <summary>
        /// Evaluates the specified result.
        /// </summary>
        /// <param name="result">The result.</param>
        /// <exception cref="InvalidOperationException">cannot write to given result - it is not a valid lvalue</exception>
        public void Evaluate(Variable result)
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
        public static Variable FromArray(Array array, IAllocator allocator)
        {
            return new Variable(new FromArrayExpression(allocator, array));
        }

        /// <summary>
        /// Selects the specified dimension.
        /// </summary>
        /// <param name="dimension">The dimension.</param>
        /// <param name="index">The index.</param>
        /// <returns>TVar.</returns>
        public Variable Select(int dimension, long index) { return new Variable(new ViewExpression(this.Expression, src => src.Select(dimension, index))); }
        /// <summary>
        /// Narrows the specified dimension.
        /// </summary>
        /// <param name="dimension">The dimension.</param>
        /// <param name="startIndex">The start index.</param>
        /// <param name="size">The size.</param>
        /// <returns>TVar.</returns>
        public Variable Narrow(int dimension, long startIndex, long size) { return new Variable(new ViewExpression(this.Expression, src => src.Narrow(dimension, startIndex, size))); }
        /// <summary>
        /// Transposes this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public Variable Transpose() { return new Variable(new ViewExpression(this.Expression, src => src.Transpose())); }
        /// <summary>
        /// Transposes the specified dim1.
        /// </summary>
        /// <param name="dim1">The dim1.</param>
        /// <param name="dim2">The dim2.</param>
        /// <returns>TVar.</returns>
        public Variable Transpose(int dim1, int dim2) { return new Variable(new ViewExpression(this.Expression, src => src.Transpose(dim1, dim2))); }
        /// <summary>
        /// Permutes the specified dims.
        /// </summary>
        /// <param name="dims">The dims.</param>
        /// <returns>TVar.</returns>
        public Variable Permute(params int[] dims) { return new Variable(new ViewExpression(this.Expression, src => src.Transpose(dims))); }
        /// <summary>
        /// Views the specified sizes.
        /// </summary>
        /// <param name="sizes">The sizes.</param>
        /// <returns>TVar.</returns>
        public Variable View(params long[] sizes) { return new Variable(new ViewExpression(this.Expression, src => src.View(sizes))); }
        /// <summary>
        /// Expands the specified sizes.
        /// </summary>
        /// <param name="sizes">The sizes.</param>
        /// <returns>TVar.</returns>
        public Variable Expand(params long[] sizes) { return new Variable(new ViewExpression(this.Expression, src => src.Expand(sizes))); }
        /// <summary>
        /// Squeezes this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public Variable Squeeze() { return new Variable(new ViewExpression(this.Expression, src => src.Squeeze())); }

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
        public static Variable TVar(this NDArray value)
        {
            return new Expression.Variable(new TensorValueExpression(value));
        }

      
    }
}
