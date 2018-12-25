// ***********************************************************************
// Assembly         : TensorSharp
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="SVar.cs" company="TensorSharp">
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
    /// Class SVar.
    /// </summary>
    public class SVar
    {
        /// <summary>
        /// The expression
        /// </summary>
        private SExpression expression;


        /// <summary>
        /// Initializes a new instance of the <see cref="SVar"/> class.
        /// </summary>
        /// <param name="expression">The expression.</param>
        public SVar(SExpression expression)
        {
            this.expression = expression;
        }


        /// <summary>
        /// Evaluates this instance.
        /// </summary>
        /// <returns>System.Single.</returns>
        public float Evaluate()
        {
            return expression.Evaluate();
        }

        /// <summary>
        /// Gets the expression.
        /// </summary>
        /// <value>The expression.</value>
        public SExpression Expression { get { return expression; } }


        /// <summary>
        /// Performs an implicit conversion from <see cref="System.Single"/> to <see cref="SVar"/>.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <returns>The result of the conversion.</returns>
        public static implicit operator SVar(float value) { return new SVar(new ConstScalarExpression(value)); }

        /// <summary>
        /// Implements the - operator.
        /// </summary>
        /// <param name="src">The source.</param>
        /// <returns>The result of the operator.</returns>
        public static SVar operator -(SVar src) { return new SVar(new UnaryScalarExpression(src.expression, val => -val)); }

        /// <summary>
        /// Implements the + operator.
        /// </summary>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>The result of the operator.</returns>
        public static SVar operator +(SVar lhs, SVar rhs) { return new SVar(new BinaryScalarExpression(lhs.expression, rhs.expression, (l, r) => l + r)); }
        /// <summary>
        /// Implements the - operator.
        /// </summary>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>The result of the operator.</returns>
        public static SVar operator -(SVar lhs, SVar rhs) { return new SVar(new BinaryScalarExpression(lhs.expression, rhs.expression, (l, r) => l - r)); }
        /// <summary>
        /// Implements the * operator.
        /// </summary>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>The result of the operator.</returns>
        public static SVar operator *(SVar lhs, SVar rhs) { return new SVar(new BinaryScalarExpression(lhs.expression, rhs.expression, (l, r) => l * r)); }
        /// <summary>
        /// Implements the / operator.
        /// </summary>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>The result of the operator.</returns>
        public static SVar operator /(SVar lhs, SVar rhs) { return new SVar(new BinaryScalarExpression(lhs.expression, rhs.expression, (l, r) => l / r)); }
        /// <summary>
        /// Implements the % operator.
        /// </summary>
        /// <param name="lhs">The LHS.</param>
        /// <param name="rhs">The RHS.</param>
        /// <returns>The result of the operator.</returns>
        public static SVar operator %(SVar lhs, SVar rhs) { return new SVar(new BinaryScalarExpression(lhs.expression, rhs.expression, (l, r) => l % r)); }


        /// <summary>
        /// Abses this instance.
        /// </summary>
        /// <returns>SVar.</returns>
        public SVar Abs() { return new SVar(new UnaryScalarExpression(this.expression, val => Math.Abs(val))); }
        /// <summary>
        /// Signs this instance.
        /// </summary>
        /// <returns>SVar.</returns>
        public SVar Sign() { return new SVar(new UnaryScalarExpression(this.expression, val => Math.Sign(val))); }

        /// <summary>
        /// SQRTs this instance.
        /// </summary>
        /// <returns>SVar.</returns>
        public SVar Sqrt() { return new SVar(new UnaryScalarExpression(this.expression, val => (float)Math.Sqrt(val))); }
        /// <summary>
        /// Exps this instance.
        /// </summary>
        /// <returns>SVar.</returns>
        public SVar Exp() { return new SVar(new UnaryScalarExpression(this.expression, val => (float)Math.Exp(val))); }
        /// <summary>
        /// Logs this instance.
        /// </summary>
        /// <returns>SVar.</returns>
        public SVar Log() { return new SVar(new UnaryScalarExpression(this.expression, val => (float)Math.Log(val))); }
        /// <summary>
        /// Floors this instance.
        /// </summary>
        /// <returns>SVar.</returns>
        public SVar Floor() { return new SVar(new UnaryScalarExpression(this.expression, val => (float)Math.Floor(val))); }
        /// <summary>
        /// Ceils this instance.
        /// </summary>
        /// <returns>SVar.</returns>
        public SVar Ceil() { return new SVar(new UnaryScalarExpression(this.expression, val => (float)Math.Ceiling(val))); }
        /// <summary>
        /// Rounds this instance.
        /// </summary>
        /// <returns>SVar.</returns>
        public SVar Round() { return new SVar(new UnaryScalarExpression(this.expression, val => (float)Math.Round(val))); }
        /// <summary>
        /// Truncs this instance.
        /// </summary>
        /// <returns>SVar.</returns>
        public SVar Trunc() { return new SVar(new UnaryScalarExpression(this.expression, val => (float)Math.Truncate(val))); }


        /// <summary>
        /// Sins this instance.
        /// </summary>
        /// <returns>SVar.</returns>
        public SVar Sin() { return new SVar(new UnaryScalarExpression(this.expression, val => (float)Math.Sin(val))); }
        /// <summary>
        /// Coses this instance.
        /// </summary>
        /// <returns>SVar.</returns>
        public SVar Cos() { return new SVar(new UnaryScalarExpression(this.expression, val => (float)Math.Cos(val))); }
        /// <summary>
        /// Tans this instance.
        /// </summary>
        /// <returns>SVar.</returns>
        public SVar Tan() { return new SVar(new UnaryScalarExpression(this.expression, val => (float)Math.Tan(val))); }

        /// <summary>
        /// Asins this instance.
        /// </summary>
        /// <returns>SVar.</returns>
        public SVar Asin() { return new SVar(new UnaryScalarExpression(this.expression, val => (float)Math.Asin(val))); }
        /// <summary>
        /// Acoses this instance.
        /// </summary>
        /// <returns>SVar.</returns>
        public SVar Acos() { return new SVar(new UnaryScalarExpression(this.expression, val => (float)Math.Acos(val))); }
        /// <summary>
        /// Atans this instance.
        /// </summary>
        /// <returns>SVar.</returns>
        public SVar Atan() { return new SVar(new UnaryScalarExpression(this.expression, val => (float)Math.Atan(val))); }

        /// <summary>
        /// Sinhes this instance.
        /// </summary>
        /// <returns>SVar.</returns>
        public SVar Sinh() { return new SVar(new UnaryScalarExpression(this.expression, val => (float)Math.Sinh(val))); }
        /// <summary>
        /// Coshes this instance.
        /// </summary>
        /// <returns>SVar.</returns>
        public SVar Cosh() { return new SVar(new UnaryScalarExpression(this.expression, val => (float)Math.Cosh(val))); }
        /// <summary>
        /// Tanhes this instance.
        /// </summary>
        /// <returns>SVar.</returns>
        public SVar Tanh() { return new SVar(new UnaryScalarExpression(this.expression, val => (float)Math.Tanh(val))); }


        /// <summary>
        /// Pows the specified y.
        /// </summary>
        /// <param name="y">The y.</param>
        /// <returns>SVar.</returns>
        public SVar Pow(SVar y) { return new SVar(new BinaryScalarExpression(this.expression, y.expression, (xVal, yVal) => (float)Math.Pow(xVal, yVal))); }
        /// <summary>
        /// Clamps the specified minimum.
        /// </summary>
        /// <param name="min">The minimum.</param>
        /// <param name="max">The maximum.</param>
        /// <returns>SVar.</returns>
        public SVar Clamp(SVar min, SVar max) { return new SVar(new DelegateScalarExpression(() => ClampFloat(this.expression.Evaluate(), min.expression.Evaluate(), max.expression.Evaluate()))); }

        /// <summary>
        /// Pows the specified y.
        /// </summary>
        /// <param name="y">The y.</param>
        /// <returns>TVar.</returns>
        public TVar Pow(TVar y) { return new TVar(new BinaryScalarTensorExpression(this.Expression, y.Expression, Ops.Tpow)); }


        /// <summary>
        /// Atan2s the specified y.
        /// </summary>
        /// <param name="y">The y.</param>
        /// <param name="x">The x.</param>
        /// <returns>SVar.</returns>
        public static SVar Atan2(SVar y, SVar x) { return new SVar(new DelegateScalarExpression(() => (float)Math.Atan2(y.Evaluate(), x.Evaluate()))); }
        /// <summary>
        /// Lerps the specified a.
        /// </summary>
        /// <param name="a">a.</param>
        /// <param name="b">The b.</param>
        /// <param name="weight">The weight.</param>
        /// <returns>SVar.</returns>
        public static SVar Lerp(SVar a, SVar b, SVar weight) { return new SVar(new DelegateScalarExpression(() => (float)LerpFloat(a.Evaluate(), b.Evaluate(), weight.Evaluate()))); }


        /// <summary>
        /// Lerps the float.
        /// </summary>
        /// <param name="a">a.</param>
        /// <param name="b">The b.</param>
        /// <param name="weight">The weight.</param>
        /// <returns>System.Single.</returns>
        private static float LerpFloat(float a, float b, float weight)
        {
            return a + weight * (b - a);
        }

        /// <summary>
        /// Clamps the float.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <param name="min">The minimum.</param>
        /// <param name="max">The maximum.</param>
        /// <returns>System.Single.</returns>
        private static float ClampFloat(float value, float min, float max)
        {
            if (value < min)
                return min;
            else if (value > max)
                return max;
            else
                return value;
        }
    }
}
