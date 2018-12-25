// ***********************************************************************
// Assembly         : TensorSharp
// Author           : Community
// Created          : 12-09-2018
//
// Last Modified By : Deepak Battini
// Last Modified On : 11-25-2018
// ***********************************************************************
// <copyright file="SExpression.cs" company="TensorSharp">
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
    /// Class SExpression.
    /// </summary>
    public abstract class SExpression
    {
        /// <summary>
        /// Evaluates this instance.
        /// </summary>
        /// <returns>System.Single.</returns>
        public abstract float Evaluate();
    }


    /// <summary>
    /// Class ConstScalarExpression.
    /// Implements the <see cref="TensorSharp.Expression.SExpression" />
    /// </summary>
    /// <seealso cref="TensorSharp.Expression.SExpression" />
    public class ConstScalarExpression : SExpression
    {
        /// <summary>
        /// The value
        /// </summary>
        private readonly float value;

        /// <summary>
        /// Initializes a new instance of the <see cref="ConstScalarExpression"/> class.
        /// </summary>
        /// <param name="value">The value.</param>
        public ConstScalarExpression(float value)
        {
            this.value = value;
        }

        /// <summary>
        /// Evaluates this instance.
        /// </summary>
        /// <returns>System.Single.</returns>
        public override float Evaluate()
        {
            return value;
        }
    }

    /// <summary>
    /// Class DelegateScalarExpression.
    /// Implements the <see cref="TensorSharp.Expression.SExpression" />
    /// </summary>
    /// <seealso cref="TensorSharp.Expression.SExpression" />
    public class DelegateScalarExpression : SExpression
    {
        /// <summary>
        /// The evaluate
        /// </summary>
        private readonly Func<float> evaluate;

        /// <summary>
        /// Initializes a new instance of the <see cref="DelegateScalarExpression"/> class.
        /// </summary>
        /// <param name="evaluate">The evaluate.</param>
        public DelegateScalarExpression(Func<float> evaluate)
        {
            this.evaluate = evaluate;
        }

        /// <summary>
        /// Evaluates this instance.
        /// </summary>
        /// <returns>System.Single.</returns>
        public override float Evaluate()
        {
            return evaluate();
        }
    }

    /// <summary>
    /// Class UnaryScalarExpression.
    /// Implements the <see cref="TensorSharp.Expression.SExpression" />
    /// </summary>
    /// <seealso cref="TensorSharp.Expression.SExpression" />
    public class UnaryScalarExpression : SExpression
    {
        /// <summary>
        /// The source
        /// </summary>
        private readonly SExpression src;
        /// <summary>
        /// The evaluate
        /// </summary>
        private readonly Func<float, float> evaluate;


        /// <summary>
        /// Initializes a new instance of the <see cref="UnaryScalarExpression"/> class.
        /// </summary>
        /// <param name="src">The source.</param>
        /// <param name="evaluate">The evaluate.</param>
        public UnaryScalarExpression(SExpression src, Func<float, float> evaluate)
        {
            this.src = src;
            this.evaluate = evaluate;
        }

        /// <summary>
        /// Evaluates this instance.
        /// </summary>
        /// <returns>System.Single.</returns>
        public override float Evaluate()
        {
            return evaluate(src.Evaluate());
        }
    }

    /// <summary>
    /// Class BinaryScalarExpression.
    /// Implements the <see cref="TensorSharp.Expression.SExpression" />
    /// </summary>
    /// <seealso cref="TensorSharp.Expression.SExpression" />
    public class BinaryScalarExpression : SExpression
    {
        /// <summary>
        /// The left
        /// </summary>
        private readonly SExpression left;
        /// <summary>
        /// The right
        /// </summary>
        private readonly SExpression right;
        /// <summary>
        /// The evaluate
        /// </summary>
        private readonly Func<float, float, float> evaluate;


        /// <summary>
        /// Initializes a new instance of the <see cref="BinaryScalarExpression"/> class.
        /// </summary>
        /// <param name="left">The left.</param>
        /// <param name="right">The right.</param>
        /// <param name="evaluate">The evaluate.</param>
        public BinaryScalarExpression(SExpression left, SExpression right, Func<float, float, float> evaluate)
        {
            this.left = left;
            this.right = right;
            this.evaluate = evaluate;
        }

        /// <summary>
        /// Evaluates this instance.
        /// </summary>
        /// <returns>System.Single.</returns>
        public override float Evaluate()
        {
            return evaluate(left.Evaluate(), right.Evaluate());
        }
    }
}
