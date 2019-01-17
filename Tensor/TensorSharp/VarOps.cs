using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp.Expression;

namespace TensorSharp
{
    public class VarOps
    {
        /// <summary>
        /// Dots the specified RHS.
        /// </summary>
        /// <param name="rhs">The RHS.</param>
        /// <returns>TVar.</returns>
        public static Variable Dot(Variable lhs, Variable rhs) { return new Variable(new BinaryTensorTensorExpression(lhs.Expression, rhs.Expression, Ops.Dot)); }

        // Returns beta * this + alpha * m1 * m2
        /// <summary>
        /// Addmms the specified beta.
        /// </summary>
        /// <param name="beta">The beta.</param>
        /// <param name="alpha">The alpha.</param>
        /// <param name="m1">The m1.</param>
        /// <param name="m2">The m2.</param>
        /// <returns>TVar.</returns>
        public static Variable Addmm(float beta, Variable lhs, float alpha, Variable m1, Variable m2) { return new Variable(new AddmmExpression(beta, lhs.Expression, alpha, m1.Expression, m2.Expression)); }

        /// <summary>
        /// cs the mul.
        /// </summary>
        /// <param name="rhs">The RHS.</param>
        /// <returns>TVar.</returns>
        public static Variable CMul(Variable lhs, Variable rhs) { return new Variable(new BinaryTensorTensorExpression(lhs.Expression, rhs.Expression, Ops.Mul)); }
        /// <summary>
        /// cs the div.
        /// </summary>
        /// <param name="rhs">The RHS.</param>
        /// <returns>TVar.</returns>
        public static Variable CDiv(Variable lhs, Variable rhs) { return new Variable(new BinaryTensorTensorExpression(lhs.Expression, rhs.Expression, Ops.Div)); }

        /// <summary>
        /// Divs the specified RHS.
        /// </summary>
        /// <param name="rhs">The RHS.</param>
        /// <returns>TVar.</returns>
        public static Variable Div(Variable lhs, ScalarVar rhs) { return new Variable(new BinaryTensorScalarExpression(lhs.Expression, rhs.Expression, Ops.Div)); }


        /// <summary>
        /// Abses this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public static Variable Abs(Variable t) { return new Variable(new UnaryTensorExpression(t.Expression, Ops.Abs)); }
        /// <summary>
        /// Signs this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public static Variable Sign(Variable t) { return new Variable(new UnaryTensorExpression(t.Expression, Ops.Sign)); }

        /// <summary>
        /// SQRTs this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public static Variable Sqrt(Variable t) { return new Variable(new UnaryTensorExpression(t.Expression, Ops.Abs)); }
        /// <summary>
        /// Exps this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public static Variable Exp(Variable t) { return new Variable(new UnaryTensorExpression(t.Expression, Ops.Exp)); }
        /// <summary>
        /// Logs this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public static Variable Log(Variable t) { return new Variable(new UnaryTensorExpression(t.Expression, Ops.Log)); }
        /// <summary>
        /// Log1ps this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public static Variable Log1p(Variable t) { return new Variable(new UnaryTensorExpression(t.Expression, Ops.Log1p)); }
        /// <summary>
        /// Floors this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public static Variable Floor(Variable t) { return new Variable(new UnaryTensorExpression(t.Expression, Ops.Floor)); }
        /// <summary>
        /// Ceils this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public static Variable Ceil(Variable t) { return new Variable(new UnaryTensorExpression(t.Expression, Ops.Ceil)); }
        /// <summary>
        /// Rounds this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public static Variable Round(Variable t) { return new Variable(new UnaryTensorExpression(t.Expression, Ops.Round)); }
        /// <summary>
        /// Truncs this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public static Variable Trunc(Variable t) { return new Variable(new UnaryTensorExpression(t.Expression, Ops.Trunc)); }
        /// <summary>
        /// Fracs this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public static Variable Frac(Variable t) { return new Variable(new UnaryTensorExpression(t.Expression, Ops.Frac)); }

        /// <summary>
        /// Sins this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public static Variable Sin(Variable t) { return new Variable(new UnaryTensorExpression(t.Expression, Ops.Abs)); }
        /// <summary>
        /// Coses this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public static Variable Cos(Variable t) { return new Variable(new UnaryTensorExpression(t.Expression, Ops.Cos)); }
        /// <summary>
        /// Tans this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public static Variable Tan(Variable t) { return new Variable(new UnaryTensorExpression(t.Expression, Ops.Tan)); }

        /// <summary>
        /// Asins this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public static Variable Asin(Variable t) { return new Variable(new UnaryTensorExpression(t.Expression, Ops.Asin)); }
        /// <summary>
        /// Acoses this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public static Variable Acos(Variable t) { return new Variable(new UnaryTensorExpression(t.Expression, Ops.Acos)); }
        /// <summary>
        /// Atans this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public static Variable Atan(Variable t) { return new Variable(new UnaryTensorExpression(t.Expression, Ops.Atan)); }

        /// <summary>
        /// Sinhes this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public static Variable Sinh(Variable t) { return new Variable(new UnaryTensorExpression(t.Expression, Ops.Sinh)); }
        /// <summary>
        /// Coshes this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public static Variable Cosh(Variable t) { return new Variable(new UnaryTensorExpression(t.Expression, Ops.Cosh)); }
        /// <summary>
        /// Tanhes this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public static Variable Tanh(Variable t) { return new Variable(new UnaryTensorExpression(t.Expression, Ops.Tanh)); }

        /// <summary>
        /// Sigmoids this instance.
        /// </summary>
        /// <returns>TVar.</returns>
        public static Variable Sigmoid(Variable t) { return new Variable(new UnaryTensorExpression(t.Expression, Ops.Sigmoid)); }

        /// <summary>
        /// Pows the specified y.
        /// </summary>
        /// <param name="y">The y.</param>
        /// <returns>TVar.</returns>
        public static Variable Pow(Variable t, ScalarVar y) { return new Variable(new BinaryTensorScalarExpression(t.Expression, y.Expression, Ops.Pow)); }

        public static Variable Square(Variable t)
        {
            return Pow(t, 2);
        }

        /// <summary>
        /// Clamps the specified minimum.
        /// </summary>
        /// <param name="min">The minimum.</param>
        /// <param name="max">The maximum.</param>
        /// <returns>TVar.</returns>
        public static Variable Clip(Variable t, ScalarVar min, ScalarVar max) { return new Variable(new UnaryTensorExpression(t.Expression, (res, src) => Ops.Clamp(res, src, min.Evaluate(), max.Evaluate()))); }

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
        public static Variable Sum(Variable t, int dimension)
        {
            if(dimension < 0)
            {
                dimension = t.Shape.Length - dimension;
            }

            return new Variable(new UnaryTensorExpression(t.Expression, (result, src) => Ops.Sum(result, src, dimension)));
        }

        public static Variable Sum(Variable t)
        {
            return new Variable(new UnaryTensorExpression(t.Expression, (result, src) => Ops.SumAll(result, src)));
        }

        /// <summary>
        /// Products the specified dimension.
        /// </summary>
        /// <param name="dimension">The dimension.</param>
        /// <returns>TVar.</returns>
        public static Variable Prod(Variable t, int dimension)
        {
            if (dimension < 0)
            {
                dimension = t.Shape.Length - dimension;
            }

            return new Variable(new UnaryTensorExpression(t.Expression, (result, src) => Ops.Prod(result, src, dimension)));
        }

        public static Variable Prod(Variable t)
        {
            return new Variable(new UnaryTensorExpression(t.Expression, (result, src) => Ops.ProdAll(result, src)));
        }

        /// <summary>
        /// Determines the minimum of the parameters.
        /// </summary>
        /// <param name="dimension">The dimension.</param>
        /// <returns>TVar.</returns>
        public static Variable Min(Variable t, int dimension)
        {
            if (dimension < 0)
            {
                dimension = t.Shape.Length - dimension;
            }

            return new Variable(new UnaryTensorExpression(t.Expression, (result, src) => Ops.Min(result, src, dimension)));
        }

        public static Variable Min(Variable t)
        {
            return new Variable(new UnaryTensorExpression(t.Expression, (result, src) => Ops.MinAll(result, src)));
        }

        /// <summary>
        /// Determines the maximun of the parameters.
        /// </summary>
        /// <param name="dimension">The dimension.</param>
        /// <returns>TVar.</returns>
        public static Variable Max(Variable t, int dimension)
        {
            if (dimension < 0)
            {
                dimension = t.Shape.Length - dimension;
            }

            return new Variable(new UnaryTensorExpression(t.Expression, (result, src) => Ops.Max(result, src, dimension)));
        }

        public static Variable Max(Variable t)
        {
            return new Variable(new UnaryTensorExpression(t.Expression, (result, src) => Ops.MaxAll(result, src)));
        }

        /// <summary>
        /// Argmins the specified dimension.
        /// </summary>
        /// <param name="dimension">The dimension.</param>
        /// <returns>TVar.</returns>
        public static Variable Argmin(Variable t, int dimension)
        {
            if (dimension < 0)
            {
                dimension = t.Shape.Length - dimension;
            }

            return new Variable(new UnaryTensorExpression(t.Expression, (result, src) => Ops.Argmin(result, src, dimension)));
        }

        /// <summary>
        /// Argmaxes the specified dimension.
        /// </summary>
        /// <param name="dimension">The dimension.</param>
        /// <returns>TVar.</returns>
        public static Variable Argmax(Variable t, int dimension)
        {
            if (dimension < 0)
            {
                dimension = t.Shape.Length - dimension;
            }

            return new Variable(new UnaryTensorExpression(t.Expression, (result, src) => Ops.Argmax(result, src, dimension)));
        }

        /// <summary>
        /// Means the specified dimension.
        /// </summary>
        /// <param name="dimension">The dimension.</param>
        /// <returns>TVar.</returns>
        public static Variable Mean(Variable t, int dimension)
        {
            if (dimension < 0)
            {
                dimension = t.Shape.Length - dimension;
            }

            return new Variable(new UnaryTensorExpression(t.Expression, (result, src) => Ops.Mean(result, src, dimension)));
        }

        public static Variable Mean(Variable t)
        {
            return new Variable(new UnaryTensorExpression(t.Expression, (result, src) => Ops.MeanAll(result, src)));
        }


        /// <summary>
        /// Norms the specified dimension.
        /// </summary>
        /// <param name="dimension">The dimension.</param>
        /// <param name="value">The value.</param>
        /// <returns>TVar.</returns>
        public static Variable Norm(Variable t, int dimension, float value)
        {
            if (dimension < 0)
            {
                dimension = t.Shape.Length - dimension;
            }

            return new Variable(new UnaryTensorExpression(t.Expression, (result, src) => Ops.Norm(result, src, dimension, value)));
        }

        public static Variable Norm(Variable t, float value)
        {
            return new Variable(new UnaryTensorExpression(t.Expression, (result, src) => Ops.NormAll(result, src, value)));
        }


        /// <summary>
        /// Standards the specified dimension.
        /// </summary>
        /// <param name="dimension">The dimension.</param>
        /// <param name="normByN">if set to <c>true</c> [norm by n].</param>
        /// <returns>TVar.</returns>
        public static Variable Std(Variable t, int dimension, bool normByN = false)
        {
            if (dimension < 0)
            {
                dimension = t.Shape.Length - dimension;
            }

            return new Variable(new UnaryTensorExpression(t.Expression, (result, src) => Ops.Std(result, src, dimension, normByN)));
        }

        public static Variable Std(Variable t, bool normByN = false)
        {
            return new Variable(new UnaryTensorExpression(t.Expression, (result, src) => Ops.StdAll(result, src)));
        }

        /// <summary>
        /// Variables the specified dimension.
        /// </summary>
        /// <param name="dimension">The dimension.</param>
        /// <param name="normByN">if set to <c>true</c> [norm by n].</param>
        /// <returns>TVar.</returns>
        public static Variable Var(Variable t, int dimension, bool normByN = false)
        {
            if (dimension < 0)
            {
                dimension = t.Shape.Length - dimension;
            }

            return new Variable(new UnaryTensorExpression(t.Expression, (result, src) => Ops.Var(result, src, dimension, normByN)));
        }

        public static Variable Var(Variable t, bool normByN = false)
        {
            return new Variable(new UnaryTensorExpression(t.Expression, (result, src) => Ops.VarAll(result, src)));
        }
    }
}
