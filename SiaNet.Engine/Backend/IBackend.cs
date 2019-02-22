using SiaNet.Engine;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Text;

namespace SiaNet.Engine
{
    public interface IBackend
    {
        float Epsilon();

        long[] GetShape(Tensor x);

        object Eval(Tensor x);

        void Print(Tensor x, string title = "");

        string UUID(string name);

        void SetDevice(DeviceType device);

        DataType GetDataType(Tensor x);

        #region Initializers
        Tensor CreateVariable(float[] data, long[] shape, string name = "");

        Tensor Reshape(Tensor x, params long[] shape);

        Tensor Constant(float value, long[] shape);

        Tensor RandomUniform(long[] shape, float min, float max, int? seed = null);

        Tensor RandomNormal(long[] shape, float mean, float stddev, int? seed = null);

        Tensor RandomBernoulli(long[] shape, float p);
        #endregion

        #region BasicOps
        Tensor Add(Tensor a, Tensor b);

        Tensor Add(Tensor a, float b);

        Tensor Add(float a, Tensor b);

        Tensor Sub(Tensor a, Tensor b);

        Tensor Sub(Tensor a, float b);

        Tensor Sub(float a, Tensor b);

        Tensor Mul(Tensor a, Tensor b);

        Tensor Mul(Tensor a, float b);

        Tensor Mul(float a, Tensor b);

        Tensor Div(Tensor a, Tensor b);

        Tensor Div(Tensor a, float b);

        Tensor Div(float a, Tensor b);

        Tensor GreaterThan(Tensor a, Tensor b);

        Tensor GreaterThanEqual(Tensor a, Tensor b);

        Tensor LessThan(Tensor a, Tensor b);

        Tensor LessThanEqual(Tensor a, Tensor b);

        Tensor GreaterThan(float a, Tensor b);

        Tensor GreaterThanEqual(float a, Tensor b);

        Tensor LessThan(float a, Tensor b);

        Tensor LessThanEqual(float a, Tensor b);

        Tensor GreaterThan(Tensor a, float b);

        Tensor GreaterThanEqual(Tensor a, float b);

        Tensor LessThan(Tensor a, float b);

        Tensor LessThanEqual(Tensor a, float b);

        Tensor EqualTo(Tensor a, Tensor b);

        Tensor Tile(Tensor x, int n, int axis = 0);
        #endregion

        #region Math Ops
        Tensor Abs(Tensor x);

        Tensor Neg(Tensor x);

        Tensor Sign(Tensor x);

        Tensor Sqrt(Tensor x);

        Tensor Exp(Tensor x);

        Tensor Log(Tensor x);

        Tensor Log10(Tensor x);

        Tensor Log1p(Tensor x);

        Tensor Floor(Tensor x);

        Tensor Ceil(Tensor x);

        Tensor Round(Tensor x);

        Tensor Trunc(Tensor x);

        Tensor Sin(Tensor x);

        Tensor Cos(Tensor x);

        Tensor Tan(Tensor x);

        Tensor Asin(Tensor x);

        Tensor Acos(Tensor x);

        Tensor Atan(Tensor x);

        Tensor Sinh(Tensor x);

        Tensor Cosh(Tensor x);

        Tensor Tanh(Tensor x);

        Tensor Sigmoid(Tensor x);

        Tensor Atan2(Tensor lhs, Tensor rhs);

        Tensor Pow(Tensor x, float value);

        Tensor Square(Tensor x);

        Tensor Tpow(float value, Tensor x);

        Tensor Clip(Tensor x, float min, float max);
        #endregion

        #region Aggregate Ops
        float Sum(Tensor x);

        Tensor Sum(Tensor x, int dim);

        Tensor Sum(Tensor x, params int[] dim);

        float Prod(Tensor x);

        Tensor Prod(Tensor x, int dim);

        Tensor Prod(Tensor x, params int[] dim);

        float Max(Tensor x);

        Tensor Max(Tensor x, int dim);

        Tensor Max(Tensor x, params int[] dim);

        float Min(Tensor x);

        Tensor Min(Tensor x, int dim);

        Tensor Min(Tensor x, params int[] dim);

        float Mean(Tensor x);

        Tensor Mean(Tensor x, int dim);

        Tensor Mean(Tensor x, params int[] dim);

        float Var(Tensor x);

        Tensor Var(Tensor x, int dim);

        Tensor Var(Tensor x, params int[] dim);

        float StdDev(Tensor x);

        Tensor StdDev(Tensor x, int dim);

        Tensor StdDev(Tensor x, params int[] dim);

        Tensor Argmax(Tensor x, int dim = 0);

        Tensor Argmin(Tensor x, int dim = 0);

        Tensor Maximum(Tensor a, Tensor b);

        Tensor Maximum(Tensor a, float b);

        Tensor Minimum(Tensor a, Tensor b);

        Tensor Minimum(Tensor a, float b);
        #endregion

        #region Matrix Ops
        Tensor Transpose(Tensor x);

        Tensor Transpose(Tensor x, params int[] dims);

        Tensor Dot(Tensor a, Tensor b);

        Tensor Diag(Tensor x);
        #endregion

        #region NN Ops
        Tensor Softmax(Tensor x, int axis = -1);

        Tensor Softplus(Tensor x, int axis = -1);

        Tensor L2Normalize(Tensor x, int axis = -1);

        Tensor Im2Col(Tensor x, Tuple<int, int> kernalSize, int padding = 1, int stride = 1);

        Tensor Col2Im(Tensor cols, long[] x_shape, Tuple<int, int> kernalSize, int padding = 1, int stride = 1);
        #endregion

        #region Misc
        Tensor SliceRows(Tensor x, long start, long end);

        Tensor SliceCols(Tensor x, long start, long end);

        Array GetArray(Tensor x);

        void Dispose(Tensor x);
        #endregion
    }
}
