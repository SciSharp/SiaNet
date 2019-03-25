using SiaNet.Engine;
using SiaNet.Engine.Layers;
using System;
using System.Linq;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Backend.MxNetLib
{
    public class SiaNetBackend : IBackend
    {
        int counter = 0;
        public static SiaNetBackend Instance
        {
            get
            {
                return new SiaNetBackend();
            }
        }

        private NDArray In(Tensor x)
        {
            return ((NDArrayTensor)x).InternalTensor;
        }

        private NDArray In(float value, params long[] shape)
        {
            if(shape.Length == 0)
            {
                shape = new long[] { 1, 1 };
            }

            var array = new NDArray(NDShape(shape));
            array.Set(value);

            return array;
        }

        private NDArrayTensor Out(NDArray x)
        {
            NDArrayTensor tensor = new NDArrayTensor();
            tensor.InternalTensor = x;
            return tensor;
        }

        private Shape NDShape(params long[] shape)
        {
            return new Shape(BackendUtil.CastShapeUInt(shape));
        }

        private Shape NDShape(params int[] shape)
        {
            return new Shape(BackendUtil.CastShapeUInt(shape));
        }

        public Tensor Abs(Tensor x)
        {
            return Out(NDArray.Abs(In(x)));
        }

        public Tensor Acos(Tensor x)
        {
            return Out(NDArray.Arccos(In(x)));
        }

        public Tensor Add(Tensor a, Tensor b)
        {
            return Out(NDArray.ElemwiseAdd(In(a), In(b)));
        }

        public Tensor Add(Tensor a, float b)
        {
            return Out(NDArray.ElemwiseAdd(In(a), In(b, a.Shape)));
        }

        public Tensor Add(float a, Tensor b)
        {
            return Out(NDArray.ElemwiseAdd(In(a, b.Shape), In(b)));
        }

        public Tensor Argmax(Tensor x, int dim = 0)
        {
            dim = dim < 0 ? x.DimCount + dim : dim;
            return Out(NDArray.Argmax(In(x), dim, true));
        }

        public Tensor Argmin(Tensor x, int dim = 0)
        {
            dim = dim < 0 ? x.DimCount + dim : dim;
            return Out(NDArray.Argmin(In(x), dim, true));
        }

        public Tensor Asin(Tensor x)
        {
            return Out(NDArray.Arcsin(In(x)));
        }

        public Tensor Atan(Tensor x)
        {
            return Out(NDArray.Arctan(In(x)));
        }

        public Tensor Atan2(Tensor lhs, Tensor rhs)
        {
            throw new NotImplementedException();
        }

        public Tensor Ceil(Tensor x)
        {
            return Out(NDArray.Ceil(In(x)));
        }

        public Tensor Clip(Tensor x, float min, float max)
        {
            return Out(NDArray.Clip(In(x), min, max));
        }

        public Tensor Col2Im(Tensor cols, long[] x_shape, Tuple<int, int> kernalSize, int padding = 1, int stride = 1)
        {
            throw new NotImplementedException();
        }

        public Tensor Constant(float value, long[] shape)
        {
            var array = new NDArray(NDShape(shape));
            array.Set(value);

            return Out(array);
        }

        public Tensor Cos(Tensor x)
        {
            return Out(NDArray.Cos(In(x)));
        }

        public Tensor Cosh(Tensor x)
        {
            return Out(NDArray.Cosh(In(x)));
        }

        public Tensor CreateVariable(float[] data, long[] shape, string name = "")
        {
            return Out(new NDArray(data, NDShape(shape)));
        }

        public Tensor Diag(Tensor x)
        {
            throw new NotImplementedException();
        }

        public void Dispose(Tensor x)
        {
            In(x).Dispose();
        }

        public Tensor Div(Tensor a, Tensor b)
        {
            return Out(NDArray.ElemwiseDiv(In(a), In(b)));
        }

        public Tensor Div(Tensor a, float b)
        {
            return Out(NDArray.ElemwiseDiv(In(a), In(b, a.Shape)));
        }

        public Tensor Div(float a, Tensor b)
        {
            return Out(NDArray.ElemwiseDiv(In(a, b.Shape), In(b)));
        }

        public Tensor Dot(Tensor a, Tensor b)
        {
            return Out(NDArray.Dot(In(a), In(b)));
        }

        public float Epsilon()
        {
            return 1e-7f;
        }

        public Tensor EqualTo(Tensor a, Tensor b)
        {
            return Out(NDArray.Equal(In(a), In(b)));
        }

        public object Eval(Tensor x)
        {
            return In(x).AsArray();
        }

        public Tensor Exp(Tensor x)
        {
            return Out(NDArray.Exp(In(x)));
        }

        public Tensor Floor(Tensor x)
        {
            return Out(NDArray.Floor(In(x)));
        }

        public ActivationFunc GetActFunc()
        {
            return new SiaNetActivations(this);
        }

        public Array GetArray(Tensor x)
        {
            return In(x).AsArray();
        }

        public DataType GetDataType(Tensor x)
        {
            int dtype = In(x).GetDType();
            DataType result = DataType.Float32;
            switch (dtype)
            {
                case 0:
                    result = DataType.Float32;
                    break;
                case 1:
                    result = DataType.Float64;
                    break;
                case 4:
                    result = DataType.Int32;
                    break;
                case 5:
                    result = DataType.Int8;
                    break;
            }

            return result;
        }

        public long[] GetShape(Tensor x)
        {
            var shape = In(x).GetShape();
            long[] result = new long[shape.Count];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = shape[i];
            }

            return result;
        }

        public Tensor GreaterThan(Tensor a, Tensor b)
        {
            return Out(NDArray.Greater(In(a), In(b)));
        }

        public Tensor GreaterThan(float a, Tensor b)
        {
            return Out(NDArray.Greater(In(a, b.Shape), In(b)));
        }

        public Tensor GreaterThan(Tensor a, float b)
        {
            return Out(NDArray.Greater(In(a), In(b, a.Shape)));
        }

        public Tensor GreaterThanEqual(Tensor a, Tensor b)
        {
            return Out(NDArray.GreaterEqual(In(a), In(b)));
        }

        public Tensor GreaterThanEqual(float a, Tensor b)
        {
            return Out(NDArray.GreaterEqual(In(a, b.Shape), In(b)));
        }

        public Tensor GreaterThanEqual(Tensor a, float b)
        {
            return Out(NDArray.GreaterEqual(In(a), In(b, a.Shape)));
        }

        public Tensor Im2Col(Tensor x, Tuple<int, int> kernalSize, int padding = 1, int stride = 1)
        {
            throw new NotImplementedException();
        }

        public Tensor L2Normalize(Tensor x, int axis = -1)
        {
            var y = Max(Sum(Square(x), axis), axis);
            return x / Sqrt(y);
        }

        public Tensor LessThan(Tensor a, Tensor b)
        {
            return Out(NDArray.Lesser(In(a), In(b)));
        }

        public Tensor LessThan(float a, Tensor b)
        {
            return Out(NDArray.Lesser(In(a, b.Shape), In(b)));
        }

        public Tensor LessThan(Tensor a, float b)
        {
            return Out(NDArray.Lesser(In(a), In(b, a.Shape)));
        }

        public Tensor LessThanEqual(Tensor a, Tensor b)
        {
            return Out(NDArray.LesserEqual(In(a), In(b)));
        }

        public Tensor LessThanEqual(float a, Tensor b)
        {
            return Out(NDArray.LesserEqual(In(a, b.Shape), In(b)));
        }

        public Tensor LessThanEqual(Tensor a, float b)
        {
            return Out(NDArray.LesserEqual(In(a), In(b, a.Shape)));
        }

        public Tensor Log(Tensor x)
        {
            return Out(NDArray.Log(In(x)));
        }

        public Tensor Log10(Tensor x)
        {
            return Out(NDArray.Log10(In(x)));
        }

        public Tensor Log1p(Tensor x)
        {
            return Out(NDArray.Log1P(In(x)));
        }

        public float Max(Tensor x)
        {
            return Out(NDArray.Max(In(x))).ToScalar();
        }

        public Tensor Max(Tensor x, int dim)
        {
            dim = dim < 0 ? x.DimCount + dim : dim;
            return Out(NDArray.Max(In(x), NDShape(dim), true));
        }

        public Tensor Max(Tensor x, params int[] dims)
        {
            for (int i = 0; i < dims.Length; i++)
            {
                dims[i] = dims[i] < 0 ? x.DimCount + dims[i] : dims[i];
            }

            return Out(NDArray.Max(In(x), NDShape(dims), true));
        }

        public Tensor Maximum(Tensor a, Tensor b)
        {
            return Out(NDArray.Maximum(In(a), In(b)));
        }

        public Tensor Maximum(Tensor a, float b)
        {
            return Out(NDArray.MaximumScalar(In(a), b));
        }

        public float Mean(Tensor x)
        {
            return Out(NDArray.Mean(In(x))).ToScalar();
        }

        public Tensor Mean(Tensor x, int dim)
        {
            dim = dim < 0 ? x.DimCount + dim : dim;
            return Out(NDArray.Mean(In(x), NDShape(dim), true));
        }

        public Tensor Mean(Tensor x, params int[] dims)
        {
            for (int i = 0;  i < dims.Length; i++)
            {
                dims[i] = dims[i] < 0 ? x.DimCount + dims[i] : dims[i];
            }

            return Out(NDArray.Mean(In(x), NDShape(dims), true));
        }

        public float Min(Tensor x)
        {
            return Out(NDArray.Min(In(x))).ToScalar();
        }

        public Tensor Min(Tensor x, int dim)
        {
            dim = dim < 0 ? x.DimCount + dim : dim;
            return Out(NDArray.Min(In(x), NDShape(dim), true));
        }

        public Tensor Min(Tensor x, params int[] dims)
        {
            for (int i = 0; i < dims.Length; i++)
            {
                dims[i] = dims[i] < 0 ? x.DimCount + dims[i] : dims[i];
            }

            return Out(NDArray.Min(In(x), NDShape(dims), true));
        }

        public Tensor Minimum(Tensor a, Tensor b)
        {
            return Out(NDArray.Minimum(In(a), In(b)));
        }

        public Tensor Minimum(Tensor a, float b)
        {
            return Out(NDArray.MinimumScalar(In(a), b));
        }

        public Tensor Mul(Tensor a, Tensor b)
        {
            return Out(NDArray.ElemwiseMul(In(a), In(b)));
        }

        public Tensor Mul(Tensor a, float b)
        {
            return Out(NDArray.ElemwiseMul(In(a), In(b, a.Shape)));
        }

        public Tensor Mul(float a, Tensor b)
        {
            return Out(NDArray.ElemwiseMul(In(a, b.Shape), In(b)));
        }

        public Tensor Neg(Tensor x)
        {
            return Out(NDArray.Negative(In(x)));
        }

        public Tensor Pow(Tensor x, float value)
        {
            return Out(NDArray.PowerScalar(In(x), value));
        }

        public void Print(Tensor x, string title = "")
        {
            Console.WriteLine(In(x).ToValueString());
        }

        public Tensor RandomBernoulli(long[] shape, float p)
        {
            var result = RandomUniform(shape, 0, 1);
            result = result > p;
            return result;
        }

        public Tensor RandomNormal(long[] shape, float mean, float stddev, int? seed = null)
        {
            NDArray data = new NDArray(NDShape(shape));
            NDArray.SampleGaussian(mean, stddev, data);
            return Out(data);
        }

        public Tensor RandomUniform(long[] shape, float min, float max, int? seed = null)
        {
            NDArray data = new NDArray(NDShape(shape));
            NDArray.SampleUniform(min, max, data);
            return Out(data);
        }

        public Tensor Reshape(Tensor x, params long[] shape)
        {
            return Out(In(x).Reshape(NDShape(shape)));
        }

        public Tensor Round(Tensor x)
        {
            return Out(NDArray.Round(In(x)));
        }

        public void SetDevice(Engine.DeviceType device)
        {
            switch (device)
            {
                case Engine.DeviceType.Default:
                    DeviceManager.Current = Context.Cpu();
                    break;
                case Engine.DeviceType.CPU:
                    DeviceManager.Current = Context.Cpu();
                    break;
                case Engine.DeviceType.CUDA:
                    DeviceManager.Current = Context.Gpu(0);
                    break;
                case Engine.DeviceType.OpenCL:
                    throw new NotSupportedException("OpenCL is not supported. Please use ArrayFire backend.");
                default:
                    break;
            }
        }

        public Tensor Sigmoid(Tensor x)
        {
            return Out(NDArray.Sigmoid(In(x)));
        }

        public Tensor Sign(Tensor x)
        {
            return Out(NDArray.Sign(In(x)));
        }

        public Tensor Sin(Tensor x)
        {
            return Out(NDArray.Sin(In(x)));
        }

        public Tensor Sinh(Tensor x)
        {
            return Out(NDArray.Sinh(In(x)));
        }

        public Tensor SliceCols(Tensor x, long start, long end)
        {
            return Out(NDArray.SliceAxis(In(x), 1, (int)start, (int)end));
        }

        public Tensor SliceRows(Tensor x, long start, long end)
        {
            end = end + 1;
            return Out(In(x).Slice((uint)start, (uint)end));
        }

        public Tensor Softmax(Tensor x, int axis = -1)
        {
            return Out(NDArray.Softmax(In(x), axis));
        }

        public Tensor Softplus(Tensor x, int axis = -1)
        {
            return Log((Exp(x) + 1));
        }

        public Tensor Sqrt(Tensor x)
        {
            return Out(NDArray.Sqrt(In(x)));
        }

        public Tensor Square(Tensor x)
        {
            return Out(NDArray.Square(In(x)));
        }

        public Tensor Sub(Tensor a, Tensor b)
        {
            return Out(NDArray.ElemwiseSub(In(a), In(b)));
        }

        public Tensor Sub(Tensor a, float b)
        {
            return Out(NDArray.ElemwiseSub(In(a), In(b, a.Shape)));
        }

        public Tensor Sub(float a, Tensor b)
        {
            return Out(NDArray.ElemwiseSub(In(a, b.Shape), In(b)));
        }

        public float Sum(Tensor x)
        {
            return Out(NDArray.Sum(In(x))).ToScalar();
        }

        public Tensor Sum(Tensor x, int dim)
        {
            dim = dim < 0 ? x.DimCount + dim : dim;
            return Out(NDArray.Sum(In(x), NDShape(dim), true));
        }

        public Tensor Sum(Tensor x, params int[] dims)
        {
            for (int i = 0; i < dims.Length; i++)
            {
                dims[i] = dims[i] < 0 ? x.DimCount + dims[i] : dims[i];
            }

            return Out(NDArray.Sum(In(x), NDShape(dims), true));
        }

        public Tensor Tan(Tensor x)
        {
            return Out(NDArray.Tan(In(x)));
        }

        public Tensor Tanh(Tensor x)
        {
            return Out(NDArray.Tanh(In(x)));
        }

        public Tensor Tile(Tensor x, int n, int axis = 0)
        {
            if (axis < 0)
                throw new ArgumentException("Axis >= 0");

            int[] times = new int[x.Shape.Length];
            for (int i = 0; i < times.Length; i++)
            {
                if (i == axis)
                {
                    times[i] = 1;
                    continue;
                }

                times[i] = n;
            }

            return Out(NDArray.Tile(In(x), NDShape(times)));
        }

        public Tensor Transpose(Tensor x)
        {
            return Out(NDArray.Transpose(In(x)));
        }

        public Tensor Transpose(Tensor x, params int[] dims)
        {
            return Out(NDArray.Transpose(In(x), NDShape(dims)));
        }

        public Tensor Trunc(Tensor x)
        {
            return Out(NDArray.Trunc(In(x)));
        }

        public string UUID(string name)
        {
            return name + "_" + counter++;
        }
    }
}
