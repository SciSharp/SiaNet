using SiaNet.Engine;
using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using System.Drawing;

namespace SiaNet.Backend.TensorSharp
{
    public class TensorSharpBackend : IBackend
    {
        int counter = 0;

        private NDArray In(Tensor x)
        {
            NDArrayTensor tensor = (NDArrayTensor)x;
            return tensor.InternalTensor;
        }

        private NDArrayTensor Out(NDArray x)
        {
            NDArrayTensor tensor = new NDArrayTensor
            {
                InternalTensor = x
            };

            return tensor;
        }

        public Tensor Abs(Tensor x)
        {
            return Out(TOps.Abs(In(x)));
        }

        public Tensor Acos(Tensor x)
        {
            return Out(TOps.Acos(In(x)));
        }

        public Tensor Add(Tensor a, Tensor b)
        {
            return Out(In(a) + In(b));
        }

        public Tensor Add(Tensor a, float b)
        {
            return Out(In(a) + b);
        }

        public Tensor Add(float a, Tensor b)
        {
            return Out(a + In(b));
        }

        public Tensor Asin(Tensor x)
        {
            return Out(TOps.Asin(In(x)));
        }

        public Tensor Atan(Tensor x)
        {
            return Out(TOps.Atan(In(x)));
        }

        public Tensor Atan2(Tensor lhs, Tensor rhs)
        {
            return Out(TOps.Atan2(In(lhs), In(rhs)));
        }

        public Tensor Ceil(Tensor x)
        {
            return Out(TOps.Ceil(In(x)));
        }

        public Tensor Clip(Tensor x, float min, float max)
        {
            return Out(TOps.Clip(In(x), min, max));
        }

        public Tensor Cos(Tensor x)
        {
            return Out(TOps.Cos(In(x)));
        }

        public Tensor Cosh(Tensor x)
        {
            return Out(TOps.Cosh(In(x)));
        }

        public Tensor CreateVariable(float[] data, long[] shape, string name = "")
        {
            var result = Out(NDArray.FromArray(DeviceManager.Current, data.ToArray()));
            result.Name = name != "" ? UUID(name) : UUID("V");
            return result.Reshape(shape);
        }

        public Tensor Div(Tensor a, Tensor b)
        {
            return Out(In(a) / In(b));
        }

        public Tensor Div(Tensor a, float b)
        {
            return Out(In(a) / b);
        }

        public Tensor Div(float a, Tensor b)
        {
            return Out(a / In(b));
        }

        public Tensor Dot(Tensor a, Tensor b)
        {
            return Out(TOps.Dot(In(a), In(b)));
        }

        public object Eval(Tensor x)
        {
            return In(x);
        }

        public Tensor Exp(Tensor x)
        {
            return Out(TOps.Exp(In(x)));
        }

        public Tensor Floor(Tensor x)
        {
            return Out(TOps.Floor(In(x)));
        }

        public Tensor Factorial(Tensor x)
        {
            return Out(TOps.Frac(In(x)));
        }

        public DataType GetDataType(Tensor x)
        {
            DType t = In(x).ElementType;
            DataType result = DataType.Float32;
            switch (t)
            {
                case DType.Float32:
                    result = DataType.Float32;
                    break;
                case DType.Float64:
                    result = DataType.Float64;
                    break;
                case DType.Int32:
                    result = DataType.Int32;
                    break;
                case DType.UInt8:
                    result = DataType.Int8;
                    break;
                default:
                    break;
            }

            return result;
        }

        public long[] GetShape(Tensor x)
        {
            return Array.ConvertAll(In(x).Shape, i => ((long)i));
        }

        public Tensor Log(Tensor x)
        {
            return Out(TOps.Log(In(x)));
        }

        public Tensor Log10(Tensor x)
        {
            throw new NotImplementedException();
        }

        public Tensor Log1p(Tensor x)
        {
            return Out(TOps.Log1p(In(x)));
        }

        public Tensor Mul(Tensor a, Tensor b)
        {
            return Out(In(a) * In(b));
        }

        public Tensor Mul(Tensor a, float b)
        {
            return Out(In(a) * b);
        }

        public Tensor Mul(float a, Tensor b)
        {
            return Out(a * In(b));
        }

        public Tensor Neg(Tensor x)
        {
            return Out(TOps.Neg(In(x)));
        }

        public Tensor Pow(Tensor x, float value)
        {
            return Out(TOps.Pow(In(x), value));
        }

        public Tensor Round(Tensor x)
        {
            return Out(TOps.Round(In(x)));
        }

        public Tensor Sigmoid(Tensor x)
        {
            return Out(TOps.Sigmoid(In(x)));
        }

        public Tensor Sign(Tensor x)
        {
            return Out(TOps.Sign(In(x)));
        }

        public Tensor Sin(Tensor x)
        {
            return Out(TOps.Sin(In(x)));
        }

        public Tensor Sinh(Tensor x)
        {
            return Out(TOps.Sinh(In(x)));
        }

        public Tensor Sqrt(Tensor x)
        {
            return Out(TOps.Sqrt(In(x)));
        }

        public Tensor Square(Tensor x)
        {
            return Out(TOps.Square(In(x)));
        }

        public Tensor Sub(Tensor a, Tensor b)
        {
            return Out(In(a) - In(b));
        }

        public Tensor Sub(Tensor a, float b)
        {
            return Out(In(a) - b);
        }

        public Tensor Sub(float a, Tensor b)
        {
            return Out(a - In(b));
        }

        public Tensor Tan(Tensor x)
        {
            return Out(TOps.Tan(In(x)));
        }

        public Tensor Tanh(Tensor x)
        {
            return Out(TOps.Tanh(In(x)));
        }

        public Tensor Tpow(float value, Tensor x)
        {
            return Out(TOps.Tpow(value, In(x)));
        }

        public Tensor Transpose(Tensor x)
        {
            if (x.Shape.Length != 2)
                throw new InvalidOperationException("Parameterless Transpose is only valid on 2d tensors");
            return Out(In(x).Transpose());
        }

        public Tensor Transpose(Tensor x, params int[] dims)
        {
            return Out(In(x).Transpose(dims));
        }

        public Tensor Trunc(Tensor x)
        {
            return Out(TOps.Trunc(In(x)));
        }

        public string UUID(string name)
        {
            return name + "_" + counter++;
        }

        public void Print(Tensor x, string title = "")
        {
            In(x).Print(title: title);
        }

        public void SetDevice(DeviceType device)
        {
            switch (device)
            {
                case DeviceType.Default:
                    DeviceManager.SetBackend(Backend.CPU);
                    break;
                case DeviceType.CPU:
                    DeviceManager.SetBackend(Backend.CPU);
                    break;
                case DeviceType.CUDA:
                    DeviceManager.SetBackend(Backend.CUDA);
                    break;
                case DeviceType.OpenCL:
                    throw new NotSupportedException();
                default:
                    break;
            }
        }

        public Tensor Reshape(Tensor x, params long[] shape)
        {
            return Out(In(x).Reshape(shape));
        }

        public float Epsilon()
        {
            return 1e-7f;
        }

        public float Sum(Tensor x)
        {
            return TOps.SumF(In(x));
        }

        public Tensor Sum(Tensor x, int dim)
        {
            return Out(TOps.Sum(In(x), dim));
        }

        public Tensor Sum(Tensor x, params int[] dim)
        {
            return Out(TOps.Sum(In(x), dim));
        }

        public float Prod(Tensor x)
        {
            return TOps.ProdF(In(x));
        }

        public Tensor Prod(Tensor x, int dim)
        {
            return Out(TOps.Prod(In(x), dim));
        }

        public Tensor Prod(Tensor x, params int[] dim)
        {
            foreach (var item in dim)
            {
                x = Prod(x, item);
            }

            return x;
        }

        public float Max(Tensor x)
        {
            return TOps.MaxF(In(x));
        }

        public Tensor Max(Tensor x, int dim)
        {
            return Out(TOps.Max(In(x), dim));
        }

        public Tensor Max(Tensor x, params int[] dim)
        {
            return Out(TOps.Max(In(x), dim));
        }

        public float Min(Tensor x)
        {
            return TOps.MinF(In(x));
        }

        public Tensor Min(Tensor x, int dim)
        {
            return Out(TOps.Min(In(x), dim));
        }

        public Tensor Min(Tensor x, params int[] dim)
        {
            foreach (var item in dim)
            {
                x = Min(x, item);
            }

            return x;
        }

        public float Mean(Tensor x)
        {
            return TOps.MeanF(In(x));
        }

        public Tensor Mean(Tensor x, int dim)
        {
            return Out(TOps.Mean(In(x), dim));
        }

        public Tensor Mean(Tensor x, params int[] dim)
        {
            return Out(TOps.Mean(In(x), dim));
        }

        public float Var(Tensor x)
        {
            return TOps.VarF(In(x));
        }

        public Tensor Var(Tensor x, int dim)
        {
            return Out(TOps.Var(In(x), dim, false));
        }

        public Tensor Var(Tensor x, params int[] dim)
        {
            foreach (var item in dim)
            {
                x = Mean(x, item);
            }

            return x;
        }

        public float StdDev(Tensor x)
        {
            return TOps.StdF(In(x));
        }

        public Tensor StdDev(Tensor x, int dim)
        {
            if (dim < 0)
                dim = x.Shape.Length + dim;

            return Out(TOps.Std(In(x), dim, false));
        }

        public Tensor StdDev(Tensor x, params int[] dim)
        {
            foreach (var item in dim)
            {
                x = StdDev(x, item);
            }

            return x;
        }

        public Tensor Argmax(Tensor x, int dim)
        {
            return Max(x, dim);
        }

        public Tensor Argmin(Tensor x, int dim)
        {
            return Min(x, dim);
        }

        public Tensor Maximum(Tensor a, Tensor b)
        {
            return Out(TOps.Maximum(In(a), In(b)));
        }

        public Tensor Maximum(Tensor a, float b)
        {
            return Out(TOps.Maximum(In(a), b));
        }

        public Tensor Minimum(Tensor a, Tensor b)
        {
            throw new NotImplementedException();
        }

        public Tensor Minimum(Tensor a, float b)
        {
            throw new NotImplementedException();
        }

        public Tensor GreaterThan(Tensor a, Tensor b)
        {
            return Out(In(a) > In(b));
        }

        public Tensor GreaterThanEqual(Tensor a, Tensor b)
        {
            return Out(In(a) >= In(b));
        }

        public Tensor LessThan(Tensor a, Tensor b)
        {
            return Out(In(a) < In(b));
        }

        public Tensor LessThanEqual(Tensor a, Tensor b)
        {
            return Out(In(a) <= In(b));
        }

        public Tensor GreaterThan(float a, Tensor b)
        {
            var b_arr = In(b);
            var a_arr = NDArray.Constant(a, b_arr.Allocator, b_arr.ElementType, b_arr.Shape);
            return Out(a_arr > b_arr);
        }

        public Tensor GreaterThanEqual(float a, Tensor b)
        {
            var b_arr = In(b);
            var a_arr = NDArray.Constant(a, b_arr.Allocator, b_arr.ElementType, b_arr.Shape);
            return Out(a_arr >= b_arr);
        }

        public Tensor LessThan(float a, Tensor b)
        {
            var b_arr = In(b);
            var a_arr = NDArray.Constant(a, b_arr.Allocator, b_arr.ElementType, b_arr.Shape);
            return Out(a_arr < b_arr);
        }

        public Tensor LessThanEqual(float a, Tensor b)
        {
            var b_arr = In(b);
            var a_arr = NDArray.Constant(a, b_arr.Allocator, b_arr.ElementType, b_arr.Shape);
            return Out(a_arr <= b_arr);
        }

        public Tensor GreaterThan(Tensor a, float b)
        {
            var a_arr = In(a);
            var b_arr = NDArray.Constant(b, a_arr.Allocator, a_arr.ElementType, a_arr.Shape);
            return Out(a_arr > b_arr);
        }

        public Tensor GreaterThanEqual(Tensor a, float b)
        {
            var a_arr = In(a);
            var b_arr = NDArray.Constant(b, a_arr.Allocator, a_arr.ElementType, a_arr.Shape);
            return Out(a_arr >= b_arr);
        }

        public Tensor LessThan(Tensor a, float b)
        {
            var a_arr = In(a);
            var b_arr = NDArray.Constant(b, a_arr.Allocator, a_arr.ElementType, a_arr.Shape);
            return Out(a_arr < b_arr);
        }

        public Tensor LessThanEqual(Tensor a, float b)
        {
            var a_arr = In(a);
            var b_arr = NDArray.Constant(b, a_arr.Allocator, a_arr.ElementType, a_arr.Shape);
            return Out(a_arr <= b_arr);
        }

        public Tensor EqualTo(Tensor a, Tensor b)
        {
            return Out(TOps.EqualTo(In(a), In(b)));
        }

        public Tensor Constant(float value, long[] shape)
        {
            return Out(NDArray.Constant(value, DeviceManager.Current, DType.Float32, shape));
        }

        public Tensor RandomUniform(long[] shape, float min, float max, int? seed = null)
        {
            var result = new NDArray(DeviceManager.Current, DType.Float32, shape);
            var seedSource = new SeedSource();
            if (seed.HasValue)
                seedSource = new SeedSource(seed.Value);

            TOps.RandomUniform(result, seedSource, min, max);
            return Out(result);
        }

        public Tensor RandomNormal(long[] shape, float mean, float stddev, int? seed = null)
        {
            var result = new NDArray(DeviceManager.Current, DType.Float32, shape);
            var seedSource = new SeedSource();
            if (seed.HasValue)
                seedSource = new SeedSource(seed.Value);
            TOps.RandomNormal(result, seedSource, mean, stddev);
            return Out(result);
        }

        public Tensor Softmax(Tensor x, int axis = -1)
        {
            var e = Exp(x - Max(x, axis));
            var s = Sum(e, axis);
            return e / s;
        }

        public Tensor Softplus(Tensor x, int axis = -1)
        {
            return Log((Exp(x) + 1));
        }

        public Tensor L1Normalize(Tensor x, int axis = -1)
        {
            throw new NotImplementedException();
        }

        public Tensor L2Normalize(Tensor x, int axis = -1)
        {
            var y = Max(Sum(Square(x), axis), axis);
            return x / Sqrt(y);
        }

        public Tensor RandomBernoulli(long[] shape, float p)
        {
            throw new NotImplementedException();
        }

        public Tensor Tile(Tensor x, int n, int axis = 0)
        {
            return Out(In(x).Tile(n));
        }

        public Tensor Diag(Tensor x)
        {
            return Out(TOps.Diag(In(x)));
        }

        public Tensor Im2Col(Tensor x, Tuple<int, int> kernalSize, int padding = 1, int stride = 1)
        {
            return Out(ImgUtil.Im2Col(In(x), Tuple.Create<uint, uint>((uint)kernalSize.Item1, (uint)kernalSize.Item2), padding, (uint)stride));
        }

        public Tensor Col2Im(Tensor cols, long[] x_shape, Tuple<int, int> kernalSize, int padding = 1, int stride = 1)
        {
            return Out(ImgUtil.Col2Im(In(cols), x_shape, Tuple.Create<uint, uint>((uint)kernalSize.Item1, (uint)kernalSize.Item2), padding, (uint)stride));
        }

        public Tensor SliceRows(Tensor x, long start, long end)
        {
            return Out(In(x).Narrow(0, start, end - start + 1));
        }

        public Tensor SliceCols(Tensor x, long start, long end)
        {
            return Out(In(x).Narrow(1, start, end - start + 1));
        }

        public void Dispose(Tensor x)
        {
            In(x).Dispose();
        }

        public Array GetArray(Tensor x)
        {
            return In(x).ToArray();
        }
    }
}
