using SiaNet.Engine;
using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using SiaNet.Backend.ArrayFire.Interop;
using System.Drawing;
using SiaNet.Engine.Layers;

namespace SiaNet.Backend.ArrayFire
{
    public class SiaNetBackend : IBackend
    {
        int counter = 0;

        public SiaNetBackend()
        {
        }

        public static SiaNetBackend Instance
        {
            get
            {
                return new SiaNetBackend();
            }
        }

        private NDArray In(Tensor x)
        {
            NDArrayTensor tensor = (NDArrayTensor)x;
            return tensor.InternalTensor;
        }

        private NDArray In(float value, params long[] shape)
        {
            NDArrayTensor tensor = new NDArrayTensor(Data.Constant(value, BackendUtil.CastShapeInt(shape)));
            return tensor.InternalTensor;
        }

        private NDArrayTensor Out(NDArray x)
        {
            NDArrayTensor tensor = new NDArrayTensor();
            tensor.InternalTensor = x;
            return tensor;
        }

        public Tensor Abs(Tensor x)
        {
            return Out(Arith.Abs(In(x)));
        }

        public Tensor Acos(Tensor x)
        {
            return Out(Arith.Acos(In(x)));
        }

        public Tensor Add(Tensor a, Tensor b)
        {
            return Out(In(a) + In(b));
        }

        public Tensor Add(Tensor a, float b)
        {
            return Out(In(a) + In(b, a.Shape));
        }

        public Tensor Add(float a, Tensor b)
        {
            return Out(In(a, b.Shape) + In(b));
        }

        public Tensor Asin(Tensor x)
        {
            return Out(Arith.Asin(In(x)));
        }

        public Tensor Atan(Tensor x)
        {
            return Out(Arith.Atan(In(x)));
        }

        public Tensor Atan2(Tensor lhs, Tensor rhs)
        {
            return Out(Arith.Atan2(In(lhs), In(rhs)));
        }

        public Tensor Ceil(Tensor x)
        {
            return Out(Arith.Ceil(In(x)));
        }

        public Tensor Clip(Tensor x, float min, float max)
        {
            return Out(Arith.Clamp(In(x), In(min, x.Shape), In(max, x.Shape)));
        }

        public Tensor Cos(Tensor x)
        {
            return Out(Arith.Cos(In(x)));
        }

        public Tensor Cosh(Tensor x)
        {
            return Out(Arith.Cosh(In(x)));
        }

        public Tensor CreateVariable(float[] data, long[] shape, string name = "")
        {
            var result = Out(Data.CreateArray(data));

            result.Name = name != "" ? UUID(name) : UUID("V");
            if (shape.Length == 2)
                result = (NDArrayTensor)result.Reshape(shape).Transpose();

            if (shape.Length == 3)
                result = (NDArrayTensor)result.Reshape(shape).Transpose(0, 2, 1);

            if (shape.Length == 4)
                result = (NDArrayTensor)result.Reshape(shape).Transpose(0, 1, 3, 2);

            return result;
        }

        public Tensor Div(Tensor a, Tensor b)
        {
            return Out(In(a) / In(b));
        }

        public Tensor Div(Tensor a, float b)
        {
            return Out(In(a) / In(b, a.Shape));
        }

        public Tensor Div(float a, Tensor b)
        {
            return Out(In(a, b.Shape) / In(b));
        }

        public Tensor Dot(Tensor a, Tensor b)
        {
            return Out(Matrix.Multiply(In(a), In(b)));
        }

        public object Eval(Tensor x)
        {
            return In(x);
        }

        public Tensor Exp(Tensor x)
        {
            return Out(Arith.Exp(In(x)));
        }

        public Tensor Floor(Tensor x)
        {
            return Out(Arith.Floor(In(x)));
        }

        public Tensor Factorial(Tensor x)
        {
            return Out(Arith.Factorial(In(x)));
        }

        public DataType GetDataType(Tensor x)
        {
            Type t = In(x).ElemType;
            DataType result = DataType.Float32;
            switch (t.Name.ToLower())
            {
                case "single":
                    result = DataType.Float32;
                    break;
                case "double":
                    result = DataType.Float64;
                    break;
                case "int32":
                    result = DataType.Int32;
                    break;
                case "byte":
                    result = DataType.Int8;
                    break;
                default:
                    break;
            }

            return result;
        }

        public long[] GetShape(Tensor x)
        {
            return In(x).Dimensions.Select(i => ((long)i)).ToArray();
        }

        public Tensor Log(Tensor x)
        {
            return Out(Arith.Log(In(x)));
        }

        public Tensor Log10(Tensor x)
        {
            return Out(Arith.Log10(In(x)));
        }

        public Tensor Log1p(Tensor x)
        {
            return Out(Arith.Log1p(In(x)));
        }

        public Tensor Mul(Tensor a, Tensor b)
        {
            return Out(In(a) * In(b));
        }

        public Tensor Mul(Tensor a, float b)
        {
            return Out(In(a) * In(b, a.Shape));
        }

        public Tensor Mul(float a, Tensor b)
        {
            return Out(In(a, b.Shape) * In(b));
        }

        public Tensor Neg(Tensor x)
        {
            return Out(In(-1, x.Shape) * In(x));
        }

        public Tensor Pow(Tensor x, float value)
        {
            return Out(Arith.Pow(In(x), In(value, x.Shape)));
        }

        public Tensor Round(Tensor x)
        {
            return Out(Arith.Round(In(x)));
        }

        public Tensor Sigmoid(Tensor x)
        {
            return Out(Arith.Sigmoid(In(x)));
        }

        public Tensor Sign(Tensor x)
        {
            return Out(Arith.Sign(In(x)));
        }

        public Tensor Sin(Tensor x)
        {
            return Out(Arith.Sin(In(x)));
        }

        public Tensor Sinh(Tensor x)
        {
            return Out(Arith.Sinh(In(x)));
        }

        public Tensor Sqrt(Tensor x)
        {
            return Out(Arith.Sqrt(In(x)));
        }

        public Tensor Square(Tensor x)
        {
            return Out(Arith.Pow2(In(x)));
        }

        public Tensor Sub(Tensor a, Tensor b)
        {
            return Out(In(a) - In(b));
        }

        public Tensor Sub(Tensor a, float b)
        {
            return Out(In(a) - In(b, a.Shape));
        }

        public Tensor Sub(float a, Tensor b)
        {
            return Out(In(a, b.Shape) - In(b));
        }

        public Tensor Tan(Tensor x)
        {
            return Out(Arith.Tan(In(x)));
        }

        public Tensor Tanh(Tensor x)
        {
            return Out(Arith.Tanh(In(x)));
        }

        public Tensor Tpow(float value, Tensor x)
        {
            return Out(Arith.Pow(In(value, x.Shape), In(x)));
        }

        public Tensor Transpose(Tensor x)
        {
            if (x.Shape.Length > 2) throw new InvalidOperationException("Parameterless Transpose is only valid on 2d tensors");
            return Out(Matrix.Transpose(In(x), false));
        }

        public Tensor Transpose(Tensor x, params int[] dims)
        {
            dims = dims.Select(i => (dims.Length - i - 1)).Reverse().ToArray();
            return Out(Data.Reorder(In(x), dims.Select(i => ((uint)i)).ToArray()));
        }

        public Tensor Trunc(Tensor x)
        {
            return Out(Arith.Trunc(In(x)));
        }

        public string UUID(string name)
        {
            return name + "_" + counter++;
        }

        public void Print(Tensor x, string title = "")
        {
            Util.Print(In(x), title);
        }

        public void SetDevice(DeviceType device)
        {
            switch (device)
            {
                case DeviceType.Default:
                    Device.SetBackend(Backend.DEFAULT);
                    break;
                case DeviceType.CPU:
                    Device.SetBackend(Backend.CPU);
                    break;
                case DeviceType.CUDA:
                    Device.SetBackend(Backend.CUDA);
                    break;
                case DeviceType.OpenCL:
                    Device.SetBackend(Backend.OPENCL);
                    break;
                default:
                    break;
            }
        }

        public Tensor Reshape(Tensor x, params long[] shape)
        {
            shape = shape.Reverse().ToArray();
            long prod = -1 * shape.Aggregate(1L, (a, b) => a * b);
            for (int i = 0; i < shape.Length; i++)
            {
                if (shape[i] == -1)
                {
                    shape[i] = x.ElementCount / prod;
                    break;
                }
            }

            return Out(Data.ModDims(In(x), shape));
        }

        public float Epsilon()
        {
            return 1e-7f;
        }

        public float Sum(Tensor x)
        {
            return (float)Algorithm.Sum(In(x)).Real;
        }

        public Tensor Sum(Tensor x, int dim)
        {
            if (dim < 0)
                dim = CorrDim(x.DimCount, dim);

            return Out(Algorithm.Sum(In(x), dim));
        }

        public Tensor Sum(Tensor x, params int[] dims)
        {
            dims = dims.Reverse().ToArray();
            foreach (var item in dims)
            {
                x = Sum(x, item);
            }

            return x;
        }

        public float Prod(Tensor x)
        {
            return (float)Algorithm.Prod(In(x)).Real;
        }

        public Tensor Prod(Tensor x, int dim)
        {
            if (dim < 0)
                dim = CorrDim(x.DimCount, dim);

            return Out(Algorithm.Prod(In(x), dim));
        }

        public Tensor Prod(Tensor x, params int[] dims)
        {
            dims = dims.Reverse().ToArray();
            foreach (var item in dims)
            {
                x = Prod(x, item);
            }

            return x;
        }

        public float Max(Tensor x)
        {
            return (float)Algorithm.Max(In(x)).Real;
        }

        public Tensor Max(Tensor x, int dim)
        {
            if (dim < 0)
                dim = CorrDim(x.DimCount, dim);

            return Out(Algorithm.Max(In(x), dim));
        }

        public Tensor Max(Tensor x, params int[] dims)
        {
            dims = dims.Reverse().ToArray();

            foreach (var item in dims)
            {
                x = Max(x, item);
            }

            return x;
        }

        public float Min(Tensor x)
        {
            return (float)Algorithm.Min(In(x)).Real;
        }

        public Tensor Min(Tensor x, int dim)
        {
            if (dim < 0)
                dim = CorrDim(x.DimCount, dim);

            return Out(Algorithm.Min(In(x), dim));
        }

        public Tensor Min(Tensor x, params int[] dims)
        {
            dims = dims.Reverse().ToArray();
            foreach (var item in dims)
            {
                x = Min(x, item);
            }

            return x;
        }

        public float Mean(Tensor x)
        {
            return (float)Algorithm.Mean(In(x)).Real;
        }

        public Tensor Mean(Tensor x, int dim)
        {
            if (dim < 0)
                dim = CorrDim(x.DimCount, dim);

            return Out(Algorithm.Mean(In(x), dim));
        }

        public Tensor Mean(Tensor x, params int[] dims)
        {
            dims = dims.Reverse().ToArray();
            foreach (var item in dims)
            {
                x = Mean(x, item);
            }

            return x;
        }

        public float Var(Tensor x)
        {
            return (float)Algorithm.Var(In(x)).Real;
        }

        public Tensor Var(Tensor x, int dim)
        {
            if (dim < 0)
                dim = CorrDim(x.DimCount, dim);

            return Out(Algorithm.Var(In(x), dim));
        }

        public Tensor Var(Tensor x, params int[] dims)
        {
            dims = dims.Reverse().ToArray();
            foreach (var item in dims)
            {
                x = Var(x, item);
            }

            return x;
        }

        public float StdDev(Tensor x)
        {
            return (float)Algorithm.StdDev(In(x)).Real;
        }

        public Tensor StdDev(Tensor x, int dim)
        {
            if (dim < 0)
                dim = CorrDim(x.DimCount, dim);

            return Out(Algorithm.StdDev(In(x), dim));
        }

        public Tensor StdDev(Tensor x, params int[] dims)
        {
            dims = dims.Reverse().ToArray();
            foreach (var item in dims)
            {
                x = StdDev(x, item);
            }

            return x;
        }

        public Tensor Argmax(Tensor x, int dim = 0)
        {
            if (dim < 0)
                dim = CorrDim(x.DimCount, dim);

            if (dim == 1)
            {
                if (x.DimCount == 2)
                    x = x.Transpose();
                else if (x.DimCount == 3)
                    x = x.Transpose(0, 2, 1);
                else if (x.DimCount == 4)
                    x = x.Transpose(0, 1, 3, 2);
            }

            Tensor result = Out(Algorithm.TopK(In(x), 1, 0, 2));
            if(dim == 1)
            {
                if (result.DimCount == 2)
                    result = result.Transpose();
                else if (result.DimCount == 3)
                    result = result.Transpose(0, 2, 1);
                else if (result.DimCount == 4)
                    result = result.Transpose(0, 1, 3, 2);
            }

            return result;
        }

        public Tensor Argmin(Tensor x, int dim = 0)
        {
            if (dim < 0)
                dim = CorrDim(x.DimCount, dim);

            if (dim == 1)
            {
                if (x.DimCount == 2)
                    x = x.Transpose();
                else if (x.DimCount == 3)
                    x = x.Transpose(0, 2, 1);
                else if (x.DimCount == 4)
                    x = x.Transpose(0, 1, 3, 2);
            }

            Tensor result = Out(Algorithm.TopK(In(x), 1, 0, 1));
            if (dim == 1)
            {
                if (result.DimCount == 2)
                    result = result.Transpose();
                else if (result.DimCount == 3)
                    result = result.Transpose(0, 2, 1);
                else if (result.DimCount == 4)
                    result = result.Transpose(0, 1, 3, 2);
            }

            return result;
        }

        public Tensor Maximum(Tensor a, Tensor b)
        {
            return Out(Arith.MaxOf(In(a), In(b)));
        }

        public Tensor Maximum(Tensor a, float b)
        {
            return Out(Arith.MaxOf(In(a), In(b, a.Shape)));
        }

        public Tensor Minimum(Tensor a, Tensor b)
        {
            return Out(Arith.MinOf(In(a), In(b)));
        }

        public Tensor Minimum(Tensor a, float b)
        {
            return Out(Arith.MinOf(In(a), In(b, a.Shape)));
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
            return Out(In(a, b.Shape) > In(b));
        }

        public Tensor GreaterThanEqual(float a, Tensor b)
        {
            return Out(In(a, b.Shape) >= In(b));
        }

        public Tensor LessThan(float a, Tensor b)
        {
            return Out(In(a, b.Shape) < In(b));
        }

        public Tensor LessThanEqual(float a, Tensor b)
        {
            return Out(In(a, b.Shape) <= In(b));
        }

        public Tensor GreaterThan(Tensor a, float b)
        {
            return Out(In(a) > In(b, a.Shape));
        }

        public Tensor GreaterThanEqual(Tensor a, float b)
        {
            return Out(In(a) >= In(b, a.Shape));
        }

        public Tensor LessThan(Tensor a, float b)
        {
            return Out(In(a) < In(b, a.Shape));
        }

        public Tensor LessThanEqual(Tensor a, float b)
        {
            return Out(In(a) <= In(b, a.Shape));
        }

        public Tensor EqualTo(Tensor a, Tensor b)
        {
            return Out(Arith.EqualTo(In(a), In(b)));
        }

        public Tensor Constant(float value, long[] shape)
        {
            return Out(Data.Constant<float>(value, BackendUtil.CastShapeInt(shape)));
        }

        public Tensor RandomUniform(long[] shape, float min, float max, int? seed = null)
        {
            var result = Data.RandUniform<float>(BackendUtil.CastShapeInt(shape));
            if (min != 0 || max != 1)
            {
                result = Data.Constant<float>((max - min), result.Dimensions) * result + Data.Constant<float>(min, result.Dimensions);
            }

            return Out(result);
        }

        public Tensor RandomNormal(long[] shape, float mean, float stddev, int? seed = null)
        {
            var result = Data.RandNormal<float>(BackendUtil.CastShapeInt(shape));
            if (mean != 0 || stddev != 1)
            {
                result = Data.Constant<float>(stddev, result.Dimensions) * result + Data.Constant<float>(mean, result.Dimensions);
            }

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
            //shape = shape.Reverse().ToArray();
            var result = RandomUniform(shape, 0, 1);
            result = result > p;
            return result;
        }

        public Tensor Tile(Tensor x, int n, int axis = 0)
        {
            if (axis < 0)
                axis = x.DimCount + axis;

            uint[] dims = new uint[x.DimCount];
            for (int i = 0; i < dims.Length; i++)
            {
                if (i == axis)
                    dims[i] = (uint)n;
                else
                    dims[i] = 1;
            }

            dims = dims.Reverse().ToArray();

            return Out(Data.Tile(In(x), dims));
        }

        public Tensor Diag(Tensor x)
        {
            return Out(Matrix.CreateDiagonal(In(x)));
        }

        public Tensor Im2Col(Tensor x, Tuple<int, int> kernalSize, int padding = 1, int stride = 1)
        {
            return Out(ImgPro.Unwrap(In(x), (uint)kernalSize.Item1, (uint)kernalSize.Item2, (uint)padding, (uint)padding, (uint)stride, (uint)stride, true));
        }

        public Tensor Col2Im(Tensor cols, long[] x_shape, Tuple<int, int> kernalSize, int padding = 1, int stride = 1)
        {
            uint ox = (uint)x_shape[2];
            uint oy = (uint)x_shape[3];

            return Out(ImgPro.Wrap(In(cols), (uint)kernalSize.Item1, (uint)kernalSize.Item2, ox, oy, (uint)padding, (uint)padding, (uint)stride, (uint)stride, true));
        }

        private int CorrDim(int dimCount, int dim)
        {
            if (dim < 0)
            {
                switch (dim)
                {
                    case -1:
                        dim = 1;
                        break;
                    case -2:
                        dim = 0;
                        break;
                    case -3:
                        dim = 2;
                        break;
                    case -4:
                        dim = 3;
                        break;
                    default:
                        break;
                }
            }

            return dim;
        }

        public Tensor SliceRows(Tensor x, long start, long end)
        {
            return Out(In(x).Rows((int)start, (int)end));
        }

        public Tensor SliceCols(Tensor x, long start, long end)
        {
            return Out(In(x).Cols((int)start, (int)end));
        }

        public void Dispose(Tensor x)
        {
            In(x).Dispose();
        }

        public Array GetArray(Tensor x)
        {
            return Data.GetData<float>(In(x)).ToArray();
        }

        public ActivationFunc GetActFunc()
        {
            return new SiaNetActivations(this);
        }
    }
}
