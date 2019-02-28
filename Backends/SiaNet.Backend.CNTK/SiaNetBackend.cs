using CNTK;
using SiaNet.Engine;
using SiaNet.Engine.Layers;
using System;
using System.Linq;
using C = CNTK.CNTKLib;

namespace SiaNet.Backend.CNTKLib
{
    public class SiaNetBackend : IBackend
    {
        private Variable In(Tensor x)
        {
            return ((NDArrayTensor)x).InternalTensor;
        }

        private Variable In(float value, params long[] shape)
        {
            Constant c = _constant(value, BackendUtil.CastShapeInt(shape));
            return c;
        }

        private Constant _constant(float value, int[] shape)
        {
            return new Constant(shape, CNTK.DataType.Float, initValue: (double)value, device: DeviceDescriptor.CPUDevice);
        }

        private NDArrayTensor Out(Variable x)
        {
            NDArrayTensor tensor = new NDArrayTensor
            {
                InternalTensor = x
            };
            return tensor;
        }

        public Tensor Abs(Tensor x)
        {
            return Out(C.Abs(In(x)));
        }

        public Tensor Acos(Tensor x)
        {
            return Out(C.Acos(In(x)));
        }

        public Tensor Add(Tensor a, Tensor b)
        {
            return Out(C.Plus(In(a), In(b)));
        }

        public Tensor Add(Tensor a, float b)
        {
            return Out(In(a) +In(b, a.Shape));
        }

        public Tensor Add(float a, Tensor b)
        {
            return Out(In(a, b.Shape) + In(b));
        }

        public Tensor Argmax(Tensor x, int dim = -1)
        {
            return Out(C.Argmax(In(x), new Axis(dim)));
        }

        public Tensor Argmin(Tensor x, int dim = -1)
        {
            return Out(C.Argmin(In(x), new Axis(dim)));
        }

        public Tensor Asin(Tensor x)
        {
            return Out(C.Asin(In(x)));
        }

        public Tensor Atan(Tensor x)
        {
            return Out(C.Atan(In(x)));
        }

        public Tensor Atan2(Tensor lhs, Tensor rhs)
        {
            throw new NotImplementedException();
        }

        public Tensor Ceil(Tensor x)
        {
            return Out(C.Ceil(In(x)));
        }

        public Tensor Clip(Tensor x, float min, float max)
        {
            return Out(C.Clip(In(x), In(min, x.Shape), In(max, x.Shape)));
        }

        public Tensor Col2Im(Tensor cols, long[] x_shape, Tuple<int, int> kernalSize, int padding = 1, int stride = 1)
        {
            throw new NotImplementedException();
        }

        public Tensor Constant(float value, long[] shape)
        {
            return Out(In(value, shape));
        }

        public Tensor Cos(Tensor x)
        {
            return Out(C.Cos(In(x)));
        }

        public Tensor Cosh(Tensor x)
        {
            return Out(C.Cosh(In(x)));
        }

        public Tensor CreateVariable(float[] data, long[] shape, string name = "")
        {
            throw new NotImplementedException();
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
            return Out(C.ElementDivide(In(a), In(b)));
        }

        public Tensor Div(Tensor a, float b)
        {
            return Out(C.ElementDivide(In(a), In(b, a.Shape)));
        }

        public Tensor Div(float a, Tensor b)
        {
            return Out(C.ElementDivide(In(a, b.Shape), In(b)));
        }

        public Tensor Dot(Tensor a, Tensor b)
        {
            return Out(C.MatMul(In(a), In(b)));
        }

        public float Epsilon()
        {
            return 1e-07f;
        }

        public Tensor EqualTo(Tensor a, Tensor b)
        {
            return Out(C.Equal(In(a), In(b)));
        }

        public object Eval(Tensor x)
        {
            return In(x);
        }

        public Tensor Exp(Tensor x)
        {
            return Out(C.Exp(In(x)));
        }

        public Tensor Factorial(Tensor x)
        {
            throw new NotImplementedException();
        }

        public Tensor Floor(Tensor x)
        {
            return Out(C.Floor(In(x)));
        }

        public Array GetArray(Tensor x)
        {
            throw new NotImplementedException();
        }

        public Engine.DataType GetDataType(Tensor x)
        {
            var d = In(x).DataType;
            Engine.DataType result = Engine.DataType.Float32;
            switch (d)
            {
                case CNTK.DataType.Unknown:
                    throw new NotSupportedException();
                    
                case CNTK.DataType.Float:
                    result = Engine.DataType.Float32;
                    break;
                case CNTK.DataType.Double:
                    result = Engine.DataType.Float64;
                    break;
                case CNTK.DataType.UChar:
                    throw new NotSupportedException();
                    
                case CNTK.DataType.Float16:
                    throw new NotSupportedException();
                    
                case CNTK.DataType.Int8:
                    result = Engine.DataType.Int8;
                    break;
                case CNTK.DataType.Int16:
                    result = Engine.DataType.Int32;
                    break;
                default:
                    break;
            }

            return result;
        }

        public long[] GetShape(Tensor x)
        {
            return Array.ConvertAll(In(x).Shape.Dimensions.ToArray(), i => (long)i);
        }

        public Tensor GreaterThan(Tensor a, Tensor b)
        {
            return Out(C.Greater(In(a), In(b)));
        }

        public Tensor GreaterThan(float a, Tensor b)
        {
            return Out(C.Greater(In(a, b.Shape), In(b)));
        }

        public Tensor GreaterThan(Tensor a, float b)
        {
            return Out(C.Greater(In(a), In(b, a.Shape)));
        }

        public Tensor GreaterThanEqual(Tensor a, Tensor b)
        {
            return Out(C.GreaterEqual(In(a), In(b)));
        }

        public Tensor GreaterThanEqual(float a, Tensor b)
        {
            return Out(C.GreaterEqual(In(a, b.Shape), In(b)));
        }

        public Tensor GreaterThanEqual(Tensor a, float b)
        {
            return Out(C.GreaterEqual(In(a), In(b, a.Shape)));
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
            return Out(C.Less(In(a), In(b)));
        }

        public Tensor LessThan(float a, Tensor b)
        {
            return Out(C.Less(In(a, b.Shape), In(b)));
        }

        public Tensor LessThan(Tensor a, float b)
        {
            return Out(C.Less(In(a), In(b, a.Shape)));
        }

        public Tensor LessThanEqual(Tensor a, Tensor b)
        {
            return Out(C.LessEqual(In(a), In(b)));
        }

        public Tensor LessThanEqual(float a, Tensor b)
        {
            return Out(C.LessEqual(In(a, b.Shape), In(b)));
        }

        public Tensor LessThanEqual(Tensor a, float b)
        {
            return Out(C.LessEqual(In(a), In(b, a.Shape)));
        }

        public Tensor Log(Tensor x)
        {
            return Out(C.Log(In(x)));
        }

        public Tensor Log10(Tensor x)
        {
            throw new NotImplementedException();
        }

        public Tensor Log1p(Tensor x)
        {
            throw new NotImplementedException();
        }

        public float Max(Tensor x)
        {
            throw new NotImplementedException();
        }

        public Tensor Max(Tensor x, int dim)
        {
            throw new NotImplementedException();
        }

        public Tensor Max(Tensor x, params int[] dim)
        {
            throw new NotImplementedException();
        }

        public Tensor Maximum(Tensor a, Tensor b)
        {
            throw new NotImplementedException();
        }

        public Tensor Maximum(Tensor a, float b)
        {
            throw new NotImplementedException();
        }

        public float Mean(Tensor x)
        {
            throw new NotImplementedException();
        }

        public Tensor Mean(Tensor x, int dim)
        {
            throw new NotImplementedException();
        }

        public Tensor Mean(Tensor x, params int[] dim)
        {
            throw new NotImplementedException();
        }

        public float Min(Tensor x)
        {
            throw new NotImplementedException();
        }

        public Tensor Min(Tensor x, int dim)
        {
            throw new NotImplementedException();
        }

        public Tensor Min(Tensor x, params int[] dim)
        {
            throw new NotImplementedException();
        }

        public Tensor Minimum(Tensor a, Tensor b)
        {
            throw new NotImplementedException();
        }

        public Tensor Minimum(Tensor a, float b)
        {
            throw new NotImplementedException();
        }

        public Tensor Mul(Tensor a, Tensor b)
        {
            throw new NotImplementedException();
        }

        public Tensor Mul(Tensor a, float b)
        {
            throw new NotImplementedException();
        }

        public Tensor Mul(float a, Tensor b)
        {
            throw new NotImplementedException();
        }

        public Tensor Neg(Tensor x)
        {
            throw new NotImplementedException();
        }

        public Tensor Pow(Tensor x, float value)
        {
            throw new NotImplementedException();
        }

        public void Print(Tensor x, string title = "")
        {
            throw new NotImplementedException();
        }

        public float Prod(Tensor x)
        {
            throw new NotImplementedException();
        }

        public Tensor Prod(Tensor x, int dim)
        {
            throw new NotImplementedException();
        }

        public Tensor Prod(Tensor x, params int[] dim)
        {
            throw new NotImplementedException();
        }

        public Tensor RandomBernoulli(long[] shape, float p)
        {
            throw new NotImplementedException();
        }

        public Tensor RandomNormal(long[] shape, float mean, float stddev, int? seed = null)
        {
            throw new NotImplementedException();
        }

        public Tensor RandomUniform(long[] shape, float min, float max, int? seed = null)
        {
            throw new NotImplementedException();
        }

        public Tensor Reshape(Tensor x, params long[] shape)
        {
            throw new NotImplementedException();
        }

        public Tensor Round(Tensor x)
        {
            throw new NotImplementedException();
        }

        public void SetDevice(DeviceType device)
        {
            throw new NotImplementedException();
        }

        public Tensor Sigmoid(Tensor x)
        {
            throw new NotImplementedException();
        }

        public Tensor Sign(Tensor x)
        {
            throw new NotImplementedException();
        }

        public Tensor Sin(Tensor x)
        {
            throw new NotImplementedException();
        }

        public Tensor Sinh(Tensor x)
        {
            throw new NotImplementedException();
        }

        public Tensor SliceCols(Tensor x, long start, long end)
        {
            throw new NotImplementedException();
        }

        public Tensor SliceRows(Tensor x, long start, long end)
        {
            throw new NotImplementedException();
        }

        public Tensor Softmax(Tensor x, int axis = -1)
        {
            throw new NotImplementedException();
        }

        public Tensor Softplus(Tensor x, int axis = -1)
        {
            throw new NotImplementedException();
        }

        public Tensor Sqrt(Tensor x)
        {
            throw new NotImplementedException();
        }

        public Tensor Square(Tensor x)
        {
            throw new NotImplementedException();
        }

        public float StdDev(Tensor x)
        {
            throw new NotImplementedException();
        }

        public Tensor StdDev(Tensor x, int dim)
        {
            throw new NotImplementedException();
        }

        public Tensor StdDev(Tensor x, params int[] dim)
        {
            throw new NotImplementedException();
        }

        public Tensor Sub(Tensor a, Tensor b)
        {
            throw new NotImplementedException();
        }

        public Tensor Sub(Tensor a, float b)
        {
            throw new NotImplementedException();
        }

        public Tensor Sub(float a, Tensor b)
        {
            throw new NotImplementedException();
        }

        public float Sum(Tensor x)
        {
            throw new NotImplementedException();
        }

        public Tensor Sum(Tensor x, int dim)
        {
            throw new NotImplementedException();
        }

        public Tensor Sum(Tensor x, params int[] dim)
        {
            throw new NotImplementedException();
        }

        public Tensor Tan(Tensor x)
        {
            throw new NotImplementedException();
        }

        public Tensor Tanh(Tensor x)
        {
            throw new NotImplementedException();
        }

        public Tensor Tile(Tensor x, int n, int axis = 0)
        {
            throw new NotImplementedException();
        }

        public Tensor Tpow(float value, Tensor x)
        {
            throw new NotImplementedException();
        }

        public Tensor Transpose(Tensor x)
        {
            throw new NotImplementedException();
        }

        public Tensor Transpose(Tensor x, params int[] dims)
        {
            throw new NotImplementedException();
        }

        public Tensor Trunc(Tensor x)
        {
            throw new NotImplementedException();
        }

        public string UUID(string name)
        {
            throw new NotImplementedException();
        }

        public float Var(Tensor x)
        {
            throw new NotImplementedException();
        }

        public Tensor Var(Tensor x, int dim)
        {
            throw new NotImplementedException();
        }

        public Tensor Var(Tensor x, params int[] dim)
        {
            throw new NotImplementedException();
        }

        public ActivationFunc GetActFunc()
        {
            throw new NotImplementedException();
        }
    }
}
