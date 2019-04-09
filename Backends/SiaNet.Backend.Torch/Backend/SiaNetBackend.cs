using SiaNet.Engine;
using SiaNet.Engine.Layers;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using System.Linq;

namespace SiaNet.Backend.Torch
{
    public class SiaNetBackend : IBackend
    {
        int counter = 0;

        internal NDArrayTensor Out(FloatTensor x)
        {
            return new NDArrayTensor(x);
        }

        internal FloatTensor In(Tensor x)
        {
            return ((NDArrayTensor)x).InternalTensor;
        }

        internal FloatTensor In(float value, long[] shape)
        {
            FloatTensor floatTensor = new FloatTensor(shape);
            floatTensor.Fill(value);
            return floatTensor;
        }

        public Tensor Abs(Tensor x)
        {
            return Out(In(x).Abs());
        }

        public Tensor Acos(Tensor x)
        {
            return Out(In(x).Acos());
        }

        public Tensor Add(Tensor a, Tensor b)
        {
            return Out(In(a).CAdd(1, In(b)));
        }

        public Tensor Add(Tensor a, float b)
        {
            return Out(In(a).Add(b));
        }

        public Tensor Add(float a, Tensor b)
        {
            return Out(In(b).Add(a));
        }

        public Tensor Argmax(Tensor x, int dim = 0)
        {
            throw new NotImplementedException();
        }

        public Tensor Argmin(Tensor x, int dim = 0)
        {
            throw new NotImplementedException();
        }

        public Tensor Asin(Tensor x)
        {
            return Out(In(x).Asin());
        }

        public Tensor Atan(Tensor x)
        {
            return Out(In(x).Atan());
        }

        public Tensor Ceil(Tensor x)
        {
            return Out(In(x).Ceil());
        }

        public Tensor Clip(Tensor x, float min, float max)
        {
            return Out(In(x).Clamp(min, max));
        }

        public Tensor Col2Im(Tensor cols, long[] x_shape, Tuple<int, int> kernalSize, int padding = 1, int stride = 1)
        {
            throw new NotImplementedException();
        }

        public Tensor Constant(float value, long[] shape)
        {
            FloatTensor floatTensor = new FloatTensor(shape);
            floatTensor.Fill(value);
            return Out(floatTensor);
        }

        public Tensor Cos(Tensor x)
        {
            return Out(In(x).Cos());
        }

        public Tensor Cosh(Tensor x)
        {
            return Out(In(x).Cosh());
        }

        public Tensor CreateVariable(float[] data, long[] shape, string name = "")
        {
            FloatTensor floatTensor = new FloatTensor(shape);
            var ptr = Marshal.UnsafeAddrOfPinnedArrayElement(data, 0);
            var tensor = FloatTensor.NewWithStorage1d(new FloatTensor.FloatStorage(new FloatTensor.FloatStorage.HType(ptr, false)), UIntPtr.Zero, data.Length, 1);
            return Reshape(Out(tensor), shape);
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
            return Out(In(a).CDiv(In(b)));
        }

        public Tensor Div(Tensor a, float b)
        {
            return Out(In(a).Div(b));
        }

        public Tensor Div(float a, Tensor b)
        {
            return Out(In(a, b.Shape).CDiv(In(b)));
        }

        public Tensor Dot(Tensor a, Tensor b)
        {
            throw new NotImplementedException();
        }

        public float Epsilon()
        {
            return 1e-7f;
        }

        public Tensor EqualTo(Tensor a, Tensor b)
        {
            return Out(In(a).EqTensorT(In(b)));
        }

        public object Eval(Tensor x)
        {
            return In(x);
        }

        public Tensor Exp(Tensor x)
        {
            return Out(In(x).Exp());
        }

        public Tensor Floor(Tensor x)
        {
            return Out(In(x).Floor());
        }

        public ActivationFunc GetActFunc()
        {
            return new SiaNetActivations(this);
        }

        public Array GetArray(Tensor x)
        {
            Array result = Array.CreateInstance(typeof(float), x.ElementCount);

            var datagch = GCHandle.Alloc(result, GCHandleType.Pinned);
            var tensor = In(x);

            return result;
        }

        public DataType GetDataType(Tensor x)
        {
            throw new NotImplementedException();
        }

        public long[] GetShape(Tensor x)
        {
            throw new NotImplementedException();
        }

        public Tensor GreaterThan(Tensor a, Tensor b)
        {
            throw new NotImplementedException();
        }

        public Tensor GreaterThan(float a, Tensor b)
        {
            throw new NotImplementedException();
        }

        public Tensor GreaterThan(Tensor a, float b)
        {
            throw new NotImplementedException();
        }

        public Tensor GreaterThanEqual(Tensor a, Tensor b)
        {
            throw new NotImplementedException();
        }

        public Tensor GreaterThanEqual(float a, Tensor b)
        {
            throw new NotImplementedException();
        }

        public Tensor GreaterThanEqual(Tensor a, float b)
        {
            throw new NotImplementedException();
        }

        public Tensor Im2Col(Tensor x, Tuple<int, int> kernalSize, int padding = 1, int stride = 1)
        {
            throw new NotImplementedException();
        }

        public Tensor L2Normalize(Tensor x, int axis = -1)
        {
            throw new NotImplementedException();
        }

        public Tensor LessThan(Tensor a, Tensor b)
        {
            throw new NotImplementedException();
        }

        public Tensor LessThan(float a, Tensor b)
        {
            throw new NotImplementedException();
        }

        public Tensor LessThan(Tensor a, float b)
        {
            throw new NotImplementedException();
        }

        public Tensor LessThanEqual(Tensor a, Tensor b)
        {
            throw new NotImplementedException();
        }

        public Tensor LessThanEqual(float a, Tensor b)
        {
            throw new NotImplementedException();
        }

        public Tensor LessThanEqual(Tensor a, float b)
        {
            throw new NotImplementedException();
        }

        public Tensor Log(Tensor x)
        {
            throw new NotImplementedException();
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
            var tensor = In(x);
            if(shape.Length == 2)
            {
                tensor.Resize2d(shape[0], shape[1]);
            }
            else if (shape.Length == 3)
            {
                tensor.Resize3d(shape[0], shape[1], shape[2]);
            }
            else if (shape.Length == 4)
            {
                tensor.Resize4d(shape[0], shape[1], shape[2], shape[3]);
            }
            else if (shape.Length == 5)
            {
                tensor.Resize5d(shape[0], shape[1], shape[2], shape[3], shape[4]);
            }

            return Out(tensor);
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

        public Tensor Transpose(Tensor x)
        {
            throw new NotImplementedException();
        }

        public Tensor Transpose(Tensor x, params int[] dims)
        {
            throw new NotImplementedException();
        }

        public string UUID(string name)
        {
            throw new NotImplementedException();
        }
    }
}
