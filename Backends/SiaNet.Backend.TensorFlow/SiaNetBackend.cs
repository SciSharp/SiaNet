using SiaNet.Engine;
using SiaNet.Engine.Layers;
using System;
using System.Collections.Generic;
using System.Text;
using TensorFlow;
using DeviceType = SiaNet.Engine.DeviceType;

namespace SiaNet.Backend.TensorFlowLib
{
    public class SiaNetBackend : IBackend
    {
        int counter = 0;
        internal TFGraph tf;
        internal TFSession session;
        private TFSession.Runner runner;

        public SiaNetBackend()
        {
            
            this.tf = new TFGraph();
            this.session = new TFSession(tf);

            runner = session.GetRunner();
        }

        public static SiaNetBackend Instance
        {
            get
            {
                return new SiaNetBackend();
            }
        }

        private TFOutput In(Tensor x)
        {
            return ((NDArrayTensor)x).InternalTensor;
        }

        private TFOutput In(float value, params long[] shape)
        {
            return tf.Const(new TFTensor(value));
        }

        private TFOutput In(int value)
        {
            return tf.Const(new TFTensor(value));
        }

        private NDArrayTensor Out(TFOutput x)
        {
            NDArrayTensor tensor = new NDArrayTensor
            {
                InternalTensor = x
            };

            return tensor;
        }

        private NDArrayTensor Out(TFTensor x)
        {
            NDArrayTensor tensor = new NDArrayTensor
            {
                InternalTensor = tf.Const(x)
            };

            return tensor;
        }

        public Tensor Abs(Tensor x)
        {
            return Out(tf.Abs(In(x)));
        }

        public Tensor Acos(Tensor x)
        {
            return Out(tf.Acos(In(x)));
        }

        public Tensor Add(Tensor a, Tensor b)
        {
            return Out(tf.Add(In(a), In(b)));
        }

        public Tensor Add(Tensor a, float b)
        {
            return Out(tf.Add(In(a), In(b)));
        }

        public Tensor Add(float a, Tensor b)
        {
            return Out(tf.Add(In(a), In(b)));
        }

        public Tensor Argmax(Tensor x, int dim = 0)
        {
            return Out(tf.ArgMax(In(x), In(dim)));
        }

        public Tensor Argmin(Tensor x, int dim = 0)
        {
            return Out(tf.ArgMin(In(x), In(dim)));
        }

        public Tensor Asin(Tensor x)
        {
            return Out(tf.Asin(In(x)));
        }

        public Tensor Atan(Tensor x)
        {
            return Out(tf.Atan(In(x)));
        }

        public Tensor Atan2(Tensor lhs, Tensor rhs)
        {
            return Out(tf.Atan2(In(lhs), In(rhs)));
        }

        public Tensor Ceil(Tensor x)
        {
            return Out(tf.Ceil(In(x)));
        }

        public Tensor Clip(Tensor x, float min, float max)
        {
            return Out(tf.ClipByValue(In(x), In(min), In(max)));
        }

        public Tensor Col2Im(Tensor cols, long[] x_shape, Tuple<int, int> kernalSize, int padding = 1, int stride = 1)
        {
            throw new NotImplementedException();
        }

        public Tensor Constant(float value, long[] shape)
        {
            return Out(tf.Constant(value, new TFShape(shape)));
        }

        public Tensor Cos(Tensor x)
        {
            return Out(tf.Cos(In(x)));
        }

        public Tensor Cosh(Tensor x)
        {
            return Out(tf.Cosh(In(x)));
        }

        public Tensor CreateVariable(float[] data, long[] shape, string name = "")
        {
            var t = tf.Const(new TFTensor(data));
            t = tf.Reshape(t, tf.Const(new TFShape(shape).AsTensor()));
            return Out(t);
        }

        public Tensor Diag(Tensor x)
        {
            return Out(tf.Diag(In(x)));
        }

        public void Dispose(Tensor x)
        {
            x.Dispose();
        }

        public Tensor Div(Tensor a, Tensor b)
        {
            return Out(tf.Div(In(a), In(b)));
        }

        public Tensor Div(Tensor a, float b)
        {
            return Out(tf.Div(In(a), In(b)));
        }

        public Tensor Div(float a, Tensor b)
        {
            return Out(tf.Div(In(a), In(b)));
        }

        public Tensor Dot(Tensor a, Tensor b)
        {
            return Out(tf.MatMul(In(a), In(b)));
        }

        public float Epsilon()
        {
            return 1e-7f;
        }

        public Tensor EqualTo(Tensor a, Tensor b)
        {
            return Out(tf.Equal(In(a), In(b)));
        }

        public object Eval(Tensor x)
        {
            return runner.Run(In(x));
        }

        public Tensor Exp(Tensor x)
        {
            return Out(tf.Exp(In(x)));
        }

        public Tensor Floor(Tensor x)
        {
            return Out(tf.Floor(In(x)));
        }

        public Array GetArray(Tensor x)
        {
            throw new NotImplementedException();
        }

        public DataType GetDataType(Tensor x)
        {
            DataType r = DataType.Float32;

            switch (In(x).OutputType)
            {
                case TFDataType.Unknown:
                    break;
                case TFDataType.Float:
                    r = DataType.Float32;
                    break;
                case TFDataType.Double:
                    r = DataType.Float64;
                    break;
                case TFDataType.Int32:
                    r = DataType.Int32;
                    break;
                case TFDataType.UInt8:
                    r = DataType.Int8;
                    break;
                case TFDataType.Int16:
                    break;
                case TFDataType.Int8:
                    r = DataType.Int8;
                    break;
                default:
                    throw new NotSupportedException();
            }

            return r;
        }

        public long[] GetShape(Tensor x)
        {
            //ToDo: Check the code again
            return runner.Run(In(x)).Shape;
        }

        public Tensor GreaterThan(Tensor a, Tensor b)
        {
            return Out(tf.Greater(In(a), In(b)));
        }

        public Tensor GreaterThan(float a, Tensor b)
        {
            return Out(tf.Greater(In(a), In(b)));
        }

        public Tensor GreaterThan(Tensor a, float b)
        {
            return Out(tf.Greater(In(a), In(b)));
        }

        public Tensor GreaterThanEqual(Tensor a, Tensor b)
        {
            return Out(tf.GreaterEqual(In(a), In(b)));
        }

        public Tensor GreaterThanEqual(float a, Tensor b)
        {
            return Out(tf.GreaterEqual(In(a), In(b)));
        }

        public Tensor GreaterThanEqual(Tensor a, float b)
        {
            return Out(tf.GreaterEqual(In(a), In(b)));
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
            var result = RandomUniform(shape, 0, 1);
            result = result > p;
            return result;
        }

        public Tensor RandomNormal(long[] shape, float mean, float stddev, int? seed = null)
        {
            return Out(tf.RandomNormal(new TFShape(shape), mean, stddev, seed));
        }

        public Tensor RandomUniform(long[] shape, float min, float max, int? seed = null)
        {
            return Out(tf.RandomUniform(new TFShape(shape), min, max, seed));
        }

        public Tensor Reshape(Tensor x, params long[] shape)
        {
            return Out(tf.Reshape(In(x), tf.Const(new TFTensor(shape))));
        }

        public Tensor Round(Tensor x)
        {
            return Out(tf.Round(In(x)));
        }

        public void SetDevice(DeviceType device)
        {
            switch (device)
            {
                case DeviceType.Default:
                    break;
                case DeviceType.CPU:
                    break;
                case DeviceType.CUDA:
                    break;
                case DeviceType.OpenCL:
                    throw new NotSupportedException("Supported device CPU and CUDA");
                default:
                    break;
            }
        }

        public Tensor Sigmoid(Tensor x)
        {
            return Out(tf.Sigmoid(In(x)));
        }

        public Tensor Sign(Tensor x)
        {
            return Out(tf.Sign(In(x)));
        }

        public Tensor Sin(Tensor x)
        {
            return Out(tf.Sin(In(x)));
        }

        public Tensor Sinh(Tensor x)
        {
            return Out(tf.Sinh(In(x)));
        }

        public Tensor SliceCols(Tensor x, long start, long end)
        {
            throw new NotImplementedException();
        }

        public Tensor SliceRows(Tensor x, long start, long end)
        {
            return Out(tf.Slice(In(x), In(start), In(end)));
        }

        public Tensor Softmax(Tensor x, int axis = -1)
        {
            return Out(tf.Softmax(In(x)));
        }

        public Tensor Softplus(Tensor x, int axis = -1)
        {
            return Out(tf.Softplus(In(x)));
        }

        public Tensor Sqrt(Tensor x)
        {
            return Out(tf.Sqrt(In(x)));
        }

        public Tensor Square(Tensor x)
        {
            return Out(tf.Square(In(x)));
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
            return Out(tf.Sub(In(a), In(b)));
        }

        public Tensor Sub(Tensor a, float b)
        {
            return Out(tf.Sub(In(a), In(b)));
        }

        public Tensor Sub(float a, Tensor b)
        {
            return Out(tf.Sub(In(a), In(b)));
        }

        public float Sum(Tensor x)
        {
            return Out(tf.ReduceSum(In(x))).ToScalar();
        }

        public Tensor Sum(Tensor x, int dim)
        {
            dim = dim < 0 ? x.DimCount + dim : dim;
            return Out(tf.ReduceSum(In(x), In(dim)));
        }

        public Tensor Sum(Tensor x, params int[] dim)
        {
            foreach (var item in dim)
            {
                x = Sum(x, item);
            }

            return x;
        }

        public Tensor Tan(Tensor x)
        {
            return Out(tf.Tan(In(x)));
        }

        public Tensor Tanh(Tensor x)
        {
            return Out(tf.Tanh(In(x)));
        }

        public Tensor Tile(Tensor x, int n, int axis = 0)
        {
            return Out(tf.Tile(In(x), In(n)));
        }

        public Tensor Tpow(float value, Tensor x)
        {
            return Out(tf.Pow(In(value), In(x)));
        }

        public Tensor Transpose(Tensor x)
        {
            return Out(tf.Transpose(In(x)));
        }

        public Tensor Transpose(Tensor x, params int[] dims)
        {
            return Out(tf.Transpose(In(x), tf.Const(new TFTensor(dims))));
        }

        public Tensor Trunc(Tensor x)
        {
            return Out(tf.TruncatedNormal(In(x), TFDataType.Float));
        }

        public string UUID(string name)
        {
            return name + "_" + counter++;
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
