using SiaNet.Engine;
using SiaNet.Engine.Layers;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using System.Linq;
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
            this.session = new TFSession(new TFGraph());
            this.tf = session.Graph;

            runner = session.GetRunner();
        }

        public static SiaNetBackend Instance
        {
            get
            {
                return new SiaNetBackend();
            }
        }

        internal TFOutput In(Tensor x)
        {
            return tf.Const(((NDArrayTensor)x).InternalTensor, TFDataType.Float);
        }

        internal TFOutput In(float value, long[] shape)
        {
            return tf.Constant(value, new TFShape(shape), TFDataType.Float);
        }

        internal TFOutput In(params int[] data)
        {
            return tf.Const(new TFTensor(data), TFDataType.Int32);
        }

        internal TFOutput In(params long[] data)
        {
            return tf.Const(new TFTensor(data), TFDataType.Int64);
        }

        internal TFOutput In(int value)
        {
            return tf.Constant(1, new TFShape(1), TFDataType.Int32);
        }

        internal TFOutput In(long value)
        {
            return tf.Constant(1, new TFShape(1), TFDataType.Int64);
        }

        internal NDArrayTensor Out(TFOutput x, bool castFloat = false)
        {
            NDArrayTensor tensor = new NDArrayTensor();
            if(castFloat)
            {
                tensor.InternalTensor = InternalEval(tf.Cast(x, TFDataType.Float));
            }
            else
            {
                tensor.InternalTensor = InternalEval(x);
            }

            return tensor;
        }

        internal NDArrayTensor Out(TFTensor x)
        {
            NDArrayTensor tensor = new NDArrayTensor
            {
                InternalTensor = x
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
            return Out(tf.Add(In(a), In(b, a.Shape)));
        }

        public Tensor Add(float a, Tensor b)
        {
            return Out(tf.Add(In(a, b.Shape), In(b)));
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
            return Out(tf.ClipByValue(In(x), In(min, x.Shape), In(max, x.Shape)));
        }

        public Tensor Col2Im(Tensor cols, long[] x_shape, Tuple<int, int> kernalSize, int padding = 1, int stride = 1)
        {
            throw new NotImplementedException();
        }

        public Tensor Constant(float value, long[] shape)
        {
            return Out(tf.Constant(value, new TFShape(shape), TFDataType.Float));
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
            
        }

        public Tensor Div(Tensor a, Tensor b)
        {
            return Out(tf.Div(In(a), In(b)));
        }

        public Tensor Div(Tensor a, float b)
        {
            return Out(tf.Div(In(a), In(b, a.Shape)));
        }

        public Tensor Div(float a, Tensor b)
        {
            return Out(tf.Div(In(a, b.Shape), In(b)));
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
            return Out(tf.Equal(In(a), In(b)), true);
        }

        public object Eval(Tensor x)
        {
            TFTensor[] result = session.Run(new TFOutput[] { }, new TFTensor[] { }, new[] { In(x) });
            return result[0].GetValue();
        }

        public TFTensor InternalEval(TFOutput x)
        {
            TFTensor[] result = session.Run(new TFOutput[] { }, new TFTensor[] { }, new[] { x });
            return result[0];
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
            return (Array)((NDArrayTensor)x).InternalTensor.GetValue();
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
            long[] shape = tf.GetShape(In(x));
            if(shape == null)
            {
                shape = new long[] { 1 };
            }

            return shape;
        }

        public Tensor GreaterThan(Tensor a, Tensor b)
        {
            return Out(tf.Greater(In(a), In(b)), true);
        }

        public Tensor GreaterThan(float a, Tensor b)
        {
            return Out(tf.Greater(In(a, b.Shape), In(b)), true);
        }

        public Tensor GreaterThan(Tensor a, float b)
        {
            return Out(tf.Greater(In(a), In(b, a.Shape)), true);
        }

        public Tensor GreaterThanEqual(Tensor a, Tensor b)
        {
            return Out(tf.GreaterEqual(In(a), In(b)), true);
        }

        public Tensor GreaterThanEqual(float a, Tensor b)
        {
            return Out(tf.GreaterEqual(In(a, b.Shape), In(b)), true);
        }

        public Tensor GreaterThanEqual(Tensor a, float b)
        {
            return Out(tf.GreaterEqual(In(a), In(b, a.Shape)), true);
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
            return Out(tf.Less(In(a), In(b)), true);
        }

        public Tensor LessThan(float a, Tensor b)
        {
            return Out(tf.Less(In(a, b.Shape), In(b)), true);
        }

        public Tensor LessThan(Tensor a, float b)
        {
            return Out(tf.Less(In(a), In(b, a.Shape)), true);
        }

        public Tensor LessThanEqual(Tensor a, Tensor b)
        {
            return Out(tf.LessEqual(In(a), In(b)), true);
        }

        public Tensor LessThanEqual(float a, Tensor b)
        {
            return Out(tf.LessEqual(In(a, b.Shape), In(b)), true);
        }

        public Tensor LessThanEqual(Tensor a, float b)
        {
            return Out(tf.LessEqual(In(a), In(b, a.Shape)), true);
        }

        public Tensor Log(Tensor x)
        {
            return Out(tf.Log(In(x)));
        }

        public Tensor Log10(Tensor x)
        {
            throw new NotImplementedException();
        }

        public Tensor Log1p(Tensor x)
        {
            return Out(tf.Log1p(In(x)));
        }

        public float Max(Tensor x)
        {
            return Out(tf.Max(In(x), In(x.Shape))).ToScalar();
        }

        public Tensor Max(Tensor x, int dim)
        {
            return Out(tf.Max(In(x), In(dim), true));
        }

        public Tensor Max(Tensor x, params int[] dim)
        {
            return Out(tf.Max(In(x), In(dim), true));
        }

        public Tensor Maximum(Tensor a, Tensor b)
        {
            return Out(tf.Maximum(In(a), In(b)));
        }

        public Tensor Maximum(Tensor a, float b)
        {
            return Out(tf.Maximum(In(a), In(b, a.Shape)));
        }

        public float Mean(Tensor x)
        {
            return (float)InternalEval(tf.ReduceMean(In(x))).GetValue();
        }

        public Tensor Mean(Tensor x, int dim)
        {
            return Out(tf.Mean(In(x), In(dim), true));
        }

        public Tensor Mean(Tensor x, params int[] dim)
        {
            return Out(tf.Mean(In(x), In(dim), true));
        }

        public float Min(Tensor x)
        {
            return Out(tf.Min(In(x), In(x.Shape), true)).ToScalar();
        }

        public Tensor Min(Tensor x, int dim)
        {
            return Out(tf.Min(In(x), In(dim), true));
        }

        public Tensor Min(Tensor x, params int[] dim)
        {
            return Out(tf.Min(In(x), In(dim), true));
        }

        public Tensor Minimum(Tensor a, Tensor b)
        {
            return Out(tf.Minimum(In(a), In(b)));
        }

        public Tensor Minimum(Tensor a, float b)
        {
            return Out(tf.Minimum(In(a), In(b, a.Shape)));
        }

        public Tensor Mul(Tensor a, Tensor b)
        {
            return Out(tf.Mul(In(a), In(b)));
        }

        public Tensor Mul(Tensor a, float b)
        {
            return Out(tf.Mul(In(a), In(b, a.Shape)));
        }

        public Tensor Mul(float a, Tensor b)
        {
            return Out(tf.Mul(In(a, b.Shape), In(b)));
        }

        public Tensor Neg(Tensor x)
        {
            return Out(tf.Neg(In(x)));
        }

        public Tensor Pow(Tensor x, float value)
        {
            return Out(tf.Pow(In(x), In(value, x.Shape)));
        }

        public void Print(Tensor x, string title = "")
        {
            string message = tf.Print(In(x), new TFOutput[] { In(x) }).ToString() ;
        }

        public Tensor RandomBernoulli(long[] shape, float p)
        {
            var result = RandomUniform(shape, 0, 1);
            result = result > p;
            return result;
        }

        public Tensor RandomNormal(long[] shape, float mean, float stddev, int? seed = null)
        {
            return Out(tf.Cast(tf.RandomNormal(new TFShape(shape), mean, stddev, seed), TFDataType.Float));
        }

        public Tensor RandomUniform(long[] shape, float min, float max, int? seed = null)
        {
            return Out(tf.Cast(tf.RandomUniform(new TFShape(shape), min, max, seed), TFDataType.Float));
        }

        public Tensor Reshape(Tensor x, params long[] shape)
        {
            shape = shape.ToArray();
            long prod = -1 * shape.Aggregate(1L, (a, b) => a * b);
            for (int i = 0; i < shape.Length; i++)
            {
                if (shape[i] == -1)
                {
                    shape[i] = x.ElementCount / prod;
                    break;
                }
            }

            return Out(tf.Reshape(In(x), In(shape)));
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
            end = end + 1;
            long size = end - start;
            long[] shape = x.Shape;
            return Out(tf.Slice(In(x), In(0, start), In(x.Shape[0], size)));
        }

        public Tensor SliceRows(Tensor x, long start, long end)
        {
            end = end + 1;
            long size = end - start;
            long[] shape = x.Shape;
            long[] startIndex = new long[shape.Length];
            startIndex[0] = start;

            long[] endIndex = new long[shape.Length];
            for (int i = 0; i < endIndex.Length; i++)
            {
                if(i==0)
                {
                    endIndex[i] = size;
                    continue;
                }

                endIndex[i] = shape[i];
            }

            return Out(tf.Slice(In(x), In(startIndex), In(endIndex)));
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

        public Tensor Sub(Tensor a, Tensor b)
        {
            return Out(tf.Sub(In(a), In(b)));
        }

        public Tensor Sub(Tensor a, float b)
        {
            return Out(tf.Sub(In(a), In(b, a.Shape)));
        }

        public Tensor Sub(float a, Tensor b)
        {
            return Out(tf.Sub(In(a, b.Shape), In(b)));
        }

        public float Sum(Tensor x)
        {
            return (float)InternalEval(tf.ReduceSum(In(x))).GetValue();
        }

        public Tensor Sum(Tensor x, int dim)
        {
            dim = dim < 0 ? x.DimCount + dim : dim;
            return Out(tf.Sum(In(x), In(dim), true));
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
            if (axis < 0)
                throw new ArgumentException("Axis >= 0");

            int[] times = new int[x.Shape.Length];
            for (int i = 0; i < times.Length; i++)
            {
                if(i == axis)
                {
                    times[i] = 1;
                    continue;
                }

                times[i] = n;
            }

            return Out(tf.Tile(In(x), In(times)));
        }

        public Tensor Tpow(float value, Tensor x)
        {
            return Out(tf.Pow(In(value, x.Shape), In(x)));
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

        public ActivationFunc GetActFunc()
        {
            return new SiaNetActivations(this);
        }
    }
}
