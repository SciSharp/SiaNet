using SiaNet.Engine;
using SiaNet.Engine.Layers;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using System.Linq;
using Tensorflow;
using DeviceType = SiaNet.Engine.DeviceType;
using SiaTensor = SiaNet.Engine.Tensor;

namespace SiaNet.Backend.TensorFlowLib
{
    public class SiaNetBackend : IBackend
    {
        int counter = 0;
        internal Graph graph;
        internal Session sess;

        public SiaNetBackend()
        {
            graph = tf.Graph();
            sess = tf.Session(graph);
        }

        public static SiaNetBackend Instance
        {
            get
            {
                return new SiaNetBackend();
            }
        }

        internal Tensorflow.Tensor In<T>(T value, long[] shape = null)
        {
            return tf.constant(value, 
                shape: shape?.Select(x => Convert.ToInt32(x))?.ToArray(), 
                dtype: TF_DataType.TF_FLOAT);
        }

        public float Epsilon()
        {
            return 1e-7f;
        }

        public long[] GetShape(SiaTensor x)
        {
            return BackendUtil.Int2Long(In(x).getShape().Dimensions);
        }

        public object Eval(SiaTensor x)
        {
            return In(x).eval();
        }

        public void Print(SiaTensor x, string title = "")
        {
            throw new NotImplementedException();
        }

        public string UUID(string name)
        {
            throw new NotImplementedException();
        }

        public void SetDevice(DeviceType device)
        {
            throw new NotImplementedException();
        }

        public Engine.DataType GetDataType(SiaTensor x)
        {
            throw new NotImplementedException();
        }

        public SiaTensor CreateVariable(float[] data, long[] shape, string name = "")
        {
            throw new NotImplementedException();
        }

        public SiaTensor Reshape(SiaTensor x, params long[] shape)
        {
            throw new NotImplementedException();
        }

        public SiaTensor Constant(float value, long[] shape)
        {
            throw new NotImplementedException();
        }

        public SiaTensor RandomUniform(long[] shape, float min, float max, int? seed = null)
        {
            throw new NotImplementedException();
        }

        public SiaTensor RandomNormal(long[] shape, float mean, float stddev, int? seed = null)
        {
            throw new NotImplementedException();
        }

        public SiaTensor RandomBernoulli(long[] shape, float p)
        {
            throw new NotImplementedException();
        }

        public SiaTensor Add(SiaTensor a, SiaTensor b)
        {
            throw new NotImplementedException();
        }

        public SiaTensor Add(SiaTensor a, float b)
        {
            throw new NotImplementedException();
        }

        public SiaTensor Add(float a, SiaTensor b)
        {
            throw new NotImplementedException();
        }

        public SiaTensor Sub(SiaTensor a, SiaTensor b)
        {
            throw new NotImplementedException();
        }

        public SiaTensor Sub(SiaTensor a, float b)
        {
            throw new NotImplementedException();
        }

        public SiaTensor Sub(float a, SiaTensor b)
        {
            throw new NotImplementedException();
        }

        public SiaTensor Mul(SiaTensor a, SiaTensor b)
        {
            throw new NotImplementedException();
        }

        public SiaTensor Mul(SiaTensor a, float b)
        {
            throw new NotImplementedException();
        }

        public SiaTensor Mul(float a, SiaTensor b)
        {
            throw new NotImplementedException();
        }

        public SiaTensor Div(SiaTensor a, SiaTensor b)
        {
            throw new NotImplementedException();
        }

        public SiaTensor Div(SiaTensor a, float b)
        {
            throw new NotImplementedException();
        }

        public SiaTensor Div(float a, SiaTensor b)
        {
            throw new NotImplementedException();
        }

        public SiaTensor GreaterThan(SiaTensor a, SiaTensor b)
        {
            throw new NotImplementedException();
        }

        public SiaTensor GreaterThanEqual(SiaTensor a, SiaTensor b)
        {
            throw new NotImplementedException();
        }

        public SiaTensor LessThan(SiaTensor a, SiaTensor b)
        {
            throw new NotImplementedException();
        }

        public SiaTensor LessThanEqual(SiaTensor a, SiaTensor b)
        {
            throw new NotImplementedException();
        }

        public SiaTensor GreaterThan(float a, SiaTensor b)
        {
            throw new NotImplementedException();
        }

        public SiaTensor GreaterThanEqual(float a, SiaTensor b)
        {
            throw new NotImplementedException();
        }

        public SiaTensor LessThan(float a, SiaTensor b)
        {
            throw new NotImplementedException();
        }

        public SiaTensor LessThanEqual(float a, SiaTensor b)
        {
            throw new NotImplementedException();
        }

        public SiaTensor GreaterThan(SiaTensor a, float b)
        {
            throw new NotImplementedException();
        }

        public SiaTensor GreaterThanEqual(SiaTensor a, float b)
        {
            throw new NotImplementedException();
        }

        public SiaTensor LessThan(SiaTensor a, float b)
        {
            throw new NotImplementedException();
        }

        public SiaTensor LessThanEqual(SiaTensor a, float b)
        {
            throw new NotImplementedException();
        }

        public SiaTensor EqualTo(SiaTensor a, SiaTensor b)
        {
            throw new NotImplementedException();
        }

        public SiaTensor Tile(SiaTensor x, int n, int axis = 0)
        {
            throw new NotImplementedException();
        }

        public SiaTensor Abs(SiaTensor x)
        {
            throw new NotImplementedException();
        }

        public SiaTensor Neg(SiaTensor x)
        {
            throw new NotImplementedException();
        }

        public SiaTensor Sqrt(SiaTensor x)
        {
            throw new NotImplementedException();
        }

        public SiaTensor Exp(SiaTensor x)
        {
            throw new NotImplementedException();
        }

        public SiaTensor Log(SiaTensor x)
        {
            throw new NotImplementedException();
        }

        public SiaTensor Log10(SiaTensor x)
        {
            throw new NotImplementedException();
        }

        public SiaTensor Log1p(SiaTensor x)
        {
            throw new NotImplementedException();
        }

        public SiaTensor Floor(SiaTensor x)
        {
            throw new NotImplementedException();
        }

        public SiaTensor Ceil(SiaTensor x)
        {
            throw new NotImplementedException();
        }

        public SiaTensor Round(SiaTensor x)
        {
            throw new NotImplementedException();
        }

        public SiaTensor Sin(SiaTensor x)
        {
            throw new NotImplementedException();
        }

        public SiaTensor Cos(SiaTensor x)
        {
            throw new NotImplementedException();
        }

        public SiaTensor Tan(SiaTensor x)
        {
            throw new NotImplementedException();
        }

        public SiaTensor Asin(SiaTensor x)
        {
            throw new NotImplementedException();
        }

        public SiaTensor Acos(SiaTensor x)
        {
            throw new NotImplementedException();
        }

        public SiaTensor Atan(SiaTensor x)
        {
            throw new NotImplementedException();
        }

        public SiaTensor Sinh(SiaTensor x)
        {
            throw new NotImplementedException();
        }

        public SiaTensor Cosh(SiaTensor x)
        {
            throw new NotImplementedException();
        }

        public SiaTensor Tanh(SiaTensor x)
        {
            throw new NotImplementedException();
        }

        public SiaTensor Sigmoid(SiaTensor x)
        {
            throw new NotImplementedException();
        }

        public SiaTensor Pow(SiaTensor x, float value)
        {
            throw new NotImplementedException();
        }

        public SiaTensor Square(SiaTensor x)
        {
            throw new NotImplementedException();
        }

        public SiaTensor Clip(SiaTensor x, float min, float max)
        {
            throw new NotImplementedException();
        }

        public float Sum(SiaTensor x)
        {
            throw new NotImplementedException();
        }

        public SiaTensor Sum(SiaTensor x, int dim)
        {
            throw new NotImplementedException();
        }

        public SiaTensor Sum(SiaTensor x, params int[] dim)
        {
            throw new NotImplementedException();
        }

        public float Max(SiaTensor x)
        {
            throw new NotImplementedException();
        }

        public SiaTensor Max(SiaTensor x, int dim)
        {
            throw new NotImplementedException();
        }

        public SiaTensor Max(SiaTensor x, params int[] dim)
        {
            throw new NotImplementedException();
        }

        public float Min(SiaTensor x)
        {
            throw new NotImplementedException();
        }

        public SiaTensor Min(SiaTensor x, int dim)
        {
            throw new NotImplementedException();
        }

        public SiaTensor Min(SiaTensor x, params int[] dim)
        {
            throw new NotImplementedException();
        }

        public float Mean(SiaTensor x)
        {
            throw new NotImplementedException();
        }

        public SiaTensor Mean(SiaTensor x, int dim)
        {
            throw new NotImplementedException();
        }

        public SiaTensor Mean(SiaTensor x, params int[] dim)
        {
            throw new NotImplementedException();
        }

        public SiaTensor Argmax(SiaTensor x, int dim = 0)
        {
            throw new NotImplementedException();
        }

        public SiaTensor Argmin(SiaTensor x, int dim = 0)
        {
            throw new NotImplementedException();
        }

        public SiaTensor Maximum(SiaTensor a, SiaTensor b)
        {
            throw new NotImplementedException();
        }

        public SiaTensor Maximum(SiaTensor a, float b)
        {
            throw new NotImplementedException();
        }

        public SiaTensor Minimum(SiaTensor a, SiaTensor b)
        {
            throw new NotImplementedException();
        }

        public SiaTensor Minimum(SiaTensor a, float b)
        {
            throw new NotImplementedException();
        }

        public SiaTensor Transpose(SiaTensor x)
        {
            throw new NotImplementedException();
        }

        public SiaTensor Transpose(SiaTensor x, params int[] dims)
        {
            throw new NotImplementedException();
        }

        public SiaTensor Dot(SiaTensor a, SiaTensor b)
        {
            throw new NotImplementedException();
        }

        public SiaTensor Diag(SiaTensor x)
        {
            throw new NotImplementedException();
        }

        public SiaTensor Softmax(SiaTensor x, int axis = -1)
        {
            throw new NotImplementedException();
        }

        public SiaTensor Softplus(SiaTensor x, int axis = -1)
        {
            throw new NotImplementedException();
        }

        public SiaTensor L2Normalize(SiaTensor x, int axis = -1)
        {
            throw new NotImplementedException();
        }

        public SiaTensor Im2Col(SiaTensor x, Tuple<int, int> kernalSize, int padding = 1, int stride = 1)
        {
            throw new NotImplementedException();
        }

        public SiaTensor Col2Im(SiaTensor cols, long[] x_shape, Tuple<int, int> kernalSize, int padding = 1, int stride = 1)
        {
            throw new NotImplementedException();
        }

        public SiaTensor SliceRows(SiaTensor x, long start, long end)
        {
            throw new NotImplementedException();
        }

        public SiaTensor SliceCols(SiaTensor x, long start, long end)
        {
            throw new NotImplementedException();
        }

        public Array GetArray(SiaTensor x)
        {
            throw new NotImplementedException();
        }

        public void Dispose(SiaTensor x)
        {
            throw new NotImplementedException();
        }

        public ActivationFunc GetActFunc()
        {
            throw new NotImplementedException();
        }
    }
}
