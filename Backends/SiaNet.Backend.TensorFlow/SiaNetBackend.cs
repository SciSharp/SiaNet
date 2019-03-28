using SiaNet.Engine;
using SiaNet.Engine.Layers;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using System.Linq;
using Tensorflow;
using DeviceType = SiaNet.Engine.DeviceType;
using DataType = SiaNet.Engine.Tensor;

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
            throw new NotImplementedException();
        }

        public long[] GetShape(DataType x)
        {
            throw new NotImplementedException();
        }

        public object Eval(DataType x)
        {
            throw new NotImplementedException();
        }

        public void Print(DataType x, string title = "")
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

        public Engine.DataType GetDataType(DataType x)
        {
            throw new NotImplementedException();
        }

        public DataType CreateVariable(float[] data, long[] shape, string name = "")
        {
            throw new NotImplementedException();
        }

        public DataType Reshape(DataType x, params long[] shape)
        {
            throw new NotImplementedException();
        }

        public DataType Constant(float value, long[] shape)
        {
            throw new NotImplementedException();
        }

        public DataType RandomUniform(long[] shape, float min, float max, int? seed = null)
        {
            throw new NotImplementedException();
        }

        public DataType RandomNormal(long[] shape, float mean, float stddev, int? seed = null)
        {
            throw new NotImplementedException();
        }

        public DataType RandomBernoulli(long[] shape, float p)
        {
            throw new NotImplementedException();
        }

        public DataType Add(DataType a, DataType b)
        {
            throw new NotImplementedException();
        }

        public DataType Add(DataType a, float b)
        {
            throw new NotImplementedException();
        }

        public DataType Add(float a, DataType b)
        {
            throw new NotImplementedException();
        }

        public DataType Sub(DataType a, DataType b)
        {
            throw new NotImplementedException();
        }

        public DataType Sub(DataType a, float b)
        {
            throw new NotImplementedException();
        }

        public DataType Sub(float a, DataType b)
        {
            throw new NotImplementedException();
        }

        public DataType Mul(DataType a, DataType b)
        {
            throw new NotImplementedException();
        }

        public DataType Mul(DataType a, float b)
        {
            throw new NotImplementedException();
        }

        public DataType Mul(float a, DataType b)
        {
            throw new NotImplementedException();
        }

        public DataType Div(DataType a, DataType b)
        {
            throw new NotImplementedException();
        }

        public DataType Div(DataType a, float b)
        {
            throw new NotImplementedException();
        }

        public DataType Div(float a, DataType b)
        {
            throw new NotImplementedException();
        }

        public DataType GreaterThan(DataType a, DataType b)
        {
            throw new NotImplementedException();
        }

        public DataType GreaterThanEqual(DataType a, DataType b)
        {
            throw new NotImplementedException();
        }

        public DataType LessThan(DataType a, DataType b)
        {
            throw new NotImplementedException();
        }

        public DataType LessThanEqual(DataType a, DataType b)
        {
            throw new NotImplementedException();
        }

        public DataType GreaterThan(float a, DataType b)
        {
            throw new NotImplementedException();
        }

        public DataType GreaterThanEqual(float a, DataType b)
        {
            throw new NotImplementedException();
        }

        public DataType LessThan(float a, DataType b)
        {
            throw new NotImplementedException();
        }

        public DataType LessThanEqual(float a, DataType b)
        {
            throw new NotImplementedException();
        }

        public DataType GreaterThan(DataType a, float b)
        {
            throw new NotImplementedException();
        }

        public DataType GreaterThanEqual(DataType a, float b)
        {
            throw new NotImplementedException();
        }

        public DataType LessThan(DataType a, float b)
        {
            throw new NotImplementedException();
        }

        public DataType LessThanEqual(DataType a, float b)
        {
            throw new NotImplementedException();
        }

        public DataType EqualTo(DataType a, DataType b)
        {
            throw new NotImplementedException();
        }

        public DataType Tile(DataType x, int n, int axis = 0)
        {
            throw new NotImplementedException();
        }

        public DataType Abs(DataType x)
        {
            throw new NotImplementedException();
        }

        public DataType Neg(DataType x)
        {
            throw new NotImplementedException();
        }

        public DataType Sqrt(DataType x)
        {
            throw new NotImplementedException();
        }

        public DataType Exp(DataType x)
        {
            throw new NotImplementedException();
        }

        public DataType Log(DataType x)
        {
            throw new NotImplementedException();
        }

        public DataType Log10(DataType x)
        {
            throw new NotImplementedException();
        }

        public DataType Log1p(DataType x)
        {
            throw new NotImplementedException();
        }

        public DataType Floor(DataType x)
        {
            throw new NotImplementedException();
        }

        public DataType Ceil(DataType x)
        {
            throw new NotImplementedException();
        }

        public DataType Round(DataType x)
        {
            throw new NotImplementedException();
        }

        public DataType Sin(DataType x)
        {
            throw new NotImplementedException();
        }

        public DataType Cos(DataType x)
        {
            throw new NotImplementedException();
        }

        public DataType Tan(DataType x)
        {
            throw new NotImplementedException();
        }

        public DataType Asin(DataType x)
        {
            throw new NotImplementedException();
        }

        public DataType Acos(DataType x)
        {
            throw new NotImplementedException();
        }

        public DataType Atan(DataType x)
        {
            throw new NotImplementedException();
        }

        public DataType Sinh(DataType x)
        {
            throw new NotImplementedException();
        }

        public DataType Cosh(DataType x)
        {
            throw new NotImplementedException();
        }

        public DataType Tanh(DataType x)
        {
            throw new NotImplementedException();
        }

        public DataType Sigmoid(DataType x)
        {
            throw new NotImplementedException();
        }

        public DataType Pow(DataType x, float value)
        {
            throw new NotImplementedException();
        }

        public DataType Square(DataType x)
        {
            throw new NotImplementedException();
        }

        public DataType Clip(DataType x, float min, float max)
        {
            throw new NotImplementedException();
        }

        public float Sum(DataType x)
        {
            throw new NotImplementedException();
        }

        public DataType Sum(DataType x, int dim)
        {
            throw new NotImplementedException();
        }

        public DataType Sum(DataType x, params int[] dim)
        {
            throw new NotImplementedException();
        }

        public float Max(DataType x)
        {
            throw new NotImplementedException();
        }

        public DataType Max(DataType x, int dim)
        {
            throw new NotImplementedException();
        }

        public DataType Max(DataType x, params int[] dim)
        {
            throw new NotImplementedException();
        }

        public float Min(DataType x)
        {
            throw new NotImplementedException();
        }

        public DataType Min(DataType x, int dim)
        {
            throw new NotImplementedException();
        }

        public DataType Min(DataType x, params int[] dim)
        {
            throw new NotImplementedException();
        }

        public float Mean(DataType x)
        {
            throw new NotImplementedException();
        }

        public DataType Mean(DataType x, int dim)
        {
            throw new NotImplementedException();
        }

        public DataType Mean(DataType x, params int[] dim)
        {
            throw new NotImplementedException();
        }

        public DataType Argmax(DataType x, int dim = 0)
        {
            throw new NotImplementedException();
        }

        public DataType Argmin(DataType x, int dim = 0)
        {
            throw new NotImplementedException();
        }

        public DataType Maximum(DataType a, DataType b)
        {
            throw new NotImplementedException();
        }

        public DataType Maximum(DataType a, float b)
        {
            throw new NotImplementedException();
        }

        public DataType Minimum(DataType a, DataType b)
        {
            throw new NotImplementedException();
        }

        public DataType Minimum(DataType a, float b)
        {
            throw new NotImplementedException();
        }

        public DataType Transpose(DataType x)
        {
            throw new NotImplementedException();
        }

        public DataType Transpose(DataType x, params int[] dims)
        {
            throw new NotImplementedException();
        }

        public DataType Dot(DataType a, DataType b)
        {
            throw new NotImplementedException();
        }

        public DataType Diag(DataType x)
        {
            throw new NotImplementedException();
        }

        public DataType Softmax(DataType x, int axis = -1)
        {
            throw new NotImplementedException();
        }

        public DataType Softplus(DataType x, int axis = -1)
        {
            throw new NotImplementedException();
        }

        public DataType L2Normalize(DataType x, int axis = -1)
        {
            throw new NotImplementedException();
        }

        public DataType Im2Col(DataType x, Tuple<int, int> kernalSize, int padding = 1, int stride = 1)
        {
            throw new NotImplementedException();
        }

        public DataType Col2Im(DataType cols, long[] x_shape, Tuple<int, int> kernalSize, int padding = 1, int stride = 1)
        {
            throw new NotImplementedException();
        }

        public DataType SliceRows(DataType x, long start, long end)
        {
            throw new NotImplementedException();
        }

        public DataType SliceCols(DataType x, long start, long end)
        {
            throw new NotImplementedException();
        }

        public Array GetArray(DataType x)
        {
            throw new NotImplementedException();
        }

        public void Dispose(DataType x)
        {
            throw new NotImplementedException();
        }

        public ActivationFunc GetActFunc()
        {
            throw new NotImplementedException();
        }
    }
}
