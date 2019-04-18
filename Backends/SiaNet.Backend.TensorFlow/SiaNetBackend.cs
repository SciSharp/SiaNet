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

        internal NDArrayTensor Out(Tensorflow.Tensor x)
        {
            return new NDArrayTensor(x);
        }

        internal Tensorflow.Tensor In(SiaTensor x)
        {
            return ((NDArrayTensor)x).InternalTensor;
        }

        internal Tensorflow.Tensor In(float value, long[] shape = null)
        {
            return tf.constant(value, 
                shape: shape?.Select(x => Convert.ToInt32(x))?.ToArray(), 
                dtype: TF_DataType.TF_FLOAT);
        }

        internal Tensorflow.Tensor In(params int[] data)
        {
            return tf.Variable<Array>(data, dtype: TF_DataType.TF_INT32);
        }

        internal Tensorflow.Tensor In(params long[] data)
        {
            return tf.Variable<Array>(data, dtype: TF_DataType.TF_INT64);
        }

        internal Tensorflow.Tensor In(int value)
        {
            return tf.constant(1, TF_DataType.TF_INT32, new int[] { 1 });
        }

        internal Tensorflow.Tensor In(long value)
        {
            return tf.constant(1, TF_DataType.TF_INT64, new int[] { 1 });
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
            return name + "_" + counter++;
        }

        public void SetDevice(DeviceType device)
        {
            
        }

        public Engine.DataType GetDataType(SiaTensor x)
        {
            switch (In(x).dtype)
            {
                case TF_DataType.DtInvalid:
                    break;
                case TF_DataType.TF_FLOAT:
                    return Engine.DataType.Float32;
                case TF_DataType.TF_DOUBLE:
                    return Engine.DataType.Float64;
                case TF_DataType.TF_INT32:
                    return Engine.DataType.Int32;
                case TF_DataType.TF_UINT8:
                    return Engine.DataType.Int8;
                case TF_DataType.TF_INT8:
                    return Engine.DataType.Int8;
                default:
                    return Engine.DataType.Float32;
            }

            return Engine.DataType.Float32;
        }

        public SiaTensor CreateVariable(float[] data, long[] shape, string name = "")
        {
            return Out(tf.Variable<Array>(data).value());
        }

        public SiaTensor Reshape(SiaTensor x, params long[] shape)
        {
            return Out(tf.reshape(In(x), BackendUtil.CastShapeInt(shape)));
        }

        public SiaTensor Constant(float value, long[] shape)
        {
            return Out(In(value, shape));
        }

        public SiaTensor RandomUniform(long[] shape, float min, float max, int? seed = null)
        {
            return Out(tf.random_uniform(BackendUtil.CastShapeInt(shape), min, max, seed: seed));
        }

        public SiaTensor RandomNormal(long[] shape, float mean, float stddev, int? seed = null)
        {
            return Out(tf.random_normal(BackendUtil.CastShapeInt(shape), mean, stddev, seed: seed));
        }

        public SiaTensor RandomBernoulli(long[] shape, float p)
        {
            var result = RandomUniform(shape, 0, 1);
            result = result > p;
            return result;
        }

        public SiaTensor Add(SiaTensor a, SiaTensor b)
        {
            return Out(tf.add(In(a), In(b)));
        }

        public SiaTensor Add(SiaTensor a, float b)
        {
            return Out(tf.add(In(a), In(b, a.Shape)));
        }

        public SiaTensor Add(float a, SiaTensor b)
        {
            return Out(tf.add(In(a, b.Shape), In(b)));
        }

        public SiaTensor Sub(SiaTensor a, SiaTensor b)
        {
            return Out(tf.sub(In(a), In(b)));
        }

        public SiaTensor Sub(SiaTensor a, float b)
        {
            return Out(tf.sub(In(a), In(b, a.Shape)));
        }

        public SiaTensor Sub(float a, SiaTensor b)
        {
            return Out(tf.sub(In(a, b.Shape), In(b)));
        }

        public SiaTensor Mul(SiaTensor a, SiaTensor b)
        {
            return Out(tf.multiply(In(a), In(b)));
        }

        public SiaTensor Mul(SiaTensor a, float b)
        {
            return Out(tf.multiply(In(a), In(b, a.Shape)));
        }

        public SiaTensor Mul(float a, SiaTensor b)
        {
            return Out(tf.multiply(In(a, b.Shape), In(b)));
        }

        public SiaTensor Div(SiaTensor a, SiaTensor b)
        {
            return Out(In(a) / In(b));
        }

        public SiaTensor Div(SiaTensor a, float b)
        {
            return Out(In(a) / In(b, a.Shape));
        }

        public SiaTensor Div(float a, SiaTensor b)
        {
            return Out(In(a, b.Shape) / In(b));
        }

        public SiaTensor GreaterThan(SiaTensor a, SiaTensor b)
        {
            return Out(tf.greater(In(a), In(b)));
        }

        public SiaTensor GreaterThanEqual(SiaTensor a, SiaTensor b)
        {
            return Out(tf.greater_equal(In(a), In(b)));
        }

        public SiaTensor LessThan(SiaTensor a, SiaTensor b)
        {
            return Out(tf.less(In(a), In(b)));
        }

        public SiaTensor LessThanEqual(SiaTensor a, SiaTensor b)
        {
            return Out(tf.less_equal(In(a), In(b)));
        }

        public SiaTensor GreaterThan(float a, SiaTensor b)
        {
            return Out(tf.greater(In(a, b.Shape), In(b)));
        }

        public SiaTensor GreaterThanEqual(float a, SiaTensor b)
        {
            return Out(tf.greater_equal(In(a, b.Shape), In(b)));
        }

        public SiaTensor LessThan(float a, SiaTensor b)
        {
            return Out(tf.less(In(a, b.Shape), In(b)));
        }

        public SiaTensor LessThanEqual(float a, SiaTensor b)
        {
            return Out(tf.less_equal(In(a, b.Shape), In(b)));
        }

        public SiaTensor GreaterThan(SiaTensor a, float b)
        {
            return Out(tf.greater(In(a), In(b, a.Shape)));
        }

        public SiaTensor GreaterThanEqual(SiaTensor a, float b)
        {
            return Out(tf.greater_equal(In(a), In(b, a.Shape)));
        }

        public SiaTensor LessThan(SiaTensor a, float b)
        {
            return Out(tf.less(In(a), In(b, a.Shape)));
        }

        public SiaTensor LessThanEqual(SiaTensor a, float b)
        {
            return Out(tf.less_equal(In(a), In(b, a.Shape)));
        }

        public SiaTensor EqualTo(SiaTensor a, SiaTensor b)
        {
            return Out(tf.equal(In(a), In(b)));
        }

        public SiaTensor Tile(SiaTensor x, int n, int axis = 0)
        {
            axis = axis < 0 ? x.DimCount + axis : axis;
            int[] multiples = new int[x.DimCount];
            for(int i=0;i<multiples.Length;i++)
            {
                if(i == axis)
                {
                    multiples[i] = n;
                    continue;
                }

                multiples[i] = 1;
            }
            
            return Out(tf.tile(In(x), In(multiples)));
        }

        public SiaTensor Abs(SiaTensor x)
        {
            return Out(tf.abs(In(x)));
        }

        public SiaTensor Neg(SiaTensor x)
        {
            return Out(tf.negative(In(x)));
        }

        public SiaTensor Sqrt(SiaTensor x)
        {
            return Out(tf.sqrt(In(x)));
        }

        public SiaTensor Exp(SiaTensor x)
        {
            return Out(tf.exp(In(x)));
        }

        public SiaTensor Log(SiaTensor x)
        {
            return Out(tf.log(In(x)));
        }

        public SiaTensor Log10(SiaTensor x)
        {
            return Out(tf.log(In(x)));
        }

        public SiaTensor Log1p(SiaTensor x)
        {
            return Out(tf.log1p(In(x)));
        }

        public SiaTensor Floor(SiaTensor x)
        {
            return Out(tf.floor(In(x)));
        }

        public SiaTensor Ceil(SiaTensor x)
        {
            return Out(tf.ceil(In(x)));
        }

        public SiaTensor Round(SiaTensor x)
        {
            return Out(tf.round(In(x)));
        }

        public SiaTensor Sin(SiaTensor x)
        {
            return Out(tf.sin(In(x)));
        }

        public SiaTensor Cos(SiaTensor x)
        {
            return Out(tf.cos(In(x)));
        }

        public SiaTensor Tan(SiaTensor x)
        {
            return Out(tf.tan(In(x)));
        }

        public SiaTensor Asin(SiaTensor x)
        {
            return Out(tf.asin(In(x)));
        }

        public SiaTensor Acos(SiaTensor x)
        {
            return Out(tf.acos(In(x)));
        }

        public SiaTensor Atan(SiaTensor x)
        {
            return Out(tf.atan(In(x)));
        }

        public SiaTensor Sinh(SiaTensor x)
        {
            return Out(tf.sinh(In(x)));
        }

        public SiaTensor Cosh(SiaTensor x)
        {
            return Out(tf.cosh(In(x)));
        }

        public SiaTensor Tanh(SiaTensor x)
        {
            return Out(tf.tanh(In(x)));
        }

        public SiaTensor Sigmoid(SiaTensor x)
        {
            return Out(tf.sigmoid(In(x)));
        }

        public SiaTensor Pow(SiaTensor x, float value)
        {
            return Out(tf.pow(In(x), In(value, x.Shape)));
        }

        public SiaTensor Square(SiaTensor x)
        {
            return Out(tf.pow(In(x), 2));
        }

        public SiaTensor Clip(SiaTensor x, float min, float max)
        {
            return Out(tf._clip_by_value(In(x), In(min, x.Shape), In(max, x.Shape)));
        }

        public float Sum(SiaTensor x)
        {
            return tf.reduce_sum(In(x));
        }

        public SiaTensor Sum(SiaTensor x, int dim)
        {
            dim = dim < 0 ? x.DimCount + dim : dim;
            return Out(tf.sum(In(x), dim, true));
        }

        public SiaTensor Sum(SiaTensor x, params int[] dims)
        {
            foreach (var item in dims)
            {
                int dim = item < 0 ? x.DimCount + item : item;
                x = Sum(x, item);
            }

            return x;
        }

        public float Max(SiaTensor x)
        {
            return math_ops.reduce_max(In(x));
        }

        public SiaTensor Max(SiaTensor x, int dim)
        {
            dim = dim < 0 ? x.DimCount + dim : dim;
            return Out(math_ops.reduce_max(In(x), new int[] { dim }, true));
        }

        public SiaTensor Max(SiaTensor x, params int[] dims)
        {
            for (int i = 0; i < dims.Length; i++)
            {
                dims[i] = dims[i] < 0 ? x.DimCount + dims[i] : dims[i];
            }

            return Out(math_ops.reduce_max(In(x), dims, true));
        }

        public float Min(SiaTensor x)
        {
            throw new NotImplementedException();
        }

        public SiaTensor Min(SiaTensor x, int dim)
        {
            return Out(tf.min(In(x), dim)));
        }

        public SiaTensor Min(SiaTensor x, params int[] dims)
        {
            for (int i = 0; i < dims.Length; i++)
            {
                dims[i] = dims[i] < 0 ? x.DimCount + dims[i] : dims[i];
                x = Min(x, dims[i]);
            }

            return x;
        }

        public float Mean(SiaTensor x)
        {
            return math_ops.reduce_mean(In(x));
        }

        public SiaTensor Mean(SiaTensor x, int dim)
        {
            dim = dim < 0 ? x.DimCount + dim : dim;
            return Out(math_ops.reduce_mean(In(x), new int[] { dim }, true));
        }

        public SiaTensor Mean(SiaTensor x, params int[] dims)
        {
            for (int i = 0; i < dims.Length; i++)
            {
                dims[i] = dims[i] < 0 ? x.DimCount + dims[i] : dims[i];
            }

            return Out(math_ops.reduce_mean(In(x), dims, true));
        }

        public SiaTensor Argmax(SiaTensor x, int dim = 0)
        {
            dim = dim < 0 ? x.DimCount + dim : dim;
            return Out(tf.arg_max(In(x), dim));
        }

        public SiaTensor Argmin(SiaTensor x, int dim = 0)
        {
            dim = dim < 0 ? x.DimCount + dim : dim;
            return Out(tf.arg_min(In(x), dim));
        }

        public SiaTensor Maximum(SiaTensor a, SiaTensor b)
        {
            return Out(tf.maximum(In(a), In(b)));
        }

        public SiaTensor Maximum(SiaTensor a, float b)
        {
            return Out(tf.maximum(In(a), In(b, a.Shape)));
        }

        public SiaTensor Minimum(SiaTensor a, SiaTensor b)
        {
            return Out(tf.minimum(In(a), In(b)));
        }

        public SiaTensor Minimum(SiaTensor a, float b)
        {
            return Out(tf.minimum(In(a), In(b, a.Shape)));
        }

        public SiaTensor Transpose(SiaTensor x)
        {
            return Out(tf.transpose(In(x), new int[] { 1, 0 }));
        }

        public SiaTensor Transpose(SiaTensor x, params int[] dims)
        {
            return Out(tf.transpose(In(x), dims));
        }

        public SiaTensor Dot(SiaTensor a, SiaTensor b)
        {
            return Out(tf.matmul(In(a), In(b)));
        }

        public SiaTensor Diag(SiaTensor x)
        {
            return Out(tf.diag(In(x)));
        }

        public SiaTensor Softmax(SiaTensor x, int axis = -1)
        {
            return Out(tf.nn.softmax(In(x), axis));
        }

        public SiaTensor Softplus(SiaTensor x, int axis = -1)
        {
            return Log((Exp(x) + 1));
        }

        public SiaTensor L2Normalize(SiaTensor x, int axis = -1)
        {
            var y = Max(Sum(Square(x), axis), axis);
            return x / Sqrt(y);
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
            return In(x).Data<float>();
        }

        public void Dispose(SiaTensor x)
        {
            In(x).Dispose();
        }

        public ActivationFunc GetActFunc()
        {
            return new SiaNetActivations(this);
        }
    }
}
