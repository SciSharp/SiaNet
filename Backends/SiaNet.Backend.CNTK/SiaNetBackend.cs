using CNTK;
using SiaNet.Engine;
using SiaNet.Engine.Layers;
using System;
using System.Collections.Generic;
using System.Linq;
using C = CNTK.CNTKLib;

namespace SiaNet.Backend.CNTKLib
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

        private Variable In(Tensor x)
        {
            return ((NDArrayTensor)x).InternalTensor;
        }

        private Variable In(float value, params long[] shape)
        {
            return new Parameter(BackendUtil.CastShapeInt(shape), CNTK.DataType.Float, value);
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
            return Out(C.Plus(In(a), In(b, a.Shape)));
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
            var arr = new CNTK.NDArrayView(BackendUtil.CastShapeInt(shape), data, DeviceManager.Current);
            var v = new CNTK.Variable(BackendUtil.CastShapeInt(shape), VariableKind.Parameter, CNTK.DataType.Float, arr, false, new AxisVector(), false, name, name);
            return Out(v);
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
            return Out(C.Times(In(a), In(b)));
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
            Variable xvar = In(x);
            Value v = null;
            if (xvar.IsOutput)
            {
                var f = xvar.ToFunction();
                
                var plist = f.Parameters();
                Dictionary<Variable, Value> inputs = new Dictionary<Variable, Value>();
                Dictionary<Variable, Value> outputs = new Dictionary<Variable, Value>()
                {
                    { f, null}
                };

                //foreach (var item in plist)
                //{
                //    inputs.Add(item, new Value(item.GetValue()));
                //}

                f.Evaluate(inputs, outputs, DeviceManager.Current);
                v = outputs.FirstOrDefault().Value;
            }
            else
            {
                v = new Value(xvar.GetValue());
            }

            return v.GetDenseData<float>(xvar)[0].ToArray();
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
            return Out(C.ReduceMax(In(x), Axis.AllAxes())).ToScalar();
        }

        public Tensor Max(Tensor x, int dim)
        {
            dim = dim < 0 ? x.DimCount + dim : dim;
            return Out(C.ReduceMax(In(x), new Axis(dim)));
        }

        public Tensor Max(Tensor x, params int[] dim)
        {
            foreach (var item in dim)
            {
                x = Max(x, item);
            }

            return x;
        }

        public Tensor Maximum(Tensor a, Tensor b)
        {
            return Out(C.ElementMax(In(a), In(b), ""));
        }

        public Tensor Maximum(Tensor a, float b)
        {
            return Out(C.ElementMax(In(a), In(b, a.Shape), ""));
        }

        public float Mean(Tensor x)
        {
            return Out(C.ReduceMean(In(x), Axis.AllAxes())).ToScalar();
        }

        public Tensor Mean(Tensor x, int dim)
        {
            dim = dim < 0 ? x.DimCount + dim : dim;
            return Out(C.ReduceMean(In(x), new Axis(dim)));
        }

        public Tensor Mean(Tensor x, params int[] dim)
        {
            foreach (var item in dim)
            {
                x = Mean(x, item);
            }

            return x;
        }

        public float Min(Tensor x)
        {
            return Out(C.ReduceMin(In(x), Axis.AllAxes())).ToScalar();
        }

        public Tensor Min(Tensor x, int dim)
        {
            dim = dim < 0 ? x.DimCount + dim : dim;
            return Out(C.ReduceMin(In(x), new Axis(dim)));
        }

        public Tensor Min(Tensor x, params int[] dim)
        {
            foreach (var item in dim)
            {
                x = Min(x, item);
            }

            return x;
        }

        public Tensor Minimum(Tensor a, Tensor b)
        {
            return Out(C.ElementMin(In(a), In(b), ""));
        }

        public Tensor Minimum(Tensor a, float b)
        {
            return Out(C.ElementMin(In(a), In(b, a.Shape), ""));
        }

        public Tensor Mul(Tensor a, Tensor b)
        {
            return Out(C.ElementTimes(In(a), In(b), ""));
        }

        public Tensor Mul(Tensor a, float b)
        {
            return Out(C.ElementTimes(In(a), In(b, a.Shape), ""));
        }

        public Tensor Mul(float a, Tensor b)
        {
            return Out(C.ElementTimes(In(a, b.Shape), In(b), ""));
        }

        public Tensor Neg(Tensor x)
        {
            return Out(C.Negate(In(x), ""));
        }

        public Tensor Pow(Tensor x, float value)
        {
            return Out(C.Pow(In(x), In(value, x.Shape)));
        }

        public void Print(Tensor x, string title = "")
        {
            BackendUtil.Print(x.Shape, x.ToArray());
        }

        public float Prod(Tensor x)
        {
            return Out(C.ReduceProd(In(x), Axis.AllAxes())).ToScalar();
        }

        public Tensor Prod(Tensor x, int dim)
        {
            dim = dim < 0 ? x.DimCount + dim : dim;
            return Out(C.ReduceProd(In(x), new Axis(dim)));
        }

        public Tensor Prod(Tensor x, params int[] dim)
        {
            foreach (var item in dim)
            {
                x = Prod(x, item);
            }

            return x;
        }

        public Tensor RandomBernoulli(long[] shape, float p)
        {
            return Out(C.BernoulliRandom(BackendUtil.CastShapeInt(shape), CNTK.DataType.Float, p));
        }

        public Tensor RandomNormal(long[] shape, float mean, float stddev, int? seed = null)
        {
            if(seed.HasValue)
                return Out(C.NormalRandom(BackendUtil.CastShapeInt(shape), CNTK.DataType.Float, mean, stddev, (uint)seed.Value));
            else
                return Out(C.NormalRandom(BackendUtil.CastShapeInt(shape), CNTK.DataType.Float, mean, stddev));
        }

        public Tensor RandomUniform(long[] shape, float min, float max, int? seed = null)
        {
            if(seed.HasValue)
                return Out(C.UniformRandom(BackendUtil.CastShapeInt(shape), CNTK.DataType.Float, min, max, (uint)seed.Value));
            else
                return Out(C.UniformRandom(BackendUtil.CastShapeInt(shape), CNTK.DataType.Float, min, max));
        }

        public Tensor Reshape(Tensor x, params long[] shape)
        {
            return Out(C.Reshape(In(x), BackendUtil.CastShapeInt(shape)));
        }

        public Tensor Round(Tensor x)
        {
            return Out(C.Round(In(x)));
        }

        public void SetDevice(DeviceType device)
        {
            switch (device)
            {
                case DeviceType.Default:
                    DeviceManager.Current = DeviceDescriptor.UseDefaultDevice();
                    break;
                case DeviceType.CPU:
                    DeviceManager.Current = DeviceDescriptor.CPUDevice;
                    break;
                case DeviceType.CUDA:
                    DeviceManager.Current = DeviceDescriptor.GPUDevice(0);
                    break;
                case DeviceType.OpenCL:
                    throw new NotSupportedException("CNTK doesn't support OpenCL. Please use ArrayFire backend.");
                default:
                    break;
            }
        }

        public Tensor Sigmoid(Tensor x)
        {
            return Out(C.Sigmoid(In(x)));
        }

        public Tensor Sign(Tensor x)
        {
            throw new NotImplementedException();
        }

        public Tensor Sin(Tensor x)
        {
            return Out(C.Sin(In(x)));
        }

        public Tensor Sinh(Tensor x)
        {
            return Out(C.Sinh(In(x)));
        }

        public Tensor SliceCols(Tensor x, long start, long end)
        {
            return Out(C.Slice(In(x), AxisVector.Repeat(new Axis(1), 1), IntVector.Repeat((int)start, 1), IntVector.Repeat((int)end, 1)));
        }

        public Tensor SliceRows(Tensor x, long start, long end)
        {
            return Out(C.Slice(In(x), AxisVector.Repeat(new Axis(0), 1), IntVector.Repeat((int)start, 1), IntVector.Repeat((int)end, 1)));
        }

        public Tensor Softmax(Tensor x, int axis = -1)
        {
            axis = axis < 0 ? x.DimCount + axis : axis;
            return Out(C.Softmax(In(x), new Axis(axis)));
        }

        public Tensor Softplus(Tensor x, int axis = -1)
        {
            axis = axis < 0 ? x.DimCount + axis : axis;
            return Out(C.Softplus(In(x)));
        }

        public Tensor Sqrt(Tensor x)
        {
            return Out(C.Sqrt(In(x)));
        }

        public Tensor Square(Tensor x)
        {
            return Out(C.Square(In(x)));
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
            return Out(C.Minus(In(a), In(b)));
        }

        public Tensor Sub(Tensor a, float b)
        {
            return Out(C.Minus(In(a), In(b, a.Shape)));
        }

        public Tensor Sub(float a, Tensor b)
        {
            return Out(C.Minus(In(a, b.Shape), In(b)));
        }

        public float Sum(Tensor x)
        {
            return Out(C.ReduceSum(In(x), Axis.AllAxes())).ToScalar();
        }

        public Tensor Sum(Tensor x, int dim)
        {
            dim = dim < 0 ? x.DimCount + dim : dim;
            return Out(C.ReduceSum(In(x), new Axis(dim)));
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
            return Out(C.Tan(In(x)));
        }

        public Tensor Tanh(Tensor x)
        {
            return Out(C.Tanh(In(x)));
        }

        public Tensor Tile(Tensor x, int n, int axis = 0)
        {
            return x;
        }

        public Tensor Tpow(float value, Tensor x)
        {
            throw new NotImplementedException();
        }

        public Tensor Transpose(Tensor x)
        {
            return Out(C.Transpose(In(x)));
        }

        public Tensor Transpose(Tensor x, params int[] dims)
        {
            if (dims.Length != x.DimCount)
                throw new InvalidOperationException("The number of permutation indices must equal the number of tensor dimensions");

            var result = In(x);
            foreach (var swap in SwapsForPermutation(dims))
            {
                var resultOld = result;
                result = C.TransposeAxes(result, new Axis(swap.Item1), new Axis(swap.Item2));
                resultOld.Dispose();
            }

            return Out(result);
        }

        private static IEnumerable<Tuple<int, int>> SwapsForPermutation(int[] perm)
        {
            int j;
            for (int i = 0; i < perm.Length; ++i)
            {
                var p = perm[i];
                if (p != i && p != -1)
                {
                    j = i;
                    do
                    {
                        if (perm[j] < 0 || perm[j] >= perm.Length)
                            throw new InvalidOperationException("Invalid permutation");

                        yield return Tuple.Create(j, perm[j]);


                        var jOld = j;
                        j = perm[j];
                        perm[jOld] = -1;
                    } while (perm[j] != i);
                    perm[j] = j;
                }
            }
        }

        public Tensor Trunc(Tensor x)
        {
            throw new NotImplementedException();
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
            return new SiaNetActivations(this);
        }
    }
}
