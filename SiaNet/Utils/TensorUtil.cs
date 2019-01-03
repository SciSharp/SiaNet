using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp.Expression;
using System.Linq;
using TensorSharp;

namespace SiaNet
{
    public static class TensorUtil
    {
        public static TVar Softplus(this TVar x)
        {
            return (x.Exp() + 1).Log();
        }

        public static TVar Softmax(this TVar x)
        {
            long[] shape = x.Evaluate().Shape;
            List<float> data = new List<float>();
            for (long i = 0; i < shape[0]; i++)
            {
                var s_x = x.Select(0, i);
                var exp = s_x.Exp();
                var sum = exp.SumAll();
                var s_t = (exp / sum.ToScalar()).View(1, shape[1]).Evaluate();
                data.AddRange(s_t.ToArray().Cast<float>());
            }

            return TVar.FromArray(data.ToArray(), Global.Device).View(shape);
        }

        public static TVar L2Normalize(this TVar x, int axis = -1)
        {
            TVar y = null;
            if (axis == -1)
            {
                y = x.Pow(2).SumAll().MaxAll();
            }
            else
            {
                y = x.Pow(2).Sum(axis).Max(axis);
            }

            return x.CDiv(y.Sqrt());
        }

        public static Variable ToVariable(this Tensor t, string name = "")
        {
            return Variable.Create(t, name);
        }

        public static List<Tensor> NDto2DList(this Tensor tensor)
        {
            List<Tensor> result = new List<Tensor>();
            if (tensor.DimensionCount <= 2)
            {
                throw new ArgumentException("Expecting more than 2-D.");
            }

            var counter = Enumerable.Repeat((long)0, tensor.DimensionCount - 2).ToArray();
            bool finished = false;
            counter[0] = -1;
            while (true)
            {
                for (int i = 0; i < tensor.DimensionCount - 2; ++i)
                {
                    counter[i]++;
                    if (counter[i] >= tensor.Shape[i])
                    {
                        if (i == tensor.DimensionCount - 3)
                        {
                            finished = true;
                            break;
                        }
                        counter[i] = 0;
                    }
                    else
                    {
                        break;
                    }
                }

                if (finished)
                    break;

                var tensorCopy = tensor.CopyRef();
                for (int i = 0; i < tensor.DimensionCount - 2; ++i)
                {
                    var newCopy = tensorCopy.Select(0, counter[i]);
                    tensorCopy.Dispose();
                    tensorCopy = newCopy;
                }

                result.Add(tensorCopy);
                tensorCopy.Dispose();
            }

            return result;
        }

        public static Tensor PadAll(this Tensor tensor, uint n = 1, float value = 0)
        {
            List<float> data = new List<float>();
            var tlist = tensor.NDto2DList();
            foreach (var item in tlist)
            {
                data.AddRange(Pad2D(item).Cast<float>());
            }

            var shape = tensor.Shape;

            shape[shape.Length - 1] = shape[shape.Length - 1] + 2;
            shape[shape.Length - 2] = shape[shape.Length - 2] + 2;

            tensor = new Tensor(Global.Device, tensor.ElementType, shape);
            tensor.CopyFrom(data.ToArray());
            return tensor;
        }

        private static Array Pad2D(Tensor d, uint n = 1)
        {
            long[] shape = d.Shape;
            Array data = d.ToArray();

            for (int i = 0; i < n; i++)
            {
                Tensor lr = new Tensor(Global.Device, DType.Float32, shape[0], 1);
                Ops.Fill(lr, 0);

                Tensor t = new Tensor(Global.Device, DType.Float32, shape[0], shape[1] + 2);
                Ops.Concat(t, 1, lr, d, lr);
                d = t;
                shape = d.Shape;

                Tensor tb = new Tensor(Global.Device, DType.Float32, 1, shape[1]);
                Ops.Fill(tb, 0);
                
                t = new Tensor(Global.Device, DType.Float32, shape[0] + 2, shape[1]);
                Ops.Concat(t, 0, tb, d, tb);
                d = t;
                shape = d.Shape;
                lr.Dispose();
                tb.Dispose();
            }
           

            return d.ToArray();
        }
    }
}
