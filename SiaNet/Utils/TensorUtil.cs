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
        public static Variable Softplus(this Variable x)
        {
            return (x.Exp() + 1).Log();
        }

        public static Variable Softmax(this Variable x)
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

            return Variable.FromArray(data.ToArray(), Global.Device).View(shape);
        }

        public static Variable L2Normalize(this Variable x, int axis = -1)
        {
            Variable y = null;
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

        public static Parameter ToParameter(this Tensor t, string name = "")
        {
            return Parameter.Create(t, name);
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

        public static Tensor Pad(this Tensor t, uint n = 1, float value = 0)
        {
            List<float> data = new List<float>();
            Tensor tensor = t.CopyRef(); // new Tensor(t.Allocator, t.ElementType, t.Shape);
            //tensor.CopyFrom(t.ToArray());
            var tlist = tensor.NDto2DList();

            bool fill3d = false;
            long fillAfterBeforeCount = 0;
            List<float> fillData = new List<float>();
            if(tensor.DimensionCount == 5)
            {
                fill3d = true;
                fillAfterBeforeCount = tensor.Shape[2];
                long dataCount = (tensor.Shape[3] + (n * 2)) * (tensor.Shape[4] + (n * 2));
                for (int i = 0; i < dataCount; i++) fillData.Add(value);
            }

             int counter = 1;

            foreach (var item in tlist)
            {
                if(fill3d)
                {
                    if(counter == 1)
                    {
                        data.AddRange(fillData);
                    }
                }

                if (tensor.DimensionCount <= 3)
                {
                    data.AddRange(PadLR(item).Cast<float>());
                }
                else
                {
                    data.AddRange(Pad2D(item).Cast<float>());
                }

                if (fill3d)
                {
                    if (counter == fillAfterBeforeCount)
                    {
                        data.AddRange(fillData);
                        counter = 0;
                    }
                }

                counter++;
            }

            var shape = tensor.Shape;

            shape[shape.Length - 1] = shape[shape.Length - 1] + (n * 2);
            if (tensor.DimensionCount >= 4)
               shape[shape.Length - 2] = shape[shape.Length - 2] + (n * 2);

            if (tensor.DimensionCount >= 5)
                shape[shape.Length - 3] = shape[shape.Length - 3] + (n * 2);

            tensor = new Tensor(Global.Device, tensor.ElementType, shape);
            tensor.CopyFrom(data.ToArray());
            return tensor;
        }

        private static Array Pad2D(Tensor d, uint n = 1)
        {
            d = TOps.NewContiguous(d);
            long[] shape = d.Shape;
            Array data = d.ToArray();

            for (int i = 0; i < n; i++)
            {
                Tensor lr = new Tensor(Global.Device, DType.Float32, shape[0], 1);
                Ops.Fill(lr, 0);

                Tensor t = new Tensor(Global.Device, DType.Float32, shape[0], shape[1] + (n * 2));
                Ops.Concat(t, 1, lr, d, lr);
                d = t;
                shape = d.Shape;

                Tensor tb = new Tensor(Global.Device, DType.Float32, 1, shape[1]);
                Ops.Fill(tb, 0);
                
                t = new Tensor(Global.Device, DType.Float32, shape[0] + (n * 2), shape[1]);
                Ops.Concat(t, 0, tb, d, tb);
                d = t;
                shape = d.Shape;

                lr.Dispose();
                tb.Dispose();
            }

            return d.ToArray();
        }

        private static Array PadLR(Tensor d, uint n = 1)
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
            }

            return d.ToArray();
        }
    }
}
