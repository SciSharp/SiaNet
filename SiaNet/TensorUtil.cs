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
            long[] shape = x.Evaluate().Sizes;
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

        

        public static Tensor Maximum(TVar tensor1, TVar tensor2)
        {
            var t1 = (tensor1 >= tensor2);
            var t2 = (tensor2 > tensor1);

            return (t1.CMul(tensor1) + t2.CMul(tensor2)).Evaluate();
        }

        public static Variable ToVariable(this Tensor t, string name = "")
        {
            return Variable.Create(t, name);
        }
    }
}
