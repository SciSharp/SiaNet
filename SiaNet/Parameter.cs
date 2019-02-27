using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using SiaNet.Constraints;
using SiaNet.Regularizers;
using SiaNet.Engine;

namespace SiaNet
{
    public class Parameter : BaseParameter
    {
        private BaseConstraint constraint;

        private BaseRegularizer regularizer;

        public Parameter(string name, params long[] shape)
            :base (name, shape)
        {
        }

        public Parameter(string name, DataType dataType, params long[] shape)
            : base(name, dataType, shape)
        {
        }

        public static Parameter Create(Tensor data, string name = "")
        {
            if (string.IsNullOrWhiteSpace(name))
                name = "v";

            Parameter x = new Parameter(name, data.ElementType, data.Shape);
            x.Data = data;

            return x;
        }

        public bool HaveRegularizer
        {
            get
            {
                return regularizer != null;
            }
        }

        public void SetConstraint(BaseConstraint fn)
        {
            constraint = fn;
        }

        public void SetRegularizer(BaseRegularizer fn)
        {
            regularizer = fn;
        }

        public void ApplyConstraint()
        {
            if (constraint != null)
            {
                Data = constraint.Call(Data);
            }
        }

        public float ApplyRegularizer()
        {
            float r = 0;
            if (regularizer != null)
            {
                r = regularizer.Call(Data);
            }

            return r;
        }

        public void ApplyDeltaRegularizer()
        {
            if (regularizer != null)
            {
                Grad += regularizer.CalcGrad(Data);
            }
        }
    }
}
