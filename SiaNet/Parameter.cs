using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using TensorSharp;
using SiaNet.Constraints;
using SiaNet.Regularizers;

namespace SiaNet
{
    public class Parameter : IDisposable
    {
        public Tensor Data { get; set; }

        public Tensor Grad { get; set; }

        public string Name { get; set; }

        private BaseConstraint constraint;

        private BaseRegularizer regularizer;

        public bool HaveRegularizer
        {
            get
            {
                return regularizer != null;
            }
        }

        public Parameter(string name, params long[] shape)
        {
            //Name = UUID.GetID(name);
            Name = name;
            Data = new Tensor(Global.Device, DType.Float32, shape);
            Grad = new Tensor(Global.Device, DType.Float32, shape);
        }

        public Parameter(string name, DType dataType, params long[] shape)
        {
            //Name = UUID.GetID(name);
            Name = name;
            Data = new Tensor(Global.Device, dataType, shape);
            Grad = new Tensor(Global.Device, dataType, shape);
        }

        public static Parameter Create(Tensor data, string name = "")
        {
            if (string.IsNullOrWhiteSpace(name))
                name = "v";

            Parameter x = new Parameter(name, data.ElementType, data.Shape);
            x.Data = data;

            return x;
        }

        public void Dispose()
        {
            Data.Dispose();
            Grad.Dispose();
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
