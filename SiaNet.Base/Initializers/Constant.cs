using SiaNet.Engine;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Initializers
{
    public class Constant : BaseInitializer
    {
        public float Value { get; set; }

        public Constant(float value)
            :base("constant")
        {
            Value = value;
        }

        public Constant(string name, float value)
           : base(name)
        {
            Value = value;
        }

        public override Tensor Operator(params long[] shape)
        {
            Tensor tensor = K.Constant(Value, shape);
            return tensor;
        }
    }
}
