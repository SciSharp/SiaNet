using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;
using TensorSharp.Expression;

namespace SiaNet.Initializers
{
    public class RandomUniform : BaseInitializer
    {
        public float MinVal { get; set; }

        public float MaxVal { get; set; }

        public RandomUniform(float minval = 0f, float maxval = 0.05f)
            :base("random_uniform")
        {
            MinVal = minval;
            MaxVal = maxval;
        }

        public override Tensor Operator(Tensor array)
        {
            Ops.RandomUniform(array, new SeedSource(), MinVal, MaxVal);
            return array;
        }
    }
}
