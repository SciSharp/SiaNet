using SiaNet.Engine;
using System;
using System.Collections.Generic;
using System.Text;

namespace SiaNet.Initializers
{
    public class RandomUniform : BaseInitializer
    {
        public float MinVal { get; set; }

        public float MaxVal { get; set; }

        public int? Seed { get; set; }

        public RandomUniform(float minval = 0f, float maxval = 0.05f, int? seed = null)
            :base("random_uniform")
        {
            MinVal = minval;
            MaxVal = maxval;
            Seed = seed;
        }

        public override Tensor Operator(params long[] shape)
        {
            return K.RandomUniform(shape, MinVal, MaxVal, Seed);
        }
    }
}
