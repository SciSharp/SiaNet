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
            SeedSource seedSource = new SeedSource();
            if (Seed.HasValue)
                seedSource = new SeedSource(Seed.Value);

            Tensor tensor = new Tensor(Global.Device, DType.Float32, shape);
            Ops.RandomUniform(tensor, seedSource, MinVal, MaxVal);
            return tensor;
        }
    }
}
