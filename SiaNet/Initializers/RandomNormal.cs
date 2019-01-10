using System;
using System.Collections.Generic;
using System.Text;
using TensorSharp;
using TensorSharp.Expression;

namespace SiaNet.Initializers
{
    public class RandomNormal : BaseInitializer
    {
        public float Mean { get; set; }

        public float StdDev { get; set; }

        public int? Seed { get; set; }

        public RandomNormal(float mean = 0f, float stddev = 0.05f, int? seed = null)
            :base ("random_normal")
        {
            Mean = mean;
            StdDev = stddev;
            Seed = seed;
        }

        public override Tensor Operator(params long[] shape)
        {
            SeedSource seedSource = new SeedSource();
            if (Seed.HasValue)
                seedSource = new SeedSource(Seed.Value);

            Tensor tensor = new Tensor(Global.Device, DType.Float32, shape);
            Ops.RandomNormal(tensor, seedSource, Mean, StdDev);
            return tensor;
        }

    }
}
